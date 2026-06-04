"""FastAPI gateway — task intake, health check, metrics, task management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_orchestrator.models.account import Account, AccountState
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES, TaskRequirements
from ai_orchestrator.models.lease import Lease
from ai_orchestrator.models.task import Task, TaskPriority, TaskStatus, TaskType
from ai_orchestrator.orchestrator.lease_manager import LeaseManager
from ai_orchestrator.orchestrator.provider_router import ProviderRouter
from ai_orchestrator.orchestrator.resource_scheduler import ResourceScheduler, SystemResources, WatermarkLevel
from ai_orchestrator.orchestrator.workflow_engine import WorkflowEngine, WorkflowState

app = FastAPI(title="AI Orchestrator", version="0.1.0")

# ── Singleton state ──────────────────────────────────────────────
lease_manager = LeaseManager()
provider_router = ProviderRouter()
resource_scheduler = ResourceScheduler(configured_max_agents=10)
workflow_engine = WorkflowEngine()
_active_tasks: dict[str, Task] = {}


# ── Request / Response schemas ───────────────────────────────────

class SubmitTaskRequest(BaseModel):
    prompt: str
    task_type: TaskType = TaskType.INTERACTIVE
    priority: TaskPriority = TaskPriority.NORMAL
    context_length: int = 4_096

class TaskResponse(BaseModel):
    id: str
    status: TaskStatus
    prompt: str
    priority: TaskPriority
    task_type: TaskType
    current_step: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

class AccountResponse(BaseModel):
    id: str
    provider: str
    state: str
    health_score: float
    consecutive_failures: int
    context_limit: int
    is_available: bool

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    watermark_level: str
    memory_available_gb: float
    cpu_percent: float
    active_tasks: int
    active_leases: int
    registered_accounts: int

class LeaseResponse(BaseModel):
    id: str
    account_id: str
    task_id: str
    agent_id: str
    state: str
    acquired_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_alive: bool


# ── Startup / shutdown ───────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    """Register default provider accounts on startup."""
    default_accounts = [
        Account(id="openai:prod-01", provider="chatgpt_api", state=AccountState.IDLE,
                context_limit=32_768, rate_limit_rpm=60),
        Account(id="openai:prod-02", provider="chatgpt_api", state=AccountState.IDLE,
                context_limit=32_768, rate_limit_rpm=60),
        Account(id="qwen:prod-01", provider="qwen", state=AccountState.IDLE,
                context_limit=131_072, rate_limit_rpm=100),
        Account(id="deepseek:prod-01", provider="deepseek", state=AccountState.IDLE,
                context_limit=1_000_000, rate_limit_rpm=120),
        Account(id="kimi:prod-01", provider="kimi", state=AccountState.IDLE,
                context_limit=128_000, rate_limit_rpm=50),
        Account(id="local:dev-01", provider="local_llm", state=AccountState.IDLE,
                context_limit=256_000, rate_limit_rpm=30),
    ]
    lease_manager.register_accounts(default_accounts)
    app.state.start_time = datetime.now(timezone.utc)


# ── System resources helper ──────────────────────────────────────

def _get_system_resources() -> SystemResources:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    return SystemResources(
        total_ram_gb=mem.total / (1024**3),
        available_ram_gb=mem.available / (1024**3),
        total_cores=psutil.cpu_count(logical=True) or 4,
        available_cores=max(0, (psutil.cpu_count(logical=True) or 4) * (100 - cpu) / 100),
        memory_usage_percent=mem.percent,
        cpu_usage_percent=cpu,
    )


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check with system resource snapshot."""
    resources = _get_system_resources()
    uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()
    wl = resource_scheduler.get_watermark_level(resources.available_ram_gb)
    return HealthResponse(
        status="ok" if wl < WatermarkLevel.CRITICAL else "degraded",
        version="0.1.0",
        uptime_seconds=uptime,
        watermark_level=wl.name,
        memory_available_gb=round(resources.available_ram_gb, 1),
        cpu_percent=round(resources.cpu_usage_percent, 1),
        active_tasks=len([t for t in _active_tasks.values() if not t.is_terminal]),
        active_leases=len(lease_manager.get_active_leases()),
        registered_accounts=len(lease_manager.list_accounts()),
    )


@app.post("/tasks", response_model=TaskResponse, status_code=201)
async def submit_task(req: SubmitTaskRequest) -> TaskResponse:
    """Submit a new task for orchestration."""
    resources = _get_system_resources()
    if not resource_scheduler.can_accept_task(resources, req.priority):
        raise HTTPException(
            status_code=503,
            detail=f"System at watermark {resource_scheduler.get_watermark_level(resources.available_ram_gb).name} — cannot accept new tasks",
        )

    task = Task(prompt=req.prompt, priority=req.priority, type=req.task_type)
    _active_tasks[task.id] = task

    # Start workflow
    await workflow_engine.start_task(task)
    plan = await workflow_engine.plan_task(task, req.prompt)
    task.current_step = plan.steps[0] if plan.steps else ""

    return _task_to_response(task)


@app.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(status: Optional[TaskStatus] = None) -> list[TaskResponse]:
    """List all tasks, optionally filtered by status."""
    tasks = _active_tasks.values()
    if status:
        tasks = [t for t in tasks if t.status == status]
    return [_task_to_response(t) for t in tasks]


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Get a single task by ID."""
    task = _active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)


@app.post("/tasks/{task_id}/execute", response_model=TaskResponse)
async def execute_task_step(task_id: str, step_name: str = "", agent_type: str = "executor") -> TaskResponse:
    """Execute the next step in a task's workflow."""
    task = _active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.is_terminal:
        raise HTTPException(status_code=400, detail="Task is already in a terminal state")

    step = step_name or task.current_step
    if not step:
        raise HTTPException(status_code=400, detail="No step to execute — set step_name or task.current_step")

    result = await workflow_engine.execute_step(task, step, agent_type)
    if not result:
        raise HTTPException(status_code=409, detail=f"Execution halted: {task.error_message}")

    return _task_to_response(task)


@app.get("/accounts", response_model=list[AccountResponse])
async def list_accounts(provider: Optional[str] = None, state: Optional[AccountState] = None) -> list[AccountResponse]:
    """List registered accounts with current state."""
    accounts = lease_manager.list_accounts(provider=provider, state=state)
    return [_account_to_response(a) for a in accounts]


@app.get("/leases", response_model=list[LeaseResponse])
async def list_leases() -> list[LeaseResponse]:
    """List all active leases."""
    return [_lease_to_response(l) for l in lease_manager.get_active_leases()]


@app.post("/leases", response_model=LeaseResponse)
async def request_lease(task_id: str, agent_id: str, provider: Optional[str] = None) -> LeaseResponse:
    """Request a lease for an agent-task pair."""
    try:
        lease = lease_manager.request_lease(task_id=task_id, agent_id=agent_id, preferred_provider=provider)
        return _lease_to_response(lease)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/leases/{lease_id}/heartbeat")
async def heartbeat_lease(lease_id: str) -> dict:
    """Heartbeat a lease to keep it alive."""
    ok = lease_manager.heartbeat(lease_id)
    return {"lease_id": lease_id, "alive": ok}


@app.delete("/leases/{lease_id}")
async def release_lease(lease_id: str) -> dict:
    """Release a lease back to the pool."""
    account = lease_manager.release_lease(lease_id)
    return {"lease_id": lease_id, "account_released": account.id if account else None}


@app.post("/tasks/{task_id}/halt")
async def halt_task(task_id: str, reason: str = "manually halted") -> TaskResponse:
    """Halt a running task."""
    task = _active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    await workflow_engine.halt_task(task, reason)
    return _task_to_response(task)


@app.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str) -> TaskResponse:
    """Resume a halted task."""
    task = _active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    ok = await workflow_engine.resume_task(task)
    if not ok:
        raise HTTPException(status_code=400, detail="Task cannot be resumed (not halted)")
    return _task_to_response(task)


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Runtime metrics for Prometheus scraping."""
    resources = _get_system_resources()
    return {
        "active_tasks": len([t for t in _active_tasks.values() if not t.is_terminal]),
        "active_leases": len(lease_manager.get_active_leases()),
        "registered_accounts": len(lease_manager.list_accounts()),
        "memory_available_gb": round(resources.available_ram_gb, 1),
        "memory_percent": resources.memory_usage_percent,
        "cpu_percent": resources.cpu_usage_percent,
        "watermark": resource_scheduler.get_watermark_level(resources.available_ram_gb).name,
        "pool_stats": lease_manager.get_pool_stats(),
    }


@app.get("/providers")
async def list_providers() -> dict[str, Any]:
    """List configured provider profiles."""
    return {name: {
        "transport": p.transport,
        "context_limit": p.context_limit,
        "supports_streaming": p.supports_streaming,
        "supports_tools": p.supports_tools,
        "capabilities": p.capabilities.model_dump(),
    } for name, p in PROVIDER_PROFILES.items()}


# ── Response helpers ─────────────────────────────────────────────

def _task_to_response(task: Task) -> TaskResponse:
    return TaskResponse(
        id=task.id,
        status=task.status,
        prompt=task.prompt,
        priority=task.priority,
        task_type=task.type,
        current_step=task.current_step,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
    )

def _account_to_response(account: Account) -> AccountResponse:
    return AccountResponse(
        id=account.id,
        provider=account.provider,
        state=account.state.value if hasattr(account.state, 'value') else account.state,
        health_score=account.health_score,
        consecutive_failures=account.consecutive_failures,
        context_limit=account.context_limit,
        is_available=account.is_available,
    )

def _lease_to_response(lease: Lease) -> LeaseResponse:
    return LeaseResponse(
        id=lease.id,
        account_id=lease.account_id,
        task_id=lease.task_id,
        agent_id=lease.agent_id,
        state=lease.state.value if hasattr(lease.state, 'value') else lease.state,
        acquired_at=lease.acquired_at,
        expires_at=lease.expires_at,
        is_alive=lease.is_alive,
    )
