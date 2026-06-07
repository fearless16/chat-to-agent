"""FastAPI gateway — task intake, health check, metrics, task management."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.adapters.chatgpt_ui import ChatGPTUIAdapter
from ai_orchestrator.adapters.deepseek_ui import DeepSeekUIAdapter
from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter
from ai_orchestrator.adapters.kimi_ui import KimiUIAdapter
from ai_orchestrator.adapters.local_llm import LocalLLMAdapter
from ai_orchestrator.adapters.minimax_ui import MiniMaxUIAdapter
from ai_orchestrator.adapters.qwen_ui import QwenUIAdapter
from ai_orchestrator.adapters.xiaomimimo_ui import XiaomiMiMoUIAdapter
from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter
from ai_orchestrator.models.account import Account, AccountState
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES, TaskRequirements
from ai_orchestrator.models.lease import Lease
from ai_orchestrator.models.task import Task, TaskPriority, TaskStatus, TaskType
from ai_orchestrator.orchestrator.lease_manager import LeaseManager
from ai_orchestrator.orchestrator.provider_router import ProviderRouter
from ai_orchestrator.orchestrator.resource_scheduler import (
    ResourceScheduler,
    SystemResources,
    WatermarkLevel,
)
from ai_orchestrator.orchestrator.workflow_engine import WorkflowEngine

app = FastAPI(title="AI Orchestrator", version="0.1.0")

# ── OpenAI-compatible API ────────────────────────────────────────
from ai_orchestrator.orchestrator.openai_compat import router as openai_router

app.include_router(openai_router)

from ai_orchestrator.orchestrator.provider_health import health_tracker

# ── Singleton state ──────────────────────────────────────────────
lease_manager = LeaseManager()
provider_router = ProviderRouter()
resource_scheduler = ResourceScheduler(configured_max_agents=10)
workflow_engine = WorkflowEngine()
_active_tasks: dict[str, Task] = {}
_workspaces: dict[str, FileWorkspace] = {}  # task_id → workspace

# Lazy-loaded singletons (created on first use)
_sandbox: Sandbox | None = None
_test_runner: TestRunner | None = None
_runtime_loop: RuntimeLoop | None = None
_default_workspace_root: Path | None = None


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
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

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
    acquired_at: datetime | None = None
    expires_at: datetime | None = None
    is_alive: bool

class WorkspaceResponse(BaseModel):
    task_id: str
    workspace_root: str
    git_initialized: bool

class RunLoopResponse(BaseModel):
    task_id: str
    success: bool
    iterations_used: int
    total_duration_ms: float
    passed: int
    failed: int
    total_tests: int


class ChatRequest(BaseModel):
    prompt: str
    provider: str | None = None
    providers: list[str] | None = None
    parallel: bool = False
    mock_mode: bool = True
    context: list[dict] | None = None


class ChatResult(BaseModel):
    provider: str
    success: bool
    content: str
    model: str
    latency_ms: float
    error: str | None = None


class ChatResponse(BaseModel):
    prompt: str
    results: list[ChatResult]
    total_latency_ms: float


_PROVIDER_CLASS_MAP: dict[str, type[ProviderAdapter]] = {
    "chatgpt_ui": ChatGPTUIAdapter,
    "deepseek_ui": DeepSeekUIAdapter,
    "kimi_ui": KimiUIAdapter,
    "qwen_ui": QwenUIAdapter,
    "z_ai_ui": ZAIUIAdapter,
    "xiaomimimo_ui": XiaomiMiMoUIAdapter,
    "minimax_ui": MiniMaxUIAdapter,
    "local_llm": LocalLLMAdapter,
}


def _load_auth_for(provider: str) -> dict | None:
    auth_path = Path(f"profiles/{provider}_auth.json")
    if auth_path.exists():
        import json
        with auth_path.open() as fh:
            return json.load(fh)
    cookie_path = Path(f"profiles/{provider}_cookies.txt")
    if cookie_path.exists():
        from ai_orchestrator.adapters.cookie_to_storage_state import (
            netscape_cookies_to_storage_state,
        )
        return netscape_cookies_to_storage_state(cookie_path)
    return None


# Map of provider → persistent browser profile directory on disk.
_PERSISTENT_PROFILE_MAP: dict[str, str] = {
    "chatgpt_ui": "chatgpt_browser_profile",
    "qwen_ui": "qwen_browser_profile",
}


# Map of provider → browser channel to use.
# Chromium is required for CDP support (HMM engine, SSE capture, traffic classifier).
# Firefox is only used where Cloudflare blocks Chromium automation (e.g. ChatGPT).
_PROVIDER_CHANNEL_MAP: dict[str, str] = {
    "chatgpt_ui": "firefox",    # Cloudflare blocks Chromium automation
    # All others default to "chromium" for full CDP support.
}


def _build_adapter(
    provider: str,
    mock_mode: bool,
) -> ProviderAdapter:
    """Construct an adapter for *provider* with auth wired in from disk.

    For browser providers in real mode:
      - Uses a persistent browser profile if one exists on disk (preserves
        full login session state including localStorage, IndexedDB, etc.)
      - Falls back to storage_state (cookies) when no profile directory exists
      - Runs ``headless=False`` so chat UIs don't detect and block automation
      - Uses Chromium by default for CDP support; Firefox only for ChatGPT
        (Cloudflare evasion)
    """
    cls = _PROVIDER_CLASS_MAP.get(provider)
    if cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider {provider!r}; available: {sorted(_PROVIDER_CLASS_MAP)}",
        )
    if cls is LocalLLMAdapter:
        return cls(mock_mode=mock_mode)

    # In real mode, prefer persistent profile → then cookies.
    # When a persistent browser profile exists, it contains live cookies
    # from the user's actual browser session.  Do NOT overwrite those
    # with (potentially stale) cookie-file cookies.
    persistent_profile = None
    storage_state = None
    if not mock_mode:
        profile_dir = _PERSISTENT_PROFILE_MAP.get(provider)
        if profile_dir and Path(profile_dir).exists():
            persistent_profile = profile_dir
        else:
            storage_state = _load_auth_for(provider)

    channel = _PROVIDER_CHANNEL_MAP.get(provider, "chromium")

    return cls(
        mock_mode=mock_mode,
        headless=False if not mock_mode else True,
        stealth=True,
        timeout_ms=120_000,
        storage_state=storage_state,
        persistent_profile=persistent_profile,
        channel=channel,
    )


def _to_chat_result(resp: ProviderResponse, provider: str) -> ChatResult:
    return ChatResult(
        provider=provider,
        success=resp.success,
        content=resp.content,
        model=resp.model,
        latency_ms=resp.latency_ms,
        error=resp.error,
    )


# ── Startup / shutdown ───────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    """Register default provider accounts on startup."""
    default_accounts = [
        Account(id="chatgpt:ui-01", provider="chatgpt_ui", state=AccountState.IDLE,
                context_limit=32_768, rate_limit_rpm=20),
        Account(id="qwen:ui-01", provider="qwen_ui", state=AccountState.IDLE,
                context_limit=131_072, rate_limit_rpm=20),
        Account(id="local:dev-01", provider="local_llm", state=AccountState.IDLE,
                context_limit=256_000, rate_limit_rpm=30),
        Account(id="deepseek:ui-01", provider="deepseek_ui", state=AccountState.IDLE,
                context_limit=1_048_576, rate_limit_rpm=20),
        Account(id="zai:ui-01", provider="z_ai_ui", state=AccountState.IDLE,
                context_limit=131_072, rate_limit_rpm=20),
        Account(id="xiaomimimo:ui-01", provider="xiaomimimo_ui", state=AccountState.IDLE,
                context_limit=131_072, rate_limit_rpm=20),
        Account(id="minimax:ui-01", provider="minimax_ui", state=AccountState.IDLE,
                context_limit=131_072, rate_limit_rpm=20),
        Account(id="kimi:ui-01", provider="kimi_ui", state=AccountState.IDLE,
                context_limit=128_000, rate_limit_rpm=20),
    ]
    lease_manager.register_accounts(default_accounts)
    app.state.start_time = datetime.now(UTC)


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
    uptime = (datetime.now(UTC) - app.state.start_time).total_seconds()
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

    # Pre-select a provider via the capability router so the lease
    # request that follows is scored against actual provider fit, not
    # the previous "highest health_score wins" heuristic.  If no account
    # satisfies the context-length / capability requirements the
    # ``request_lease`` call later will surface the problem.
    preferred = _select_preferred_provider(req)
    task.assigned_agent = "planner"
    if preferred is not None:
        task.assigned_account_id = preferred

    # Start workflow
    await workflow_engine.start_task(task)
    plan = await workflow_engine.plan_task(task, req.prompt)
    task.current_step = plan.steps[0] if plan.steps else ""

    return _task_to_response(task)


def _select_preferred_provider(req: SubmitTaskRequest) -> str | None:
    """Use the :class:`ProviderRouter` to pick a preferred provider for *req*.

    Returns the ``provider_name`` of the highest-scoring account (or
    ``None`` if no account can satisfy the context-length requirement).
    """
    requirements = TaskRequirements(
        context_length=req.context_length,
        requires_reasoning=True,
        requires_coding=False,
    )
    accounts = lease_manager.list_accounts()
    if not accounts:
        return None
    selected = provider_router.select_account(accounts, requirements)
    return selected.provider if selected is not None else None


@app.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(status: TaskStatus | None = None) -> list[TaskResponse]:
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
async def list_accounts(provider: str | None = None, state: AccountState | None = None) -> list[AccountResponse]:
    """List registered accounts with current state."""
    accounts = lease_manager.list_accounts(provider=provider, state=state)
    return [_account_to_response(a) for a in accounts]


@app.get("/leases", response_model=list[LeaseResponse])
async def list_leases() -> list[LeaseResponse]:
    """List all active leases."""
    return [_lease_to_response(l) for l in lease_manager.get_active_leases()]


@app.post("/leases", response_model=LeaseResponse)
async def request_lease(task_id: str, agent_id: str, provider: str | None = None) -> LeaseResponse:
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


# ── Chat endpoints (real LLM fan-out) ────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send *prompt* to one provider, or fan out to many in parallel.

    Body fields:
        provider:   single-provider mode (returns one result)
        providers:  multi-provider mode (returns one result per provider)
        parallel:   when ``providers`` is given, run them concurrently
                    (default ``True``); ignored in single-provider mode
        mock_mode:  return canned responses without launching a browser
        context:    optional chat history passed to the adapter
    """
    t0 = asyncio.get_event_loop().time()

    if req.providers:
        provider_names = req.providers
        do_parallel = req.parallel
    elif req.provider:
        provider_names = [req.provider]
        do_parallel = False
    else:
        raise HTTPException(
            status_code=400,
            detail="specify either 'provider' or 'providers'",
        )

    adapters: list[ProviderAdapter] = []
    for name in provider_names:
        try:
            adapters.append(_build_adapter(name, req.mock_mode))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"{name}: {exc}") from exc

    if do_parallel and len(adapters) > 1:
        responses = await EngineUIAdapter.fan_out(adapters, req.prompt, req.context)
    else:
        responses = []
        for adapter in adapters:
            try:
                responses.append(await adapter.send(req.prompt, req.context))
            except Exception as exc:
                responses.append(
                    ProviderResponse(
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        model=adapter.provider_name,
                    )
                )
            finally:
                import contextlib
                with contextlib.suppress(Exception):
                    await adapter.close()

    results = [_to_chat_result(r, n) for r, n in zip(responses, provider_names)]

    # Record health metrics for each result.
    for r, name in zip(responses, provider_names):
        if r.success:
            health_tracker.record_success(name, r.latency_ms)
            health_tracker.record_auth_success(name)
        else:
            error_str = r.error or ""
            health_tracker.record_failure(name, error_str)
            if "AuthenticationError" in error_str:
                health_tracker.record_auth_failure(name)
            elif "CloudflareBlockError" in error_str:
                health_tracker.record_captcha(name)

    return ChatResponse(
        prompt=req.prompt,
        results=results,
        total_latency_ms=round((asyncio.get_event_loop().time() - t0) * 1000, 1),
    )


@app.get("/provider-health")
async def provider_health() -> dict:
    """Provider health dashboard — per-provider metrics.

    Shows: success_rate, auth_rate, avg_latency, captcha_count,
    popup_count, recovery_count, status (healthy/degraded/unhealthy),
    cooldown state.
    """
    from ai_orchestrator.adapters.recovery_engine import recovery_engine

    dashboard = health_tracker.get_dashboard()
    dashboard["cooldowns"] = recovery_engine.get_all_cooldowns()
    return dashboard


# ── Workspace & runtime endpoints ────────────────────────────────

def _get_runtime() -> tuple:
    """Lazy-init sandbox, test runner, and runtime loop."""
    global _sandbox, _test_runner, _runtime_loop, _default_workspace_root
    from ai_orchestrator.runtime.loop import RuntimeLoop
    from ai_orchestrator.security.sandbox import Sandbox
    from ai_orchestrator.testrunner.runner import TestRunner
    from ai_orchestrator.workspace.manager import DEFAULT_WORKSPACE_ROOT
    if _sandbox is None:
        _sandbox = Sandbox()
    if _test_runner is None:
        _test_runner = TestRunner(_sandbox)
    if _runtime_loop is None:
        _runtime_loop = RuntimeLoop(_sandbox, test_runner=_test_runner)
    if _default_workspace_root is None:
        _default_workspace_root = DEFAULT_WORKSPACE_ROOT
    return _sandbox, _test_runner, _runtime_loop, _default_workspace_root


@app.post("/tasks/{task_id}/workspace", response_model=WorkspaceResponse)
async def create_task_workspace(task_id: str) -> WorkspaceResponse:
    """Create a workspace for an existing task.

    The workspace is created under the default workspace root
    (``<cwd>/workspaces/<task_id>/``) and optionally initialized with git.
    Idempotent — repeated calls return the same workspace.
    """
    if task_id not in _active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if task_id in _workspaces:
        ws = _workspaces[task_id]
    else:
        from ai_orchestrator.workspace.manager import FileWorkspace
        _, _, _, root = _get_runtime()
        ws = FileWorkspace.for_task(task_id, root=root)
        _workspaces[task_id] = ws

    # Initialize git for version tracking
    from ai_orchestrator.workspace.git import GitWorkspace
    git_ws = GitWorkspace(ws)
    try:
        await git_ws.init()
        git_init = True
    except Exception:
        git_init = False

    return WorkspaceResponse(
        task_id=task_id,
        workspace_root=str(ws.workspace_root),
        git_initialized=git_init,
    )


@app.post("/tasks/{task_id}/run-loop", response_model=RunLoopResponse)
async def run_task_loop(
    task_id: str,
    max_iterations: int = 5,
) -> RunLoopResponse:
    """Run the full fix-analysis-test loop for a task.

    Requires a workspace (call ``POST /tasks/{id}/workspace`` first).
    Code must already exist in the workspace (e.g. written by an agent).
    """
    if task_id not in _active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    ws = _workspaces.get(task_id)
    if not ws:
        raise HTTPException(
            status_code=400,
            detail="No workspace for this task — call POST /tasks/{id}/workspace first",
        )

    sandbox, test_runner, runtime_loop, _ = _get_runtime()

    # Check sandbox availability before starting
    if not await sandbox.check_sandbox_available():
        raise HTTPException(status_code=503, detail="Sandbox unavailable — no Python interpreter found")

    try:
        result = await runtime_loop.run(
            ws,
            max_iterations=max_iterations,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Runtime loop failed: {exc}")

    return RunLoopResponse(
        task_id=task_id,
        success=result.success,
        iterations_used=result.iterations_used,
        total_duration_ms=round(result.total_duration_ms, 2),
        passed=result.test_run.passed if result.test_run else 0,
        failed=result.test_run.failed if result.test_run else 0,
        total_tests=result.test_run.total if result.test_run else 0,
    )


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
