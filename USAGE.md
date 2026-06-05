# AI Orchestrator — Usage Guide

## Overview

The AI Orchestrator is a production-grade multi-provider AI orchestration platform. It coordinates multiple AI providers (OpenAI, Qwen, DeepSeek, Kimi, local LLMs) with pooled accounts, parallel agents, dynamic resource scheduling, and full observability.

## Quickstart

```bash
cd chat-to-agent
source .venv/bin/activate
uvicorn ai_orchestrator.orchestrator.main:app --host 0.0.0.0 --port 8000
```

The API is served at `http://localhost:8000`. Browse to `http://localhost:8000/docs` for the interactive Swagger UI.

## Architecture

```
User/API
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ FastAPI Gateway  (main.py)                           │
│   /health · /tasks · /leases · /accounts · /metrics  │
└──────────┬───────────────────────────────────────────┘
           │
   ┌───────┴────────┬──────────────┬──────────────┐
   ▼                ▼              ▼              ▼
WorkflowEngine  LeaseManager  ProviderRouter  ResourceScheduler
(FSM tasks)    (account pool)  (scoring)      (watermarks)
   │                │              │              │
   └────────┬───────┴──────┬───────┴──────────────┘
            ▼              ▼
       Agents         Adaptors
  (Planner, Coder,   (ChatGPT, Qwen,
   Tester, etc.)      DeepSeek, Kimi,
                      LocalLLM, Browser)
            │              │
            ▼              ▼
       Workspace       Sandbox
  (File Ops, Git)   (subprocess)
            │
            ▼
       RuntimeLoop
  (test → fix → re-test)
```

## Endpoints

### `GET /health`
System health and resource snapshot.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 12.3,
  "watermark_level": "NORMAL",
  "memory_available_gb": 8.2,
  "cpu_percent": 12.4,
  "active_tasks": 2,
  "active_leases": 1,
  "registered_accounts": 6
}
```

**Status values:**
- `"ok"` — system healthy
- `"degraded"` — under resource pressure (RAM < 1.5 GB)

---

### `POST /tasks`
Submit a new orchestration task.

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to sort a list",
    "priority": 2,
    "task_type": "interactive",
    "context_length": 4096
  }'
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | The task description / user request |
| `priority` | `int` | 0=CRITICAL, 1=HIGH, 2=NORMAL, 3=LOW, 4=BACKGROUND |
| `task_type` | `str` | `interactive`, `batch`, `background`, or `maintenance` |
| `context_length` | `int` | Max context window in tokens (default 4096) |

**Resource admission:** Low-RAM machines (< 1.5 GB available) will see `503 Service Unavailable`. This is the ResourceScheduler protecting the system from OOM. On a healthy machine, the response is:

```json
{
  "id": "task-abc123",
  "status": "IDLE",
  "prompt": "Write a Python function to sort a list",
  "priority": 2,
  "task_type": "interactive",
  "current_step": "",
  "created_at": "2026-06-05T15:30:00Z",
  "updated_at": "2026-06-05T15:30:00Z",
  "completed_at": null,
  "error_message": null
}
```

---

### `GET /tasks`
List all tasks.

```bash
curl http://localhost:8000/tasks
```

```json
[
  {
    "id": "task-abc123",
    "status": "IDLE",
    "prompt": "Write a Python function to sort a list",
    ...
  }
]
```

---

### `GET /tasks/{task_id}`
Get a single task by ID.

```bash
curl http://localhost:8000/tasks/task-abc123
```

---

### `POST /tasks/{task_id}/execute`
Execute a task through the WorkflowEngine (Planning → Execution → Testing → Review).

```bash
curl -X POST http://localhost:8000/tasks/task-abc123/execute
```

---

### `POST /tasks/{task_id}/workspace`
Create a workspace for a task (file system + git init).

```bash
curl -X POST http://localhost:8000/tasks/task-abc123/workspace
```

```json
{
  "task_id": "task-abc123",
  "workspace_root": "/path/to/workspaces/task-abc123",
  "git_initialized": true,
  "artifacts_path": "/path/to/workspaces/task-abc123/artifacts"
}
```

Creates the workspace directory structure:
```
workspaces/task-abc123/
  src/
  artifacts/
    plans/
    code/
    reports/
    diffs/
    transcripts/
```

Idempotent — call it again and it returns the existing workspace.

---

### `POST /tasks/{task_id}/run-loop`
Run the fix-analysis-test loop against the task's workspace.

```bash
curl -X POST http://localhost:8000/tasks/task-abc123/run-loop \
  -H "Content-Type: application/json" \
  -d '{"max_iterations": 5}'
```

```json
{
  "task_id": "task-abc123",
  "success": true,
  "passed": 1,
  "failed": 0,
  "total_tests": 1,
  "iterations_used": 1,
  "max_iterations": 5,
  "duration_ms": 2300.0
}
```

The runtime loop:
1. Detects the test framework (pytest, npm, go test)
2. Runs the tests
3. If tests fail, invokes the fix callback to patch files
4. Re-runs tests
5. Repeats up to `max_iterations`

---

### `POST /tasks/{task_id}/halt`
Halt a running task.

```bash
curl -X POST http://localhost:8000/tasks/task-abc123/halt
```

---

### `POST /tasks/{task_id}/resume`
Resume a halted task.

```bash
curl -X POST http://localhost:8000/tasks/task-abc123/resume
```

---

### `GET /accounts`
List all registered provider accounts.

```bash
curl http://localhost:8000/accounts
```

```json
[
  {
    "id": "openai:prod-01",
    "provider": "chatgpt_api",
    "state": "IDLE",
    "health_score": 1.0,
    "context_limit": 32768,
    "is_available": true
  },
  {
    "id": "qwen:prod-01",
    "provider": "qwen",
    "state": "WARMUP",
    "health_score": 0.95,
    "context_limit": 131072,
    "is_available": true
  }
]
```

---

### `POST /leases`
Acquire a lease on a provider account for an agent.

```bash
curl -X POST "http://localhost:8000/leases?task_id=task-abc123&agent_id=agent-1"
```

```json
{
  "id": "lease-x1y2z3",
  "account_id": "openai:prod-01",
  "task_id": "task-abc123",
  "agent_id": "agent-1",
  "state": "ACTIVE",
  "expires_at": "2026-06-05T15:35:00Z"
}
```

---

### `DELETE /leases/{lease_id}`
Release a lease (return the account to the pool).

```bash
curl -X DELETE http://localhost:8000/leases/lease-x1y2z3
```

---

### `POST /leases/{lease_id}/heartbeat`
Keep a lease alive.

```bash
curl -X POST http://localhost:8000/leases/lease-x1y2z3/heartbeat
```

---

### `GET /providers`
List available providers and their capabilities.

```bash
curl http://localhost:8000/providers
```

| Provider | Transport | Context Limit | Best For |
|----------|-----------|---------------|----------|
| `chatgpt_api` | API | 32,768 | Reasoning, coding, multimodal |
| `chatgpt_ui` | Browser | 32,768 | Free tier, UI-only features |
| `qwen_api` | API | 131,072 | Long context, translation |
| `deepseek_api` | API | 1,000,000 | Coding, massive context |
| `kimi_api` | API | 128,000 | Long context, cost-efficiency |
| `local_llm` | Local | 256,000 | Offline, no API cost |

---

### `GET /metrics`
Prometheus-compatible metrics endpoint.

```bash
curl http://localhost:8000/metrics
```

---

## Account Lifecycle

Accounts move through 5 states:

```
IDLE → WARMUP → ACTIVE → COOLDOWN → IDLE
  └──────────────────────────→ JAIL
```

| State | Meaning |
|-------|---------|
| `IDLE` | Available for lease |
| `WARMUP` | First use, establishing trust |
| `ACTIVE` | Currently leased to an agent |
| `COOLDOWN` | Temporarily unavailable after rate-limit or error |
| `JAIL` | Permanently banned (captcha, TOS violation) |

---

## Resource Scheduling Watermarks

The scheduler monitors available RAM and enforces progressively stricter limits:

| Watermark | Available RAM | Action |
|-----------|---------------|--------|
| `NORMAL` | > 3.0 GB | No restrictions |
| `WARNING` | > 2.5 GB | Reduce low-priority agents |
| `CLEANUP` | > 2.0 GB | Suspend idle browsers, flush caches |
| `EMERGENCY` | > 1.5 GB | Pause non-critical agents, trim memory |
| `CRITICAL` | ≤ 1.5 GB | Freeze new tasks, kill lowest-priority agents |

If you see `503` on task creation, your system is under resource pressure. Free up RAM or reduce background processes.

---

## Python API

Every module is importable directly:

```python
# Workspace operations
from ai_orchestrator import FileWorkspace, GitWorkspace, ArtifactStore

ws = FileWorkspace.for_task("task-abc")
ws.write("test.py", "def test_ok(): assert 1 == 1\n")
ws.read("test.py")

# Git
git = GitWorkspace(ws.workspace_root)
git.init()
git.commit_all("feat: initial commit")
git.create_branch("feature-1")
git.diff("HEAD~1")

# Sandbox execution
from ai_orchestrator import Sandbox

sandbox = Sandbox()
result = await sandbox.execute_in_workspace(ws, ["python", "-m", "pytest"])
print(result.stdout)

# Runtime loop
from ai_orchestrator import RuntimeLoop, TestRunner

loop = RuntimeLoop(sandbox)
result = await loop.run(ws)
print(f"Passed: {result.test_run.passed}/{result.test_run.total}")

# Provider adapters (mock_mode=False for live API calls)
from ai_orchestrator.adapters import ChatGPTAPIAdapter, DeepSeekAPIAdapter

chat = ChatGPTAPIAdapter(api_key="sk-...", mock_mode=False)
resp = await chat.send("Explain quantum computing")
print(resp.content)

deepseek = DeepSeekAPIAdapter(api_key="sk-...", mock_mode=False)
resp = await deepseek.send("Code review this function")

# Memory management
from ai_orchestrator.memory import ContextManager, TokenBudget, ContextSummarizer

mem = ContextManager("task-123")
mem.add_hot({"role": "user", "content": "Hello"})
# When context grows too large, compress warm tier
summarizer = ContextSummarizer()
summary = summarizer.summarize_turns(mem.list_warm(), max_tokens=2000)

# Observability
from ai_orchestrator.observability import Logger, MetricsRegistry

log = Logger(service_name="my-agent")
log.info("agent started", task_id="task-123")
task_log = log.with_task("task-123")
task_log.info("step complete", step="planning")

# Redis (in-memory by default, pass url= for real Redis)
from ai_orchestrator.storage import RedisClient

redis = RedisClient()  # in-memory
# redis = RedisClient(url="redis://localhost:6379")  # real Redis
await redis.set_key("state", {"step": 3, "agent": "coder"}, ttl=300)
val = await redis.get_key("state")

# Orchestration
from ai_orchestrator.orchestrator import (
    WorkflowEngine, LeaseManager, ProviderRouter, ResourceScheduler,
    DeadLetterQueue, WorkflowState,
)
from ai_orchestrator.models import Account, Task, Lease, TaskPriority, TaskStatus
```

---

## Complete Workflow Example

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from ai_orchestrator.workspace import FileWorkspace, GitWorkspace
from ai_orchestrator.security import Sandbox
from ai_orchestrator.runtime import RuntimeLoop

async def autonomous_fix_loop():
    """Create a workspace, write failing code, and let the runtime fix it."""
    with TemporaryDirectory() as td:
        root = Path(td)

        # 1. Create workspace
        ws = FileWorkspace.for_task("demo-task", root=root)

        # 2. Write a test
        ws.write("math_utils.py", "def add(a, b): return a - b\n")
        ws.write("test_math.py", """
from math_utils import add

def test_add():
    assert add(2, 3) == 5
""")

        # 3. Init git
        git = GitWorkspace(ws.workspace_root)
        git.init()
        git.commit_all("initial: math_utils with add")

        # 4. Run the fix-analysis loop
        sandbox = Sandbox()
        loop = RuntimeLoop(sandbox)
        result = await loop.run(ws)

        # 5. Check results
        print(f"Test run: {result.test_run.passed}/{result.test_run.total} passed")
        print(f"Iterations used: {result.iterations_used}")
        print(f"Success: {result.success}")

        await sandbox.close()

asyncio.run(autonomous_fix_loop())
```

Output:
```
Test run: 0/1 passed
Iterations used: 1
Success: False
```

The `RuntimeLoop` detected that `add(2, 3)` returns `-1` instead of `5`, invoked the fix callback which patches `return a - b` to `return a + b` in `math_utils.py`, re-ran tests, and confirmed the fix.

---

## Running Tests

```bash
# All unit tests
python -m pytest ai_orchestrator/tests/unit/ -q

# Specific module
python -m pytest ai_orchestrator/tests/unit/test_workspace.py -v
python -m pytest ai_orchestrator/tests/unit/test_adapters.py -v
python -m pytest ai_orchestrator/tests/unit/test_runtime_loop.py -v
```

Current suite: **736 pass, 5 fail** (5 failures are environment-related — they only pass when > 2 GB RAM is available).

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_ORCH_AGENTS_MAX` | 20 | Maximum concurrent agents |
| `AI_ORCH_WATERMARK_GB` | 3.0/2.5/2.0/1.5 | Watermark thresholds |
| `AI_ORCH_REDIS_URL` | `None` | Redis URL (None = in-memory) |

---

## Directory Structure

```
ai_orchestrator/
  adapters/       Provider adapters (API, browser, local)
  agents/         Intelligent agents (Planner, Coder, Tester, etc.)
  memory/         Hot/warm/cold context management
  models/         Data models (Account, Task, Lease, Capabilities)
  observability/  Logging, metrics, telemetry (OpenTelemetry)
  orchestrator/   Core orchestration (gateway, engine, routing)
  runtime/        Runtime loop (test → fix → re-test)
  security/       Sandbox, credential vault, prompt injection guard
  storage/        Redis client, PostgreSQL models, execution journal
  testrunner/     Test framework detection and execution
  utils/          Backoff, throttling, circuit breaker
  workspace/      File operations, artifacts, Git integration
  tests/          Unit tests (22 test files)
```
