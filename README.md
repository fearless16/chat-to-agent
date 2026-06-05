# chat-to-agent / ai_orchestrator v0.1

Multi-provider AI orchestration gateway — submit tasks, route to the best LLM,
drive a fix-analysis-test loop, manage leases, accounts, and resource watermarks.

```bash
# Quick start
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest ai_orchestrator/tests/unit/ -q    # 745 pass
.venv/bin/uvicorn ai_orchestrator.orchestrator.main:app --port 8000
curl http://127.0.0.1:8000/health
```

---

## What it does

1. **`POST /tasks`** — submit `{prompt, priority, task_type, context_length}`.
2. **Admission control** — `ResourceScheduler` checks RAM watermark against
   `TaskPriority`; rejects (503) when pressure exceeds the priority's threshold.

   | Priority     | Rejected at watermark ≥ | Notes |
   |--------------|-------------------------|-------|
   | `CRITICAL`   | —                       | always admitted |
   | `HIGH`       | `EMERGENCY` (< 2.0 GB free) | |
   | `NORMAL`     | `EMERGENCY` (< 2.0 GB free) | |
   | `LOW`        | `CLEANUP`  (< 2.5 GB free) | |
   | `BACKGROUND` | `WARNING`  (< 3.0 GB free) | |

3. **Provider selection** — `ProviderRouter` scores every account's
   `ProviderCapabilities` against the task's `TaskRequirements` and picks the
   best match (reasoning, coding, translation, multimodality, context length).
4. **Lease acquisition** — `LeaseManager.request_lease` reserves that account
   for `agent_id` on `task_id` with a TTL (300 s, renewable up to 5×).
5. **Workflow execution** — `WorkflowEngine` drives the FSM:

   ```
   IDLE → PLANNING → EXECUTING ─┬→ TESTING ─→ REVIEW ─→ DONE
                                │      │            │
                                │      └→ FIX ──────┘
                                └→ HALTED  (operator pause)
   ```

   `HALTED` is resumable via `POST /tasks/{id}/resume`.  `DONE` and `DLQ` are
   absorbing.
6. **Run loop** (optional) — `POST /tasks/{id}/run-loop` runs the full
   fix-analysis-test cycle: build → test → analyze → fix → repeat until all
   tests pass or the iteration budget is exhausted.
7. **Workspace** — `POST /tasks/{id}/workspace` creates an isolated,
   path-traversal-protected directory for each task, optionally initialized with
   git.
8. **Lease release** — `DELETE /leases/{id}` returns the account to the pool.
   Heartbeat thread expires idle leases on a timer.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────────────┐
│  HTTP client │────▶│  FastAPI gateway  (orchestrator/main.py)             │
└──────────────┘     │  17 endpoints: /health /tasks /accounts /leases      │
                     │  /metrics /providers /workspace /run-loop             │
                     └────────┬──────────────────────────┬───────────────────┘
                              │                          │
                ┌─────────────▼──────────┐    ┌──────────▼──────────────────┐
                │  ResourceScheduler     │    │  WorkflowEngine             │
                │  watermark admission   │    │  FSM + loop detection       │
                │  3.0/2.5/2.0/1.5 GB    │    │  halt / resume              │
                └────────────────────────┘    └──────────┬───────────────────┘
                                                         │
                              ┌──────────────────────────┼────────────────────────────┐
                              ▼                          ▼                            ▼
                       ┌──────────────┐         ┌─────────────────┐         ┌─────────────────┐
                       │  Providers   │         │  LeaseManager   │         │  Agents (7)     │
                       │  7 profiles  │         │  account pool   │         │  planner, coder, │
                       │  API/BROWSER │         │  heartbeat      │         │  tester, review, │
                       │  /LOCAL      │         │  expire         │         │  fixer, executor │
                       └──────┬───────┘         └────────┬────────┘         │  researcher      │
                              │                          │                  └─────────────────┘
                              └──────────┬───────────────┘
                                         ▼
                              ┌────────────────────────┐
                              │  Adapters (7 impls)    │
                              │  ChatGPT API + UI      │
                              │  Qwen API + UI         │
                              │  DeepSeek, Kimi, Local │
                              │  mock_mode, throttle,  │
                              │  circuit breaker       │
                              └────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  Runtime Loop + Test Runner + Workspace + Git layer                           │
│                                                                                │
│  POST /tasks/{id}/workspace  →  FileWorkspace (atomic writes, path-traversal  │
│                                  protected, thread-safe, git-aware)           │
│  POST /tasks/{id}/run-loop   →  Build → Test → Analyze → Fix → Conclude       │
│                                                                                │
│  TestRunner auto-detects pytest/unittest, runs in Sandbox (RLIMIT_AS +        │
│  process-group kill + 1 MiB output cap), returns structured TestRun results.  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Modules

| Module | Files | Description |
|--------|------:|-------------|
| `models/`         | 5 | Pydantic domain models — `Account` (IDLE→WARMUP→ACTIVE→COOLDOWN→JAIL), `Lease` (acquire→renew→release/expire), `Task` with FSM-aware `transition_to`, `ProviderCapabilities` matrix + `CapabilityVector` scoring. |
| `agents/`         | 9 | `BaseAgent` ABC + 7 role agents: `planner`, `researcher`, `coder`, `tester`, `reviewer`, `fixer`, `executor`.  Each has `max_steps=25`, `max_runtime_ms=300_000`. |
| `adapters/`       | 9 | 7 provider-specific adapters + base protocol.  API adapters (ChatGPT, Qwen, DeepSeek, Kimi, Local) use `protected_send` with token-bucket throttle + circuit breaker.  Browser adapters (ChatGPT UI, Qwen UI) use Playwright with persistent profiles, stealth anti-detection, and Cloudflare bypass.  All default to `mock_mode=True`. |
| `orchestrator/`   | 7 | `main.py` (FastAPI gateway, 17 endpoints), `workflow_engine.py` (FSM), `lease_manager.py` (account pool + heartbeats), `provider_router.py` (capability scoring), `resource_scheduler.py` (watermarks + admission), `dlq.py` (idempotent DLQ). |
| `runtime/`        | 3 | `RuntimeLoop` — build → test → analyze → fix → conclude cycle.  `FixResult` type, iteration budget (max iterations + timeout). |
| `testrunner/`     | 2 | `TestRunner` — auto-detects test framework (pytest, unittest), runs in `Sandbox`, returns structured `TestRun(passed, failed, total, output)`. |
| `workspace/`      | 5 | `FileWorkspace` — atomic writes (`os.replace`), path-traversal protection, thread-safe.  `GitWorkspace` — branch, commit, diff, rollback.  `ArtifactStore` — convention layer for plans, code, reports, media. |
| `storage/`        | 4 | `RedisClient` (coroutine-locked mutators, stream support), `PostgresModels` (async UPSERT via `session.merge`), `ExecutionJournal` (per-task + global streams). |
| `memory/`         | 4 | `ContextManager` with HOT/WARM/COLD tiers, `TokenBudget` estimator + `trim_to_limit`, `ContextSummarizer` (extractive). |
| `security/`       | 4 | `Vault` (Fernet keyring with safe `rotate_key`), `Sandbox` (env whitelist, `RLIMIT_AS` preexec, 1 MiB output cap, process-group kill), `PromptGuard` (regex-based injection detection with weighted risk scoring). |
| `observability/`  | 4 | OpenTelemetry tracing (sampled via `TraceIdRatioBased`), Prometheus `MetricsRegistry` (bounded `reason` cardinality), `structlog` JSON logger. |
| `utils/`          | 3 | `TokenBucket` (sync `threading.Lock` — works in threadpool + asyncio), `RetryConfig.compute_delay_ms` (decorrelated jitter), `CircuitBreaker` (3 failures → 60 s recovery, async HALF_OPEN probe). |
| `tests/unit/`     | 23 | 22 test modules + `conftest.py`.  745 tests, no external services required. |

---

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| `GET`    | `/health`                                     | Uptime, watermark level, free RAM, CPU%, active tasks/leases, account count. |
| `GET`    | `/metrics`                                    | Prometheus-formatted metrics + pool stats per provider. |
| `GET`    | `/providers`                                  | Capability matrix of all 7 configured providers. |
| `GET`    | `/accounts[?provider=&state=]`                | Account pool — each shows `state`, `health_score`, `consecutive_failures`. |
| `GET`    | `/leases`                                     | Active leases. |
| `POST`   | `/leases?task_id=&agent_id=&provider=`        | Acquire a lease (503 if no account available). |
| `POST`   | `/leases/{id}/heartbeat`                      | Keep a lease alive. |
| `DELETE` | `/leases/{id}`                                | Release a lease. |
| `GET`    | `/tasks[?status=]`                            | List tasks. |
| `GET`    | `/tasks/{id}`                                 | Single task. |
| `POST`   | `/tasks`                                      | Submit a task. |
| `POST`   | `/tasks/{id}/execute?step_name=&agent_type=`  | Run the next workflow step. |
| `POST`   | `/tasks/{id}/halt?reason=`                    | Pause a task (workflow → HALTED). |
| `POST`   | `/tasks/{id}/resume`                          | Resume a HALTED task. |
| `POST`   | `/tasks/{id}/workspace`                       | Create an isolated workspace for a task. |
| `POST`   | `/tasks/{id}/run-loop?max_iterations=5`       | Run the full fix-analysis-test cycle. |

### TaskPriority

Submit the integer value:

| Constant     | Value |
|--------------|------:|
| `CRITICAL`   | `0`   |
| `HIGH`       | `1`   |
| `NORMAL`     | `2`   |
| `LOW`        | `3`   |
| `BACKGROUND` | `4`   |

---

## Providers

| Key | Transport | Context | Streaming | Tools |
|-----|-----------|--------:|:---------:|:-----:|
| `chatgpt_api` | API | 32 768 | ✓ | ✓ |
| `chatgpt_ui`  | BROWSER | 32 768 | ✗ | ✗ |
| `qwen_api`    | API | 131 072 | ✓ | ✓ |
| `qwen_ui`     | BROWSER | 131 072 | ✗ | ✗ |
| `deepseek_api`| API | 1 000 000 | ✓ | ✓ |
| `kimi_api`    | API | 128 000 | ✓ | ✗ |
| `local_llm`   | LOCAL | 256 000 | ✓ | ✗ |

---

## Browser adapters

Two Playwright-based adapters (`chatgpt_ui` + `qwen_ui`) automate web-based
LLMs that lack public APIs.  Key features:

| Feature | Detail |
|---------|--------|
| **persistent_profile** | Playwright user-data directory — login once with `headless=False`, reuse the session forever.  Bypasses fingerprint-linked auth. |
| **storage_state** | Exported cookies + localStorage JSON.  Works for Qwen; ChatGPT may reject (server-side fingerprint check). |
| **stealth** | Anti-detection: custom UA (Chrome 131 macOS), viewport, `--disable-blink-features=AutomationControlled`. |
| **mock_mode** | `True` by default — returns canned responses, no browser launched.  Set `False` for real interaction. |
| **channel** | `"chromium"` (default), `"chrome"`, or `"msedge"`. |
| **ProseMirror editor** | ChatGPT's `#prompt-textarea` is `contenteditable` — uses `.click()` + `keyboard.type(delay=50)` instead of `.fill()`. |

Auth files (`chatgpt_auth.json`, `qwen_auth.json`) are gitignored and store
cookies + localStorage for each service.

---

## Quick start

```bash
# 1. Install
.venv/bin/pip install -e ".[dev]"
playwright install chromium      # for browser adapters

# 2. Run all 745 tests (no external services needed)
.venv/bin/pytest ai_orchestrator/tests/unit/ -v

# 3. Start the gateway
.venv/bin/uvicorn ai_orchestrator.orchestrator.main:app --host 127.0.0.1 --port 8000

# 4. Submit a task (priority=0 = CRITICAL, bypasses watermarks)
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2?","priority":0,"context_length":4096}'

# 5. Walk through the FSM
TASK_ID=...  # from step 4
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/execute?step_name=implement&agent_type=executor"
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/execute?step_name=test&agent_type=tester"
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/execute?step_name=review&agent_type=reviewer"

# 6. Or run the full fix-analysis-test loop
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/workspace"
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/run-loop"

# 7. Read-only introspection
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/providers
```

The bundled `scripts/run.sh {api|test|test-all|shell}` wraps the same steps.

---

## Docker

```bash
cd docker
docker compose up -d
# Gateway:     http://localhost:8000
# Prometheus:  http://localhost:9091
# Grafana:     http://localhost:3000
# Redis + Postgres + OpenTelemetry collector
```

---

## Test suite

```
745 passed, 1 skipped
├── models         47  (Account, Lease, Task, Capabilities)
├── agents         62  (BaseAgent + 7 role agents)
├── adapters       47  (7 providers + base protocol)
├── orchestrator   89  (LeaseManager, Router, Scheduler, Workflow, DLQ)
├── runtime        13  (RuntimeLoop fix-analysis-test cycle)
├── testrunner     22  (auto-detect, execute, parse test results)
├── workspace      38  (FileWorkspace, GitWorkspace, ArtifactStore)
├── storage        78  (Redis, Postgres, Journal)
├── memory         26  (Hot/Warm/Cold, Token Budget, Summarizer)
├── security       40  (Vault, Sandbox, PromptGuard)
├── observability  34  (Telemetry, Metrics, Logging)
├── utils          42  (Throttle, Backoff, CircuitBreaker)
└── main           39  (Gateway API endpoints)
```

**Known environmental 503s:** `test_main.py::TestTaskEndpoints` (4 tests)
submits NORMAL-priority tasks.  On hosts with < 2 GB free RAM those return
503 — by design, not a regression.  On a host with ≥ 3 GB free, all 4 pass.

---

## Configuration

Environment variables (all optional; sensible defaults built in):

| Variable | Default | Effect |
|----------|---------|--------|
| `REDIS_HOST` / `REDIS_PORT`               | `localhost` / `6379`     | Stream-backed journal + cross-task state. |
| `POSTGRES_HOST` / `POSTGRES_DB`           | `localhost` / `orchestrator` | Cold-storage tier. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`             | `http://localhost:4317`  | OpenTelemetry collector. |
| `LOG_LEVEL`                               | `INFO`                   | `structlog` minimum level. |
| `ORCHESTRATOR_CONFIGURED_MAX_AGENTS`      | `20`                     | Hard cap in the `MaxAgents` formula. |

---

## Design invariants

1. **Account FSM:** `IDLE → WARMUP → ACTIVE → COOLDOWN → JAIL` (IDLE reachable
   from COOLDOWN after `cooldown_until` expires).
2. **Watermarks descend monotonically:** `NORMAL (≥ 3 GB) > WARNING (≥ 2.5) >
   CLEANUP (≥ 2.0) > EMERGENCY (≥ 1.5) > CRITICAL (< 1.5)`.
3. **Workflow FSM:** every transition is in `_VALID_TRANSITIONS`; `DONE` and
   `HALTED` are absorbing on the workflow side; `DONE` and `DLQ` on the task
   side.  HALTED is resumable — `POST /resume` re-enters `PLANNING`.
4. **Lease lifecycle:** `acquire → renew (heartbeat) → release | expire`.
   TTL default 300 s, max 5 renewals.
5. **Circuit breaker:** opens after 3 consecutive failures, recovery timeout
   60 s, HALF_OPEN admits a single probe before flipping back.
6. **DLQ idempotency:** `push(entry)` with an existing `task.id` replaces the
   prior entry — safe to call from a retry loop.
7. **Vault rotation:** `rotate_key(new_key)` validates, re-encrypts atomically,
   preserves old ciphertext on per-entry decrypt failure.
8. **Token bucket:** sync `threading.Lock` — works in both threadpool and
   asyncio; lock released before any `await`.
9. **`MaxAgents` formula:**
   `min(⌊(AvailRAM − 2) / 1.5⌋, cores × 2, browser_max_contexts, provider_max_concurrent, configured_max)`,
   floored at 1.
10. **Provider context limits:** ChatGPT 32 768, Qwen 131 072, DeepSeek
    1 000 000, Kimi 128 000, Local 256 000.
