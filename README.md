# chat-to-agent / ai_orchestrator

Production-grade multi-provider AI orchestration platform. A FastAPI gateway that accepts
incoming tasks, plans them into a workflow, picks the best provider account, leases it for
the duration of the work, drives a state-machine pipeline (plan → execute → test → review
→ fix → done), and tears everything down — all under watermark-based admission control
so a single orchestrator can host many concurrent agents without exhausting the host.

## What it actually does

1. **`POST /tasks`** — client submits `{prompt, priority, task_type, context_length}`.
2. **Admission control** — `ResourceScheduler.can_accept_task` checks the current RAM
   watermark against the task's `TaskPriority`.  Rejected (HTTP 503) when pressure
   exceeds the priority's threshold.  The mapping is:

   | Priority     | Rejected at watermark ≥ | |
   |--------------|-------------------------|---|
   | `CRITICAL`   | `CRITICAL`  (< 1.5 GB free) | only when the host is about to die |
   | `HIGH`       | `EMERGENCY` (< 2.0 GB free) | |
   | `NORMAL`     | `EMERGENCY` (< 2.0 GB free) | |
   | `LOW`        | `CLEANUP`  (< 2.5 GB free) | |
   | `BACKGROUND` | `WARNING`  (< 3.0 GB free) | |

3. **Provider selection** — `ProviderRouter` scores every account's
   `ProviderCapabilities` against the task's `TaskRequirements` and picks the best fit
   (reasoning, coding, translation, multimodality, context length).
4. **Lease acquisition** — `LeaseManager.request_lease` reserves that account for
   `agent_id` on `task_id` with a TTL (`ttl_seconds=300`, renewable up to
   `max_renewals=5`).
5. **Workflow execution** — `WorkflowEngine` drives the FSM:

   ```
   IDLE → PLANNING → EXECUTING ─┬→ TESTING ─→ REVIEW ─→ DONE
                                │      │            │
                                │      └→ FIX ──────┘
                                └→ HALTED  (operator pause)
   ```

   `WorkflowState.HALTED` is the only "paused" terminal — `POST /tasks/{id}/resume`
   brings the task back to `PLANNING`.  `DONE` and `TaskStatus.DLQ` are truly
   absorbing; the same-state guard in `Task.transition_to` keeps the FSM honest.
6. **Lease release / expiry** — `DELETE /leases/{id}` returns the account to the pool.
   The `LeaseManager` heartbeat thread expires idle leases on a timer.
7. **Observability** — every step is journaled to a per-task Redis stream
   (`journal:{task_id}`) and the global `journal:all`, plus OpenTelemetry traces and
   Prometheus metrics.

When the workflow finishes, the task is `DONE`.  When `max_retries` is exhausted the
task is pushed to the DLQ (idempotent on `task.id` — re-pushing replaces the prior
entry).

## Architecture

```
┌──────────────┐     ┌────────────────────────────────────────────────┐
│  HTTP client │────▶│  FastAPI gateway  (orchestrator/main.py)       │
└──────────────┘     │  /tasks /accounts /leases /health /metrics      │
                     └────────┬─────────────────────┬──────────────────┘
                              │                     │
                ┌─────────────▼──────────┐  ┌───────▼─────────────────┐
                │  ResourceScheduler     │  │  WorkflowEngine         │
                │  watermark admission   │  │  FSM + loop detection   │
                │  3.0/2.5/2.0/1.5 GB    │  │  halt / resume          │
                └────────────────────────┘  └────────┬─────────────────┘
                                                     │
                                ┌────────────────────┼─────────────────────┐
                                ▼                    ▼                     ▼
                         ┌────────────┐       ┌───────────────┐    ┌────────────────┐
                         │ Providers  │       │ LeaseManager  │    │ Agents         │
                         │ 6 profiles │       │ account pool  │    │ 7 roles:       │
                         │ API/BROWSER│       │ heartbeat     │    │ planner, coder,│
                         │  /LOCAL    │       │ expire        │    │ tester, review,│
                         └─────┬──────┘       └───────┬───────┘    │ fixer, executor│
                               │                      │            │ researcher     │
                               └──────────┬───────────┘            └────────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │  Adapters (per provider)│
                              │  ChatGPT, Qwen, DeepSeek│
                              │  Kimi, Local LLM        │
                              │  circuit breaker + token│
                              │  bucket throttle        │
                              └────────────────────────┘

Memory:  HOT (RAM) → WARM (compressed) → COLD (Postgres/Redis)
Storage: Redis streams (journal) + Postgres (UPSERT via session.merge)
Security: Fernet (AES-128-CBC+HMAC-SHA256) vault, sandboxed exec,
          prompt-injection guard with weighted risk scoring
```

## Modules

| Module | Lines | Description |
|--------|------:|-------------|
| `models/`         | 4 files | Pydantic domain models.  `Account` state machine (IDLE → WARMUP → ACTIVE → COOLDOWN → JAIL), `Lease` lifecycle (acquire → renew → release/expire), `Task` with FSM-aware `transition_to`, `ProviderCapabilities` matrix. |
| `agents/`         | 9 files | `BaseAgent` ABC + 7 role agents: `planner`, `researcher`, `coder`, `tester`, `reviewer`, `fixer`, `executor`.  Each tracks step count, actions, runtime, and has a hard ceiling (`max_steps=25`, `max_runtime_ms=300_000`). |
| `adapters/`       | 7 files | Provider-specific LLM adapters (ChatGPT API+UI, Qwen, DeepSeek, Kimi, local LLM).  All share `protected_send`: token-bucket throttle + circuit breaker (3 fails, 60 s recovery) + call counter. |
| `orchestrator/`   | 6 files | `main.py` (FastAPI), `workflow_engine.py` (FSM), `lease_manager.py` (account pool + heartbeats), `provider_router.py` (capability scoring), `resource_scheduler.py` (watermarks + admission), `dlq.py` (idempotent DLQ). |
| `storage/`        | 3 files | `RedisClient` (coroutine-locked mutators, stream support), `postgres_models` (async UPSERT via `session.merge` + `json.dumps(default=str)`), `ExecutionJournal` (per-task + global streams, 1M-entry bulk-fetch cap). |
| `memory/`         | 3 files | `ContextManager` with HOT/WARM/COLD tiers, `trim_to_budget(max_hot_tokens)` to enforce the cap, `TokenBudget` estimator + `trim_to_limit`, `ContextSummarizer` (extractive). |
| `security/`       | 3 files | `vault.py` (Fernet keyring with safe `rotate_key`: validate-then-swap, atomic assign, preserve old ciphertext on per-entry decrypt failure), `sandbox.py` (env whitelist, `network_access` flag, `RLIMIT_AS` preexec, 1 MiB output cap, process-group kill), `prompt_guard.py` (regex-based injection detection with weighted risk scoring, single match = unsafe). |
| `observability/`  | 3 files | OpenTelemetry tracing (`record_exception` sets `Status(ERROR, str(exception))`, `sample_rate` wired via `TraceIdRatioBased`), Prometheus `MetricsRegistry` (bounded `reason` label to cap cardinality), `structlog` JSON logger. |
| `utils/`          | 3 files | `TokenBucket` (threading.Lock, `ConcurrencyLimiter` semaphore), `RetryConfig.compute_delay_ms` (decorrelated jitter `delay + uniform(0, delay)`, capped at `max_delay_ms`), `CircuitBreaker` (threading.Lock around every state mutation, async-lock for HALF_OPEN probe slot). |

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| `GET`    | `/health`                            | Uptime, watermark level, free RAM, CPU%, active tasks/leases, account count. |
| `GET`    | `/metrics`                           | Same data, formatted for Prometheus scrapers, plus `pool_stats` per provider. |
| `GET`    | `/providers`                         | Capability matrix of all configured providers. |
| `GET`    | `/accounts[?provider=&state=]`       | Account pool.  Each shows `state`, `health_score`, `consecutive_failures`, `is_available`. |
| `GET`    | `/leases`                            | Active leases. |
| `POST`   | `/leases?task_id=&agent_id=&provider=` | Acquire a lease.  Returns 503 if no account is available. |
| `POST`   | `/leases/{id}/heartbeat`             | Keep a lease alive. |
| `DELETE` | `/leases/{id}`                       | Release a lease. |
| `GET`    | `/tasks[?status=]`                   | List tasks. |
| `GET`    | `/tasks/{id}`                        | Single task. |
| `POST`   | `/tasks`                             | Submit a task. |
| `POST`   | `/tasks/{id}/execute?step_name=&agent_type=` | Run the next workflow step. |
| `POST`   | `/tasks/{id}/halt?reason=`           | Pause a task (workflow → HALTED, error_message set). |
| `POST`   | `/tasks/{id}/resume`                 | Resume a HALTED task (workflow → PLANNING, error cleared). |

### `TaskPriority` enum

It's an `int` enum — submit the integer value, not the name:

| Constant        | Value |
|-----------------|------:|
| `CRITICAL`      | `0`   |
| `HIGH`          | `1`   |
| `NORMAL`        | `2`   |
| `LOW`           | `3`   |
| `BACKGROUND`    | `4`   |

### `TaskStatus` (domain) vs `WorkflowState` (FSM)

`TaskStatus` is what the world sees (`IDLE`, `PLANNING`, `EXECUTING`, `VERIFICATION`,
`FAILED`, `DONE`, `HALTED`, `DLQ`).  `WorkflowState` is the engine's internal FSM
(`IDLE`, `PLANNING`, `EXECUTING`, `TESTING`, `REVIEW`, `FIX`, `DONE`, `HALTED`).
The engine maps them via `_state_to_task_status`; both `TESTING` and `REVIEW` map to
`TaskStatus.VERIFICATION`, and `FIX` re-enters the task as `EXECUTING`.

## Quick start

```bash
# 1. Install (use the project venv or your own)
. .venv/bin/activate
pip install -e ".[dev]"

# 2. Tests (543 tests, ~15 s, no external services required)
pytest ai_orchestrator/tests/unit/ -v

# 3. Run the gateway
uvicorn ai_orchestrator.orchestrator.main:app --host 127.0.0.1 --port 8000

# 4. Submit a task (priority=0 = CRITICAL, bypasses the EMERGENCY watermark)
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2?","priority":0,"context_length":4096}'

# 5. Walk it through the FSM
TASK_ID=...   # from step 4
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/execute?step_name=implement&agent_type=executor"
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/execute?step_name=test&agent_type=tester"
curl -X POST "http://127.0.0.1:8000/tasks/$TASK_ID/execute?step_name=review&agent_type=reviewer"

# 6. Health, metrics, accounts, providers — read-only endpoints
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

The bundled `scripts/run.sh {api|test|test-all|shell}` wraps the same steps.

## Docker

```bash
cd docker
docker compose up -d
# Gateway:     http://localhost:8000
# Prometheus:  http://localhost:9091
# Grafana:     http://localhost:3000
# Redis + Postgres + OpenTelemetry collector
```

## Test suite

```
543 tests
├── models          47  (Account, Lease, Task, Capabilities)
├── agents          62  (BaseAgent + 7 role agents)
├── adapters        22  (6 providers + base protocol)
├── orchestrator    89  (LeaseManager, Router, Scheduler, Workflow, DLQ)
├── storage         78  (Redis, Postgres, Journal)
├── memory          26  (Hot/Warm/Cold, Token Budget, Summarizer)
├── security        40  (Vault, Sandbox, PromptGuard)
├── observability   34  (Telemetry, Metrics, Logging)
├── utils           42  (Throttle, Backoff, CircuitBreaker)
└── main            39  (Gateway API endpoints)
```

**Known environmental 503s:** `test_main.py::TestTaskEndpoints` (4 tests) submits
NORMAL-priority tasks and expects 201.  Under the project's admission-control rules
NORMAL is rejected at watermark ≥ EMERGENCY, so on hosts with < 2 GB free RAM those
tests return 503 — by design, not a regression.  On a host with ≥ 3 GB free, all 4 pass.

## Configuration

Environment variables (all optional; sensible defaults are baked in):

| Variable | Default | Effect |
|----------|---------|--------|
| `REDIS_HOST` / `REDIS_PORT`               | `localhost` / `6379`     | Stream-backed journal + cross-task state. |
| `POSTGRES_HOST` / `POSTGRES_DB`           | `localhost` / `orchestrator` | Cold-storage tier + account/task persistence. |
| `OTEL_EXPORTER_OTLP_ENDPOINT`             | `http://localhost:4317`  | OpenTelemetry collector.  `sample_rate` (0–1) controls the ratio. |
| `LOG_LEVEL`                               | `INFO`                   | `structlog` minimum level. |
| `ORCHESTRATOR_CONFIGURED_MAX_AGENTS`      | `20`                     | Hard cap in the `MaxAgents` formula. |

## Design invariants

These hold across the codebase and are enforced by the test suite:

1. **Account FSM:** `IDLE → WARMUP → ACTIVE → COOLDOWN → JAIL` (with `IDLE` reachable
   from `COOLDOWN` after `cooldown_until` expires).
2. **Watermarks descend monotonically:** `NORMAL (≥ 3 GB) > WARNING (≥ 2.5) > CLEANUP
   (≥ 2.0) > EMERGENCY (≥ 1.5) > CRITICAL (< 1.5)`.
3. **Workflow FSM:** every transition is in `_VALID_TRANSITIONS`; `DONE` and
   `HALTED` are absorbing on the workflow side; `DONE` and `DLQ` are absorbing on the
   task side.  HALTED is *paused* (resumable) — the gateway's `POST /resume` re-enters
   `PLANNING`.
4. **Lease lifecycle:** `acquire → renew (heartbeat) → release | expire`.  TTL
   default 300 s, `max_renewals=5`.
5. **Circuit breaker:** opens after 3 consecutive failures, recovery timeout 60 s,
   HALF_OPEN admits a single probe before flipping back.
6. **DLQ idempotency:** `push(entry)` with an existing `task.id` replaces the prior
   entry — safe to call from a retry loop.
7. **Vault rotation:** `rotate_key(new_key)` validates the new key first, builds the
   fully re-encrypted dict, then assigns atomically; per-entry decrypt failures
   preserve the old ciphertext.
8. **Token bucket:** sync `threading.Lock` (not `asyncio.Lock`) — works in both
   threadpool and asyncio contexts; lock is released before any `await`.
9. **`MaxAgents` formula:**
   `min(⌊(AvailRAM − 2) / 1.5⌋, cores × 2, browser_max_contexts, provider_max_concurrent, configured_max)`,
   floored at 1.
10. **Provider context limits:** ChatGPT 32 768, Qwen 131 072, DeepSeek 1 000 000,
    Kimi 128 000, Local 256 000.
