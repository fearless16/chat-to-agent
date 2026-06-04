# AI Orchestrator

Production-grade multi-provider AI orchestration platform. Coordinates OpenAI, Qwen, DeepSeek, Kimi, and local LLMs with pooled accounts, parallel agents, dynamic resource scheduling, and full observability.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  User/API    │────▶│          FastAPI Gateway             │
└─────────────┘     │  /tasks  /accounts  /leases  /health  │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         Workflow Engine               │
                    │  PLANNING→EXECUTING→TESTING→DONE      │
                    └──────────┬───────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌──────────┐       ┌──────────────┐     ┌──────────────┐
   │  Agents   │       │ProviderRouter│     │ResourceSched │
   │ Planner   │       │  Capability  │     │  Watermarks  │
   │ Coder     │       │  Scoring     │     │  3→2→1.5GB   │
   │ Tester    │       │  Fallback    │     │  Admission   │
   └──────────┘       └──────┬───────┘     └──────────────┘
                              │
                     ┌────────▼────────┐
                     │  Lease Manager   │
                     │  Account Pool    │
                     │  Health Tracking │
                     └────────┬────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐       ┌──────────────┐     ┌──────────────┐
   │ChatGPT   │       │  DeepSeek    │     │   Local LLM  │
   │Qwen/Kimi │       │  API         │     │   Ollama     │
   └──────────┘       └──────────────┘     └──────────────┘

Memory: Hot (RAM) → Warm (Compressed) → Cold (Postgres/S3)
Observability: OpenTelemetry → Prometheus → Grafana
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest ai_orchestrator/tests/unit/ -v

# Start the gateway
uvicorn ai_orchestrator.orchestrator.main:app --reload --port 8000

# Submit a task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"prompt": "write a sorting function"}'

# Check health
curl http://localhost:8000/health

# View providers
curl http://localhost:8000/providers
```

## Docker

```bash
cd docker
docker compose up -d
# Gateway:    http://localhost:8000
# Prometheus: http://localhost:9091
# Grafana:    http://localhost:3000
```

## Test Suite

```
519 tests — 92% coverage across 12 modules
├── models          47 tests (Account, Lease, Task, Capabilities)
├── agents          62 tests (7 role agents + BaseAgent)
├── adapters        22 tests (6 providers + base protocol)
├── orchestrator    89 tests (LeaseManager, Router, Scheduler, Workflow)
├── storage         78 tests (Redis, Postgres, Journal)
├── memory          26 tests (Hot/Warm/Cold, Token Budget, Summarizer)
├── security        40 tests (Vault, Sandbox, PromptGuard)
├── observability   34 tests (Telemetry, Metrics, Logging)
├── utils           42 tests (Throttle, Backoff, CircuitBreaker)
└── main            39 tests (Gateway API endpoints)
```

## Modules

| Module | Description |
|--------|-------------|
| `models/` | Account state machine, Lease lifecycle, Task model, Capability vectors |
| `adapters/` | Provider adapters for ChatGPT (API + UI), Qwen, DeepSeek, Kimi, Local LLM |
| `agents/` | BaseAgent ABC + Planner, Researcher, Coder, Tester, Reviewer, Fixer, Executor |
| `orchestrator/` | WorkflowEngine FSM, LeaseManager, ProviderRouter, ResourceScheduler, FastAPI gateway |
| `storage/` | Redis client, Postgres async models, execution journal |
| `memory/` | Hot/warm/cold context management, token budget, summarizer |
| `security/` | AES-256-GCM vault, sandboxed execution, prompt injection guard |
| `observability/` | OpenTelemetry tracing, Prometheus metrics, structured logging (structlog) |
| `utils/` | Token bucket throttling, exponential backoff, circuit breaker |

## Configuration

Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `POSTGRES_HOST` | `localhost` | Postgres host |
| `POSTGRES_DB` | `orchestrator` | Postgres database |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry collector |
| `LOG_LEVEL` | `INFO` | Log level |
