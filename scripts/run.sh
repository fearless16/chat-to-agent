#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

MODE="${1:-api}"

case "$MODE" in
    api)
        echo "Starting API gateway on port 8000..."
        exec uvicorn ai_orchestrator.orchestrator.main:app \
            --host 0.0.0.0 --port 8000 \
            --reload \
            --log-level info
        ;;
    test)
        echo "Running tests..."
        exec python -m pytest ai_orchestrator/tests/unit/ -v --tb=short --cov=ai_orchestrator "$@"
        ;;
    test-all)
        echo "Running all tests (unit + integration)..."  
        exec python -m pytest ai_orchestrator/tests/ -v --tb=short "$@"
        ;;
    shell)
        exec python -c "
import asyncio
from ai_orchestrator.orchestrator.lease_manager import LeaseManager
from ai_orchestrator.orchestrator.provider_router import ProviderRouter
from ai_orchestrator.orchestrator.resource_scheduler import ResourceScheduler
from ai_orchestrator.orchestrator.workflow_engine import WorkflowEngine
from ai_orchestrator.models.account import Account
from ai_orchestrator.models.task import Task
print('AI Orchestrator interactive shell')
print('  lm = LeaseManager()')
print('  router = ProviderRouter()')
print('  scheduler = ResourceScheduler()')
print('  wf = WorkflowEngine()')
import code; code.interact(local=locals())
"
        ;;
    *)
        echo "Usage: $0 {api|test|test-all|shell}"
        exit 1
        ;;
esac
