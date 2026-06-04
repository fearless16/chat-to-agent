#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== AI Orchestrator Setup ==="

# Python version check
PYTHON=$(command -v python3 || command -v python)
PYTHON_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
MIN_VERSION="3.13"

if [ "$(echo "$PYTHON_VERSION >= $MIN_VERSION" | bc -l 2>/dev/null || echo 0)" = "0" ]; then
    echo "Error: Python $MIN_VERSION+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✔ Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi
source .venv/bin/activate
echo "✔ Virtual environment"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev]" -q
echo "✔ Dependencies"

# Install playwright browsers (for UI adapter)
if command -v playwright &>/dev/null; then
    echo "Installing Playwright browsers..."
    playwright install chromium --with-deps 2>/dev/null || true
fi

# Create data directories
mkdir -p data logs

# Git hooks
if [ -d ".git" ] && ! [ -f ".git/hooks/pre-commit" ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install 2>/dev/null || true
fi

echo ""
echo "✔ Setup complete!"
echo ""
echo "  Activate:   source .venv/bin/activate"
echo "  Run tests:  pytest ai_orchestrator/tests/unit/ -v"
echo "  Start API:  uvicorn ai_orchestrator.orchestrator.main:app --reload --port 8000"
echo "  Docker:     cd docker && docker compose up -d"
