FROM python:3.14-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --user hatchling && \
    pip install --no-cache-dir --user .

# Final stage
FROM python:3.14-slim-bookworm

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --system --no-create-home appuser

# Copy installed packages
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages

# Copy application
COPY ai_orchestrator/ /app/ai_orchestrator/
COPY pyproject.toml /app/

# Set environment
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run orchestrator gateway
CMD ["uvicorn", "ai_orchestrator.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
