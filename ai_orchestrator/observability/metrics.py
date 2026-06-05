"""Prometheus metrics — counters, gauges, histograms for the orchestration platform."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest, registry

#: Bounded set of failure-reason labels.  Anything not in this set is
#: collapsed to ``"other"`` to keep Prometheus cardinality bounded.
ALLOWED_FAILURE_REASONS: frozenset[str] = frozenset({
    "timeout",
    "rate_limit",
    "auth",
    "network",
    "parse",
    "provider_5xx",
    "provider_4xx",
    "sandbox",
    "validation",
    "internal",
    "unknown",
})


class MetricsRegistry:
    """Application-level Prometheus metric registry.

    Every method maps to a well-known metric name so that dashboards and
    alerting rules can rely on stable conventions.
    """

    def __init__(self) -> None:
        self._registry = registry.CollectorRegistry()

        # --- Counters ---
        self._task_submitted = Counter(
            "orchestrator_task_submitted_total",
            "Total tasks submitted to the orchestrator",
            labelnames=["task_type"],
            registry=self._registry,
        )
        self._task_completed = Counter(
            "orchestrator_task_completed_total",
            "Total tasks completed (by status)",
            labelnames=["status"],
            registry=self._registry,
        )
        self._task_failed = Counter(
            "orchestrator_task_failed_total",
            "Total tasks that failed (by reason)",
            labelnames=["reason"],
            registry=self._registry,
        )
        self._provider_error = Counter(
            "orchestrator_provider_error_total",
            "Total provider errors (by provider and error type)",
            labelnames=["provider", "error_type"],
            registry=self._registry,
        )

        # --- Gauges ---
        self._active_agents = Gauge(
            "orchestrator_active_agents",
            "Current number of active agents",
            registry=self._registry,
        )
        self._memory_usage = Gauge(
            "orchestrator_memory_usage_bytes",
            "Current memory usage in bytes",
            registry=self._registry,
        )
        self._pool_size = Gauge(
            "orchestrator_pool_size",
            "Current pool size per state",
            labelnames=["state"],
            registry=self._registry,
        )

        # --- Histograms ---
        self._provider_latency = Histogram(
            "orchestrator_provider_latency_ms",
            "Provider latency in milliseconds",
            labelnames=["provider"],
            buckets=(10, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000),
            registry=self._registry,
        )

    # ---- counters ---------------------------------------------------------

    def inc_task_submitted(self, task_type: str = "interactive") -> None:
        """Increment the task-submission counter."""
        self._task_submitted.labels(task_type=task_type).inc()

    def inc_task_completed(self, status: str = "success") -> None:
        """Increment the task-completion counter."""
        self._task_completed.labels(status=status).inc()

    def inc_task_failed(self, reason: str = "unknown") -> None:
        """Increment the task-failure counter.

        Unrecognized *reason* values are folded into ``"other"`` to keep
        the Prometheus label cardinality bounded (a runaway free-form
        error message would otherwise explode the time-series count).
        """
        if reason not in ALLOWED_FAILURE_REASONS:
            reason = "other"
        self._task_failed.labels(reason=reason).inc()

    def inc_provider_error(self, provider: str, error_type: str) -> None:
        """Increment the provider-error counter."""
        self._provider_error.labels(provider=provider, error_type=error_type).inc()

    # ---- gauges -----------------------------------------------------------

    def set_active_agents(self, count: int) -> None:
        """Set the active-agent gauge."""
        self._active_agents.set(count)

    def set_memory_usage(self, bytes_: int) -> None:
        """Set the memory-usage gauge."""
        self._memory_usage.set(bytes_)

    def set_pool_size(self, state: str, count: int) -> None:
        """Set the pool-size gauge for a given pool state."""
        self._pool_size.labels(state=state).set(count)

    # ---- histograms -------------------------------------------------------

    def observe_provider_latency(self, provider: str, ms: float) -> None:
        """Record a provider-latency observation (in milliseconds)."""
        self._provider_latency.labels(provider=provider).observe(ms)

    # ---- output -----------------------------------------------------------

    def get_metrics_text(self) -> str:
        """Return the Prometheus exposition-format text for all registered metrics.

        This is the string-compatible form of ``generate_latest()`` that can be
        served directly at a ``/metrics`` endpoint.
        """
        return generate_latest(self._registry).decode("utf-8")
