"""Provider health tracking — per-provider metrics for Phase 6.

Tracks per provider:
  - success_count / failure_count → success_rate
  - auth_success / auth_failure → auth_rate
  - total response_ms → avg_latency_ms
  - last_error, last_success timestamps
  - captcha_count, popup_count, recovery_count

Thread-safe (uses a lock for increments).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderMetrics:
    """Accumulated metrics for a single provider."""

    provider: str = ""
    success_count: int = 0
    failure_count: int = 0
    auth_success_count: int = 0
    auth_failure_count: int = 0
    total_latency_ms: float = 0.0
    captcha_count: int = 0
    popup_count: int = 0
    recovery_count: int = 0
    last_error: str = ""
    last_error_at: float = 0.0
    last_success_at: float = 0.0

    @property
    def total_attempts(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts

    @property
    def auth_rate(self) -> float:
        total = self.auth_success_count + self.auth_failure_count
        if total == 0:
            return 0.0
        return self.auth_success_count / total

    @property
    def avg_latency_ms(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "auth_success_count": self.auth_success_count,
            "auth_failure_count": self.auth_failure_count,
            "auth_rate": round(self.auth_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "captcha_count": self.captcha_count,
            "popup_count": self.popup_count,
            "recovery_count": self.recovery_count,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "last_success_at": self.last_success_at,
            "status": self._status_label,
        }

    @property
    def _status_label(self) -> str:
        if self.total_attempts == 0:
            return "untested"
        if self.success_rate >= 0.8:
            return "healthy"
        if self.success_rate >= 0.5:
            return "degraded"
        return "unhealthy"


class ProviderHealthTracker:
    """Singleton-style tracker for all provider health metrics."""

    def __init__(self) -> None:
        self._metrics: dict[str, ProviderMetrics] = {}
        self._lock = threading.Lock()

    def _ensure(self, provider: str) -> ProviderMetrics:
        if provider not in self._metrics:
            self._metrics[provider] = ProviderMetrics(provider=provider)
        return self._metrics[provider]

    def record_success(self, provider: str, latency_ms: float) -> None:
        with self._lock:
            m = self._ensure(provider)
            m.success_count += 1
            m.total_latency_ms += latency_ms
            m.last_success_at = time.time()

    def record_failure(self, provider: str, error: str) -> None:
        with self._lock:
            m = self._ensure(provider)
            m.failure_count += 1
            m.last_error = error
            m.last_error_at = time.time()

    def record_auth_success(self, provider: str) -> None:
        with self._lock:
            m = self._ensure(provider)
            m.auth_success_count += 1

    def record_auth_failure(self, provider: str) -> None:
        with self._lock:
            m = self._ensure(provider)
            m.auth_failure_count += 1

    def record_captcha(self, provider: str) -> None:
        with self._lock:
            m = self._ensure(provider)
            m.captcha_count += 1

    def record_popup(self, provider: str) -> None:
        with self._lock:
            m = self._ensure(provider)
            m.popup_count += 1

    def record_recovery(self, provider: str) -> None:
        with self._lock:
            m = self._ensure(provider)
            m.recovery_count += 1

    def get_metrics(self, provider: str) -> ProviderMetrics:
        with self._lock:
            return self._ensure(provider)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: m.to_dict() for name, m in self._metrics.items()}

    def get_dashboard(self) -> dict[str, Any]:
        """Return a summary dashboard for the /provider-health endpoint."""
        all_metrics = self.get_all_metrics()
        healthy = sum(1 for m in all_metrics.values() if m["status"] == "healthy")
        degraded = sum(1 for m in all_metrics.values() if m["status"] == "degraded")
        unhealthy = sum(1 for m in all_metrics.values() if m["status"] == "unhealthy")
        untested = sum(1 for m in all_metrics.values() if m["status"] == "untested")

        return {
            "summary": {
                "total_providers": len(all_metrics),
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "untested": untested,
            },
            "providers": all_metrics,
        }


# Module-level singleton.
health_tracker = ProviderHealthTracker()
