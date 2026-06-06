"""Browser Intelligence — scheduling subsystem.

Adaptive scheduler that decides how many browser workers to run, how
often the engine tick should fire, and what the per-provider
concurrency cap should be, based on live system metrics.

Inputs:
- psutil-style CPU/RAM samples (caller provides the actual numbers).
- browser_count — currently open workers.
- queue_depth — pending prompt-send jobs.
- provider_reliability — Beta posterior mean per provider.
- account_health — per-account reliability.

Outputs (computed each call to `decide()`):
- worker_count — how many browsers to keep open.
- tick_interval — engine tick period in seconds.
- concurrency_limits — per-provider max parallel sends.

No fixed concurrency: every output is derived from the inputs.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class SchedulingInputs:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    browser_count: int = 0
    queue_depth: int = 0
    provider_reliability: dict[str, float] = field(default_factory=dict)
    account_health: dict[str, float] = field(default_factory=dict)
    target_browsers: int = 4
    min_workers: int = 1
    max_workers: int = 16
    min_tick: float = 0.25
    max_tick: float = 4.0

    def health_score(self) -> float:
        """0.0 (overloaded) → 1.0 (idle) system health.

        RAM is weighted slightly higher than CPU because browser
        instances scale with RAM, not with CPU cycles.
        """
        cpu_term = max(0.0, 1.0 - max(0.0, self.cpu_percent) / 100.0)
        ram_term = max(0.0, 1.0 - max(0.0, self.ram_percent) / 100.0)
        return 0.4 * cpu_term + 0.6 * ram_term


@dataclass
class SchedulingDecision:
    worker_count: int
    tick_interval: float
    concurrency_limits: dict[str, int]
    reason: str

    def to_dict(self) -> dict:
        return {
            "worker_count": self.worker_count,
            "tick_interval": round(self.tick_interval, 3),
            "concurrency_limits": dict(self.concurrency_limits),
            "reason": self.reason,
        }


class AdaptiveScheduler:
    """Derive scheduling knobs from live inputs.

    The scheduler is stateless across calls except for the most
    recent inputs and the decision. Each call recomputes everything
    from scratch, so the engine can invoke it as often as it likes
    without model drift.
    """

    def __init__(
        self,
        *,
        default_concurrency: int = 2,
        high_reliability_threshold: float = 0.85,
        low_reliability_threshold: float = 0.4,
    ):
        self._default_concurrency = max(1, int(default_concurrency))
        self._high_rel = float(high_reliability_threshold)
        self._low_rel = float(low_reliability_threshold)
        self._last_decision: SchedulingDecision | None = None
        self._last_inputs: SchedulingInputs | None = None
        self._last_decided_at: float = 0.0

    @property
    def last_decision(self) -> SchedulingDecision | None:
        return self._last_decision

    def decide(self, inputs: SchedulingInputs) -> SchedulingDecision:
        self._last_inputs = inputs
        health = inputs.health_score()
        reasons: list[str] = []

        # ── Worker count ────────────────────────────────────────
        # If the system is idle, scale up to target; if it's loaded,
        # scale down. The default is half-target on a moderately
        # busy box.
        target = inputs.target_browsers
        if health >= 0.7:
            desired = target
            reasons.append(f"health_high={health:.2f}")
        elif health >= 0.4:
            desired = max(inputs.min_workers, target // 2)
            reasons.append(f"health_med={health:.2f}")
        else:
            desired = inputs.min_workers
            reasons.append(f"health_low={health:.2f}")

        # Queue pressure: add workers if queue is long.
        if inputs.queue_depth > desired:
            extra = min(
                inputs.max_workers - desired,
                max(1, (inputs.queue_depth - desired) // 2),
            )
            desired = min(inputs.max_workers, desired + extra)
            reasons.append(f"queue_pressure={inputs.queue_depth}")

        worker_count = max(inputs.min_workers, min(inputs.max_workers, desired))

        # ── Tick interval ───────────────────────────────────────
        # A busier system → slower tick. Idle → fastest tick.
        base_tick = (inputs.min_tick + inputs.max_tick) / 2.0
        health_factor = 1.5 - health  # 0.5 (idle) → 1.5 (busy)
        queue_factor = 1.0 + 0.25 * math.log1p(max(0, inputs.queue_depth))
        tick = base_tick * health_factor * queue_factor
        tick = max(inputs.min_tick, min(inputs.max_tick, tick))

        # ── Per-provider concurrency ────────────────────────────
        conc: dict[str, int] = {}
        for provider, rel in inputs.provider_reliability.items():
            if rel >= self._high_rel:
                conc[provider] = max(1, self._default_concurrency + 2)
                reasons.append(f"{provider}:high_rel({rel:.2f})")
            elif rel <= self._low_rel:
                conc[provider] = 1
                reasons.append(f"{provider}:low_rel({rel:.2f})")
            else:
                conc[provider] = self._default_concurrency

        decision = SchedulingDecision(
            worker_count=worker_count,
            tick_interval=tick,
            concurrency_limits=conc or {"_default": self._default_concurrency},
            reason="; ".join(reasons),
        )
        self._last_decision = decision
        self._last_decided_at = time.monotonic()
        return decision

    def reset(self) -> None:
        self._last_decision = None
        self._last_inputs = None
        self._last_decided_at = 0.0


__all__ = [
    "AdaptiveScheduler",
    "SchedulingInputs",
    "SchedulingDecision",
]
