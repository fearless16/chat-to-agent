"""Browser Intelligence — recovery subsystem.

Implements the Recovery Cascade:

    Selector Cache
        ↓
    Accessibility Recovery
        ↓
    Graph Recovery
        ↓
    Network Recovery
        ↓
    Session Recovery
        ↓
    Worker Replacement
        ↓
    Provider Replacement
        ↓
    Workflow Replan

Each step has a fixed cost. The cascade always picks the cheapest
valid recovery — the first step that returns a positive confidence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)


class RecoveryStep(str, Enum):
    """The full recovery cascade, in ascending cost order."""

    SELECTOR_CACHE = "selector_cache"
    A11Y = "accessibility"
    GRAPH = "graph"
    NETWORK = "network"
    SESSION = "session"
    WORKER = "worker"
    PROVIDER = "provider"
    WORKFLOW = "workflow"

    @property
    def cost(self) -> float:
        """Relative cost of running this step (1.0 = cheapest)."""
        table = {
            RecoveryStep.SELECTOR_CACHE: 1.0,
            RecoveryStep.A11Y: 2.0,
            RecoveryStep.GRAPH: 3.0,
            RecoveryStep.NETWORK: 4.0,
            RecoveryStep.SESSION: 6.0,
            RecoveryStep.WORKER: 8.0,
            RecoveryStep.PROVIDER: 10.0,
            RecoveryStep.WORKFLOW: 14.0,
        }
        return table[self]


# Ordered by cost, lowest first.
CASCADE_ORDER: tuple[RecoveryStep, ...] = (
    RecoveryStep.SELECTOR_CACHE,
    RecoveryStep.A11Y,
    RecoveryStep.GRAPH,
    RecoveryStep.NETWORK,
    RecoveryStep.SESSION,
    RecoveryStep.WORKER,
    RecoveryStep.PROVIDER,
    RecoveryStep.WORKFLOW,
)


@dataclass
class RecoveryOutcome:
    step: RecoveryStep
    success: bool
    confidence: float
    detail: str = ""
    duration_seconds: float = 0.0
    cost: float = 0.0
    started_at: float = field(default_factory=time.monotonic)


# Action signature: takes (state context dict) and returns a
# RecoveryOutcome. May be sync or async.
RecoveryAction = Callable[[dict], "Awaitable[RecoveryOutcome] | RecoveryOutcome"]


@dataclass
class RecoveryCascade:
    """Cheapest-first recovery executor.

    A handler can be registered for each step. `run()` walks the
    cascade in cost order and stops at the first step whose handler
    reports a confident success (confidence >= min_confidence).
    """

    handlers: dict[RecoveryStep, RecoveryAction] = field(default_factory=dict)
    min_confidence: float = 0.55
    history: list[RecoveryOutcome] = field(default_factory=list)
    max_history: int = 200

    def register(self, step: RecoveryStep, handler: RecoveryAction) -> None:
        self.handlers[step] = handler

    def reset(self) -> None:
        self.history.clear()

    async def run(self, context: dict | None = None) -> list[RecoveryOutcome]:
        """Run the cascade. Returns the chain of outcomes (the loop
        stops as soon as a step succeeds, but earlier steps are also
        recorded so the operator can see what was tried)."""
        context = context or {}
        outcomes: list[RecoveryOutcome] = []
        for step in CASCADE_ORDER:
            handler = self.handlers.get(step)
            if handler is None:
                continue
            start = time.monotonic()
            try:
                result = handler(context)
                if _is_awaitable(result):
                    result = await result  # type: ignore[assignment]
            except Exception as exc:
                log.warning("Recovery handler %s raised: %s", step, exc)
                outcomes.append(RecoveryOutcome(
                    step=step,
                    success=False,
                    confidence=0.0,
                    detail=f"exception:{exc!r}",
                    duration_seconds=time.monotonic() - start,
                    cost=step.cost,
                ))
                continue
            result.duration_seconds = time.monotonic() - start
            result.cost = step.cost
            outcomes.append(result)
            self._record(result)
            if result.success and result.confidence >= self.min_confidence:
                break
        return outcomes

    def _record(self, outcome: RecoveryOutcome) -> None:
        self.history.append(outcome)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]


# ──────────────────────────────────────────────────────────────────────
# Built-in handlers — pure-logic / no-I/O versions used by default
# and as test fixtures.
# ──────────────────────────────────────────────────────────────────────

def selector_cache_recovery(context: dict) -> RecoveryOutcome:
    """The cheapest step: re-query the selector cache before doing
    anything expensive. The handler is a no-op by default; concrete
    implementations override it."""
    if context.get("selector_cache_hit"):
        return RecoveryOutcome(
            step=RecoveryStep.SELECTOR_CACHE,
            success=True,
            confidence=0.95,
            detail="cache_hit",
        )
    return RecoveryOutcome(
        step=RecoveryStep.SELECTOR_CACHE,
        success=False,
        confidence=0.0,
        detail="cache_miss",
    )


def a11y_recovery(context: dict) -> RecoveryOutcome:
    if context.get("a11y_recovered"):
        return RecoveryOutcome(
            step=RecoveryStep.A11Y,
            success=True,
            confidence=0.80,
            detail="a11y_query_succeeded",
        )
    return RecoveryOutcome(
        step=RecoveryStep.A11Y,
        success=False,
        confidence=0.0,
        detail="no_a11y_root",
    )


def graph_recovery(context: dict) -> RecoveryOutcome:
    if context.get("graph_recovered"):
        return RecoveryOutcome(
            step=RecoveryStep.GRAPH,
            success=True,
            confidence=0.70,
            detail="graph_query_succeeded",
        )
    return RecoveryOutcome(
        step=RecoveryStep.GRAPH,
        success=False,
        confidence=0.0,
        detail="no_graph_match",
    )


def network_recovery(context: dict) -> RecoveryOutcome:
    if context.get("network_recovered"):
        return RecoveryOutcome(
            step=RecoveryStep.NETWORK,
            success=True,
            confidence=0.65,
            detail="stream_recovered",
        )
    return RecoveryOutcome(
        step=RecoveryStep.NETWORK,
        success=False,
        confidence=0.0,
        detail="stream_still_stalled",
    )


def session_recovery(context: dict) -> RecoveryOutcome:
    if context.get("session_refreshed"):
        return RecoveryOutcome(
            step=RecoveryStep.SESSION,
            success=True,
            confidence=0.85,
            detail="session_refreshed",
        )
    return RecoveryOutcome(
        step=RecoveryStep.SESSION,
        success=False,
        confidence=0.0,
        detail="session_refresh_failed",
    )


def worker_recovery(context: dict) -> RecoveryOutcome:
    if context.get("worker_replaced"):
        return RecoveryOutcome(
            step=RecoveryStep.WORKER,
            success=True,
            confidence=0.90,
            detail="worker_replaced",
        )
    return RecoveryOutcome(
        step=RecoveryStep.WORKER,
        success=False,
        confidence=0.0,
        detail="no_spare_worker",
    )


def provider_recovery(context: dict) -> RecoveryOutcome:
    if context.get("provider_switched"):
        return RecoveryOutcome(
            step=RecoveryStep.PROVIDER,
            success=True,
            confidence=0.95,
            detail="provider_switched",
        )
    return RecoveryOutcome(
        step=RecoveryStep.PROVIDER,
        success=False,
        confidence=0.0,
        detail="no_alternate_provider",
    )


def workflow_recovery(context: dict) -> RecoveryOutcome:
    if context.get("workflow_replanned"):
        return RecoveryOutcome(
            step=RecoveryStep.WORKFLOW,
            success=True,
            confidence=0.99,
            detail="workflow_replanned",
        )
    return RecoveryOutcome(
        step=RecoveryStep.WORKFLOW,
        success=False,
        confidence=0.0,
        detail="replan_failed",
    )


def build_default_cascade() -> RecoveryCascade:
    """Convenience constructor that wires up the built-in handlers."""
    c = RecoveryCascade()
    c.register(RecoveryStep.SELECTOR_CACHE, selector_cache_recovery)
    c.register(RecoveryStep.A11Y, a11y_recovery)
    c.register(RecoveryStep.GRAPH, graph_recovery)
    c.register(RecoveryStep.NETWORK, network_recovery)
    c.register(RecoveryStep.SESSION, session_recovery)
    c.register(RecoveryStep.WORKER, worker_recovery)
    c.register(RecoveryStep.PROVIDER, provider_recovery)
    c.register(RecoveryStep.WORKFLOW, workflow_recovery)
    return c


def _is_awaitable(value) -> bool:
    import inspect
    return inspect.isawaitable(value)


__all__ = [
    "RecoveryCascade",
    "RecoveryStep",
    "RecoveryOutcome",
    "RecoveryAction",
    "CASCADE_ORDER",
    "build_default_cascade",
    "selector_cache_recovery",
    "a11y_recovery",
    "graph_recovery",
    "network_recovery",
    "session_recovery",
    "worker_recovery",
    "provider_recovery",
    "workflow_recovery",
]
