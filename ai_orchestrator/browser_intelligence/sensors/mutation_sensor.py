"""MutationSensor — observes DOM mutation rate via MutationObserver."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ai_orchestrator.browser_intelligence.sensors.base import BaseSensor

log = logging.getLogger(__name__)

_MUTATION_SCRIPT = """
() => {
    if (!window.__bio_mutation_stats__) {
        window.__bio_mutation_stats__ = { count: 0, last_read: performance.now() };
        const observer = new MutationObserver(() => {
            window.__bio_mutation_stats__.count++;
        });
        observer.observe(document.body, {
            childList: true, subtree: true, characterData: true, attributes: true
        });
    }
    const s = window.__bio_mutation_stats__;
    const count = s.count;
    s.count = 0;
    const now = performance.now();
    const elapsed = (now - s.last_read) / 1000;
    s.last_read = now;
    return {
        count: count,
        rate: elapsed > 0 ? count / elapsed : 0,
    };
}
"""


@dataclass
class MutationFeatures:
    mutation_count: int = 0
    mutation_rate: float = 0.0
    mutation_acceleration: float = 0.0


class MutationSensor(BaseSensor):
    """Observes DOM mutation rate via injected MutationObserver.

    Mutation rate is the single most powerful signal for detecting
    generation activity — streaming responses cause high-frequency
    DOM mutations.

    The MutationObserver is injected once per page lifecycle and
    accumulates counts between reads.
    """

    def __init__(self):
        self._previous_rate: float = 0.0

    async def sense(self, page) -> MutationFeatures:
        features = MutationFeatures()
        try:
            stats = await page.evaluate(_MUTATION_SCRIPT)
            features.mutation_count = int(stats.get("count", 0))
            features.mutation_rate = float(stats.get("rate", 0.0))
            features.mutation_acceleration = (
                features.mutation_rate - self._previous_rate
            )
            self._previous_rate = features.mutation_rate
        except Exception as exc:
            log.debug("MutationSensor failed: %s", exc)
        return features

    def reset(self) -> None:
        self._previous_rate = 0.0
