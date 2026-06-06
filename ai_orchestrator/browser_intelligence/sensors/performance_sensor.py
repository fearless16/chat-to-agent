"""PerformanceSensor — observes browser performance metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ai_orchestrator.browser_intelligence.sensors.base import BaseSensor

log = logging.getLogger(__name__)

_PERFORMANCE_SCRIPT = """
() => {
    const m = performance.memory || {};
    return {
        js_heap_size: m.usedJSHeapSize || 0,
        js_heap_limit: m.jsHeapSizeLimit || 0,
        dom_nodes: document.querySelectorAll('*').length,
    };
}
"""


@dataclass
class PerformanceFeatures:
    js_heap_used_mb: float = 0.0
    js_heap_limit_mb: float = 0.0
    dom_node_count: int = 0
    layout_shift_count: int = 0
    page_load_stable: bool = True


class PerformanceSensor(BaseSensor):
    """Observes browser performance via JS performance API.

    Tracks JS heap usage, DOM node count, and page stability.
    """

    def __init__(self, heap_warning_threshold_mb: float = 500.0):
        self._heap_warning = heap_warning_threshold_mb
        self._prev_dom_nodes: int = 0

    async def sense(self, page) -> PerformanceFeatures:
        features = PerformanceFeatures()
        try:
            metrics = await page.evaluate(_PERFORMANCE_SCRIPT)
            features.js_heap_used_mb = float(metrics["js_heap_size"]) / (1024 * 1024)
            features.js_heap_limit_mb = float(metrics["js_heap_limit"]) / (1024 * 1024)
            features.dom_node_count = int(metrics["dom_nodes"])
            features.layout_shift_count = 0

            if features.dom_node_count > 0:
                dom_change = abs(features.dom_node_count - self._prev_dom_nodes)
                features.page_load_stable = dom_change < 50
            self._prev_dom_nodes = features.dom_node_count
        except Exception as exc:
            log.debug("PerformanceSensor failed: %s", exc)
        return features

    def reset(self) -> None:
        self._prev_dom_nodes = 0
