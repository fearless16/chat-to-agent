"""FeatureComposer — collects sensor outputs into a unified FeatureVector."""

from __future__ import annotations

import logging
import time

from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector
from ai_orchestrator.browser_intelligence.sensors.accessibility_sensor import (
    AccessibilitySensor,
)
from ai_orchestrator.browser_intelligence.sensors.dom_sensor import DOMSensor
from ai_orchestrator.browser_intelligence.sensors.mutation_sensor import MutationSensor
from ai_orchestrator.browser_intelligence.sensors.network_sensor import NetworkSensor
from ai_orchestrator.browser_intelligence.sensors.performance_sensor import (
    PerformanceSensor,
)
from ai_orchestrator.browser_intelligence.sensors.visual_sensor import VisualSensor

log = logging.getLogger(__name__)


class FeatureComposer:
    """Assembles sensor outputs into a FeatureVector each tick.

    Runs all sensors and merges their outputs into the unified
    feature vector. Sensors are independent — no sensor accesses
    another sensor's state.
    """

    def __init__(
        self,
        dom_sensor: DOMSensor | None = None,
        accessibility_sensor: AccessibilitySensor | None = None,
        network_sensor: NetworkSensor | None = None,
        mutation_sensor: MutationSensor | None = None,
        visual_sensor: VisualSensor | None = None,
        performance_sensor: PerformanceSensor | None = None,
        response_extractor=None,
    ):
        self._dom = dom_sensor or DOMSensor()
        self._a11y = accessibility_sensor or AccessibilitySensor()
        self._network = network_sensor or NetworkSensor()
        self._mutation = mutation_sensor or MutationSensor()
        self._visual = visual_sensor or VisualSensor()
        self._perf = performance_sensor or PerformanceSensor()
        self._response_extractor = response_extractor
        self._tick = 0
        self._prev_response_length = 0

    async def tick(self, page) -> FeatureVector:
        """Execute one tick: sense all features, compose vector."""
        self._tick += 1
        t0 = time.monotonic()

        dom = await self._dom.sense(page)
        a11y = await self._a11y.sense(page)
        net = await self._network.sense(page)
        mut = await self._mutation.sense(page)
        vis = await self._visual.sense(page)
        perf = await self._perf.sense(page)

        response_length = 0
        page_title = ""
        url = ""
        try:
            url = page.url
            page_title = await page.title()
            if self._response_extractor:
                length = self._response_extractor(page)
                if hasattr(length, "__await__"):
                    response_length = await length
                else:
                    response_length = length
            else:
                response_length = await self._default_extract_response_length(page)
        except Exception:
            pass

        response_length_delta = response_length - self._prev_response_length
        self._prev_response_length = response_length

        fv = FeatureVector(
            tick=self._tick,
            timestamp=time.monotonic(),
            input_visible=dom.input_visible,
            send_enabled=dom.send_visible,
            stop_button_visible=dom.stop_button_visible,
            regenerate_visible=dom.regenerate_visible,
            error_banner_visible=dom.error_banner_visible,
            auth_form_visible=dom.auth_form_visible,
            text_input_count=a11y.text_input_count,
            button_count=a11y.button_count,
            has_thinking_marker=a11y.has_thinking_marker,
            has_error_marker=a11y.has_error_marker,
            has_rate_limit_marker=a11y.has_rate_limit_marker,
            has_streaming_marker=a11y.has_streaming_marker,
            stream_active=net.stream_active,
            transport_detected=net.transport_detected,
            generation_started=net.generation_started,
            generation_completed=net.generation_completed,
            stream_closed=net.stream_closed,
            generation_stop_detected=net.generation_stop_detected,
            tokens_per_second=net.tokens_per_second,
            stream_idle_time=net.stream_idle_time,
            total_chunks=net.total_chunks,
            bytes_received=net.bytes_received,
            network_request_rate=net.request_rate,
            mutation_rate=mut.mutation_rate,
            mutation_acceleration=mut.mutation_acceleration,
            js_heap_used_mb=perf.js_heap_used_mb,
            page_stability=float(perf.page_load_stable),
            response_length=response_length,
            response_length_delta=response_length_delta,
            visual_stability=vis.visual_stability,
            a11y_extraction_success=a11y.extraction_success,
            a11y_confidence=a11y.accessibility_confidence,
            a11y_node_count=a11y.snapshot_node_count,
            page_title=page_title,
            url=url,
        )

        elapsed = (time.monotonic() - t0) * 1000
        if elapsed > 500:
            log.warning("FeatureComposer tick %d took %.0fms", self._tick, elapsed)

        return fv

    def reset(self) -> None:
        self._tick = 0
        self._prev_response_length = 0
        sensors = [self._dom, self._a11y, self._network, self._mutation, self._visual, self._perf]
        for sensor in sensors:
            sensor.reset()

    @staticmethod
    async def _default_extract_response_length(page) -> int:
        """Robust fallback: extract response text length via JS evaluate."""
        try:
            text = await page.evaluate("""() => {
                const sel = document.querySelector('[data-message-author-role="assistant"]');
                if (sel) return sel.innerText || '';
                const articles = document.querySelectorAll('article');
                if (articles.length) return articles[articles.length - 1].innerText || '';
                const messages = document.querySelectorAll('[class*="message"]');
                if (messages.length) return messages[messages.length - 1].innerText || '';
                return '';
            }""")
            return len(text.strip()) if text else 0
        except Exception:
            return 0
