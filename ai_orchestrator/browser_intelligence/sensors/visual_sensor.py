"""VisualSensor — low-frequency visual observation for major state changes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ai_orchestrator.browser_intelligence.sensors.base import BaseSensor

log = logging.getLogger(__name__)


@dataclass
class VisualFeatures:
    frame_difference: float = 0.0
    visual_stability: float = 1.0
    dominant_color_change: float = 0.0
    has_large_white_region: bool = False


class VisualSensor(BaseSensor):
    """Low-frequency visual sensor. Last resort, not primary.

    Takes periodic screenshots and computes frame differences.
    Runs every N ticks due to cost; between ticks returns cached
    features.
    """

    def __init__(self, tick_interval: int = 10, max_diff: float = 50.0):
        self._tick_interval = tick_interval
        self._max_diff = max_diff
        self._ticks_since_last = 0
        self._previous_frame = None
        self._last_features = VisualFeatures()
        self._prev_dominant: tuple[int, int, int] = (0, 0, 0)

    async def sense(self, page) -> VisualFeatures:
        self._ticks_since_last += 1

        if self._ticks_since_last < self._tick_interval:
            return self._last_features

        self._ticks_since_last = 0

        try:
            screenshot = await page.screenshot(type="jpeg", quality=30, scale="css")
            diff = self._compute_frame_diff(screenshot)
            self._last_features.frame_difference = diff
            self._last_features.visual_stability = max(
                0.0, 1.0 - min(diff / self._max_diff, 1.0)
            )
            self._previous_frame = screenshot
        except Exception as exc:
            log.debug("VisualSensor failed: %s", exc)

        return self._last_features

    def _compute_frame_diff(self, current: bytes) -> float:
        if self._previous_frame is None:
            return 0.0
        if len(current) != len(self._previous_frame):
            return 100.0
        diff_bytes = sum(
            1 for a, b in zip(current, self._previous_frame) if a != b
        )
        return (diff_bytes / max(len(current), 1)) * 100.0

    def reset(self) -> None:
        self._ticks_since_last = 0
        self._previous_frame = None
        self._last_features = VisualFeatures()
        self._prev_dominant = (0, 0, 0)
