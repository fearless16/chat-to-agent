"""CompletionEngine — response completion detection via signal processing.

Absolutely no sleep(). No fixed timers.
Observes response_length(t) → velocity → acceleration → completion confidence.
"""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.browser_intelligence.estimation.kalman_filter import ResponseKalmanFilter
from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureStore


class CompletionEngine:
    """Response completion detection via velocity and acceleration analysis.

    Response is complete when:
        velocity → 0 AND acceleration → 0 AND confidence > threshold

    Uses a Kalman filter for smoothing the noisy length signal.
    """

    def __init__(
        self,
        velocity_threshold: float = 2.0,
        stable_for_ticks: int = 3,
        confidence_threshold: float = 0.85,
        min_response_length: int = 20,
    ):
        self._velocity_threshold = velocity_threshold
        self._stable_for_ticks = stable_for_ticks
        self._confidence_threshold = confidence_threshold
        self._min_response_length = min_response_length
        self._kf = ResponseKalmanFilter()
        self._stable_count = 0

    def is_complete(self, feature_store: FeatureStore) -> tuple[bool, float]:
        """Determine if response is complete.

        Returns:
            (is_done: bool, confidence: float in [0, 1])
        """
        if feature_store.size < 5:
            return False, 0.0

        window = feature_store.window(10)
        lengths = [float(fv.response_length) for fv in window]

        self._kf.reset()
        smoothed = self._kf.smooth(lengths)
        if not smoothed:
            return False, 0.0

        velocity = self._kf.velocity()
        abs_velocity = abs(velocity)

        acceleration = self._kf.acceleration()
        abs_accel = abs(acceleration)

        if abs_velocity < self._velocity_threshold and abs_accel < 1.0:
            self._stable_count += 1
        else:
            self._stable_count = 0

        confidence = min(
            self._stable_count / self._stable_for_ticks, 1.0
        )

        latest = feature_store.latest
        has_content = (
            latest is not None
            and (latest.response_length > self._min_response_length
                 or latest.total_chunks > 5)
        )
        stream_done = latest is not None and (
            latest.generation_completed or latest.stream_closed
        )
        stream_idle_confirm = latest is not None and (
            latest.stream_idle_time > 5.0 and latest.total_chunks > 5
        )
        not_streaming = latest is not None and (
            not latest.stream_active
            and not latest.has_streaming_marker
            and not latest.stop_button_visible
        )
        network_stop = (
            latest is not None and latest.generation_stop_detected
        )

        done = (
            self._stable_count >= self._stable_for_ticks
            and confidence >= self._confidence_threshold
            and has_content
            and (not_streaming or network_stop or stream_done or stream_idle_confirm)
        )

        return done, confidence

    def completion_confidence(self, feature_store: FeatureStore) -> float:
        """Continuous confidence score [0, 1]."""
        _, conf = self.is_complete(feature_store)
        return conf

    def reset(self) -> None:
        self._kf.reset()
        self._stable_count = 0
