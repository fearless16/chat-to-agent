"""Feature vector — the unified observation produced every tick (1 Hz)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

TICK_INTERVAL: float = 1.0


@dataclass
class FeatureVector:
    """The unified observation vector produced every tick.

    Each tick, the FeatureComposer collects raw features from all
    sensors and assembles them into this vector. The vector is then
    pushed to the FeatureStore for time-series analysis and fed to
    the HMM engine for state estimation.
    """

    tick: int = 0
    timestamp: float = 0.0

    input_visible: bool = False
    send_enabled: bool = False
    stop_button_visible: bool = False
    regenerate_visible: bool = False
    error_banner_visible: bool = False
    auth_form_visible: bool = False

    text_input_count: int = 0
    button_count: int = 0
    has_thinking_marker: bool = False
    has_error_marker: bool = False
    has_rate_limit_marker: bool = False
    has_streaming_marker: bool = False

    stream_active: bool = False
    transport_detected: bool = False
    generation_started: bool = False
    generation_completed: bool = False
    stream_closed: bool = False
    generation_stop_detected: bool = False

    tokens_per_second: float = 0.0
    stream_idle_time: float = 0.0
    total_chunks: int = 0
    bytes_received: int = 0
    network_request_rate: float = 0.0

    mutation_rate: float = 0.0
    mutation_acceleration: float = 0.0

    js_heap_used_mb: float = 0.0
    page_stability: float = 1.0

    response_length: int = 0
    response_length_delta: int = 0

    visual_stability: float = 1.0

    a11y_extraction_success: bool = False
    a11y_confidence: float = 1.0
    a11y_node_count: int = 0

    page_title: str = ""
    url: str = ""

    def to_list(self) -> list[float]:
        # Binary features (19): boolean indicators
        binary = [
            float(self.input_visible),           # 0
            float(self.send_enabled),            # 1
            float(self.stop_button_visible),     # 2
            float(self.regenerate_visible),      # 3
            float(self.error_banner_visible),    # 4
            float(self.auth_form_visible),       # 5
            float(self.text_input_count > 0),    # 6: has_text_input
            float(self.button_count > 0),        # 7: has_button
            float(self.has_thinking_marker),     # 8
            float(self.has_error_marker),        # 9
            float(self.has_rate_limit_marker),   # 10
            float(self.has_streaming_marker),    # 11
            float(self.stream_active),           # 12
            float(self.transport_detected),      # 13
            float(self.generation_started),      # 14
            float(self.generation_completed),    # 15
            float(self.stream_closed),           # 16
            float(self.generation_stop_detected),# 17
            float(self.a11y_extraction_success), # 18
        ]
        # Continuous features (14): rates, lengths, counts
        continuous = [
            float(self.mutation_rate),           # 0
            float(self.mutation_acceleration),   # 1
            float(self.js_heap_used_mb),         # 2
            float(self.page_stability),          # 3
            float(self.response_length),         # 4
            float(self.response_length_delta),   # 5
            float(self.visual_stability),        # 6
            float(self.tokens_per_second),       # 7
            float(self.stream_idle_time),        # 8
            float(self.total_chunks),            # 9
            float(self.bytes_received),          # 10
            float(self.network_request_rate),    # 11
            float(self.a11y_confidence),         # 12
            float(self.a11y_node_count),         # 13
        ]
        return binary + continuous


class FeatureStore:
    """Ring buffer of FeatureVectors with time-series analysis.

    Capacity: 300 ticks = 5 minutes at 1 Hz.
    Minimum capacity: 1.
    """

    _MIN_CAPACITY = 1

    def __init__(self, capacity: int = 300):
        if capacity < self._MIN_CAPACITY:
            raise ValueError(
                f"FeatureStore capacity must be >= {self._MIN_CAPACITY}, got {capacity}"
            )
        self._capacity = capacity
        self._buffer: deque[FeatureVector] = deque(maxlen=capacity)
        self._ema_alpha: float = 0.3

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, fv: FeatureVector) -> None:
        self._buffer.append(fv)

    @property
    def latest(self) -> FeatureVector | None:
        return self._buffer[-1] if self._buffer else None

    @property
    def size(self) -> int:
        return len(self._buffer)

    def window(self, n: int) -> list[FeatureVector]:
        return list(self._buffer)[-n:]

    def ema(self, field: str, n: int = 10) -> float:
        values = [float(getattr(fv, field, 0)) for fv in self.window(n)]
        if not values:
            return 0.0
        result = values[0]
        for v in values[1:]:
            result = self._ema_alpha * v + (1 - self._ema_alpha) * result
        return result

    def aged_mean(self, field: str, n: int = 10, half_life_ticks: int = 30) -> float:
        """Weighted mean with exponential aging.

        Recent observations get higher weight via exp(-age/half_life).
        """
        window = self.window(n)
        m = len(window)
        if m == 0:
            return 0.0
        if half_life_ticks <= 0:
            raise ValueError("half_life_ticks must be positive")

        decay = math.log(2) / half_life_ticks
        total_weight = 0.0
        weighted_sum = 0.0

        for i, fv in enumerate(window):
            age = m - 1 - i
            weight = math.exp(-decay * age)
            val = float(getattr(fv, field, 0))
            weighted_sum += weight * val
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    def obsolescence_weight(self, age_ticks: int, half_life_ticks: int = 30) -> float:
        """Weight for an observation aged `age_ticks` ticks."""
        if half_life_ticks <= 0:
            raise ValueError("half_life_ticks must be positive")
        decay = math.log(2) / half_life_ticks
        return math.exp(-decay * age_ticks)

    def derivative(self, field: str, n: int = 5) -> float:
        window = self.window(n)
        if len(window) < 2:
            return 0.0
        t0 = window[0].timestamp
        tn = window[-1].timestamp
        dt = tn - t0
        if dt <= 0:
            return 0.0
        v0 = float(getattr(window[0], field, 0))
        vn = float(getattr(window[-1], field, 0))
        return (vn - v0) / dt

    def second_derivative(self, field: str, n: int = 5) -> float:
        if len(self._buffer) < n + 2:
            return 0.0
        d1 = self.derivative(field, n)
        older = list(self._buffer)[-(n + 2):-2]
        if len(older) < n:
            return 0.0
        t0 = older[0].timestamp
        tn = older[-1].timestamp
        dt = tn - t0
        if dt <= 0:
            return 0.0
        v0 = float(getattr(older[0], field, 0))
        vn = float(getattr(older[-1], field, 0))
        d0 = (vn - v0) / dt
        return (d1 - d0) / TICK_INTERVAL

    def mean(self, field: str, n: int = 10) -> float:
        values = [float(getattr(fv, field, 0)) for fv in self.window(n)]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def std(self, field: str, n: int = 10) -> float:
        values = [float(getattr(fv, field, 0)) for fv in self.window(n)]
        if len(values) < 2:
            return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    def clear(self) -> None:
        self._buffer.clear()
