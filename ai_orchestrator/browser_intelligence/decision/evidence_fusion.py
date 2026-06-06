"""EvidenceFusion — weighted sensor fusion with confidence tracking.

Fuses sensor outputs into a unified observation, weighting each
sensor by its real-time confidence score. Implements observation
aging and evidence decay for temporal coherence.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class SensorConfidence:
    """Per-sensor confidence tracking."""

    name: str
    success_count: int = 0
    failure_count: int = 0
    last_success_time: float = 0.0
    last_error_time: float = 0.0
    consecutive_failures: int = 0

    def record_success(self) -> None:
        self.success_count += 1
        self.last_success_time = time.monotonic()
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_error_time = time.monotonic()
        self.consecutive_failures += 1

    @property
    def confidence(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0

        success_rate = self.success_count / total

        if self.consecutive_failures >= 5:
            return 0.1

        if self.consecutive_failures >= 3:
            return max(0.2, success_rate * 0.5)

        now = time.monotonic()
        if self.last_success_time > 0 and (now - self.last_success_time) > 60:
            return max(0.3, success_rate * 0.5)

        return success_rate


@dataclass
class EvidenceVector:
    """Weighted evidence from multiple sensors for a single boolean signal."""

    signal_name: str
    value: bool = False
    weight: float = 0.0
    source_count: int = 0
    fused_confidence: float = 0.0
    tick: int = 0


class EvidenceFusion:
    """Multi-sensor evidence fusion with temporal decay.

    Each sensor contributes evidence with a confidence weight.
    Evidence is aged over time so stale observations have less
    influence on current state estimation.
    """

    def __init__(self, half_life_ticks: int = 30):
        if half_life_ticks <= 0:
            raise ValueError("half_life_ticks must be positive")
        self._half_life = half_life_ticks
        self._decay = math.log(2) / half_life_ticks
        self._sensor_confidence: dict[str, SensorConfidence] = {}
        self._evidence_buffer: dict[str, list[EvidenceVector]] = {}
        self._tick: int = 0

    def register_sensor(self, name: str) -> None:
        if name not in self._sensor_confidence:
            self._sensor_confidence[name] = SensorConfidence(name=name)

    def record_sensor_success(self, name: str) -> None:
        self.register_sensor(name)
        self._sensor_confidence[name].record_success()

    def record_sensor_failure(self, name: str) -> None:
        self.register_sensor(name)
        self._sensor_confidence[name].record_failure()

    def sensor_confidence(self, name: str) -> float:
        sc = self._sensor_confidence.get(name)
        if sc is None:
            return 1.0
        return sc.confidence

    def all_sensor_confidences(self) -> dict[str, float]:
        return {
            name: sc.confidence
            for name, sc in self._sensor_confidence.items()
        }

    def submit_evidence(
        self, sensor_name: str, signal_name: str, value: bool, confidence: float | None = None
    ) -> EvidenceVector:
        if confidence is None:
            confidence = self.sensor_confidence(sensor_name)

        self._tick += 1
        ev = EvidenceVector(
            signal_name=signal_name,
            value=value,
            weight=max(0.0, min(1.0, confidence)),
            source_count=1,
            tick=self._tick,
        )

        key = f"{sensor_name}:{signal_name}"
        if key not in self._evidence_buffer:
            self._evidence_buffer[key] = []
        self._evidence_buffer[key].append(ev)

        if len(self._evidence_buffer[key]) > 100:
            self._evidence_buffer[key].pop(0)

        return ev

    def fused_confidence(self, signal_name: str, max_age_ticks: int = 30) -> float:
        matches: list[EvidenceVector] = []
        for key, evs in self._evidence_buffer.items():
            if key.endswith(f":{signal_name}") and evs:
                latest = evs[-1]
                age = self._tick - latest.tick
                if age <= max_age_ticks:
                    matches.append(latest)

        if not matches:
            return 0.0

        total_weight = 0.0
        for ev in matches:
            age = self._tick - ev.tick
            age_weight = math.exp(-self._decay * age)
            total_weight += ev.weight * age_weight

        return min(total_weight / max(len(matches), 1), 1.0)

    def _count_more_recent(self, ev: EvidenceVector) -> int:
        count = 0
        for evs in self._evidence_buffer.values():
            for other in evs:
                if other.tick > ev.tick:
                    count += 1
        return count

    def clear(self) -> None:
        self._evidence_buffer.clear()

    def reset(self) -> None:
        self._sensor_confidence.clear()
        self._evidence_buffer.clear()
        self._tick = 0
