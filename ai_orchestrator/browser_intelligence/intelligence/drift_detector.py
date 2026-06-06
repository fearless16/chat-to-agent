"""ProviderDriftDetector — tracks structural drift in provider pages.

Modern chat UIs ship continuously. DOM hierarchies shift, accessibility
roles get renamed, network signatures change (new auth headers, new SSE
field names, new WebSocket sub-protocols). A model trained on last
week's structure will degrade silently unless drift is detected and
the engine relearns.

Drift signals:
- DOM topology: tag/role/class fingerprints of "this looks like the
  chat input / send button / assistant bubble" nodes.
- Accessibility topology: ARIA roles and property names.
- Network signatures: streaming content-type, delta-key names, request
  method/path shape.
- Stream signatures: chunk sizes, inter-chunk cadence, [DONE] marker.

Output: a `DriftSnapshot` with a drift score in [0, 1]. When it
exceeds a threshold, downstream code should invalidate stale
assumptions and trigger relearning.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)


@dataclass
class DriftSignal:
    """A single observation feeding the drift detector."""

    kind: str  # "dom" | "a11y" | "network" | "stream"
    fingerprint: str
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class DriftSnapshot:
    """The detector's current view of provider drift."""

    drift_score: float
    sample_count: int
    signal_kinds: dict[str, int]
    novel_signal_count: int
    threshold: float
    should_relearn: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "drift_score": round(self.drift_score, 4),
            "sample_count": self.sample_count,
            "signal_kinds": self.signal_kinds,
            "novel_signals": self.novel_signal_count,
            "threshold": self.threshold,
            "should_relearn": self.should_relearn,
            "reason": self.reason,
        }


class DriftDetector:
    """Tracks provider-side structural drift across sessions.

    The detector keeps a rolling window of fingerprints per signal
    kind and reports a drift score in [0, 1] based on two things:

    1. The fraction of recent signals that are "novel" — never seen
       before in this provider's history.
    2. The Shannon entropy of the per-kind fingerprint distribution.
       A sudden entropy drop means the provider started serving a
       different shape exclusively.
    """

    DEFAULT_WINDOW: int = 200
    DEFAULT_THRESHOLD: float = 0.35

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self._window = int(window)
        self._threshold = float(threshold)
        self._known: dict[str, set[str]] = {
            "dom": set(),
            "a11y": set(),
            "network": set(),
            "stream": set(),
        }
        self._recent: list[DriftSignal] = []
        self._counts: dict[str, int] = {
            "dom": 0, "a11y": 0, "network": 0, "stream": 0,
        }
        self._novel_total: int = 0
        self._last_snapshot: DriftSnapshot | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, float(value)))

    @property
    def known_fingerprints(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._known.items()}

    def observe(self, signal: DriftSignal) -> None:
        if signal.kind not in self._known:
            log.debug("DriftDetector: unknown kind %s, dropping", signal.kind)
            return
        bucket = self._known[signal.kind]
        if signal.fingerprint not in bucket:
            bucket.add(signal.fingerprint)
            self._novel_total += 1
        self._recent.append(signal)
        self._counts[signal.kind] += 1
        if len(self._recent) > self._window:
            old = self._recent.pop(0)
            self._counts[old.kind] = max(0, self._counts[old.kind] - 1)

    def observe_many(self, signals: Iterable[DriftSignal]) -> None:
        for s in signals:
            self.observe(s)

    def snapshot(self) -> DriftSnapshot:
        n = len(self._recent)
        kinds = {k: c for k, c in self._counts.items() if c}
        if n == 0:
            snap = DriftSnapshot(
                drift_score=0.0,
                sample_count=0,
                signal_kinds={},
                novel_signal_count=self._novel_total,
                threshold=self._threshold,
                should_relearn=False,
                reason="no_data",
            )
            self._last_snapshot = snap
            return snap

        # Novelty score: fraction of recent signals whose fingerprint
        # appears < 3 times in history.
        recent_fps = Counter(s.fingerprint for s in self._recent)
        novel = sum(1 for fp, c in recent_fps.items() if c < 3)
        novelty = novel / max(1, len(recent_fps))

        # Entropy: Shannon entropy of the kind distribution.
        ent = _shannon_entropy(self._counts)

        # Drift score = weighted combination: novelty (faster-moving
        # signals) dominates, entropy gives a slower baseline.
        score = 0.70 * novelty + 0.30 * (1.0 - ent / 2.0)
        score = max(0.0, min(1.0, score))

        should = score >= self._threshold
        reason_parts = []
        if novelty > 0.5:
            reason_parts.append(f"novel_fingerprints={novel}")
        if ent < 0.5 and n > 20:
            reason_parts.append(f"low_entropy={ent:.2f}")
        if not reason_parts:
            reason_parts.append(f"score={score:.3f}")

        snap = DriftSnapshot(
            drift_score=score,
            sample_count=n,
            signal_kinds=kinds,
            novel_signal_count=len(recent_fps),
            threshold=self._threshold,
            should_relearn=should,
            reason="; ".join(reason_parts),
        )
        self._last_snapshot = snap
        return snap

    def reset(self) -> None:
        self._known = {k: set() for k in self._known}
        self._recent.clear()
        self._counts = {k: 0 for k in self._counts}
        self._novel_total = 0
        self._last_snapshot = None

    def seed_history(self, signals: Iterable[DriftSignal]) -> None:
        """Pre-populate `known` from saved state so the very first
        post-restart observations don't all look novel."""
        for s in signals:
            if s.kind in self._known:
                self._known[s.kind].add(s.fingerprint)
                self._counts[s.kind] += 1

    def export(self) -> dict:
        return {
            "known": {k: sorted(v) for k, v in self._known.items()},
            "window": self._window,
            "threshold": self._threshold,
        }


def _log2(p: float) -> float:
    if p <= 0.0:
        return 0.0
    import math
    return math.log2(p)


def _shannon_entropy(counts: dict[str, int]) -> float:
    total = sum(c for c in counts.values() if c > 0)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * _log2(p)
    return h
