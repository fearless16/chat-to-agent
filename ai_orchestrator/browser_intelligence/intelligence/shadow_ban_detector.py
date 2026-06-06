"""ShadowBanDetector — multi-signal posterior for soft / hard shadow bans.

A shadow-banned account still appears to "work" from the user's side,
but the model returns truncated, off-topic, or low-quality replies, or
the stream stalls. We estimate P(shadow_ban | signals) using
independent Bayesian updates from:

- response length (z-score vs baseline)
- response quality heuristic (length, word diversity, code-block ratio)
- completion rate (did the stream actually finish?)
- stream latency (tokens/sec z-score)
- error / rate-limit frequency

Output:
- NORMAL: no ban signal
- DEGRADED: some signals suggest soft throttling
- SHADOW_BANNED: confident ban estimate

The detector keeps per-account history so the baseline updates slowly
and outliers don't poison it.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class ShadowBanState(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    SHADOW_BANNED = "SHADOW_BANNED"


@dataclass
class ShadowBanVerdict:
    state: ShadowBanState
    p_shadow_ban: float
    p_degraded: float
    p_normal: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "p_shadow_ban": round(self.p_shadow_ban, 4),
            "p_degraded": round(self.p_degraded, 4),
            "p_normal": round(self.p_normal, 4),
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "samples": self.sample_count,
        }


@dataclass
class _Observation:
    response_length: int
    completion_rate: float
    tokens_per_second: float
    error_count: int
    quality_score: float
    timestamp: float = field(default_factory=time.monotonic)


class ShadowBanDetector:
    """Bayesian-ish posterior over shadow-ban states per account.

    The detector is intentionally lightweight: no matrices, no solver.
    Each observation nudges the posterior by fixed weights and we
    threshold for the categorical label.
    """

    DEFAULT_BASELINE_WINDOW: int = 50

    def __init__(
        self,
        baseline_window: int = DEFAULT_BASELINE_WINDOW,
        ban_threshold: float = 0.65,
        degraded_threshold: float = 0.30,
    ):
        self._baseline_window = int(baseline_window)
        self._ban_threshold = float(ban_threshold)
        self._degraded_threshold = float(degraded_threshold)
        self._history: deque[_Observation] = deque(maxlen=self._baseline_window)
        self._posterior: tuple[float, float, float] = (1.0, 0.0, 0.0)
        self._stale_count: int = 0
        self._last_verdict: ShadowBanVerdict | None = None

    @property
    def last_verdict(self) -> ShadowBanVerdict | None:
        return self._last_verdict

    def reset(self) -> None:
        self._history.clear()
        self._posterior = (1.0, 0.0, 0.0)
        self._stale_count = 0
        self._last_verdict = None

    def observe(
        self,
        *,
        response_length: int,
        completion_rate: float,
        tokens_per_second: float,
        error_count: int = 0,
        quality_score: float = 0.0,
    ) -> ShadowBanVerdict:
        obs = _Observation(
            response_length=int(response_length),
            completion_rate=float(completion_rate),
            tokens_per_second=float(tokens_per_second),
            error_count=int(error_count),
            quality_score=float(quality_score),
        )
        self._history.append(obs)
        return self._update_posterior(obs)

    def _update_posterior(self, obs: _Observation) -> ShadowBanVerdict:
        if len(self._history) < 5:
            # Not enough data; emit prior.
            return self._emit(len(self._history), ["insufficient_data"])

        lens = [o.response_length for o in self._history]
        tps = [o.tokens_per_second for o in self._history]
        mean_len = statistics_mean(lens)
        std_len = statistics_stdev(lens, mean=mean_len)
        mean_tps = statistics_mean(tps)

        reasons: list[str] = []
        log_likelihoods = {"normal": 0.0, "degraded": 0.0, "shadow": 0.0}

        # ── Response length z-score ─────────────────────────────
        z_len = 0.0
        if std_len > 0:
            z_len = (obs.response_length - mean_len) / max(std_len, 1.0)
        # long-truncated is shadow-bannish; very long is normal.
        if z_len < -2.0:
            log_likelihoods["shadow"] += 1.5
            reasons.append(f"short_response(z={z_len:.2f})")
        elif z_len < -1.0:
            log_likelihoods["degraded"] += 1.0
            reasons.append(f"shortish_response(z={z_len:.2f})")
        else:
            log_likelihoods["normal"] += 0.5

        # ── Completion rate ─────────────────────────────────────
        cr = obs.completion_rate
        if cr < 0.4:
            log_likelihoods["shadow"] += 1.4
            reasons.append(f"low_completion({cr:.2f})")
        elif cr < 0.75:
            log_likelihoods["degraded"] += 0.8
            reasons.append(f"partial_completion({cr:.2f})")
        else:
            log_likelihoods["normal"] += 0.5

        # ── Token rate vs baseline ──────────────────────────────
        if mean_tps > 0:
            ratio = obs.tokens_per_second / mean_tps
            if ratio < 0.3:
                log_likelihoods["shadow"] += 1.0
                reasons.append(f"slow_stream(ratio={ratio:.2f})")
            elif ratio < 0.6:
                log_likelihoods["degraded"] += 0.5
            else:
                log_likelihoods["normal"] += 0.3

        # ── Error / rate-limit count ────────────────────────────
        if obs.error_count >= 3:
            log_likelihoods["shadow"] += 0.8
            reasons.append(f"errors={obs.error_count}")
        elif obs.error_count == 0:
            log_likelihoods["normal"] += 0.2

        # ── Quality ─────────────────────────────────────────────
        if obs.quality_score < 0.2:
            log_likelihoods["shadow"] += 0.7
            reasons.append(f"low_quality({obs.quality_score:.2f})")
        elif obs.quality_score < 0.5:
            log_likelihoods["degraded"] += 0.3
        else:
            log_likelihoods["normal"] += 0.2

        # Convert log-likelihoods to normalized posterior.
        # Use softmax with a temperature that keeps things in a
        # reasonable range.
        temp = 2.0
        n = math.exp(log_likelihoods["normal"] / temp)
        d = math.exp(log_likelihoods["degraded"] / temp)
        s = math.exp(log_likelihoods["shadow"] / temp)
        z = n + d + s
        if z <= 0:
            z = 1.0
        p_normal = n / z
        p_degraded = d / z
        p_shadow = s / z

        # EMA blend with previous posterior — keeps state stable.
        prev = self._posterior
        alpha = 0.6
        p_normal = alpha * p_normal + (1 - alpha) * prev[0]
        p_degraded = alpha * p_degraded + (1 - alpha) * prev[1]
        p_shadow = alpha * p_shadow + (1 - alpha) * prev[2]
        # Renormalize.
        s2 = p_normal + p_degraded + p_shadow
        p_normal /= s2
        p_degraded /= s2
        p_shadow /= s2

        self._posterior = (p_normal, p_degraded, p_shadow)

        if p_shadow >= self._ban_threshold:
            state = ShadowBanState.SHADOW_BANNED
        elif p_shadow + p_degraded >= self._degraded_threshold:
            state = ShadowBanState.DEGRADED
        else:
            state = ShadowBanState.NORMAL

        confidence = max(p_normal, p_degraded, p_shadow)
        verdict = ShadowBanVerdict(
            state=state,
            p_shadow_ban=p_shadow,
            p_degraded=p_degraded,
            p_normal=p_normal,
            confidence=confidence,
            reasons=reasons,
            sample_count=len(self._history),
        )
        self._last_verdict = verdict
        return verdict

    def _emit(self, n: int, reasons: list[str]) -> ShadowBanVerdict:
        p_normal, p_degraded, p_shadow = self._posterior
        if p_shadow >= self._ban_threshold:
            state = ShadowBanState.SHADOW_BANNED
        elif p_shadow + p_degraded >= self._degraded_threshold:
            state = ShadowBanState.DEGRADED
        else:
            state = ShadowBanState.NORMAL
        verdict = ShadowBanVerdict(
            state=state,
            p_shadow_ban=p_shadow,
            p_degraded=p_degraded,
            p_normal=p_normal,
            confidence=max(p_normal, p_degraded, p_shadow),
            reasons=reasons,
            sample_count=n,
        )
        self._last_verdict = verdict
        return verdict


def statistics_mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def statistics_stdev(xs: list[float], mean: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean if mean is not None else statistics_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)
