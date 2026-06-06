"""Browser Intelligence — learning subsystem.

Provides:
- BayesianReliability — Beta-binomial posterior reliability tracker.
- ProviderReliabilityStore — per-provider / per-account / per-selector
  reliability table with persistence.
- BayesianUpdate — pure-function Bayesian update helpers.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Pure math
# ──────────────────────────────────────────────────────────────────────

def beta_pdf(x: float, alpha: float, beta_param: float) -> float:
    """Unnormalized Beta PDF (the B(alpha,beta) normalization constant
    is omitted since posteriors are renormalized when compared)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    if alpha <= 0.0 or beta_param <= 0.0:
        return 0.0
    logp = (alpha - 1.0) * math.log(x) + (beta_param - 1.0) * math.log(1.0 - x)
    return math.exp(logp)


def beta_mean(alpha: float, beta_param: float) -> float:
    return alpha / (alpha + beta_param)


def beta_variance(alpha: float, beta_param: float) -> float:
    a, b = alpha, beta_param
    return (a * b) / ((a + b) ** 2 * (a + b + 1))


def beta_update(
    alpha: float,
    beta_param: float,
    success: bool,
) -> tuple[float, float]:
    """Bayesian update of Beta(alpha, beta) with a Bernoulli outcome.

    Returns the new (alpha, beta).
    """
    if success:
        return alpha + 1.0, beta_param
    return alpha, beta_param + 1.0


def beta_sample_mean(alpha: float, beta_param: float) -> float:
    """Estimate of the mean of the Beta posterior.

    The Beta posterior's expected value is alpha / (alpha + beta).
    """
    return beta_mean(alpha, beta_param)


def log_odds(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return math.log(p / (1.0 - p))


def probability_from_log_odds(lo: float) -> float:
    if lo == math.inf:
        return 1.0
    if lo == -math.inf:
        return 0.0
    return 1.0 / (1.0 + math.exp(-lo))


# ──────────────────────────────────────────────────────────────────────
# Reliability tracker
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BayesianReliability:
    """Beta-binomial posterior reliability tracker.

    Prior is Beta(prior_alpha, prior_beta). On every observation we
    update and can read off:
    - `posterior_mean`: the expected success probability.
    - `variance`: posterior uncertainty.
    - `credible_interval(p)`: equal-tailed interval.

    Exponential decay: old observations lose weight over time.
    The `apply_decay()` method shrinks successes and failures toward
    the prior at the specified rate, ensuring stale data doesn't
    dominate the posterior.
    """

    key: str
    successes: int = 0
    failures: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    last_updated: float = field(default_factory=time.time)
    _effective_successes: float = 0.0
    _effective_failures: float = 0.0
    _decay_rate: float = 0.0
    _decay_interval: float = 3600.0

    def update(self, success: bool) -> None:
        now = time.time()
        self._apply_decay_if_stale(now)
        if success:
            self.successes += 1
            self._effective_successes += 1.0
        else:
            self.failures += 1
            self._effective_failures += 1.0
        self.last_updated = now

    def update_many(self, outcomes: Iterable[bool]) -> None:
        for ok in outcomes:
            self.update(ok)

    def configure_decay(self, rate: float = 0.001, interval_seconds: float = 3600.0) -> None:
        """Set exponential decay.
        
        Args:
            rate: decay constant per interval (0.001 means ~0.1% per hour).
            interval_seconds: how often to apply decay (default 1 hour).
        
        With rate=0.001, after 10 hours effective count ≈ count * e^(-0.01).
        """
        self._decay_rate = float(rate)
        self._decay_interval = float(interval_seconds)
        self._effective_successes = float(self.successes)
        self._effective_failures = float(self.failures)

    def apply_decay(self) -> None:
        """Manually trigger decay now."""
        now = time.time()
        self._apply_decay_if_stale(now)

    def _apply_decay_if_stale(self, now: float) -> None:
        if self._decay_rate <= 0:
            return
        if self._effective_successes <= 0 and self._effective_failures <= 0:
            self._effective_successes = float(self.successes)
            self._effective_failures = float(self.failures)
            return
        elapsed = max(0.0, now - self.last_updated)
        if elapsed < self._decay_interval:
            return
        periods = elapsed / self._decay_interval
        factor = math.exp(-self._decay_rate * periods)
        self._effective_successes *= factor
        self._effective_failures *= factor

    @property
    def alpha(self) -> float:
        if self._decay_rate > 0:
            return self.prior_alpha + self._effective_successes
        return self.prior_alpha + self.successes

    @property
    def beta_param(self) -> float:
        if self._decay_rate > 0:
            return self.prior_beta + self._effective_failures
        return self.prior_beta + self.failures

    @property
    def posterior_mean(self) -> float:
        return beta_mean(self.alpha, self.beta_param)

    @property
    def variance(self) -> float:
        return beta_variance(self.alpha, self.beta_param)

    @property
    def total_observations(self) -> int:
        return self.successes + self.failures

    def credible_interval(self, p: float = 0.94) -> tuple[float, float]:
        """Equal-tailed credible interval. Falls back to a wide range
        when we have very few observations."""
        if self.total_observations < 2:
            return 0.0, 1.0
        alpha = self.alpha
        beta_param = self.beta_param
        # Simple closed-form approximation:
        # mu = alpha / (alpha + beta)
        # var = alpha*beta / ((a+b)^2 (a+b+1))
        mu = self.posterior_mean
        var = self.variance
        # 1.96 ≈ z_{0.975} for the 94% CI heuristic.
        z = 1.88
        half = z * math.sqrt(max(var, 0.0))
        lo = max(0.0, mu - half)
        hi = min(1.0, mu + half)
        return lo, hi

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "successes": self.successes,
            "failures": self.failures,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "last_updated": self.last_updated,
            "posterior_mean": self.posterior_mean,
            "effective_successes": self._effective_successes,
            "effective_failures": self._effective_failures,
            "decay_rate": self._decay_rate,
            "decay_interval": self._decay_interval,
        }


# ──────────────────────────────────────────────────────────────────────
# Per-target reliability store (provider / account / selector / action)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _Targets:
    providers: dict[str, BayesianReliability] = field(default_factory=dict)
    accounts: dict[str, BayesianReliability] = field(default_factory=dict)
    selectors: dict[str, BayesianReliability] = field(default_factory=dict)
    actions: dict[str, BayesianReliability] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "providers": {k: _serialize(v) for k, v in self.providers.items()},
            "accounts": {k: _serialize(v) for k, v in self.accounts.items()},
            "selectors": {k: _serialize(v) for k, v in self.selectors.items()},
            "actions": {k: _serialize(v) for k, v in self.actions.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "_Targets":
        out = cls()
        for k, v in (raw.get("providers") or {}).items():
            out.providers[k] = _deserialize(k, v)
        for k, v in (raw.get("accounts") or {}).items():
            out.accounts[k] = _deserialize(k, v)
        for k, v in (raw.get("selectors") or {}).items():
            out.selectors[k] = _deserialize(k, v)
        for k, v in (raw.get("actions") or {}).items():
            out.actions[k] = _deserialize(k, v)
        return out


def _serialize(r: BayesianReliability) -> dict:
    return {
        "successes": r.successes,
        "failures": r.failures,
        "prior_alpha": r.prior_alpha,
        "prior_beta": r.prior_beta,
        "last_updated": r.last_updated,
        "effective_successes": r._effective_successes,
        "effective_failures": r._effective_failures,
        "decay_rate": r._decay_rate,
        "decay_interval": r._decay_interval,
    }


def _deserialize(key: str, d: dict) -> BayesianReliability:
    r = BayesianReliability(
        key=key,
        successes=int(d.get("successes", 0)),
        failures=int(d.get("failures", 0)),
        prior_alpha=float(d.get("prior_alpha", 1.0)),
        prior_beta=float(d.get("prior_beta", 1.0)),
        last_updated=float(d.get("last_updated", time.time())),
    )
    r._effective_successes = float(d.get("effective_successes", 0) or 0)
    r._effective_failures = float(d.get("effective_failures", 0) or 0)
    r._decay_rate = float(d.get("decay_rate", 0) or 0)
    r._decay_interval = float(d.get("decay_interval", 0) or 0)
    return r


class ProviderReliabilityStore:
    """Aggregated Bayesian reliability table for the engine.

    The store keeps separate reliability objects for providers,
    accounts, selectors, and recovery actions, and can persist them
    to disk so cross-session learning survives a process restart.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self._path = Path(path) if path else None
        self._prior_alpha = float(prior_alpha)
        self._prior_beta = float(prior_beta)
        self._targets = _Targets()
        self._loaded = False
        if self._path is not None and self._path.exists():
            try:
                self._load()
            except Exception as exc:
                log.warning("Failed to load reliability store %s: %s", self._path, exc)
                self._targets = _Targets()
        self._loaded = True

    # ── Accessors ───────────────────────────────────────────────

    def _bucket(self, kind: str) -> dict[str, BayesianReliability]:
        if kind == "provider":
            return self._targets.providers
        if kind == "account":
            return self._targets.accounts
        if kind == "selector":
            return self._targets.selectors
        if kind == "action":
            return self._targets.actions
        raise KeyError(kind)

    def get(self, kind: str, key: str) -> BayesianReliability:
        bucket = self._bucket(kind)
        rel = bucket.get(key)
        if rel is None:
            rel = BayesianReliability(
                key=key,
                prior_alpha=self._prior_alpha,
                prior_beta=self._prior_beta,
            )
            bucket[key] = rel
        return rel

    def record(self, kind: str, key: str, success: bool) -> BayesianReliability:
        rel = self.get(kind, key)
        rel.update(success)
        return rel

    def snapshot(self) -> dict:
        return {
            "providers": {k: v.posterior_mean for k, v in self._targets.providers.items()},
            "accounts": {k: v.posterior_mean for k, v in self._targets.accounts.items()},
            "selectors": {k: v.posterior_mean for k, v in self._targets.selectors.items()},
            "actions": {k: v.posterior_mean for k, v in self._targets.actions.items()},
        }

    # ── Persistence ─────────────────────────────────────────────

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._targets.to_dict(), indent=2, sort_keys=True)
        )
        tmp.replace(self._path)

    def _load(self) -> None:
        raw = json.loads(self._path.read_text())  # type: ignore[union-attr]
        self._targets = _Targets.from_dict(raw or {})

    def reset(self) -> None:
        self._targets = _Targets()


__all__ = [
    "BayesianReliability",
    "ProviderReliabilityStore",
    "beta_update",
    "beta_mean",
    "beta_variance",
    "beta_pdf",
    "log_odds",
    "probability_from_log_odds",
]
