"""Browser Intelligence — persistent per-provider "brain".

A brain survives process restarts. It captures everything a provider
session has learned so the next time we attach to ChatGPT, Qwen, Kimi,
DeepSeek, or z.ai, the engine starts with calibrated priors rather
than from scratch.

Persisted fields:
- emission_calibration: per-state calibration score the engine feeds
  to the HMM.
- selector_reliability: per-selector Bayesian posterior.
- provider_drift: snapshot from the last session.
- recovery_history: which recovery steps succeeded / failed.
- account_health: per-account Bayesian posterior.
- session_count: how many times we've attached to this provider.

Brains are stored as JSON files in a directory (one per provider).
The path is configurable so tests can use a tmp_path.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ProviderName(str, Enum):
    CHATGPT = "chatgpt"
    QWEN = "qwen"
    KIMI = "kimi"
    DEEPSEEK = "deepseek"
    ZAI = "zai"
    XIAOMI_MIMO = "xiaomimimo"
    MINIMAX = "minimax"
    UNKNOWN = "unknown"


@dataclass
class ProviderBrainState:
    """Serializable brain state."""

    provider: str
    session_count: int = 0
    emission_calibration: dict[str, float] = field(default_factory=dict)
    selector_reliability: dict[str, dict[str, float]] = field(default_factory=dict)
    account_health: dict[str, dict[str, float]] = field(default_factory=dict)
    recovery_history: list[dict[str, Any]] = field(default_factory=list)
    drift_fingerprints: dict[str, list[str]] = field(default_factory=dict)
    state_priors: dict[str, float] = field(default_factory=dict)
    last_calibration: dict[str, float] = field(default_factory=dict)
    last_updated: float = 0.0
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ProviderBrainState":
        return cls(
            provider=str(raw.get("provider", "unknown")),
            session_count=int(raw.get("session_count", 0)),
            emission_calibration=dict(raw.get("emission_calibration") or {}),
            selector_reliability=dict(raw.get("selector_reliability") or {}),
            account_health=dict(raw.get("account_health") or {}),
            recovery_history=list(raw.get("recovery_history") or []),
            drift_fingerprints=dict(raw.get("drift_fingerprints") or {}),
            state_priors=dict(raw.get("state_priors") or {}),
            last_calibration=dict(raw.get("last_calibration") or {}),
            last_updated=float(raw.get("last_updated", 0.0)),
            schema_version=int(raw.get("schema_version", 1)),
        )


class ProviderBrain:
    """A long-lived brain for one provider.

    The brain is a thin wrapper over `ProviderBrainState` plus a
    path on disk. All write operations go through `save()` so a
    crash never leaves a half-written file.
    """

    def __init__(self, provider: str | ProviderName, path: Path):
        self._provider = str(provider)
        self._path = Path(path)
        self._state = ProviderBrainState(provider=self._provider)
        self._loaded = False

    # ── File I/O ────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._state = ProviderBrainState.from_dict(raw)
            except Exception as exc:
                log.warning("Brain load failed for %s: %s", self._provider, exc)
                self._state = ProviderBrainState(provider=self._provider)
        self._loaded = True

    def save(self) -> None:
        self._state.last_updated = time.time()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state.to_dict(), indent=2, sort_keys=True))
        tmp.replace(self._path)

    # ── Read API ────────────────────────────────────────────────

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def state(self) -> ProviderBrainState:
        if not self._loaded:
            self.load()
        return self._state

    @property
    def session_count(self) -> int:
        return self.state.session_count

    @property
    def emission_calibration(self) -> dict[str, float]:
        return dict(self.state.emission_calibration)

    @property
    def state_priors(self) -> dict[str, float]:
        return dict(self.state.state_priors)

    # ── Write API ───────────────────────────────────────────────

    def record_session_start(self) -> None:
        self.state.session_count += 1

    def record_emission_calibration(self, calibration: dict[str, float]) -> None:
        if not calibration:
            return
        # Keep a moving average; old entries blend with new at α=0.3.
        alpha = 0.3
        merged = dict(self.state.emission_calibration)
        for k, v in calibration.items():
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            vf = max(0.0, min(1.0, vf))
            if k in merged:
                merged[k] = (1 - alpha) * float(merged[k]) + alpha * vf
            else:
                merged[k] = vf
        self.state.emission_calibration = merged
        self.state.last_calibration = dict(calibration)

    def record_selector_outcome(self, selector: str, success: bool) -> None:
        sel = self.state.selector_reliability.setdefault(
            selector, {"successes": 0.0, "failures": 0.0, "mean": 0.5}
        )
        if success:
            sel["successes"] = float(sel.get("successes", 0.0)) + 1.0
        else:
            sel["failures"] = float(sel.get("failures", 0.0)) + 1.0
        s = float(sel["successes"])
        f = float(sel["failures"])
        # Beta(1,1) posterior mean.
        sel["mean"] = (s + 1.0) / (s + f + 2.0)

    def record_account_outcome(self, account: str, success: bool) -> None:
        acc = self.state.account_health.setdefault(
            account, {"successes": 0.0, "failures": 0.0, "mean": 0.5}
        )
        if success:
            acc["successes"] = float(acc.get("successes", 0.0)) + 1.0
        else:
            acc["failures"] = float(acc.get("failures", 0.0)) + 1.0
        s = float(acc["successes"])
        f = float(acc["failures"])
        acc["mean"] = (s + 1.0) / (s + f + 2.0)

    def record_recovery_outcome(
        self,
        step: str,
        success: bool,
        confidence: float,
    ) -> None:
        self.state.recovery_history.append(
            {
                "step": str(step),
                "success": bool(success),
                "confidence": float(confidence),
                "ts": time.time(),
            }
        )
        # Keep the last 100 entries so the file doesn't grow forever.
        if len(self.state.recovery_history) > 100:
            self.state.recovery_history = self.state.recovery_history[-100:]

    def record_drift_fingerprints(self, fingerprints: dict[str, list[str]]) -> None:
        if not fingerprints:
            return
        merged = dict(self.state.drift_fingerprints)
        for kind, fps in fingerprints.items():
            seen = set(merged.get(kind, []))
            seen.update(fps)
            merged[kind] = sorted(seen)
        self.state.drift_fingerprints = merged

    def record_state_priors(self, priors: dict[str, float]) -> None:
        if not priors:
            return
        merged = dict(self.state.state_priors)
        for k, v in priors.items():
            try:
                vf = max(0.0, float(v))
            except (TypeError, ValueError):
                continue
            merged[k] = vf
        # Renormalize to a probability distribution.
        z = sum(merged.values())
        if z > 0:
            merged = {k: v / z for k, v in merged.items()}
        self.state.state_priors = merged

    # ── Brain factory helpers ───────────────────────────────────

    @classmethod
    def chatgpt(cls, base_dir: Path) -> "ProviderBrain":
        return cls(ProviderName.CHATGPT.value, base_dir / "chatgpt.brain.json")

    @classmethod
    def qwen(cls, base_dir: Path) -> "ProviderBrain":
        return cls(ProviderName.QWEN.value, base_dir / "qwen.brain.json")

    @classmethod
    def kimi(cls, base_dir: Path) -> "ProviderBrain":
        return cls(ProviderName.KIMI.value, base_dir / "kimi.brain.json")

    @classmethod
    def deepseek(cls, base_dir: Path) -> "ProviderBrain":
        return cls(ProviderName.DEEPSEEK.value, base_dir / "deepseek.brain.json")

    @classmethod
    def zai(cls, base_dir: Path) -> "ProviderBrain":
        return cls(ProviderName.ZAI.value, base_dir / "zai.brain.json")


__all__ = [
    "ProviderName",
    "ProviderBrain",
    "ProviderBrainState",
]
