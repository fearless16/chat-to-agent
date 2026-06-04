"""ProviderRouter — scores, ranks, and selects provider accounts for task requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ai_orchestrator.models.account import Account
from ai_orchestrator.models.capabilities import (
    PROVIDER_PROFILES,
    CapabilityVector,
    TaskRequirements,
)


@dataclass
class ScoredAccount:
    """A scored account with a reason for the score."""
    account: Account
    score: float
    reason: str


DEFAULT_WEIGHTS = {
    "reasoning": 0.3,
    "coding": 0.3,
    "translation": 0.2,
    "multimodality": 0.2,
    "health_penalty": 2.0,
    "rate_penalty": 1.5,
    "latency_penalty": 1.0,
    "failures_penalty": 0.5,
}

_CAPABILITY_NAMES = ("reasoning", "coding", "translation", "multimodality")


class ProviderRouter:
    """Routes tasks to the best provider account based on capability scoring
    and health/rate/latency/failures penalties.

    Scoring flow:
      1. Context-window check — reject if context_limit < required context_length.
      2. Capability vector lookup from PROVIDER_PROFILES by provider name.
      3. Capability score = Σ(priority_weight × capability_value) for active
         requirements, or Σ(DEFAULT_WEIGHTS × capability_value) when none are set.
      4. Penalties subtracted: health, rate usage, latency, consecutive failures.
      5. Final score = max(0, capability_score - total_penalty).
    """

    def __init__(self, weights: dict | None = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    @staticmethod
    def _get_capability_vector(account: Account) -> CapabilityVector | None:
        """Look up the capability vector for an account's provider.

        Matches ``account.provider`` against ``provider_name`` in the
        pre-defined ``PROVIDER_PROFILES`` registry. Returns ``None`` when
        the provider is unknown.
        """
        for profile in PROVIDER_PROFILES.values():
            if profile.provider_name == account.provider:
                return profile.capabilities
        return None

    def score_account(
        self, account: Account, requirements: TaskRequirements
    ) -> ScoredAccount:
        """Score a single account against the given task requirements.

        Returns a ``ScoredAccount`` with a score in ``[0.0, …)`` and a
        human-readable reason string.
        """
        # --- 1. Context-window guard ---
        if account.context_limit < requirements.context_length:
            return ScoredAccount(account, 0.0, "context_exceeded")

        caps = self._get_capability_vector(account)
        if caps is None:
            return ScoredAccount(account, 0.0, "unknown_provider")

        # --- 2. Capability score ---
        capability_score = self._compute_capability_score(caps, requirements)

        # --- 3. Penalties ---
        total_penalty, reason_parts = self._compute_penalties(account)

        final_score = max(0.0, capability_score - total_penalty)

        reason = f"score={final_score:.3f}"
        if reason_parts:
            reason += " penalties: " + ", ".join(reason_parts)

        return ScoredAccount(account, final_score, reason)

    def _compute_capability_score(
        self, caps: CapabilityVector, requirements: TaskRequirements
    ) -> float:
        """Compute the raw capability score from requirements and provider vector."""
        active: list[str] = []
        flag_map = {
            "reasoning": requirements.requires_reasoning,
            "coding": requirements.requires_coding,
            "translation": requirements.requires_translation,
            "multimodality": requirements.requires_multimodality,
        }
        for name, required in flag_map.items():
            if required:
                active.append(name)

        if not active:
            # No specific requirements — use default weights on all dimensions
            return sum(
                self.weights.get(name, 0.0) * getattr(caps, name, 0.0)
                for name in _CAPABILITY_NAMES
            )

        score = 0.0
        for name in active:
            weight = requirements.priority.get(name, self.weights.get(name, 0.0))
            cap_value = getattr(caps, name, 0.0)
            score += weight * cap_value
        return score

    def _compute_penalties(self, account: Account) -> tuple[float, list[str]]:
        """Compute total penalty and a list of human-readable penalty descriptors."""
        health_penalty = (1.0 - account.health_score) * self.weights["health_penalty"]
        rate_penalty = account.current_rate_usage * self.weights["rate_penalty"]
        latency_penalty = (
            min(account.avg_latency_ms / 1000.0, 5.0)
            * self.weights["latency_penalty"]
        )
        failures_penalty = account.consecutive_failures * self.weights["failures_penalty"]

        total = health_penalty + rate_penalty + latency_penalty + failures_penalty

        parts: list[str] = []
        if health_penalty > 0:
            parts.append(f"health_penalty={health_penalty:.2f}")
        if rate_penalty > 0:
            parts.append(f"rate_penalty={rate_penalty:.2f}")
        if latency_penalty > 0:
            parts.append(f"latency_penalty={latency_penalty:.2f}")
        if failures_penalty > 0:
            parts.append(f"failures_penalty={failures_penalty:.2f}")

        return total, parts

    def rank_accounts(
        self, accounts: list[Account], requirements: TaskRequirements
    ) -> list[ScoredAccount]:
        """Score and sort accounts by descending score.

        Returns an empty list when ``accounts`` is empty.
        """
        if not accounts:
            return []
        scored = [self.score_account(a, requirements) for a in accounts]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def select_account(
        self, accounts: list[Account], requirements: TaskRequirements
    ) -> Account | None:
        """Return the single highest-scoring account, or ``None`` if empty."""
        ranked = self.rank_accounts(accounts, requirements)
        if not ranked:
            return None
        return ranked[0].account

    def select_provider(
        self,
        requirements: TaskRequirements,
        provider_pool: dict[str, list[Account]],
    ) -> tuple[str, Account] | None:
        """Select the best ``(provider_name, Account)`` pair across a provider pool.

        For each provider in the pool the best account is scored; the provider
        with the highest-scoring account wins. Returns ``None`` when the pool
        is empty or all providers have empty account lists.
        """
        if not provider_pool:
            return None

        best_provider: str | None = None
        best_account: Account | None = None
        best_score: float = -1.0

        for provider_name, accounts in provider_pool.items():
            if not accounts:
                continue
            account = self.select_account(accounts, requirements)
            if account is None:
                continue
            scored = self.score_account(account, requirements)
            if scored.score > best_score:
                best_score = scored.score
                best_provider = provider_name
                best_account = account

        if best_provider is not None and best_account is not None:
            return (best_provider, best_account)
        return None
