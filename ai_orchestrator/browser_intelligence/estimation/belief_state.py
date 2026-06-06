"""Hidden states — the provider as a hidden-state system."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class HiddenState(StrEnum):
    BOOTING = "booting"
    AUTH_REQUIRED = "auth_required"
    READY = "ready"
    PROMPT_SENT = "prompt_sent"
    THINKING = "thinking"
    GENERATING = "generating"
    COMPLETE = "complete"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    SHADOW_BANNED = "shadow_banned"


@dataclass
class BeliefState:
    """Probability distribution over hidden states.

    Invariant: sum(probabilities) == 1.0
    """

    probabilities: dict[HiddenState, float]

    def __post_init__(self):
        total = sum(self.probabilities.values())
        if total > 0 and abs(total - 1.0) > 0.001:
            for s in self.probabilities:
                self.probabilities[s] /= total

    @property
    def most_likely(self) -> HiddenState:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.most_likely]

    @property
    def entropy(self) -> float:
        return -sum(
            p * math.log2(p) for p in self.probabilities.values() if p > 0
        )

    def is_confident(self, threshold: float = 0.85) -> bool:
        return self.confidence >= threshold

    def is_uncertain(self, threshold: float = 0.5) -> bool:
        return self.confidence < threshold

    @classmethod
    def uniform(cls) -> BeliefState:
        n = len(HiddenState)
        return cls({s: 1.0 / n for s in HiddenState})

    @classmethod
    def certain(cls, state: HiddenState) -> BeliefState:
        probs = {s: 0.0 for s in HiddenState}
        probs[state] = 1.0
        return cls(probs)
