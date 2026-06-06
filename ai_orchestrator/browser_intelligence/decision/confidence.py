"""ConfidenceEngine — computes confidence scores for decisions.

Confidence = weighted blend of observation quality, historical success,
selector reliability, accessibility reliability, and network reliability.
"""

from __future__ import annotations

from ai_orchestrator.browser_intelligence.estimation.belief_state import BeliefState


class ConfidenceEngine:
    """Computes confidence scores for every decision.

    Confidence ∈ [0, 1]. Higher confidence means the system's
    current belief and observations support the decision.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights or {
            "observation": 0.25,
            "historical": 0.25,
            "selector": 0.20,
            "accessibility": 0.15,
            "network": 0.15,
        }

    def compute(
        self,
        observation_quality: float,
        historical_success: float = 1.0,
        selector_reliability: float = 1.0,
        accessibility_reliability: float = 1.0,
        network_reliability: float = 1.0,
    ) -> float:
        """Weighted confidence score ∈ [0, 1]."""
        conf = (
            self._weights["observation"] * observation_quality
            + self._weights["historical"] * historical_success
            + self._weights["selector"] * selector_reliability
            + self._weights["accessibility"] * accessibility_reliability
            + self._weights["network"] * network_reliability
        )
        return max(0.0, min(1.0, conf))

    def from_belief(self, belief: BeliefState) -> float:
        """Confidence derived from belief state quality."""
        import math

        max_entropy = math.log2(10)
        if max_entropy == 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - belief.entropy / max_entropy))
