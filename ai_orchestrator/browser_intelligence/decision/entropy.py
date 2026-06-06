"""EntropyEngine — measures system uncertainty from belief state."""

from __future__ import annotations

import math

from ai_orchestrator.browser_intelligence.estimation.belief_state import (
    BeliefState,
    HiddenState,
)


class EntropyEngine:
    """Measures uncertainty in the system's belief state.

    High entropy → system is confused → avoid expensive actions.
    Low entropy → system is confident → proceed.

    Entropy gates action selection by modifying exploration rate
    and triggering recovery when uncertainty exceeds thresholds.
    """

    def __init__(
        self,
        high_entropy_threshold: float = 2.0,
        critical_entropy_threshold: float = 3.0,
    ):
        self._high = high_entropy_threshold
        self._critical = critical_entropy_threshold
        self._max_entropy = math.log2(len(HiddenState))

    def compute(self, belief: BeliefState) -> float:
        return belief.entropy

    def is_confused(self, belief: BeliefState) -> bool:
        return belief.entropy > self._high

    def is_critical(self, belief: BeliefState) -> bool:
        return belief.entropy > self._critical

    def should_recover(self, belief: BeliefState) -> bool:
        return (
            belief.entropy > self._high
            and belief.probabilities.get(HiddenState.GENERATING, 0) < 0.3
            and belief.probabilities.get(HiddenState.READY, 0) < 0.3
        )

    def exploration_factor(self, belief: BeliefState) -> float:
        """Maps entropy to exploration rate ∈ [0.1, 1.0].

        High entropy → explore more.
        Low entropy → exploit.
        """
        if self._max_entropy == 0:
            return 0.1
        return max(0.1, min(belief.entropy / self._max_entropy, 1.0))

    def normalized(self, belief: BeliefState) -> float:
        """Entropy normalized to [0, 1]."""
        if self._max_entropy == 0:
            return 0.0
        return min(belief.entropy / self._max_entropy, 1.0)
