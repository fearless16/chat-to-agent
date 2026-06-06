"""UtilityEngine — computes expected utility for candidate actions.

U(a | b) = Σ_s P(s | b) × R(a, s)

where b = belief state, s = true hidden state, R(a,s) = reward.
"""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.browser_intelligence.estimation.belief_state import (
    BeliefState,
    HiddenState,
)


class UtilityEngine:
    """Computes expected utility for actions given a belief state.

    Actions are selected by maximizing expected utility:
        a* = argmax_a Σ_s b(s) · R(a, s)
    """

    def __init__(self, reward_matrix: Optional[dict[tuple[str, HiddenState], float]] = None):
        self._reward = reward_matrix or self._default_rewards()

    @staticmethod
    def _default_rewards() -> dict[tuple[str, HiddenState], float]:
        return {
            ("type_prompt", HiddenState.READY): 10.0,
            ("type_prompt", HiddenState.ERROR): -5.0,
            ("type_prompt", HiddenState.RATE_LIMITED): -10.0,
            ("type_prompt", HiddenState.GENERATING): -20.0,
            ("click_send", HiddenState.PROMPT_SENT): 10.0,
            ("click_send", HiddenState.GENERATING): -20.0,
            ("click_send", HiddenState.READY): -5.0,
            ("extract_response", HiddenState.COMPLETE): 10.0,
            ("extract_response", HiddenState.GENERATING): -5.0,
            ("extract_response", HiddenState.THINKING): -10.0,
            ("extract_response", HiddenState.READY): -5.0,
            ("wait", HiddenState.GENERATING): 2.0,
            ("wait", HiddenState.THINKING): 2.0,
            ("wait", HiddenState.PROMPT_SENT): 1.0,
            ("wait", HiddenState.COMPLETE): -1.0,
            ("wait", HiddenState.READY): -2.0,
            ("recover", HiddenState.ERROR): 5.0,
            ("recover", HiddenState.SHADOW_BANNED): 5.0,
            ("recover", HiddenState.READY): -10.0,
            ("recover", HiddenState.GENERATING): -30.0,
            ("refresh", HiddenState.ERROR): 5.0,
            ("refresh", HiddenState.AUTH_REQUIRED): 3.0,
            ("refresh", HiddenState.GENERATING): -30.0,
            ("refresh", HiddenState.READY): -5.0,
            ("relogin", HiddenState.AUTH_REQUIRED): 10.0,
            ("relogin", HiddenState.READY): -15.0,
            ("relogin", HiddenState.GENERATING): -50.0,
            ("quarantine", HiddenState.SHADOW_BANNED): 5.0,
            ("quarantine", HiddenState.RATE_LIMITED): 5.0,
            ("quarantine", HiddenState.ERROR): 3.0,
            ("quarantine", HiddenState.GENERATING): -50.0,
        }

    def expected_utility(self, action: str, belief: BeliefState) -> float:
        """E[U(a)] = Σ_s P(s) · R(a, s)."""
        total = 0.0
        for state, prob in belief.probabilities.items():
            reward = self._reward.get((action, state), 0.0)
            total += prob * reward
        return total

    def best_action(
        self, actions: list[str], belief: BeliefState
    ) -> tuple[str, float]:
        """Return (best_action, expected_utility)."""
        if not actions:
            return ("wait", 0.0)
        scored = [(a, self.expected_utility(a, belief)) for a in actions]
        return max(scored, key=lambda x: x[1])

    def all_utilities(
        self, actions: list[str], belief: BeliefState
    ) -> dict[str, float]:
        """Return utility for each action."""
        return {a: self.expected_utility(a, belief) for a in actions}
