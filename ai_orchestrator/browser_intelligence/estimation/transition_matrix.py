"""Transition matrix — learned transition probabilities between hidden states."""

from __future__ import annotations

import math

from ai_orchestrator.browser_intelligence.estimation.belief_state import HiddenState

_DEFAULT_EPSILON = 1e-9
_STRONG_EPSILON = 1e-3


class TransitionMatrix:
    """Learned transition probabilities A[i][j] = P(S_{t+1}=s_j | S_t=s_i).

    Stored as log-probabilities for numerical stability.
    Initial values from empirical observation. Refined via Baum-Welch.

    Invariants:
        Σ_j A[i][j] = 1.0 for all i    (stochastic rows)
        A[i][j] > 0 for all i, j       (ergodic)


    """

    def __init__(self, epsilon: float = _DEFAULT_EPSILON):
        self._epsilon = max(epsilon, _DEFAULT_EPSILON)
        self._log_probs: dict[HiddenState, dict[HiddenState, float]] = {}
        self._init_defaults()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def _init_defaults(self) -> None:
        defaults: dict[HiddenState, list[tuple[HiddenState, float]]] = {
            HiddenState.BOOTING: [
                (HiddenState.BOOTING, 0.10),
                (HiddenState.AUTH_REQUIRED, 0.50),
                (HiddenState.READY, 0.35),
                (HiddenState.ERROR, 0.05),
            ],
            HiddenState.AUTH_REQUIRED: [
                (HiddenState.AUTH_REQUIRED, 0.30),
                (HiddenState.READY, 0.60),
                (HiddenState.ERROR, 0.10),
            ],
            HiddenState.READY: [
                (HiddenState.READY, 0.80),
                (HiddenState.PROMPT_SENT, 0.15),
                (HiddenState.RATE_LIMITED, 0.02),
                (HiddenState.ERROR, 0.03),
            ],
            HiddenState.PROMPT_SENT: [
                (HiddenState.PROMPT_SENT, 0.05),
                (HiddenState.THINKING, 0.40),
                (HiddenState.GENERATING, 0.40),
                (HiddenState.RATE_LIMITED, 0.05),
                (HiddenState.ERROR, 0.10),
            ],
            HiddenState.THINKING: [
                (HiddenState.THINKING, 0.30),
                (HiddenState.GENERATING, 0.60),
                (HiddenState.ERROR, 0.10),
            ],
            HiddenState.GENERATING: [
                (HiddenState.GENERATING, 0.70),
                (HiddenState.COMPLETE, 0.20),
                (HiddenState.ERROR, 0.08),
                (HiddenState.RATE_LIMITED, 0.02),
            ],
            HiddenState.COMPLETE: [
                (HiddenState.COMPLETE, 0.40),
                (HiddenState.READY, 0.55),
                (HiddenState.ERROR, 0.05),
            ],
            HiddenState.RATE_LIMITED: [
                (HiddenState.RATE_LIMITED, 0.50),
                (HiddenState.READY, 0.30),
                (HiddenState.ERROR, 0.20),
            ],
            HiddenState.ERROR: [
                (HiddenState.ERROR, 0.30),
                (HiddenState.READY, 0.30),
                (HiddenState.BOOTING, 0.40),
            ],
            HiddenState.SHADOW_BANNED: [
                (HiddenState.SHADOW_BANNED, 0.60),
                (HiddenState.COMPLETE, 0.30),
                (HiddenState.RATE_LIMITED, 0.10),
            ],
        }

        for from_state in HiddenState:
            self._log_probs[from_state] = {}
            row: dict[HiddenState, float] = {}

            if from_state in defaults:
                for to_state, prob in defaults[from_state]:
                    row[to_state] = prob + self._epsilon

            for to_state in HiddenState:
                if to_state not in row:
                    row[to_state] = self._epsilon

            self._normalize_row(row)

            for to_state in HiddenState:
                self._log_probs[from_state][to_state] = math.log(
                    max(row[to_state], _DEFAULT_EPSILON)
                )

    @staticmethod
    def _normalize_row(row: dict[HiddenState, float]) -> None:
        total = sum(row.values())
        if total > 0:
            inv = 1.0 / total
            for s in row:
                row[s] *= inv

    def transition_prob(self, from_state: HiddenState, to_state: HiddenState) -> float:
        default_log = math.log(_DEFAULT_EPSILON)
        log_prob = self._log_probs.get(from_state, {}).get(to_state, default_log)
        return math.exp(log_prob)

    def to_prob_matrix(self) -> list[list[float]]:
        states = list(HiddenState)
        return [
            [self.transition_prob(si, sj) for sj in states]
            for si in states
        ]

    def update_from_counts(
        self, counts: dict[tuple[HiddenState, HiddenState], int]
    ) -> None:
        """Update from (from_state, to_state) → transition count.

        Laplace smoothing on increment, then full renormalization.
        """
        for from_state in HiddenState:
            total = sum(
                counts.get((from_state, s), 0) for s in HiddenState
            )
            if total == 0:
                continue

            row: dict[HiddenState, float] = {}
            for to_state in HiddenState:
                c = counts.get((from_state, to_state), 0)
                row[to_state] = c + 1.0

            self._normalize_row(row)

            for to_state in HiddenState:
                self._log_probs[from_state][to_state] = math.log(
                    max(row[to_state], _DEFAULT_EPSILON)
                )

    def validate_stochastic(self, tolerance: float = 1e-6) -> bool:
        """Verify every row sums to 1.0 within tolerance."""
        states = list(HiddenState)
        for si in states:
            row_sum = sum(self.transition_prob(si, sj) for sj in states)
            if abs(row_sum - 1.0) > tolerance:
                return False
        return True

    def enforce_stochastic(self) -> None:
        """Re-normalize every row to enforce Σ_j A[i][j] = 1.0."""
        for from_state in HiddenState:
            row: dict[HiddenState, float] = {}
            for to_state in HiddenState:
                row[to_state] = self.transition_prob(from_state, to_state)
            self._normalize_row(row)
            for to_state in HiddenState:
                self._log_probs[from_state][to_state] = math.log(
                    max(row[to_state], _DEFAULT_EPSILON)
                )

    def is_ergodic(self) -> bool:
        """Check ergodicity: every state reachable from every other state."""
        states = list(HiddenState)
        n = len(states)
        {s: i for i, s in enumerate(states)}
        reachable = [[False] * n for _ in range(n)]
        for i, si in enumerate(states):
            for j, sj in enumerate(states):
                reachable[i][j] = self.transition_prob(si, sj) > 0

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    reachable[i][j] = reachable[i][j] or (reachable[i][k] and reachable[k][j])

        for i in range(n):
            for j in range(n):
                if not reachable[i][j]:
                    return False
        return True

    def row_sums(self) -> dict[HiddenState, float]:
        states = list(HiddenState)
        return {
            si: sum(self.transition_prob(si, sj) for sj in states)
            for si in states
        }

    def min_entry(self) -> float:
        return min(
            self.transition_prob(si, sj)
            for si in HiddenState
            for sj in HiddenState
        )

    def max_entry(self) -> float:
        return max(
            self.transition_prob(si, sj)
            for si in HiddenState
            for sj in HiddenState
        )
