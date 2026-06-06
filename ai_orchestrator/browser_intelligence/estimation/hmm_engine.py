"""HMM Engine — forward algorithm for real-time probabilistic state estimation."""

from __future__ import annotations

import math

from ai_orchestrator.browser_intelligence.estimation.belief_state import (
    BeliefState,
    HiddenState,
)
from ai_orchestrator.browser_intelligence.estimation.emission_model import EmissionModel
from ai_orchestrator.browser_intelligence.estimation.transition_matrix import TransitionMatrix
from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector


class HMMEngine:
    """Hidden Markov Model engine for probabilistic state estimation.

    Uses the forward algorithm to compute:
        b_t(s) = P(S_t = s | O_{1:t})

    Supports online emission model learning via soft assignment
    and adaptive readiness thresholds.
    """

    _MIN_OBS_FOR_LEARNING = 5

    def __init__(
        self, transition: TransitionMatrix | None = None, emission: EmissionModel | None = None
    ):
        self._transition = transition or TransitionMatrix()
        self._emission = emission or EmissionModel()
        self._belief: BeliefState | None = None
        self._states = list(HiddenState)
        self._learning_enabled: bool = True
        self._tick_count: int = 0

    @property
    def belief(self) -> BeliefState | None:
        return self._belief

    @property
    def emission(self) -> EmissionModel:
        return self._emission

    @property
    def learning_enabled(self) -> bool:
        return self._learning_enabled

    @learning_enabled.setter
    def learning_enabled(self, value: bool) -> None:
        self._learning_enabled = value

    def initialize(self) -> BeliefState:
        self._belief = BeliefState.uniform()
        return self._belief

    def update(self, observation: FeatureVector) -> BeliefState:
        if self._belief is None:
            self._belief = BeliefState.uniform()

        prev_belief = self._belief
        new_probs: dict[HiddenState, float] = {}
        total = 0.0

        for s_t in HiddenState:
            pred = 0.0
            for s_prev in HiddenState:
                pred += (
                    prev_belief.probabilities[s_prev]
                    * self._transition.transition_prob(s_prev, s_t)
                )
            emission_prob = self._emission.emission_prob(observation, s_t)
            new_probs[s_t] = pred * emission_prob
            total += new_probs[s_t]

        if total > 1e-30:
            for s in HiddenState:
                new_probs[s] /= total
        else:
            n = len(HiddenState)
            for s in HiddenState:
                new_probs[s] = 1.0 / n

        self._belief = BeliefState(probabilities=new_probs)

        self._tick_count += 1
        if self._learning_enabled and self._tick_count >= self._MIN_OBS_FOR_LEARNING:
            self._emission.update_from_soft_assignment(
                observation, new_probs,
                min_learning_rate=0.01,
                max_learning_rate=0.08,
            )

        return self._belief

    def adaptive_readiness_threshold(
        self,
        base_threshold: float = 0.45,
        min_threshold: float = 0.30,
        max_threshold: float = 0.75,
    ) -> float:
        """Compute adaptive threshold for readiness detection.

        Low when model is uncalibrated (few observations) → allows
        faster detection with lower confidence requirement.
        Rises as model learns → requires higher confidence.

        threshold = base + (max - base) * calibration_score
        clamped to [min_threshold, max_threshold].
        """
        calib = self._emission.calibration_score(HiddenState.READY)
        threshold = base_threshold + (max_threshold - base_threshold) * calib
        return max(min_threshold, min(threshold, max_threshold))

    def viterbi(self, observations: list[FeatureVector]) -> list[HiddenState]:
        if not observations:
            return []

        T = len(observations)  # noqa: N806
        N = len(self._states)  # noqa: N806
        {s: i for i, s in enumerate(self._states)}

        dp: list[list[float]] = [[float("-inf")] * N for _ in range(T)]
        backpointer: list[list[int]] = [[-1] * N for _ in range(T)]

        prior = 1.0 / N
        for j, s in enumerate(self._states):
            emission = self._emission.emission_prob(observations[0], s)
            dp[0][j] = math.log(prior) + math.log(max(emission, 1e-30))

        for t in range(1, T):
            for j, s_j in enumerate(self._states):
                emission = self._emission.emission_prob(observations[t], s_j)
                log_emission = math.log(max(emission, 1e-30))

                best_score = float("-inf")
                best_prev = -1
                for i, s_i in enumerate(self._states):
                    trans = self._transition.transition_prob(s_i, s_j)
                    score = dp[t - 1][i] + math.log(max(trans, 1e-30))
                    if score > best_score:
                        best_score = score
                        best_prev = i

                dp[t][j] = best_score + log_emission
                backpointer[t][j] = best_prev

        last_best = max(range(N), key=lambda j: dp[T - 1][j])
        path = [self._states[last_best]]
        for t in range(T - 1, 0, -1):
            last_best = backpointer[t][last_best]
            path.insert(0, self._states[last_best])

        return path

    def learn_from_observation(self, observation: FeatureVector, state: HiddenState) -> None:
        self._emission.update_from_observation(observation, state)

    def learn_transitions(self, state_sequence: list[HiddenState]) -> None:
        counts: dict[tuple[HiddenState, HiddenState], int] = {}
        for i in range(len(state_sequence) - 1):
            pair = (state_sequence[i], state_sequence[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        self._transition.update_from_counts(counts)

    def reset(self) -> None:
        self._belief = None
        self._tick_count = 0
