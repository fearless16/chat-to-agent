"""Emission model — P(O_t | S_t = s) for each hidden state."""

from __future__ import annotations

import math
import time

from ai_orchestrator.browser_intelligence.estimation.belief_state import HiddenState
from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector


def _feature_is_binary(idx: int) -> bool:
    return idx < 18


class EmissionModel:
    """Learned emission probabilities P(O|S).

    Uses a mixture:
    - Bernoulli for binary features (visible/invisible indicators)
    - Gaussian for continuous features (rates, lengths)

    Parameters are initialized with default signatures per state
    and refined via online learning from observations.
    """

    FEATURE_DIM = 30

    def __init__(self):
        self._binary_params: dict[HiddenState, list[float]] = {}
        self._continuous_mu: dict[HiddenState, list[float]] = {}
        self._continuous_sigma: dict[HiddenState, list[float]] = {}
        self._observation_counts: dict[HiddenState, int] = {s: 0 for s in HiddenState}
        self._last_update_time: dict[HiddenState, float] = {
            s: 0.0 for s in HiddenState
        }
        self._init_defaults()

    def _init_defaults(self) -> None:
        d = self.FEATURE_DIM

        self._binary_params = {
            s: [0.4] * 18 for s in HiddenState
        }
        self._continuous_mu = {
            s: [0.0] * (d - 18) for s in HiddenState
        }
        self._continuous_sigma = {
            s: [50.0] * (d - 18) for s in HiddenState
        }

        defaults: dict[HiddenState, list[float]] = {
            HiddenState.READY: [
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 1.0,
                0.0, 10.0, 0.0, 0.0, 0.1,
            ],
            HiddenState.GENERATING: [
                0.7, 0.3, 0.7, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.3, 0.3, 0.3, 0.7,
                0.7, 0.7, 0.7, 0.3, 0.3, 0.3,
                8.0, 1.5, 300.0, 0.8, 300.0, 50.0, 0.8,
                15.0, 0.2, 50.0, 5000.0, 2.0,
            ],
            HiddenState.COMPLETE: [
                0.7, 0.7, 0.3, 0.7, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.3, 0.7, 0.7, 0.7,
                0.0, 0.0, 100.0, 1.0, 500.0, 0.0, 1.0,
                0.0, 5.0, 200.0, 20000.0, 0.1,
            ],
            HiddenState.RATE_LIMITED: [
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.3, 0.3, 0.3, 0.7, 0.7, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 20.0, 0.0, 0.0, 0.5,
            ],
            HiddenState.ERROR: [
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.3, 0.3, 0.7, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            HiddenState.THINKING: [
                0.7, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.7, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.7, 0.3, 0.3, 0.3,
                1.0, -0.1, 150.0, 1.0, 150.0, 10.0, 1.0,
                0.0, 2.0, 0.0, 0.0, 0.5,
            ],
            HiddenState.AUTH_REQUIRED: [
                0.3, 0.3, 0.3, 0.3, 0.3, 0.7,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            HiddenState.BOOTING: [
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.0, 0.0, 80.0, 0.3, 0.0, 0.0, 0.0,
                0.0, 30.0, 0.0, 0.0, 0.0,
            ],
            HiddenState.SHADOW_BANNED: [
                0.7, 0.7, 0.3, 0.7, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.3, 0.3, 0.7, 0.3,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 30.0, 0.0, 0.0, 0.0,
            ],
            HiddenState.PROMPT_SENT: [
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.7, 0.3, 0.3, 0.3,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 2.0, 0.0, 0.0, 3.0,
            ],
        }

        for state, values in defaults.items():
            for i in range(18):
                self._binary_params[state][i] = max(0.05, min(0.95, values[i]))
            for i in range(self.FEATURE_DIM - 18):
                self._continuous_mu[state][i] = values[18 + i]
                self._continuous_sigma[state][i] = 50.0

    def emission_prob(self, observation: FeatureVector, state: HiddenState) -> float:
        obs = observation.to_list()
        log_prob = 0.0

        for i in range(18):
            p = self._binary_params[state][i]
            p = max(0.01, min(0.99, p))
            if obs[i] > 0.5:
                log_prob += math.log(p)
            else:
                log_prob += math.log(1.0 - p)

        for i in range(18, self.FEATURE_DIM):
            idx = i - 18
            mu = self._continuous_mu[state][idx]
            sigma = max(self._continuous_sigma[state][idx], 0.1)
            diff = obs[i] - mu
            log_prob += -0.5 * (diff * diff) / (sigma * sigma) - math.log(
                sigma * math.sqrt(2 * math.pi)
            )

        prob = math.exp(log_prob)
        return max(prob, 1e-30)

    def update_from_observation(
        self, observation: FeatureVector, state: HiddenState, learning_rate: float = 0.05
    ) -> None:
        obs = observation.to_list()

        for i in range(18):
            p = self._binary_params[state][i]
            target = 1.0 if obs[i] > 0.5 else 0.0
            self._binary_params[state][i] = p + learning_rate * (target - p)
            self._binary_params[state][i] = max(0.01, min(0.99, self._binary_params[state][i]))

        for i in range(18, self.FEATURE_DIM):
            idx = i - 18
            old_mu = self._continuous_mu[state][idx]
            diff = obs[i] - old_mu
            self._continuous_mu[state][idx] = old_mu + learning_rate * diff
            old_sigma = self._continuous_sigma[state][idx]
            self._continuous_sigma[state][idx] = max(
                0.1,
                (1 - learning_rate) * old_sigma + learning_rate * diff * diff,
            )

        self._observation_counts[state] += 1
        self._last_update_time[state] = time.monotonic()

    def update_from_soft_assignment(
        self,
        observation: FeatureVector,
        beliefs: dict[HiddenState, float],
        min_learning_rate: float = 0.01,
        max_learning_rate: float = 0.10,
    ) -> None:
        """Update emission parameters using soft assignment."""
        for state in HiddenState:
            belief_mass = beliefs.get(state, 0.0)
            if belief_mass < 1e-6:
                continue
            lr = min_learning_rate + belief_mass * (max_learning_rate - min_learning_rate)
            self.update_from_observation(observation, state, learning_rate=lr)

    def calibration_score(self, state: HiddenState) -> float:
        """How well calibrated are the emission parameters for this state?"""
        count = self._observation_counts.get(state, 0)
        if count == 0:
            return 0.0
        return 1.0 - math.exp(-count / 100.0)

    def observation_count(self, state: HiddenState) -> int:
        return self._observation_counts.get(state, 0)

    def total_observations(self) -> int:
        return sum(self._observation_counts.values())

    def sigma_for_state(self, state: HiddenState, feature_idx: int) -> float:
        if feature_idx < 18:
            return 0.0
        idx = feature_idx - 18
        return max(self._continuous_sigma[state][idx], 0.1)
