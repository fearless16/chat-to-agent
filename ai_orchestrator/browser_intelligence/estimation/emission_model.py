"""Emission model — P(O_t | S_t = s) for each hidden state."""

from __future__ import annotations

import math
import time

from ai_orchestrator.browser_intelligence.estimation.belief_state import HiddenState
from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector


def _feature_is_binary(idx: int) -> bool:
    return idx < 19


class EmissionModel:
    """Learned emission probabilities P(O|S).

    Uses a mixture:
    - Bernoulli for binary features (visible/invisible indicators)
    - Gaussian for continuous features (rates, lengths)

    Parameters are initialized with default signatures per state
    and refined via online learning from observations.
    """

    FEATURE_DIM = 33
    CONTINUOUS_DIM = 14

    # Feature-specific sigmas for continuous features (indices 0-13):
    # 0: mutation_rate, 1: mutation_acceleration, 2: js_heap_used_mb,
    # 3: page_stability, 4: response_length, 5: response_length_delta,
    # 6: visual_stability, 7: tokens_per_second, 8: stream_idle_time,
    # 9: total_chunks, 10: bytes_received, 11: network_request_rate,
    # 12: a11y_confidence, 13: a11y_node_count
    _DEFAULT_SIGMAS = [
        10.0,   # mutation_rate
        5.0,    # mutation_acceleration
        100.0,  # js_heap_used_mb
        0.5,    # page_stability (0-1)
        500.0,  # response_length
        100.0,  # response_length_delta
        0.5,    # visual_stability (0-1)
        20.0,   # tokens_per_second
        10.0,   # stream_idle_time
        50.0,   # total_chunks
        10000.0,  # bytes_received (can be large)
        5.0,    # network_request_rate
        0.2,    # a11y_confidence (0-1)
        30.0,   # a11y_node_count
    ]

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
        cd = self.CONTINUOUS_DIM

        self._binary_params = {
            s: [0.4] * 19 for s in HiddenState
        }
        self._continuous_mu = {
            s: [0.0] * cd for s in HiddenState
        }
        self._continuous_sigma = {
            s: list(self._DEFAULT_SIGMAS) for s in HiddenState
        }

        defaults: dict[HiddenState, list[float]] = {
            HiddenState.READY: [
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.0, 0.0, 0.9,
                100.0, 1.0,
                100.0, 0.0, 1.0,
                0.0, 10.0, 0.0, 0.0, 0.1,
                0.95, 50.0,
            ],
            HiddenState.GENERATING: [
                0.7, 0.3, 0.7, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.3, 0.3, 0.3, 0.7,
                0.7, 0.7, 0.7, 0.3, 0.3, 0.3,
                8.0, 1.5, 0.8,
                300.0, 0.8,
                300.0, 50.0, 0.8,
                15.0, 0.2, 50.0, 5000.0, 2.0,
                0.9, 30.0,
            ],
            HiddenState.COMPLETE: [
                0.7, 0.7, 0.3, 0.7, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.3, 0.7, 0.7, 0.7,
                0.0, 0.0, 0.95,
                100.0, 1.0,
                500.0, 0.0, 1.0,
                0.0, 5.0, 200.0, 20000.0, 0.1,
                0.95, 40.0,
            ],
            HiddenState.RATE_LIMITED: [
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.3, 0.3, 0.3, 0.7, 0.7, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.0, 0.0, 0.3,
                0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 20.0, 0.0, 0.0, 0.5,
                0.2, 10.0,
            ],
            HiddenState.ERROR: [
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.3, 0.3, 0.7, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.7, 0.3,
                0.0, 0.0, 0.1,
                0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 5.0,
            ],
            HiddenState.THINKING: [
                0.7, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.7, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.7, 0.3, 0.3, 0.3,
                1.0, -0.1, 0.8,
                150.0, 1.0,
                150.0, 10.0, 1.0,
                0.0, 2.0, 0.0, 0.0, 0.5,
                0.7, 20.0,
            ],
            HiddenState.AUTH_REQUIRED: [
                0.3, 0.3, 0.3, 0.3, 0.3, 0.7,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.0, 0.0, 0.1,
                0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 5.0,
            ],
            HiddenState.BOOTING: [
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                0.0, 0.0, 0.3,
                80.0, 0.3,
                0.0, 0.0, 0.0,
                0.0, 30.0, 0.0, 0.0, 0.0,
                0.3, 20.0,
            ],
            HiddenState.SHADOW_BANNED: [
                0.7, 0.7, 0.3, 0.7, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.3, 0.3, 0.7, 0.3,
                0.0, 0.0, 0.6,
                0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 30.0, 0.0, 0.0, 0.0,
                0.5, 15.0,
            ],
            HiddenState.PROMPT_SENT: [
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3, 0.3, 0.3,
                0.3, 0.7, 0.7, 0.3, 0.3, 0.3,
                0.0, 0.0, 0.7,
                0.0, 0.0,
                0.0, 0.0, 0.0,
                1.0, 2.0, 0.0, 0.0, 3.0,
                0.6, 25.0,
            ],
        }

        for state, values in defaults.items():
            for i in range(19):
                self._binary_params[state][i] = max(0.05, min(0.95, values[i]))
            for i in range(self.FEATURE_DIM - 19):
                self._continuous_mu[state][i] = values[19 + i]
                # Keep feature-specific sigmas from _DEFAULT_SIGMAS

    def emission_prob(self, observation: FeatureVector, state: HiddenState) -> float:
        obs = observation.to_list()
        log_prob = 0.0

        for i in range(19):
            p = self._binary_params[state][i]
            p = max(0.01, min(0.99, p))
            if obs[i] > 0.5:
                log_prob += math.log(p)
            else:
                log_prob += math.log(1.0 - p)

        for i in range(19, self.FEATURE_DIM):
            idx = i - 19
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

        for i in range(19):
            p = self._binary_params[state][i]
            target = 1.0 if obs[i] > 0.5 else 0.0
            self._binary_params[state][i] = p + learning_rate * (target - p)
            self._binary_params[state][i] = max(0.01, min(0.99, self._binary_params[state][i]))

        for i in range(19, self.FEATURE_DIM):
            idx = i - 19
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
        """Update emission parameters using soft assignment.

        Each state gets a fraction of the update proportional to
        the belief mass assigned to it. The learning rate for each
        state is scaled by that state's belief mass:
            lr_s = belief(s) * (max_lr - min_lr) + min_lr
        """
        for state in HiddenState:
            belief_mass = beliefs.get(state, 0.0)
            if belief_mass < 1e-6:
                continue
            lr = min_learning_rate + belief_mass * (max_learning_rate - min_learning_rate)
            self.update_from_observation(observation, state, learning_rate=lr)

    def calibration_score(self, state: HiddenState) -> float:
        """How well calibrated are the emission parameters for this state?

        Returns confidence in [0, 1] based on observation count.
        score = 1 - exp(-count / 100)  → reaches 0.63 at 100 obs, 0.86 at 200.
        """
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
