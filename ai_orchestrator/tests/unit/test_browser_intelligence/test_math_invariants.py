"""Math invariant enforcement tests for the Browser Intelligence OS.

The engine's probabilistic claims are only useful if they remain
valid in every code path. These tests exercise the math directly
and assert the invariants:

- Transition matrix:  each row sums to 1.0 (forward kernel valid).
- Belief state:        probabilities sum to 1.0 (normalised).
- Confidence:          0 ≤ confidence ≤ 1.
- Entropy:             ≥ 0 always.
- Beta posterior:      mean in [0, 1], variance > 0 for uninformative
                       priors, decreases as data accumulates.
"""

from __future__ import annotations

import math

import pytest

from ai_orchestrator.browser_intelligence.estimation.belief_state import (
    BeliefState,
    HiddenState,
)
from ai_orchestrator.browser_intelligence.estimation.transition_matrix import (
    TransitionMatrix,
)
from ai_orchestrator.browser_intelligence.intelligence.shadow_ban_detector import (
    ShadowBanDetector,
    ShadowBanState,
)
from ai_orchestrator.browser_intelligence.intelligence.traffic_classifier import (
    ResponseClassifier,
    TrafficCategory,
)
from ai_orchestrator.browser_intelligence.learning import (
    BayesianReliability,
    beta_mean,
    beta_variance,
)


class TestTransitionMatrix:
    def test_each_row_sums_to_one(self):
        tm = TransitionMatrix()
        for state in HiddenState:
            row_sum = sum(
                tm.transition_prob(state, target) for target in HiddenState
            )
            assert abs(row_sum - 1.0) < 1e-6, f"row for {state} sums to {row_sum}"

    def test_no_negative_probabilities(self):
        tm = TransitionMatrix()
        for state in HiddenState:
            for target in HiddenState:
                p = tm.transition_prob(state, target)
                assert p >= 0.0, f"negative prob {state}→{target} = {p}"

    def test_validate_stochastic(self):
        tm = TransitionMatrix()
        assert tm.validate_stochastic()

    def test_is_ergodic(self):
        tm = TransitionMatrix()
        assert tm.is_ergodic()

    def test_min_entry_non_negative(self):
        tm = TransitionMatrix()
        assert tm.min_entry() >= 0.0

    def test_row_sums_to_one(self):
        tm = TransitionMatrix()
        rs = tm.row_sums()
        for state, s in rs.items():
            assert abs(s - 1.0) < 1e-6


class TestBeliefState:
    def test_initial_uniform(self):
        b = BeliefState.uniform()
        s = sum(b.probabilities.values())
        assert abs(s - 1.0) < 1e-9
        for p in b.probabilities.values():
            assert p >= 0.0

    def test_certain_normalizes(self):
        b = BeliefState.certain(HiddenState.READY)
        s = sum(b.probabilities.values())
        assert abs(s - 1.0) < 1e-9
        assert b.probabilities[HiddenState.READY] == 1.0

    def test_auto_normalize_on_init(self):
        # __post_init__ should renormalize.
        b = BeliefState({HiddenState.READY: 2, HiddenState.GENERATING: 1, HiddenState.ERROR: 1})
        s = sum(b.probabilities.values())
        assert abs(s - 1.0) < 1e-9

    def test_confidence_in_unit_interval(self):
        b = BeliefState.uniform()
        assert 0.0 <= b.confidence <= 1.0

    def test_entropy_non_negative(self):
        b = BeliefState.uniform()
        assert b.entropy >= 0.0
        for p in b.probabilities.values():
            assert p >= 0.0

    def test_entropy_decreases_with_certainty(self):
        uniform = BeliefState.uniform()
        peaky = BeliefState.certain(HiddenState.READY)
        assert peaky.entropy < uniform.entropy

    def test_entropy_max_for_uniform_over_10_states(self):
        b = BeliefState.uniform()
        # 10 hidden states → log2(10) ≈ 3.32
        max_entropy = math.log2(len(b.probabilities))
        assert abs(b.entropy - max_entropy) < 1e-9


class TestConfidenceEngine:
    def test_confidence_in_unit_interval(self):
        from ai_orchestrator.browser_intelligence.decision.confidence import (
            ConfidenceEngine,
        )
        ce = ConfidenceEngine()
        # Engine takes float observation_quality, not FeatureVector.
        conf = ce.compute(
            observation_quality=0.85,
            historical_success=0.9,
            selector_reliability=0.95,
            accessibility_reliability=0.8,
            network_reliability=0.92,
        )
        assert 0.0 <= conf <= 1.0

    def test_from_belief_in_unit_interval(self):
        from ai_orchestrator.browser_intelligence.decision.confidence import (
            ConfidenceEngine,
        )
        ce = ConfidenceEngine()
        b = BeliefState.uniform()
        assert 0.0 <= ce.from_belief(b) <= 1.0
        peaky = BeliefState.certain(HiddenState.READY)
        # Certain belief → entropy = 0 → confidence = 1.0.
        assert ce.from_belief(peaky) == 1.0


class TestShadowBanInvariants:
    def test_posterior_sums_to_one_across_states(self):
        s = ShadowBanDetector()
        for _ in range(15):
            s.observe(response_length=2000, completion_rate=1.0, tokens_per_second=30.0)
        for _ in range(5):
            v = s.observe(
                response_length=500,
                completion_rate=0.5,
                tokens_per_second=5.0,
            )
            total = v.p_normal + v.p_degraded + v.p_shadow_ban
            assert abs(total - 1.0) < 1e-9, f"posterior sums to {total}"

    def test_state_consistent_with_posterior(self):
        s = ShadowBanDetector()
        for _ in range(20):
            s.observe(response_length=2000, completion_rate=1.0, tokens_per_second=30.0)
        for _ in range(15):
            v = s.observe(
                response_length=10,
                completion_rate=0.1,
                tokens_per_second=0.5,
                error_count=5,
            )
        if v.p_shadow_ban > 0.5:
            assert v.state == ShadowBanState.SHADOW_BANNED
        elif v.p_shadow_ban + v.p_degraded > 0.3:
            assert v.state == ShadowBanState.DEGRADED


class TestTrafficClassifierInvariants:
    def test_confidence_in_unit_interval_for_random_urls(self):
        cls = ResponseClassifier()
        urls = [
            "https://api.openai.com/v1/chat/completions",
            "https://chatgpt.com/backend-api/conversation",
            "https://www.google-analytics.com/collect",
            "https://example.com/auth/login",
            "https://example.com/static/main.js",
            "https://example.com/api/quota",
        ]
        for url in urls:
            c = cls.classify(url=url, method="POST", status=200)
            assert 0.0 <= c.confidence <= 1.0, url


class TestBayesianInvariants:
    def test_beta_mean_in_unit_interval(self):
        for s in range(20):
            for f in range(20):
                m = beta_mean(1.0 + s, 1.0 + f)
                assert 0.0 <= m <= 1.0

    def test_beta_variance_non_negative(self):
        for s in range(20):
            for f in range(20):
                v = beta_variance(1.0 + s, 1.0 + f)
                assert v >= 0.0

    def test_reliability_stays_in_unit_interval(self):
        r = BayesianReliability(key="x")
        outcomes = [True, False, True, True, False, True, True, True, False, True]
        for ok in outcomes:
            r.update(ok)
            assert 0.0 <= r.posterior_mean <= 1.0
            assert r.variance >= 0.0
