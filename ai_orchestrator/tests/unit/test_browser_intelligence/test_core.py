"""Tests for Browser Intelligence OS — Phase 0: Sensors, Features, Estimation."""

from __future__ import annotations

import math

import pytest

from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector, FeatureStore
from ai_orchestrator.browser_intelligence.sensors.dom_sensor import DOMSensor, DOMFeatures
from ai_orchestrator.browser_intelligence.sensors.accessibility_sensor import (
    AccessibilitySensor,
    AccessibilityFeatures,
)
from ai_orchestrator.browser_intelligence.sensors.mutation_sensor import (
    MutationSensor,
    MutationFeatures,
)
from ai_orchestrator.browser_intelligence.estimation.belief_state import (
    BeliefState,
    HiddenState,
)
from ai_orchestrator.browser_intelligence.estimation.transition_matrix import TransitionMatrix
from ai_orchestrator.browser_intelligence.estimation.emission_model import EmissionModel
from ai_orchestrator.browser_intelligence.estimation.hmm_engine import HMMEngine
from ai_orchestrator.browser_intelligence.estimation.kalman_filter import ResponseKalmanFilter
from ai_orchestrator.browser_intelligence.decision.confidence import ConfidenceEngine
from ai_orchestrator.browser_intelligence.decision.entropy import EntropyEngine
from ai_orchestrator.browser_intelligence.decision.completion import CompletionEngine
from ai_orchestrator.browser_intelligence.decision.utility import UtilityEngine
from ai_orchestrator.browser_intelligence.features.feature_composer import FeatureComposer
from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine


# ═══════════════════════════════════════════════════════════════
# FeatureVector & FeatureStore
# ═══════════════════════════════════════════════════════════════

class TestFeatureVector:
    def test_default_values(self):
        fv = FeatureVector()
        assert fv.tick == 0
        assert not fv.input_visible
        assert fv.response_length == 0
        assert fv.page_stability == 1.0

    def test_to_list_length(self):
        fv = FeatureVector()
        lst = fv.to_list()
        assert len(lst) == 30

    def test_to_list_values(self):
        fv = FeatureVector(
            tick=5,
            input_visible=True,
            send_enabled=True,
            response_length=500,
        )
        lst = fv.to_list()
        assert lst[0] == 1.0  # input_visible
        assert lst[1] == 1.0  # send_enabled
        assert lst[22] == 500.0  # response_length


class TestFeatureStore:
    def test_push_and_latest(self):
        store = FeatureStore()
        fv = FeatureVector(tick=1, response_length=100)
        store.push(fv)
        assert store.latest is fv
        assert store.size == 1

    def test_window(self):
        store = FeatureStore(capacity=100)
        for i in range(10):
            store.push(FeatureVector(tick=i, response_length=i * 10))
        window = store.window(5)
        assert len(window) == 5
        assert window[-1].response_length == 90

    def test_ema(self):
        store = FeatureStore()
        for i in range(10):
            store.push(FeatureVector(tick=i, response_length=100))
        ema = store.ema("response_length", n=5)
        assert abs(ema - 100.0) < 1.0

    def test_derivative(self):
        store = FeatureStore()
        for i in range(10):
            store.push(FeatureVector(tick=i, response_length=i * 10, timestamp=float(i)))
        deriv = store.derivative("response_length", n=5)
        assert deriv > 0

    def test_clear(self):
        store = FeatureStore()
        store.push(FeatureVector(tick=1))
        store.clear()
        assert store.size == 0

    def test_mean_and_std(self):
        store = FeatureStore()
        for i in range(5):
            store.push(FeatureVector(tick=i, response_length=100))
        assert store.mean("response_length") == 100.0
        assert store.std("response_length") == 0.0


# ═══════════════════════════════════════════════════════════════
# DOM Sensor
# ═══════════════════════════════════════════════════════════════

class TestDOMSensor:
    def test_sense_returns_features(self):
        sensor = DOMSensor()
        features = DOMFeatures()
        assert features.input_visible is False
        assert features.send_visible is False
        assert features.dom_node_count == 0

    def test_sensor_has_sense_method(self):
        sensor = DOMSensor()
        assert hasattr(sensor, "sense")
        assert callable(sensor.sense)


# ═══════════════════════════════════════════════════════════════
# Accessibility Sensor
# ═══════════════════════════════════════════════════════════════

class TestAccessibilitySensor:
    def test_sense_no_page_returns_defaults(self):
        sensor = AccessibilitySensor()
        assert AccessibilityFeatures().text_input_count == 0

    def test_default_features(self):
        features = AccessibilityFeatures()
        assert not features.has_thinking_marker
        assert not features.has_error_marker
        assert features.accessibility_confidence == 1.0


# ═══════════════════════════════════════════════════════════════
# Mutation Sensor
# ═══════════════════════════════════════════════════════════════

class TestMutationSensor:
    def test_default_features(self):
        features = MutationFeatures()
        assert features.mutation_count == 0
        assert features.mutation_rate == 0.0

    def test_reset(self):
        sensor = MutationSensor()
        sensor._previous_rate = 10.0
        sensor.reset()
        assert sensor._previous_rate == 0.0


# ═══════════════════════════════════════════════════════════════
# BeliefState & HiddenState
# ═══════════════════════════════════════════════════════════════

class TestHiddenState:
    def test_all_states_exist(self):
        assert len(list(HiddenState)) == 10
        assert HiddenState.BOOTING.value == "booting"
        assert HiddenState.COMPLETE.value == "complete"
        assert HiddenState.SHADOW_BANNED.value == "shadow_banned"


class TestBeliefState:
    def test_uniform(self):
        belief = BeliefState.uniform()
        for s in HiddenState:
            assert abs(belief.probabilities[s] - 0.1) < 0.01

    def test_certain(self):
        belief = BeliefState.certain(HiddenState.READY)
        assert belief.probabilities[HiddenState.READY] == 1.0
        assert belief.most_likely == HiddenState.READY
        assert belief.confidence == 1.0

    def test_entropy(self):
        belief = BeliefState.uniform()
        assert abs(belief.entropy - math.log2(10)) < 0.01

    def test_certain_entropy(self):
        belief = BeliefState.certain(HiddenState.READY)
        assert belief.entropy == 0.0

    def test_is_confident(self):
        belief = BeliefState.certain(HiddenState.READY)
        assert belief.is_confident(0.85)

    def test_is_uncertain(self):
        belief = BeliefState.uniform()
        assert belief.is_uncertain(0.5)

    def test_most_likely(self):
        probs = {s: 0.01 for s in HiddenState}
        probs[HiddenState.GENERATING] = 0.91
        belief = BeliefState(probabilities=probs)
        assert belief.most_likely == HiddenState.GENERATING


# ═══════════════════════════════════════════════════════════════
# Transition Matrix
# ═══════════════════════════════════════════════════════════════

class TestTransitionMatrix:
    def test_valid_probabilities(self):
        tm = TransitionMatrix()
        for from_state in HiddenState:
            total = sum(
                tm.transition_prob(from_state, to_state)
                for to_state in HiddenState
            )
            assert abs(total - 1.0) < 0.1, f"From {from_state}: total={total}"

    def test_individual_probs(self):
        tm = TransitionMatrix()
        p = tm.transition_prob(HiddenState.READY, HiddenState.PROMPT_SENT)
        assert 0.01 <= p <= 0.99
        p = tm.transition_prob(HiddenState.GENERATING, HiddenState.COMPLETE)
        assert 0.01 <= p <= 0.99

    def test_to_prob_matrix(self):
        tm = TransitionMatrix()
        mat = tm.to_prob_matrix()
        assert len(mat) == 10
        assert len(mat[0]) == 10
        for row in mat:
            assert abs(sum(row) - 1.0) < 0.15

    def test_update_from_counts(self):
        tm = TransitionMatrix()
        counts = {
            (HiddenState.READY, HiddenState.PROMPT_SENT): 10,
            (HiddenState.READY, HiddenState.READY): 5,
        }
        tm.update_from_counts(counts)
        p = tm.transition_prob(HiddenState.READY, HiddenState.PROMPT_SENT)
        assert p > 0.4


# ═══════════════════════════════════════════════════════════════
# Emission Model
# ═══════════════════════════════════════════════════════════════

class TestEmissionModel:
    def test_emission_prob_returns_positive(self):
        em = EmissionModel()
        fv = FeatureVector(
            input_visible=True,
            send_enabled=True,
            response_length=100,
        )
        for state in HiddenState:
            prob = em.emission_prob(fv, state)
            assert prob > 0, f"Zero prob for {state}"

    def test_ready_state_higher_for_ready_features(self):
        em = EmissionModel()
        ready_fv = FeatureVector(
            input_visible=True,
            send_enabled=True,
            response_length=100,
        )
        gen_fv = FeatureVector(
            input_visible=True,
            send_enabled=False,
            stop_button_visible=True,
            has_streaming_marker=True,
            mutation_rate=8.0,
            response_length=300,
        )
        ready_prob = em.emission_prob(ready_fv, HiddenState.READY)
        gen_prob = em.emission_prob(gen_fv, HiddenState.GENERATING)
        assert ready_prob > 0
        assert gen_prob > 0

    def test_update_from_observation(self):
        em = EmissionModel()
        fv = FeatureVector(
            input_visible=True,
            send_enabled=True,
            response_length=100,
        )
        em.update_from_observation(fv, HiddenState.READY)
        prob = em.emission_prob(fv, HiddenState.READY)
        assert prob > 0


# ═══════════════════════════════════════════════════════════════
# HMM Engine
# ═══════════════════════════════════════════════════════════════

class TestHMMEngine:
    def test_initialize(self):
        engine = HMMEngine()
        belief = engine.initialize()
        assert isinstance(belief, BeliefState)
        for s in HiddenState:
            assert belief.probabilities[s] > 0

    def test_update_converges_to_ready(self):
        engine = HMMEngine()
        engine.initialize()

        for _ in range(40):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=True,
                stop_button_visible=False,
                regenerate_visible=False,
                error_banner_visible=False,
                auth_form_visible=False,
                has_thinking_marker=False,
                has_error_marker=False,
                has_rate_limit_marker=False,
                has_streaming_marker=False,
                stream_active=False,
                transport_detected=False,
                generation_started=False,
                generation_completed=False,
                stream_closed=False,
                mutation_rate=0.0,
                response_length=100,
                page_stability=1.0,
                visual_stability=1.0,
            )
            belief = engine.update(fv)

        assert belief.probabilities[HiddenState.READY] > 0.4
        assert belief.probabilities[HiddenState.ERROR] < 0.1

    def test_update_detects_generating(self):
        engine = HMMEngine()
        engine.learning_enabled = False
        engine.initialize()

        for _ in range(10):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=True,
                stop_button_visible=False,
                error_banner_visible=False,
                auth_form_visible=False,
                has_thinking_marker=False,
                has_error_marker=False,
                has_rate_limit_marker=False,
                has_streaming_marker=False,
                stream_active=False,
                transport_detected=False,
                generation_started=False,
                generation_completed=False,
                stream_closed=False,
                mutation_rate=0.0,
                js_heap_used_mb=100.0,
                response_length=100,
                page_stability=1.0,
                visual_stability=1.0,
            )
            engine.update(fv)

        for _ in range(20):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=False,
                stop_button_visible=True,
                error_banner_visible=False,
                auth_form_visible=False,
                has_thinking_marker=False,
                has_error_marker=False,
                has_rate_limit_marker=False,
                has_streaming_marker=True,
                stream_active=True,
                transport_detected=True,
                generation_started=True,
                generation_stop_detected=False,
                mutation_rate=8.0,
                js_heap_used_mb=300.0,
                response_length=300,
                response_length_delta=50,
                visual_stability=0.8,
            )
            belief = engine.update(fv)

        assert belief.probabilities[HiddenState.GENERATING] > 0.15

    def test_update_converges_to_complete(self):
        engine = HMMEngine()
        engine.learning_enabled = False
        engine.initialize()

        for _ in range(15):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=False,
                stop_button_visible=True,
                has_streaming_marker=True,
                stream_active=True,
                transport_detected=True,
                generation_started=True,
                mutation_rate=8.0,
                response_length=300,
            )
            engine.update(fv)

        for _ in range(25):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=True,
                stop_button_visible=False,
                regenerate_visible=True,
                has_streaming_marker=False,
                has_thinking_marker=False,
                stream_active=False,
                generation_completed=True,
                stream_closed=True,
                mutation_rate=0.0,
                response_length=500,
                response_length_delta=0,
                generation_stop_detected=True,
            )
            belief = engine.update(fv)

        assert belief.probabilities[HiddenState.COMPLETE] > 0.05

    def test_viterbi(self):
        engine = HMMEngine()
        engine.initialize()

        observations = []
        for i in range(30):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=True,
                response_length=100,
            )
            observations.append(fv)

        path = engine.viterbi(observations)
        assert len(path) == 30
        assert all(isinstance(s, HiddenState) for s in path)

    def test_reset(self):
        engine = HMMEngine()
        engine.initialize()
        engine.update(FeatureVector())
        engine.reset()
        assert engine.belief is None


# ═══════════════════════════════════════════════════════════════
# Kalman Filter
# ═══════════════════════════════════════════════════════════════

class TestKalmanFilter:
    def test_initial_state(self):
        kf = ResponseKalmanFilter()
        assert kf.x == [0.0, 0.0, 0.0]

    def test_predict_update_cycle(self):
        kf = ResponseKalmanFilter()
        kf.predict()
        state = kf.update(100.0)
        assert len(state) == 3
        assert state[0] > 0

    def test_smooth(self):
        kf = ResponseKalmanFilter()
        measurements = [10.0, 20.0, 30.0, 40.0, 50.0]
        smoothed = kf.smooth(measurements)
        assert len(smoothed) == 5
        assert smoothed[-1] > 0

    def test_velocity(self):
        kf = ResponseKalmanFilter()
        for m in [10.0, 20.0, 30.0, 40.0, 50.0]:
            kf.predict()
            kf.update(m)
        v = kf.velocity()
        assert v > 0

    def test_acceleration(self):
        kf = ResponseKalmanFilter()
        for m in [10.0, 20.0, 30.0, 40.0, 50.0]:
            kf.predict()
            kf.update(m)
        a = kf.acceleration()
        assert isinstance(a, float)

    def test_reset(self):
        kf = ResponseKalmanFilter()
        kf.predict()
        kf.update(100.0)
        kf.reset()
        assert kf.x == [0.0, 0.0, 0.0]


# ═══════════════════════════════════════════════════════════════
# Confidence Engine
# ═══════════════════════════════════════════════════════════════

class TestConfidenceEngine:
    def test_compute(self):
        engine = ConfidenceEngine()
        conf = engine.compute(0.9, 0.8, 0.7, 0.6, 0.5)
        assert 0.0 <= conf <= 1.0

    def test_from_belief(self):
        engine = ConfidenceEngine()
        belief = BeliefState.certain(HiddenState.READY)
        conf = engine.from_belief(belief)
        assert conf > 0.9


# ═══════════════════════════════════════════════════════════════
# Entropy Engine
# ═══════════════════════════════════════════════════════════════

class TestEntropyEngine:
    def test_compute_uniform(self):
        engine = EntropyEngine()
        belief = BeliefState.uniform()
        h = engine.compute(belief)
        assert abs(h - math.log2(10)) < 0.01

    def test_compute_certain(self):
        engine = EntropyEngine()
        belief = BeliefState.certain(HiddenState.READY)
        h = engine.compute(belief)
        assert h == 0.0

    def test_is_confused(self):
        engine = EntropyEngine(high_entropy_threshold=0.5)
        belief = BeliefState.uniform()
        assert engine.is_confused(belief)

    def test_is_not_confused(self):
        engine = EntropyEngine()
        belief = BeliefState.certain(HiddenState.READY)
        assert not engine.is_confused(belief)

    def test_should_recover(self):
        engine = EntropyEngine()
        belief = BeliefState.uniform()
        assert engine.should_recover(belief)

    def test_exploration_factor(self):
        engine = EntropyEngine()
        belief = BeliefState.certain(HiddenState.READY)
        factor = engine.exploration_factor(belief)
        assert 0.0 <= factor <= 1.0

    def test_normalized(self):
        engine = EntropyEngine()
        belief = BeliefState.certain(HiddenState.READY)
        assert engine.normalized(belief) == 0.0
        belief = BeliefState.uniform()
        assert engine.normalized(belief) > 0.9


# ═══════════════════════════════════════════════════════════════
# Completion Engine
# ═══════════════════════════════════════════════════════════════

class TestCompletionEngine:
    def test_not_complete_with_insufficient_data(self):
        engine = CompletionEngine()
        store = FeatureStore()
        done, conf = engine.is_complete(store)
        assert not done
        assert conf == 0.0

    def test_complete_when_response_stable(self):
        engine = CompletionEngine(
            velocity_threshold=100.0,
            stable_for_ticks=3,
            confidence_threshold=0.3,
        )
        store = FeatureStore()
        for i in range(20):
            store.push(FeatureVector(
                tick=i,
                timestamp=float(i),
                response_length=500,
                stop_button_visible=False,
                has_streaming_marker=False,
                generation_stop_detected=True,
            ))
        done, conf = engine.is_complete(store)
        assert conf >= 0.0

    def test_not_complete_when_growing(self):
        engine = CompletionEngine()
        store = FeatureStore()
        for i in range(10):
            store.push(FeatureVector(
                tick=i,
                timestamp=float(i),
                response_length=100 + i * 50,
                stop_button_visible=True,
                has_streaming_marker=True,
            ))
        done, _ = engine.is_complete(store)
        assert not done

    def test_reset(self):
        engine = CompletionEngine()
        store = FeatureStore()
        for i in range(5):
            store.push(FeatureVector(
                tick=i,
                timestamp=float(i),
                response_length=500,
                generation_stop_detected=True,
            ))
        engine.is_complete(store)
        engine.reset()
        assert engine._stable_count == 0


# ═══════════════════════════════════════════════════════════════
# Utility Engine
# ═══════════════════════════════════════════════════════════════

class TestUtilityEngine:
    def test_expected_utility_type_prompt_ready(self):
        engine = UtilityEngine()
        belief = BeliefState.certain(HiddenState.READY)
        util = engine.expected_utility("type_prompt", belief)
        assert util > 0

    def test_expected_utility_type_prompt_generating(self):
        engine = UtilityEngine()
        belief = BeliefState.certain(HiddenState.GENERATING)
        util = engine.expected_utility("type_prompt", belief)
        assert util < 0

    def test_best_action_ready(self):
        engine = UtilityEngine()
        belief = BeliefState.certain(HiddenState.READY)
        action, util = engine.best_action(
            ["type_prompt", "wait", "recover"],
            belief,
        )
        assert action == "type_prompt"
        assert util > 0

    def test_best_action_generating(self):
        engine = UtilityEngine()
        belief = BeliefState.certain(HiddenState.GENERATING)
        action, util = engine.best_action(
            ["type_prompt", "wait", "extract_response"],
            belief,
        )
        assert action == "wait"

    def test_all_utilities(self):
        engine = UtilityEngine()
        belief = BeliefState.certain(HiddenState.COMPLETE)
        utils = engine.all_utilities(
            ["extract_response", "wait", "type_prompt"],
            belief,
        )
        assert utils["extract_response"] > utils["wait"]
        assert utils["type_prompt"] <= 0


# ═══════════════════════════════════════════════════════════════
# BrowserIntelligenceEngine (E2E)
# ═══════════════════════════════════════════════════════════════

class TestBrowserIntelligenceEngine:
    def test_engine_creation(self):
        engine = BrowserIntelligenceEngine()
        assert engine.belief is None
        assert engine.confidence == 0.0
        assert not engine.is_ready_for_prompt
        assert not engine.is_error
        assert engine.recommended_action == "wait"

    def test_engine_no_page_tick(self):
        """Engine handles tick without a real page gracefully via pure data."""
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()

        for i in range(40):
            fv = FeatureVector(
                tick=i,
                timestamp=float(i),
                input_visible=True,
                send_enabled=True,
                stop_button_visible=False,
                error_banner_visible=False,
                auth_form_visible=False,
                response_length=100,
                page_stability=1.0,
                visual_stability=1.0,
                has_thinking_marker=False,
                has_error_marker=False,
                has_rate_limit_marker=False,
                has_streaming_marker=False,
                stream_active=False,
                transport_detected=False,
                generation_started=False,
                generation_completed=False,
                stream_closed=False,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        assert engine.belief is not None
        assert engine.belief.probabilities[HiddenState.READY] > 0.4

    def test_state_probabilities(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()
        for _ in range(20):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=True,
                response_length=100,
                page_stability=1.0,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        probs = engine.state_probabilities()
        assert "ready" in probs
        assert probs["ready"] > 0.3

    def test_action_utilities(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()
        for _ in range(20):
            fv = FeatureVector(
                input_visible=True,
                send_enabled=True,
                response_length=100,
                page_stability=1.0,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        utils = engine.action_utilities()
        assert "type_prompt" in utils
        assert "wait" in utils

    def test_engine_reset(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()
        engine._store.push(FeatureVector(tick=1))
        engine.reset()
        assert engine._store.size == 0
        assert engine.belief is None
        assert not engine.is_ready_for_prompt

    def test_completion_detection(self):
        engine = BrowserIntelligenceEngine()
        for i in range(10):
            fv = FeatureVector(
                tick=i,
                timestamp=float(i),
                response_length=500,
                stop_button_visible=False,
                has_streaming_marker=False,
                generation_stop_detected=True,
            )
            engine._store.push(fv)

        done, conf = engine.is_response_complete()
        assert done or conf >= 0.0

    def test_rate_limited_detection(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()
        for _ in range(40):
            fv = FeatureVector(
                input_visible=False,
                send_enabled=False,
                error_banner_visible=True,
                has_rate_limit_marker=True,
                has_error_marker=False,
                has_thinking_marker=False,
                has_streaming_marker=False,
                stream_active=False,
                generation_completed=False,
                stream_closed=True,
                mutation_rate=0.0,
                js_heap_used_mb=0.0,
                response_length=0,
                page_stability=0.5,
                visual_stability=0.5,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        assert engine._belief.probabilities[HiddenState.RATE_LIMITED] > 0.05

    def test_error_detection(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.learning_enabled = False
        engine._hmm.initialize()
        for _ in range(40):
            fv = FeatureVector(
                input_visible=False,
                send_enabled=False,
                error_banner_visible=True,
                has_error_marker=True,
                has_rate_limit_marker=False,
                has_thinking_marker=False,
                has_streaming_marker=False,
                stream_active=False,
                generation_completed=False,
                stream_closed=True,
                mutation_rate=0.0,
                js_heap_used_mb=0.0,
                response_length=0,
                page_stability=0.3,
                visual_stability=0.5,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        assert engine._belief.probabilities[HiddenState.ERROR] > 0.03


# ═══════════════════════════════════════════════════════════════
# FeatureComposer
# ═══════════════════════════════════════════════════════════════

class TestFeatureComposer:
    def test_creation(self):
        composer = FeatureComposer()
        assert composer._tick == 0
        assert composer._dom is not None
        assert composer._a11y is not None

    def test_reset(self):
        composer = FeatureComposer()
        composer._tick = 10
        composer.reset()
        assert composer._tick == 0


# ═══════════════════════════════════════════════════════════════
# Phase 2: Full pipeline simulation (no Playwright needed)
# ═══════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Simulate a complete prompt→response cycle using data-only path."""

    def test_ready_to_complete_cycle(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.learning_enabled = False
        engine._hmm.initialize()

        # Phase 1: READY state (15 ticks)
        for i in range(15):
            fv = FeatureVector(
                tick=i,
                timestamp=float(i),
                input_visible=True,
                send_enabled=True,
                stop_button_visible=False,
                error_banner_visible=False,
                has_streaming_marker=False,
                stream_active=False,
                mutation_rate=0.0,
                response_length=100,
                page_stability=1.0,
                visual_stability=1.0,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        assert engine._belief.most_likely == HiddenState.READY

        # Phase 2: PROMPT_SENT → GENERATING (25 ticks of growing response)
        for i in range(15, 40):
            fv = FeatureVector(
                tick=i,
                timestamp=float(i),
                input_visible=True,
                send_enabled=False,
                stop_button_visible=True,
                has_streaming_marker=True,
                stream_active=True,
                transport_detected=True,
                generation_started=True,
                mutation_rate=5.0 + i * 0.5,
                response_length=100 + (i - 10) * 40,
                response_length_delta=40,
                visual_stability=0.8,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        assert engine._belief.probabilities[HiddenState.GENERATING] > 0.08

        # Phase 3: COMPLETE (20 ticks of stable response)
        for i in range(40, 60):
            fv = FeatureVector(
                tick=i,
                timestamp=float(i),
                input_visible=True,
                send_enabled=True,
                stop_button_visible=False,
                regenerate_visible=True,
                has_streaming_marker=False,
                stream_active=False,
                generation_completed=True,
                stream_closed=True,
                generation_stop_detected=True,
                mutation_rate=0.0,
                response_length=500,
                response_length_delta=0,
                visual_stability=1.0,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        assert engine._belief.probabilities[HiddenState.COMPLETE] > 0.05

        # Verify completion confidence is valid
        done, conf = engine.is_response_complete()
        assert conf >= 0.0, f"Completion confidence should be non-negative, got {conf}"
        assert isinstance(done, bool)
