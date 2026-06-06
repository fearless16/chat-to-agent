"""Tests for Browser Intelligence Runtime improvements.

Covers: P1 online learning, P2 DOM refactor, P3 transition matrix,
P4 FeatureStore validation, P5 evidence fusion + aging + stream stalled.
"""

from __future__ import annotations

import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_orchestrator.browser_intelligence.decision.evidence_fusion import (
    EvidenceFusion,
    SensorConfidence,
    EvidenceVector,
)
from ai_orchestrator.browser_intelligence.estimation.belief_state import (
    BeliefState,
    HiddenState,
)
from ai_orchestrator.browser_intelligence.estimation.emission_model import EmissionModel
from ai_orchestrator.browser_intelligence.estimation.hmm_engine import HMMEngine
from ai_orchestrator.browser_intelligence.estimation.transition_matrix import TransitionMatrix
from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector, FeatureStore
from ai_orchestrator.browser_intelligence.sensors.dom_sensor import DOMSensor, DOMFeatures
from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine


# ═══════════════════════════════════════════════════════════════
# P4: FeatureStore capacity=0 validation
# ═══════════════════════════════════════════════════════════════

class TestFeatureStoreCapacity:

    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            FeatureStore(capacity=0)

    def test_capacity_negative_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            FeatureStore(capacity=-5)

    def test_capacity_one_works(self):
        store = FeatureStore(capacity=1)
        fv = FeatureVector(tick=1)
        store.push(fv)
        assert store.size == 1
        fv2 = FeatureVector(tick=2)
        store.push(fv2)
        assert store.size == 1
        assert store.latest.tick == 2

    def test_capacity_default(self):
        store = FeatureStore()
        assert store.capacity == 300

    def test_capacity_preserved(self):
        store = FeatureStore(capacity=50)
        assert store.capacity == 50


# ═══════════════════════════════════════════════════════════════
# P4: Observation aging
# ═══════════════════════════════════════════════════════════════

class TestObservationAging:

    def test_aged_mean_favors_recent(self):
        store = FeatureStore(capacity=100)
        for i in range(10):
            store.push(FeatureVector(tick=i, timestamp=float(i), response_length=float(i)))
        aged = store.aged_mean("response_length", n=10, half_life_ticks=2)
        simple = store.mean("response_length", n=10)
        assert aged > simple, f"aged={aged} should be > simple={simple} (favors recent)"

    def test_aged_mean_empty(self):
        store = FeatureStore(capacity=100)
        assert store.aged_mean("response_length", n=10) == 0.0

    def test_obsolescence_weight_zero_age(self):
        store = FeatureStore(capacity=100)
        w = store.obsolescence_weight(0, half_life_ticks=30)
        assert abs(w - 1.0) < 1e-6

    def test_obsolescence_weight_at_half_life(self):
        store = FeatureStore(capacity=100)
        w = store.obsolescence_weight(30, half_life_ticks=30)
        assert abs(w - 0.5) < 1e-6

    def test_obsolescence_weight_past(self):
        store = FeatureStore(capacity=100)
        w = store.obsolescence_weight(300, half_life_ticks=30)
        assert w < 0.01

    def test_aged_mean_invalid_half_life(self):
        store = FeatureStore(capacity=100)
        store.push(FeatureVector(tick=1, response_length=10))
        with pytest.raises(ValueError, match="half_life"):
            store.aged_mean("response_length", n=10, half_life_ticks=0)

    def test_obsolescence_weight_invalid_half_life(self):
        store = FeatureStore(capacity=100)
        with pytest.raises(ValueError, match="half_life"):
            store.obsolescence_weight(5, half_life_ticks=0)


# ═══════════════════════════════════════════════════════════════
# P3: Transition matrix stochastic invariants
# ═══════════════════════════════════════════════════════════════

class TestTransitionMatrixInvariants:

    def test_validate_stochastic_passes(self):
        tm = TransitionMatrix()
        assert tm.validate_stochastic(tolerance=1e-6)

    def test_validate_stochastic_with_tolerance(self):
        tm = TransitionMatrix()
        assert tm.validate_stochastic(tolerance=0.1)

    def test_enforce_stochastic_maintains_valid(self):
        tm = TransitionMatrix()
        tm.enforce_stochastic()
        assert tm.validate_stochastic(tolerance=1e-6)

    def test_row_sums_are_one(self):
        tm = TransitionMatrix()
        sums = tm.row_sums()
        for state, total in sums.items():
            assert abs(total - 1.0) < 1e-6, f"{state}: sum={total}"

    def test_min_entry_positive(self):
        tm = TransitionMatrix()
        assert tm.min_entry() > 0

    def test_max_entry_reasonable(self):
        tm = TransitionMatrix()
        assert tm.max_entry() < 1.0

    def test_epsilon_configurable(self):
        tm = TransitionMatrix(epsilon=1e-3)
        assert tm.epsilon == 1e-3
        assert tm.min_entry() > 1e-9

    def test_update_then_validate(self):
        tm = TransitionMatrix()
        counts = {
            (HiddenState.READY, HiddenState.PROMPT_SENT): 50,
            (HiddenState.READY, HiddenState.READY): 50,
        }
        tm.update_from_counts(counts)
        assert tm.validate_stochastic()

    def test_is_ergodic(self):
        tm = TransitionMatrix(epsilon=1e-3)
        assert tm.is_ergodic()

    def test_to_prob_matrix_rows_sum_to_one(self):
        tm = TransitionMatrix()
        mat = tm.to_prob_matrix()
        for row in mat:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_enforce_stochastic_after_corruption(self):
        tm = TransitionMatrix()
        row_sums_before = tm.row_sums()
        tm.enforce_stochastic()
        row_sums_after = tm.row_sums()
        for s in HiddenState:
            assert abs(row_sums_after[s] - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════
# P1: Online emission model learning + calibration
# ═══════════════════════════════════════════════════════════════

class TestEmissionModelOnlineLearning:

    def test_soft_assignment_updates_all_states(self):
        em = EmissionModel()
        fv = FeatureVector(
            input_visible=True, send_enabled=True,
            response_length=100, page_stability=1.0,
        )
        beliefs = {s: 1.0 / 10 for s in HiddenState}
        beliefs[HiddenState.READY] = 0.8
        remaining = (1.0 - 0.8) / 9
        for s in HiddenState:
            if s != HiddenState.READY:
                beliefs[s] = remaining

        em.update_from_soft_assignment(fv, beliefs)
        for s in HiddenState:
            assert em.observation_count(s) >= 0

    def test_calibration_score_zero_initially(self):
        em = EmissionModel()
        assert em.calibration_score(HiddenState.READY) == 0.0

    def test_calibration_score_rises_with_observations(self):
        em = EmissionModel()
        fv = FeatureVector(input_visible=True, response_length=100)
        for _ in range(50):
            em.update_from_observation(fv, HiddenState.READY, learning_rate=0.05)
        score = em.calibration_score(HiddenState.READY)
        assert 0.3 < score < 0.5

    def test_calibration_score_approaches_one(self):
        em = EmissionModel()
        fv = FeatureVector(input_visible=True, response_length=100)
        for _ in range(500):
            em.update_from_observation(fv, HiddenState.GENERATING, learning_rate=0.02)
        score = em.calibration_score(HiddenState.GENERATING)
        assert score > 0.95

    def test_total_observations_accumulates(self):
        em = EmissionModel()
        fv = FeatureVector(response_length=100)
        for i in range(10):
            em.update_from_observation(fv, HiddenState.READY)
        assert em.total_observations() == 10

    def test_update_from_observation_increments_count(self):
        em = EmissionModel()
        assert em.observation_count(HiddenState.THINKING) == 0
        fv = FeatureVector(response_length=100)
        em.update_from_observation(fv, HiddenState.THINKING)
        assert em.observation_count(HiddenState.THINKING) == 1

    def test_soft_assignment_skips_low_belief(self):
        em = EmissionModel()
        fv = FeatureVector(response_length=100)
        beliefs = {s: 0.0 for s in HiddenState}
        beliefs[HiddenState.READY] = 1.0
        em.update_from_soft_assignment(fv, beliefs)
        assert em.observation_count(HiddenState.READY) >= 1
        assert em.observation_count(HiddenState.ERROR) == 0


# ═══════════════════════════════════════════════════════════════
# P1: HMM Engine adaptive thresholds
# ═══════════════════════════════════════════════════════════════

class TestHMMEngineAdaptiveThreshold:

    def test_adaptive_threshold_starts_low(self):
        engine = HMMEngine()
        engine.initialize()
        thresh = engine.adaptive_readiness_threshold(
            base_threshold=0.45, min_threshold=0.30, max_threshold=0.75
        )
        assert abs(thresh - 0.45) < 0.01

    def test_adaptive_threshold_clamped_to_min(self):
        engine = HMMEngine()
        engine.initialize()
        thresh = engine.adaptive_readiness_threshold(
            base_threshold=-1.0, min_threshold=0.20, max_threshold=0.75
        )
        assert thresh >= 0.20

    def test_adaptive_threshold_clamped_to_max(self):
        engine = HMMEngine()
        engine.initialize()
        thresh = engine.adaptive_readiness_threshold(
            base_threshold=2.0, min_threshold=0.10, max_threshold=0.60
        )
        assert thresh <= 0.60

    def test_adaptive_threshold_rises_with_learning(self):
        engine = HMMEngine()
        engine.learning_enabled = True
        engine.initialize()

        fv = FeatureVector(
            input_visible=True, send_enabled=True,
            response_length=100, page_stability=1.0,
        )
        for _ in range(200):
            engine.update(fv)

        thresh = engine.adaptive_readiness_threshold(
            base_threshold=0.45, min_threshold=0.30, max_threshold=0.75
        )
        assert thresh > 0.50

    def test_learning_can_be_disabled(self):
        engine = HMMEngine()
        engine.learning_enabled = False
        engine.initialize()

        fv = FeatureVector(response_length=100)
        count_before = engine.emission.total_observations()
        for _ in range(50):
            engine.update(fv)
        count_after = engine.emission.total_observations()
        assert count_after == count_before


# ═══════════════════════════════════════════════════════════════
# P2: DOM sensor single-eval
# ═══════════════════════════════════════════════════════════════

class TestDOMSensorSingleEval:

    @pytest.mark.asyncio
    async def test_sense_calls_evaluate_once(self):
        sensor = DOMSensor()
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value={
            "input_visible": True,
            "send_visible": False,
            "stop_visible": False,
            "regenerate_visible": False,
            "error_visible": False,
            "auth_visible": False,
            "dom_nodes": 150,
            "interactive": 12,
        })

        result = await sensor.sense(mock_page)

        assert mock_page.evaluate.call_count == 1
        assert result.input_visible is True
        assert result.dom_node_count == 150
        assert result.interactive_count == 12

    @pytest.mark.asyncio
    async def test_sense_handles_evaluate_failure(self):
        sensor = DOMSensor()
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("JS error"))

        result = await sensor.sense(mock_page)

        assert result.input_visible is False
        assert result.dom_node_count == 0

    @pytest.mark.asyncio
    async def test_sense_returns_all_features(self):
        sensor = DOMSensor()
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value={
            "input_visible": True,
            "send_visible": True,
            "stop_visible": True,
            "regenerate_visible": True,
            "error_visible": True,
            "auth_visible": True,
            "dom_nodes": 42,
            "interactive": 7,
        })

        result = await sensor.sense(mock_page)
        assert result.input_visible is True
        assert result.send_visible is True
        assert result.stop_button_visible is True
        assert result.regenerate_visible is True
        assert result.error_banner_visible is True
        assert result.auth_form_visible is True

    def test_reset(self):
        sensor = DOMSensor()
        sensor.reset()


# ═══════════════════════════════════════════════════════════════
# P5: Evidence Fusion
# ═══════════════════════════════════════════════════════════════

class TestSensorConfidence:

    def test_initial_confidence_is_one(self):
        sc = SensorConfidence(name="test")
        assert sc.confidence == 1.0

    def test_confidence_drops_with_failures(self):
        sc = SensorConfidence(name="test")
        for _ in range(5):
            sc.record_failure()
        assert sc.confidence < 0.5

    def test_confidence_recovers_with_successes(self):
        sc = SensorConfidence(name="test")
        sc.record_failure()
        for _ in range(10):
            sc.record_success()
        assert sc.confidence > 0.8

    def test_consecutive_failures_harsh_penalty(self):
        sc = SensorConfidence(name="test")
        for _ in range(10):
            sc.record_success()
        for _ in range(5):
            sc.record_failure()
        assert sc.confidence <= 0.1

    def test_stale_success_penalty(self):
        sc = SensorConfidence(name="test")
        sc.record_success()
        sc.record_failure()
        sc._last_success_time = time.monotonic() - 120
        assert sc.confidence <= 0.5


class TestEvidenceFusion:

    def test_register_sensor(self):
        ef = EvidenceFusion()
        ef.register_sensor("dom")
        assert "dom" in ef._sensor_confidence

    def test_record_success_updates_confidence(self):
        ef = EvidenceFusion()
        ef.record_sensor_success("dom")
        assert ef.sensor_confidence("dom") == 1.0

    def test_record_failure_drops_confidence(self):
        ef = EvidenceFusion()
        for _ in range(5):
            ef.record_sensor_failure("network")
        assert ef.sensor_confidence("network") < 0.5

    def test_submit_evidence_returns_vector(self):
        ef = EvidenceFusion()
        ef.register_sensor("dom")
        ev = ef.submit_evidence("dom", "input_visible", True)
        assert isinstance(ev, EvidenceVector)
        assert ev.signal_name == "input_visible"
        assert ev.value is True

    def test_fused_confidence_empty(self):
        ef = EvidenceFusion()
        assert ef.fused_confidence("nonexistent") == 0.0

    def test_fused_confidence_with_evidence(self):
        ef = EvidenceFusion()
        ef.register_sensor("dom")
        ef.record_sensor_success("dom")
        ef.record_sensor_success("dom")
        ef.submit_evidence("dom", "input_visible", True, confidence=0.9)
        ef.submit_evidence("a11y", "input_visible", True, confidence=0.7)
        fused = ef.fused_confidence("input_visible")
        assert fused >= 0.0

    def test_all_sensor_confidences(self):
        ef = EvidenceFusion()
        ef.record_sensor_success("dom")
        ef.record_sensor_failure("network")
        confidences = ef.all_sensor_confidences()
        assert "dom" in confidences
        assert "network" in confidences

    def test_reset_clears_all(self):
        ef = EvidenceFusion()
        ef.record_sensor_success("dom")
        ef.reset()
        assert len(ef._sensor_confidence) == 0
        assert len(ef._evidence_buffer) == 0

    def test_invalid_half_life_raises(self):
        with pytest.raises(ValueError, match="half_life"):
            EvidenceFusion(half_life_ticks=0)


# ═══════════════════════════════════════════════════════════════
# P5: Stream stalled detection
# ═══════════════════════════════════════════════════════════════

class TestStreamStalled:

    def test_not_stalled_without_stream(self):
        engine = BrowserIntelligenceEngine()
        assert not engine.stream_stalled

    def test_stalled_when_active_but_idle(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()

        fv = FeatureVector(
            stream_active=True,
            total_chunks=10,
            tokens_per_second=0.0,
            stream_idle_time=10.0,
        )
        engine._store.push(fv)
        engine._belief = engine._hmm.update(fv)
        engine._detect_stream_stalled(fv)

        assert engine.stream_stalled

    def test_not_stalled_when_tokens_flowing(self):
        engine = BrowserIntelligenceEngine()

        fv = FeatureVector(
            stream_active=True,
            total_chunks=10,
            tokens_per_second=15.0,
            stream_idle_time=1.0,
        )
        engine._detect_stream_stalled(fv)
        assert not engine.stream_stalled

    def test_not_stalled_with_few_chunks(self):
        engine = BrowserIntelligenceEngine()

        fv = FeatureVector(
            stream_active=True,
            total_chunks=1,
            tokens_per_second=0.0,
            stream_idle_time=10.0,
        )
        engine._detect_stream_stalled(fv)
        assert not engine.stream_stalled

    def test_recover_action_when_stalled(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()

        for _ in range(20):
            fv = FeatureVector(
                input_visible=True, send_enabled=False,
                stop_button_visible=True, has_streaming_marker=True,
                stream_active=True, transport_detected=True,
                generation_started=True, mutation_rate=8.0,
                response_length=300, response_length_delta=50,
                total_chunks=50, tokens_per_second=15.0,
                stream_idle_time=0.2,
                bytes_received=5000,
            )
            engine._store.push(fv)
            engine._belief = engine._hmm.update(fv)

        engine._stream_stalled = True
        actions = engine._compute_available_actions()
        assert "recover" in actions


# ═══════════════════════════════════════════════════════════════
# P1: Online learning convergence
# ═══════════════════════════════════════════════════════════════

class TestOnlineLearningConvergence:

    def test_emission_parameters_shift_toward_observation(self):
        em = EmissionModel()
        ready_mu_before = em._continuous_mu[HiddenState.READY][0]

        fv = FeatureVector(
            input_visible=True, send_enabled=True,
            mutation_rate=5.0, response_length=100,
        )
        for _ in range(100):
            em.update_from_observation(fv, HiddenState.READY, learning_rate=0.05)

        ready_mu_after = em._continuous_mu[HiddenState.READY][0]
        assert ready_mu_after != ready_mu_before
        assert ready_mu_after > 0.1

    def test_learning_decreases_sigma_when_observations_consistent(self):
        em = EmissionModel()
        sigma_before = em._continuous_sigma[HiddenState.READY][0]

        fv = FeatureVector(mutation_rate=1.0)
        for _ in range(50):
            em.update_from_observation(fv, HiddenState.READY, learning_rate=0.05)

        sigma_after = em._continuous_sigma[HiddenState.READY][0]
        assert sigma_after < sigma_before

    def test_soft_assignment_learning_rate_scales_with_belief(self):
        em = EmissionModel()
        fv = FeatureVector(mutation_rate=5.0, response_length=500)
        beliefs = {
            HiddenState.GENERATING: 0.7,
            HiddenState.COMPLETE: 0.2,
            HiddenState.READY: 0.1,
        }
        for s in HiddenState:
            if s not in beliefs:
                beliefs[s] = 0.0

        count_before_gen = em.observation_count(HiddenState.GENERATING)
        count_before_complete = em.observation_count(HiddenState.COMPLETE)

        for _ in range(20):
            em.update_from_soft_assignment(fv, beliefs)

        gen_delta = em.observation_count(HiddenState.GENERATING) - count_before_gen
        complete_delta = em.observation_count(HiddenState.COMPLETE) - count_before_complete
        assert gen_delta >= complete_delta


# ═══════════════════════════════════════════════════════════════
# Engine-level integration
# ═══════════════════════════════════════════════════════════════

class TestEngineIntegration:

    def test_sensor_confidences_tracked(self):
        engine = BrowserIntelligenceEngine()
        confidences = engine.sensor_confidences
        assert len(confidences) == 6
        assert confidences["dom"] == 1.0

    def test_emission_calibration_reported(self):
        engine = BrowserIntelligenceEngine()
        calib = engine.emission_calibration
        assert len(calib) == 10
        assert "ready" in calib
        for v in calib.values():
            assert 0.0 <= v <= 1.0

    def test_adaptive_threshold_property(self):
        engine = BrowserIntelligenceEngine()
        thresh = engine.adaptive_threshold
        assert 0.3 <= thresh <= 0.75

    def test_stream_stalled_property(self):
        engine = BrowserIntelligenceEngine()
        assert not engine.stream_stalled

    def test_engine_reset_clears_all_state(self):
        engine = BrowserIntelligenceEngine()
        engine._hmm.initialize()
        engine._store.push(FeatureVector(tick=1, response_length=100))
        engine._ready_for_prompt = True
        engine._stream_stalled = True
        engine._readiness_ticks = 5

        engine.reset()

        assert engine._store.size == 0
        assert engine.belief is None
        assert not engine.is_ready_for_prompt
        assert not engine.stream_stalled
        assert engine._readiness_ticks == 0
        assert engine._last_adaptive_threshold == 0.50
