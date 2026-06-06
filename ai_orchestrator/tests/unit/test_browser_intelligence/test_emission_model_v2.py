"""Tests for EmissionModel v2 — 30-dim validation."""

from __future__ import annotations

import pytest

from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector
from ai_orchestrator.browser_intelligence.estimation.emission_model import EmissionModel
from ai_orchestrator.browser_intelligence.estimation.belief_state import HiddenState


class TestEmissionModelV2:
    def test_feature_dim_is_30(self):
        em = EmissionModel()
        assert EmissionModel.FEATURE_DIM == 30

    def test_emission_prob_returns_valid_float_for_all_10_states(self):
        em = EmissionModel()
        fv = FeatureVector(
            input_visible=True,
            send_enabled=True,
            stream_active=True,
            transport_detected=True,
            generation_started=True,
            tokens_per_second=10.0,
            bytes_received=5000,
        )
        for state in HiddenState:
            prob = em.emission_prob(fv, state)
            assert isinstance(prob, float)
            assert prob > 0, f"Zero probability for state {state}"
            assert prob <= 1.0, f"Probability > 1 for state {state}"

    def test_emission_prob_handles_edge_case_empty_feature_vector(self):
        em = EmissionModel()
        fv = FeatureVector()
        for state in HiddenState:
            prob = em.emission_prob(fv, state)
            assert prob > 0

    def test_emission_prob_handles_extreme_values(self):
        em = EmissionModel()
        fv = FeatureVector(
            tokens_per_second=1000.0,
            bytes_received=1_000_000,
            stream_idle_time=100.0,
            total_chunks=10_000,
            network_request_rate=100.0,
        )
        for state in HiddenState:
            prob = em.emission_prob(fv, state)
            assert prob > 0

    def test_generating_state_has_high_stream_active_binary_param(self):
        em = EmissionModel()
        stream_active_idx = 12
        assert em._binary_params[HiddenState.GENERATING][stream_active_idx] >= 0.6

    def test_generating_state_has_high_transport_detected_binary_param(self):
        em = EmissionModel()
        transport_detected_idx = 13
        assert em._binary_params[HiddenState.GENERATING][transport_detected_idx] >= 0.6

    def test_generating_state_has_high_generation_started_binary_param(self):
        em = EmissionModel()
        gen_started_idx = 14
        assert em._binary_params[HiddenState.GENERATING][gen_started_idx] >= 0.6

    def test_complete_state_has_high_generation_completed_binary_param(self):
        em = EmissionModel()
        gen_completed_idx = 15
        assert em._binary_params[HiddenState.COMPLETE][gen_completed_idx] >= 0.6

    def test_complete_state_has_high_stream_closed_binary_param(self):
        em = EmissionModel()
        stream_closed_idx = 16
        assert em._binary_params[HiddenState.COMPLETE][stream_closed_idx] >= 0.6

    def ready_state_binary_params(self):
        em = EmissionModel()
        stream_active_idx = 12
        assert em._binary_params[HiddenState.READY][stream_active_idx] <= 0.4

    def test_all_states_have_18_binary_params(self):
        em = EmissionModel()
        for state in HiddenState:
            assert len(em._binary_params[state]) == 18, f"State {state} has wrong binary param count"

    def test_all_states_have_12_continuous_params(self):
        em = EmissionModel()
        for state in HiddenState:
            assert len(em._continuous_mu[state]) == 12, f"State {state} has wrong continuous mu count"
            assert len(em._continuous_sigma[state]) == 12, f"State {state} has wrong continuous sigma count"

    def test_update_from_observation_updates_binary_params(self):
        em = EmissionModel()
        fv = FeatureVector(stream_active=True, generation_started=True)
        old_p = em._binary_params[HiddenState.GENERATING][12]
        em.update_from_observation(fv, HiddenState.GENERATING, learning_rate=0.1)
        new_p = em._binary_params[HiddenState.GENERATING][12]
        assert new_p > old_p

    def test_update_from_observation_updates_continuous_params(self):
        em = EmissionModel()
        fv = FeatureVector(tokens_per_second=50.0, bytes_received=10000)
        old_mu = em._continuous_mu[HiddenState.GENERATING][7]  # tokens_per_second index (25-18=7)
        em.update_from_observation(fv, HiddenState.GENERATING, learning_rate=0.1)
        new_mu = em._continuous_mu[HiddenState.GENERATING][7]
        assert new_mu != old_mu

    def test_binary_params_clamped_between_0_01_and_0_99(self):
        em = EmissionModel()
        for state in HiddenState:
            for p in em._binary_params[state]:
                assert 0.01 <= p <= 0.99, f"State {state} param {p} out of bounds"

    def test_continuous_sigma_minimum_0_1(self):
        em = EmissionModel()
        for state in HiddenState:
            for sigma in em._continuous_sigma[state]:
                assert sigma >= 0.1, f"State {state} sigma {sigma} below 0.1"

    def test_emission_prob_stays_valid_after_updates(self):
        em = EmissionModel()
        fv = FeatureVector(
            stream_active=True,
            tokens_per_second=25.0,
            bytes_received=10000,
            response_length=500,
        )
        for _ in range(50):
            em.update_from_observation(fv, HiddenState.GENERATING, learning_rate=0.05)
        prob = em.emission_prob(fv, HiddenState.GENERATING)
        assert prob > 0
