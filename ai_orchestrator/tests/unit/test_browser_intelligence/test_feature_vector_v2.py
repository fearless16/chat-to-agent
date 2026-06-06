"""Tests for FeatureVector v2 — 30-dim layout validation."""

from __future__ import annotations

import pytest

from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureVector


class TestFeatureVectorV2:
    def test_to_list_returns_exactly_30_elements(self):
        fv = FeatureVector()
        lst = fv.to_list()
        assert len(lst) == 30

    def test_binary_features_occupy_indices_0_to_17(self):
        fv = FeatureVector(
            input_visible=True,
            send_enabled=True,
            stop_button_visible=True,
            has_streaming_marker=True,
        )
        lst = fv.to_list()
        for i in range(18):
            assert isinstance(lst[i], float), f"Index {i} should be float"
        assert lst[0] == 1.0
        assert lst[1] == 1.0
        assert lst[2] == 1.0

    def test_continuous_features_occupy_indices_18_to_29(self):
        fv = FeatureVector(
            response_length=500,
            bytes_received=10000,
            tokens_per_second=25.0,
        )
        lst = fv.to_list()
        for i in range(18, 30):
            assert isinstance(lst[i], float), f"Index {i} should be float"
        assert len(lst) == 30

    def test_new_stream_fields_present_in_vector(self):
        fv = FeatureVector(
            stream_active=True,
            transport_detected=True,
            generation_started=True,
            generation_completed=True,
            stream_closed=False,
            tokens_per_second=15.5,
            stream_idle_time=0.2,
            total_chunks=50,
            bytes_received=5000,
            network_request_rate=2.0,
        )
        lst = fv.to_list()
        assert lst[12] == 1.0  # stream_active
        assert lst[13] == 1.0  # transport_detected
        assert lst[14] == 1.0  # generation_started
        assert lst[15] == 1.0  # generation_completed
        assert lst[16] == 0.0  # stream_closed
        assert lst[17] == 0.0  # generation_stop_detected
        assert lst[25] == 15.5  # tokens_per_second
        assert lst[26] == 0.2   # stream_idle_time
        assert lst[27] == 50.0  # total_chunks
        assert lst[28] == 5000.0  # bytes_received
        assert lst[29] == 2.0   # network_request_rate

    def test_new_fields_default_to_false_or_zero(self):
        fv = FeatureVector()
        assert fv.stream_active is False
        assert fv.transport_detected is False
        assert fv.generation_started is False
        assert fv.generation_completed is False
        assert fv.stream_closed is False
        assert fv.tokens_per_second == 0.0
        assert fv.stream_idle_time == 0.0
        assert fv.total_chunks == 0
        assert fv.bytes_received == 0
        assert fv.network_request_rate == 0.0

    def test_generation_stop_detected_is_still_present(self):
        fv = FeatureVector(generation_stop_detected=True)
        assert fv.generation_stop_detected is True
        lst = fv.to_list()
        assert lst[17] == 1.0

    def test_legacy_pass_through_fields_are_not_on_feature_vector(self):
        fv = FeatureVector()
        assert not hasattr(fv, "websocket_activity")
        assert not hasattr(fv, "sse_active")
        assert not hasattr(fv, "streaming_indicators")
        assert not hasattr(fv, "streaming_detected")
        assert not hasattr(fv, "generation_event_detected")

    def test_legacy_fields_not_in_to_list(self):
        fv = FeatureVector(stream_active=True, transport_detected=True)
        lst = fv.to_list()
        assert len(lst) == 30

    def test_all_30_indices_are_float(self):
        fv = FeatureVector()
        lst = fv.to_list()
        assert all(isinstance(v, float) for v in lst)
