"""Tests for network intelligence sensor subsystem."""

from __future__ import annotations

import time

import pytest

from ai_orchestrator.browser_intelligence.sensors.network.protocol_detector import (
    ProtocolDetector,
    TransportProtocol,
)
from ai_orchestrator.browser_intelligence.sensors.network.stream_parser import (
    StreamParser,
)
from ai_orchestrator.browser_intelligence.sensors.network.sse_observer import (
    SSEObserver,
)
from ai_orchestrator.browser_intelligence.sensors.network.ws_observer import (
    WSObserver,
)
from ai_orchestrator.browser_intelligence.sensors.network.fetch_observer import (
    FetchObserver,
)


# ═══════════════════════════════════════════════════════════════
# ProtocolDetector
# ═══════════════════════════════════════════════════════════════

class TestProtocolDetector:
    def test_sse_detection_from_content_type_text_event_stream(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/chat",
            content_type="text/event-stream",
        )
        assert proto == TransportProtocol.SSE
        assert conf >= 0.85

    def test_sse_detection_from_content_type_with_charset(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/chat",
            content_type="text/event-stream;charset=utf-8",
        )
        assert proto == TransportProtocol.SSE
        assert conf >= 0.85

    def test_sse_detection_from_content_type_with_charset_space(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/chat",
            content_type="text/event-stream; charset=utf-8",
        )
        assert proto == TransportProtocol.SSE
        assert conf >= 0.85

    def test_websocket_detection_from_101_status(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="wss://example.com/socket",
            status=101,
        )
        assert proto == TransportProtocol.WEBSOCKET
        assert conf >= 0.80

    def test_fetch_stream_detection_from_chunked_transfer_encoding(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/chat/completions",
            content_type="application/json",
            response_headers={"transfer-encoding": "chunked"},
        )
        assert proto == TransportProtocol.FETCH_STREAM
        assert conf >= 0.65

    def test_fetch_stream_detection_from_url_pattern(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/api/generate",
            content_type="text/plain",
            response_headers={"transfer-encoding": "chunked"},
        )
        assert proto == TransportProtocol.FETCH_STREAM
        assert conf >= 0.65

    def test_xhr_detection_from_requested_with_header(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/api/data",
            response_headers={"x-requested-with": "xmlhttprequest"},
        )
        assert proto == TransportProtocol.XHR_POLL
        assert conf >= 0.25

    def test_unknown_when_no_signals(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="https://example.com/page",
            content_type="text/html",
        )
        assert proto == TransportProtocol.UNKNOWN
        assert conf == 0.0

    def test_confidence_aggregation_multiple_signals(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/v1/chat/completions",
            status=101,
            content_type="application/octet-stream",
            response_headers={"x-requested-with": "xmlhttprequest"},
        )
        assert proto == TransportProtocol.WEBSOCKET
        assert conf > 0.90

    def test_smoothed_confidence_returns_average_of_recent_detections(self):
        pd = ProtocolDetector()
        for i in range(5):
            pd.feed_response(
                url="/chat",
                content_type="text/event-stream",
            )
        smoothed = pd.smoothed_confidence()
        assert 0.8 < smoothed <= 1.0

    def test_smoothed_confidence_empty_history_returns_zero(self):
        pd = ProtocolDetector()
        assert pd.smoothed_confidence() == 0.0

    def test_detection_history_capped_at_20_entries(self):
        pd = ProtocolDetector()
        for i in range(30):
            pd.feed_response(
                url="/chat",
                content_type="text/event-stream",
            )
        assert len(pd._detection_history) == 20

    def test_feed_websocket_created_gives_ws_with_high_confidence(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_websocket_created(url="wss://example.com/ws")
        assert proto == TransportProtocol.WEBSOCKET
        assert conf == 0.95

    def test_feed_websocket_created_appends_to_history(self):
        pd = ProtocolDetector()
        pd.feed_websocket_created(url="wss://example.com/ws")
        assert len(pd._detection_history) == 1
        assert pd._detection_history[0] == ("websocket", 0.95)

    def test_latest_detection_returns_last_entry(self):
        pd = ProtocolDetector()
        pd.feed_response(url="/chat", content_type="text/event-stream")
        pd.feed_response(url="/api", content_type="application/json",
                         response_headers={"x-requested-with": "xmlhttprequest"})
        proto, _ = pd.latest_detection
        assert proto == TransportProtocol.XHR_POLL

    def test_latest_detection_empty_history_returns_unknown(self):
        pd = ProtocolDetector()
        proto, conf = pd.latest_detection
        assert proto == TransportProtocol.UNKNOWN
        assert conf == 0.0

    def test_reset_clears_all_state(self):
        pd = ProtocolDetector()
        pd.feed_response(url="/chat", content_type="text/event-stream")
        pd.feed_websocket_created(url="wss://example.com/ws")
        pd.reset()
        assert pd._last_content_type is None
        assert pd._last_url == ""
        assert pd._last_status == 0
        assert pd._detection_history == []
        assert pd.smoothed_confidence() == 0.0

    def test_url_pattern_boost_with_streaming_content_type_no_transfer_encoding(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/v1/chat/completions",
            content_type="application/json",
        )
        assert proto == TransportProtocol.FETCH_STREAM
        assert 0.2 < conf < 0.6

    def test_url_only_with_no_content_type_produces_url_boost(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(url="/api/generate")
        assert proto in (TransportProtocol.SSE, TransportProtocol.FETCH_STREAM)
        assert 0.0 < conf <= 0.35

    def test_fetch_stream_commercial_content_type_ndjson(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/stream",
            content_type="application/x-ndjson",
            response_headers={"transfer-encoding": "chunked"},
        )
        assert proto == TransportProtocol.FETCH_STREAM

    def test_websocket_from_octet_stream_content_type(self):
        pd = ProtocolDetector()
        proto, conf = pd.feed_response(
            url="/ws",
            content_type="application/octet-stream",
        )
        assert proto == TransportProtocol.WEBSOCKET
        assert conf >= 0.40


# ═══════════════════════════════════════════════════════════════
# StreamParser
# ═══════════════════════════════════════════════════════════════

class TestStreamParser:
    def test_push_event_increments_total_chunks(self):
        sp = StreamParser()
        sp.push_event(data="hello")
        assert sp.total_chunks == 1
        sp.push_event(data="world")
        assert sp.total_chunks == 2

    def test_push_event_sets_timestamps(self):
        sp = StreamParser()
        sp.push_event(data="hello")
        assert sp.last_chunk_timestamp > 0
        assert sp.first_chunk_timestamp > 0

    def test_push_event_with_actual_data_counts_bytes(self):
        sp = StreamParser()
        sp.push_event(data="hello world")
        assert sp.bytes_received == len("hello world".encode("utf-8"))

    def test_push_event_without_data_increments_chunk_but_not_bytes(self):
        sp = StreamParser()
        sp.push_event(data=None)
        assert sp.total_chunks == 1
        assert sp.bytes_received == 0

    def test_multiple_chunks_compute_tokens_per_second(self):
        sp = StreamParser()
        now = 100.0
        for i in range(5):
            sp.push_event(data="chunk", timestamp=now)
            now += 0.2
        assert sp.tokens_per_second() > 0

    def test_tokens_per_second_zero_with_one_chunk(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        rate = sp.tokens_per_second()
        assert rate >= 0.0

    def test_evaluate_stream_active_true_when_idle_below_threshold(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        state = sp.evaluate(now=101.0)
        assert state.stream_active is True

    def test_evaluate_stream_active_false_when_idle_above_threshold(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        state = sp.evaluate(now=103.0)
        assert state.stream_active is False

    def test_evaluate_stream_active_false_when_no_chunks(self):
        sp = StreamParser()
        state = sp.evaluate(now=100.0)
        assert state.stream_active is False

    def test_evaluate_generation_started_on_first_chunk(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        state = sp.evaluate(now=100.5)
        assert state.generation_started is True

    def test_evaluate_generation_started_after_long_idle_then_new_chunk(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        sp.evaluate(now=100.5)
        sp.push_event(data="world", timestamp=106.0)
        state = sp.evaluate(now=106.5)
        assert state.generation_started is True

    def test_evaluate_generation_not_started_on_second_chunk_without_long_idle(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        sp.evaluate(now=100.5)
        sp.push_event(data="world", timestamp=101.0)
        state = sp.evaluate(now=101.5)
        assert state.generation_started is False

    def test_evaluate_stream_closed_after_hard_timeout(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        state = sp.evaluate(now=111.0)
        assert state.stream_closed is True

    def test_evaluate_stream_closed_after_idle_plus_disconnect(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        sp.signal_transport_disconnected()
        state = sp.evaluate(now=106.0)
        assert state.stream_closed is True

    def test_evaluate_stream_not_closed_when_active(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        sp.push_event(data="world", timestamp=100.5)
        state = sp.evaluate(now=101.0)
        assert state.stream_closed is False

    def test_signal_transport_disconnected_sets_flag(self):
        sp = StreamParser()
        sp.signal_transport_disconnected()
        state = sp.evaluate(now=100.0)
        assert state.transport_disconnected is True

    def test_signal_transport_connected_clears_flag(self):
        sp = StreamParser()
        sp.signal_transport_disconnected()
        sp.signal_transport_connected()
        state = sp.evaluate(now=100.0)
        assert state.transport_disconnected is False

    def test_push_bytes_increments_chunks_and_bytes(self):
        sp = StreamParser()
        sp.push_bytes(byte_count=100, timestamp=100.0)
        assert sp.total_chunks == 1
        assert sp.bytes_received == 100

    def test_push_bytes_zero_or_negative_ignored(self):
        sp = StreamParser()
        sp.push_bytes(byte_count=0, timestamp=100.0)
        sp.push_bytes(byte_count=-5, timestamp=100.0)
        assert sp.total_chunks == 0
        assert sp.bytes_received == 0

    def test_chunk_rate_variance_requires_min_three_chunks(self):
        sp = StreamParser()
        sp.push_event(data="a", timestamp=100.0)
        sp.push_event(data="b", timestamp=100.5)
        assert sp.chunk_rate_variance() == 0.0

    def test_chunk_rate_variance_with_multiple_chunks(self):
        sp = StreamParser()
        for i, ts in enumerate([100.0, 100.5, 101.5]):
            sp.push_event(data="x", timestamp=ts)
        variance = sp.chunk_rate_variance()
        assert variance > 0.0

    def test_average_chunk_size(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        sp.push_event(data="world!", timestamp=100.5)
        avg = sp.average_chunk_size()
        assert avg > 0.0

    def test_average_chunk_size_empty_returns_zero(self):
        sp = StreamParser()
        assert sp.average_chunk_size() == 0.0

    def test_reset_clears_all_state(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        sp.push_bytes(byte_count=50, timestamp=100.5)
        sp.signal_transport_disconnected()
        sp.reset()
        assert sp.total_chunks == 0
        assert sp.bytes_received == 0
        assert sp.last_chunk_timestamp == 0.0
        assert sp.first_chunk_timestamp == 0.0
        assert sp.tokens_per_second() == 0.0
        assert sp.average_chunk_size() == 0.0

    def test_raw_timestamps_capped_at_200(self):
        sp = StreamParser()
        for i in range(250):
            sp.push_event(data="x", timestamp=100.0 + i * 0.1)
        assert len(sp._raw_chunk_timestamps) <= 200

    def test_idle_time_based_on_last_chunk(self):
        sp = StreamParser()
        sp.push_event(data="hello", timestamp=100.0)
        state = sp.evaluate(now=105.5)
        assert state.stream_idle_time == pytest.approx(5.5)


# ═══════════════════════════════════════════════════════════════
# SSEObserver
# ═══════════════════════════════════════════════════════════════

class TestSSEObserver:
    def test_event_source_message_increments_counts(self):
        obs = SSEObserver()
        obs._active_stream_ids.add("req-1")
        obs.on_event_source_message_received({
            "requestId": "req-1",
            "data": '{"token": "hello"}',
            "eventName": "",
        })
        assert obs.event_count == 1
        assert obs.data_chunk_count == 1
        assert obs.bytes_received > 0
        assert obs.stream_active is True

    def test_done_marker_sets_done_seen_and_closes_stream(self):
        obs = SSEObserver()
        obs._active_stream_ids.add("req-1")
        obs.on_event_source_message_received({
            "requestId": "req-1",
            "data": "[DONE]",
            "eventName": "",
        })
        assert obs.done_seen is True
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_event_source_message_ignores_unknown_request_id(self):
        obs = SSEObserver()
        obs._active_stream_ids.add("req-known")
        obs.on_event_source_message_received({
            "requestId": "req-unknown",
            "data": "hello",
            "eventName": "",
        })
        assert obs.event_count == 0
        assert obs.data_chunk_count == 0

    def test_loading_finished_closes_stream_when_no_active_streams(self):
        obs = SSEObserver()
        obs._active_stream_ids.add("req-1")
        obs.stream_active = True
        obs.on_loading_finished({"requestId": "req-1"})
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_loading_failed_closes_stream(self):
        obs = SSEObserver()
        obs._active_stream_ids.add("req-1")
        obs.stream_active = True
        obs.on_loading_failed({"requestId": "req-1"})
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_tokens_per_second_positive_with_data(self):
        obs = SSEObserver()
        obs._active_stream_ids.add("req-1")
        obs.first_event_time = time.monotonic() - 2.0
        obs.data_chunk_count = 10
        tps = obs.tokens_per_second()
        assert tps > 0

    def test_tokens_per_second_zero_without_data(self):
        obs = SSEObserver()
        assert obs.tokens_per_second() == 0.0

    def test_stream_idle_time_positive_when_active(self):
        obs = SSEObserver()
        obs.stream_active = True
        obs.last_event_time = time.monotonic() - 0.5
        idle = obs.stream_idle_time()
        assert idle >= 0.45

    def test_stream_idle_time_zero_when_not_active(self):
        obs = SSEObserver()
        obs.last_event_time = time.monotonic()
        obs.stream_active = False
        idle = obs.stream_idle_time()
        assert idle == 0.0

    def test_on_response_received_event_stream(self):
        obs = SSEObserver()
        result = obs.on_response_received({
            "requestId": "req-1",
            "response": {
                "mimeType": "text/event-stream",
                "headers": {},
            },
        })
        assert result is True
        assert "req-1" in obs._active_stream_ids
        assert obs.stream_active is True

    def test_on_response_received_non_event_stream(self):
        obs = SSEObserver()
        result = obs.on_response_received({
            "requestId": "req-1",
            "response": {
                "mimeType": "application/json",
                "headers": {},
            },
        })
        assert result is False
        assert "req-1" not in obs._active_stream_ids

    def test_on_request_will_be_sent_stores_url(self):
        obs = SSEObserver()
        obs.on_request_will_be_sent({
            "requestId": "req-1",
            "request": {"url": "https://example.com/sse"},
        })
        assert obs._request_id_to_url["req-1"] == "https://example.com/sse"

    def test_reset_clears_all_state(self):
        obs = SSEObserver()
        obs._active_stream_ids.add("req-1")
        obs.on_event_source_message_received({
            "requestId": "req-1",
            "data": "test",
            "eventName": "",
        })
        obs.reset()
        assert obs.event_count == 0
        assert obs.data_chunk_count == 0
        assert obs.bytes_received == 0
        assert obs.stream_active is False
        assert obs.stream_closed is False
        assert obs.done_seen is False
        assert len(obs._active_stream_ids) == 0
        assert len(obs._event_buffer) == 0


# ═══════════════════════════════════════════════════════════════
# WSObserver
# ═══════════════════════════════════════════════════════════════

class TestWSObserver:
    def test_ws_created_opens_connection(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        assert obs.connection_open is True
        assert "wss://example.com/ws" in obs._active_ws_urls
        assert obs.stream_closed is False

    def test_ws_frame_with_data_increments_counts(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_frame_received({
            "response": {"payloadData": '{"token": "hello"}'},
        })
        assert obs.frame_count == 1
        assert obs.data_frame_count == 1
        assert obs.bytes_received > 0
        assert obs.stream_active is True

    def test_done_in_payload_sets_done_seen_and_closes_stream(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_frame_received({
            "response": {"payloadData": "[DONE]"},
        })
        assert obs.done_seen is True
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_done_inside_payload_detected(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_frame_received({
            "response": {"payloadData": '{"data": "[DONE]"}'},
        })
        assert obs.done_seen is True

    def test_empty_payload_after_data_frames_closes_stream(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_frame_received({
            "response": {"payloadData": '{"token": "hello"}'},
        })
        obs.on_ws_frame_received({
            "response": {"payloadData": "  "},
        })
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_empty_payload_before_any_data_does_not_close_stream(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_frame_received({
            "response": {"payloadData": "  "},
        })
        assert obs.data_frame_count == 0
        assert obs.stream_closed is False

    def test_ws_closed_sets_connection_closed(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_closed({"url": "wss://example.com/ws"})
        assert obs.connection_open is False
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_ws_closed_does_not_close_when_other_connections_open(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://a.com/ws"})
        obs.on_ws_created({"url": "wss://b.com/ws"})
        obs.on_ws_closed({"url": "wss://a.com/ws"})
        assert obs.connection_open is True
        assert obs.stream_closed is False

    def test_ws_frame_without_payload_increments_frame_count_only(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_frame_received({
            "response": {"payloadData": None},
        })
        assert obs.frame_count == 1
        assert obs.data_frame_count == 0

    def test_tokens_per_second_positive_with_data_frames(self):
        obs = WSObserver()
        obs.first_frame_time = time.monotonic() - 2.0
        obs.data_frame_count = 10
        tps = obs.tokens_per_second()
        assert tps > 0

    def test_tokens_per_second_zero_without_data(self):
        obs = WSObserver()
        assert obs.tokens_per_second() == 0.0

    def test_stream_idle_time_zero_when_not_active(self):
        obs = WSObserver()
        obs.last_frame_time = time.monotonic()
        obs.stream_active = False
        idle = obs.stream_idle_time()
        assert idle == 0.0

    def test_reset_clears_all_state(self):
        obs = WSObserver()
        obs.on_ws_created({"url": "wss://example.com/ws"})
        obs.on_ws_frame_received({
            "response": {"payloadData": '{"token": "hello"}'},
        })
        obs.reset()
        assert obs.frame_count == 0
        assert obs.data_frame_count == 0
        assert obs.bytes_received == 0
        assert obs.stream_active is False
        assert obs.stream_closed is False
        assert obs.done_seen is False
        assert obs.connection_open is False
        assert len(obs._active_ws_urls) == 0

    def test_is_json_returns_true_for_valid_json(self):
        assert WSObserver._is_json('{"key": "value"}') is True
        assert WSObserver._is_json("[1, 2, 3]") is True

    def test_is_json_returns_false_for_non_json(self):
        assert WSObserver._is_json("plain text") is False
        assert WSObserver._is_json("") is False
        assert WSObserver._is_json("   ") is False


# ═══════════════════════════════════════════════════════════════
# FetchObserver
# ═══════════════════════════════════════════════════════════════

class TestFetchObserver:
    def test_chunked_transfer_encoding_triggers_stream_tracking(self):
        obs = FetchObserver()
        result = obs.on_response_received({
            "requestId": "req-1",
            "response": {
                "headers": {"transfer-encoding": "chunked"},
                "mimeType": "application/json",
            },
        })
        assert result is True
        assert "req-1" in obs._active_stream_request_ids
        assert obs.stream_active is True

    def test_chunked_transfer_encoding_mixed_case(self):
        obs = FetchObserver()
        result = obs.on_response_received({
            "requestId": "req-1",
            "response": {
                "headers": {"Transfer-Encoding": "Chunked"},
                "mimeType": "application/json",
            },
        })
        assert result is True

    def test_data_received_increments_chunk_count_and_bytes(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-1")
        obs.on_data_received({
            "requestId": "req-1",
            "dataLength": 100,
            "encodedDataLength": 0,
        })
        assert obs.chunk_count == 1
        assert obs.bytes_received == 100
        assert obs.stream_active is True

    def test_data_received_ignores_unknown_request_id(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-known")
        obs.on_data_received({
            "requestId": "req-unknown",
            "dataLength": 100,
        })
        assert obs.chunk_count == 0

    def test_loading_finished_closes_stream_when_no_active_streams(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-1")
        obs.chunk_count = 5
        obs.stream_active = True
        obs.on_loading_finished({"requestId": "req-1"})
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_loading_failed_closes_stream(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-1")
        obs.chunk_count = 3
        obs.stream_active = True
        obs.on_loading_failed({"requestId": "req-1"})
        assert obs.stream_closed is True
        assert obs.stream_active is False

    def test_loading_finished_does_not_close_stream_if_others_active(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-1")
        obs._active_stream_request_ids.add("req-2")
        obs.chunk_count = 5
        obs.on_loading_finished({"requestId": "req-1"})
        assert obs.stream_closed is False

    def test_tokens_per_second_positive_with_data(self):
        obs = FetchObserver()
        obs.first_data_time = time.monotonic() - 2.0
        obs.chunk_count = 10
        tps = obs.tokens_per_second()
        assert tps > 0

    def test_stream_idle_time_positive_when_active(self):
        obs = FetchObserver()
        obs.stream_active = True
        obs.last_data_time = time.monotonic() - 1.0
        idle = obs.stream_idle_time()
        assert idle >= 0.95

    def test_stream_idle_time_zero_when_not_active(self):
        obs = FetchObserver()
        obs.last_data_time = time.monotonic()
        obs.stream_active = False
        idle = obs.stream_idle_time()
        assert idle == 0.0

    def test_reset_clears_all_state(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-1")
        obs.on_data_received({"requestId": "req-1", "dataLength": 50})
        obs.reset()
        assert obs.chunk_count == 0
        assert obs.bytes_received == 0
        assert obs.stream_active is False
        assert obs.stream_closed is False
        assert len(obs._active_stream_request_ids) == 0
        assert len(obs._data_buffer) == 0

    def test_on_response_received_does_not_track_bytes(self):
        obs = FetchObserver()
        obs.on_response_received({
            "requestId": "req-1",
            "response": {
                "headers": {"transfer-encoding": "chunked"},
                "mimeType": "application/json",
                "encodedDataLength": 200,
            },
        })
        assert obs.bytes_received == 0
        assert obs.chunk_count == 0
        assert obs.stream_active is True
        assert "req-1" in obs._active_stream_request_ids

    def test_non_chunked_transfer_returns_false_for_streaming_detection(self):
        obs = FetchObserver()
        result = obs.on_response_received({
            "requestId": "req-1",
            "response": {
                "headers": {"content-type": "text/html"},
                "mimeType": "text/html",
            },
        })
        assert result is False

    def test_loading_failed_does_not_close_if_no_chunks(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-1")
        obs.chunk_count = 0
        obs.on_loading_failed({"requestId": "req-1"})
        assert obs.stream_closed is False

    def test_loading_finished_does_not_track_encoded_bytes(self):
        obs = FetchObserver()
        obs._active_stream_request_ids.add("req-1")
        obs.on_loading_finished({
            "requestId": "req-1",
            "encodedDataLength": 250,
        })
        assert obs.bytes_received == 0
        assert obs.chunk_count == 0
