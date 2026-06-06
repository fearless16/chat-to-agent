"""Tests for transport-native response capture."""

from __future__ import annotations

import pytest

from ai_orchestrator.browser_intelligence.intelligence.response_capture import (
    CapturedResponse,
    ResponseCapture,
    parse_stream_chunk,
)
from ai_orchestrator.browser_intelligence.intelligence.traffic_classifier import (
    TrafficCategory,
)


def _make_capture():
    return ResponseCapture()


class TestResponseCapture:
    def test_begin_response_classifies_chat(self):
        rc = _make_capture()
        c = rc.begin_response(
            request_id="r1",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        assert c.is_chat

    def test_begin_response_rejects_analytics(self):
        rc = _make_capture()
        c = rc.begin_response(
            request_id="r1",
            url="https://google-analytics.com/collect",
            method="POST",
            status=200,
            content_type="image/gif",
        )
        assert not c.is_chat

    def test_append_chunk_accepted_for_chat(self):
        rc = _make_capture()
        rc.begin_response(
            request_id="r1",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        ok = rc.append_chunk("r1", "hello ")
        assert ok
        ok = rc.append_chunk("r1", "world")
        assert ok
        cap = rc.get_response("r1")
        assert cap is not None
        assert cap.text == "hello world"
        assert cap.chunks == 2

    def test_append_chunk_rejected_for_analytics(self):
        rc = _make_capture()
        rc.begin_response(
            request_id="r1",
            url="https://google-analytics.com/collect",
            method="POST",
            status=200,
            content_type="image/gif",
        )
        ok = rc.append_chunk("r1", "would-be-pollution")
        assert ok is False
        cap = rc.get_response("r1")
        # The response is still tracked (so we know the URL), but
        # chunks are rejected.
        assert cap is not None
        assert cap.text == ""

    def test_get_response_text_returns_latest_chat(self):
        rc = _make_capture()
        rc.begin_response(
            request_id="r1",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        rc.append_chunk("r1", "first response")
        rc.close_response("r1")
        rc.begin_response(
            request_id="r2",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        rc.append_chunk("r2", "second response")
        assert rc.get_response_text() == "second response"

    def test_close_response_marks_closed(self):
        rc = _make_capture()
        rc.begin_response(
            request_id="r1",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        rc.append_chunk("r1", "abc")
        cap = rc.close_response("r1")
        assert cap is not None
        assert cap.stream_closed
        assert cap.stream_active is False
        # Should be removed from active tracking.
        assert rc.get_response("r1") is None

    def test_response_text_buffer_is_bounded(self):
        rc = _make_capture()
        rc.begin_response(
            request_id="r1",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        # Append a huge chunk; ensure buffer caps.
        rc.append_chunk("r1", "x" * 3_000_000)
        cap = rc.get_response("r1")
        assert cap is not None
        assert len(cap.text) <= 2_000_000  # _MAX_BODY_CHARS cap

    def test_reset_clears_all_state(self):
        rc = _make_capture()
        rc.begin_response(
            request_id="r1",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        rc.append_chunk("r1", "abc")
        rc.reset()
        assert rc.get_response("r1") is None
        assert rc.get_response_text() == ""

    def test_stats_track_active_and_closed(self):
        rc = _make_capture()
        rc.begin_response(
            request_id="r1",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        rc.append_chunk("r1", "x")
        rc.close_response("r1")
        s = rc.stats()
        assert s["active"] == 0
        assert s["closed"] == 1


class TestParseStreamChunk:
    def test_empty_returns_empty(self):
        assert parse_stream_chunk("") == ""

    def test_sse_data_line(self):
        raw = 'data: {"choices":[{"delta":{"content":"hello"}}]}'
        out = parse_stream_chunk(raw)
        assert "hello" in out

    def test_done_marker_ignored(self):
        raw = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
        out = parse_stream_chunk(raw)
        assert "hi" in out
        assert "[DONE]" not in out

    def test_ndjson(self):
        raw = (
            '{"choices":[{"delta":{"content":"a"}}]}\n'
            '{"choices":[{"delta":{"content":"b"}}]}\n'
        )
        out = parse_stream_chunk(raw)
        assert "a" in out and "b" in out

    def test_bare_json(self):
        raw = '{"choices":[{"delta":{"content":"plain"}}]}'
        out = parse_stream_chunk(raw)
        assert "plain" in out

    def test_plain_text_fallback(self):
        raw = "this is not json"
        out = parse_stream_chunk(raw)
        assert "not json" in out

    def test_multiple_delta_paths(self):
        for raw in [
            'data: {"content":"x"}',
            'data: {"text":"x"}',
            'data: {"message":"x"}',
            'data: {"delta_content":"x"}',
        ]:
            out = parse_stream_chunk(raw)
            assert "x" in out, raw
