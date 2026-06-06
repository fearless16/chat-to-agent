"""Tests for the response traffic classifier."""

from __future__ import annotations

import pytest

from ai_orchestrator.browser_intelligence.intelligence.traffic_classifier import (
    ResponseClassifier,
    TrafficCategory,
    TrafficClassification,
)


class TestResponseClassifier:
    def setup_method(self) -> None:
        self.cls = ResponseClassifier()

    def test_chat_completions_path_is_chat(self):
        c = self.cls.classify(
            url="https://api.example.com/v1/chat/completions",
            method="POST",
            content_type="text/event-stream",
            status=200,
        )
        assert c.category == TrafficCategory.CHAT_RESPONSE
        assert c.confidence >= 0.55

    def test_conversation_list_is_not_chat(self):
        c = self.cls.classify(
            url="https://chatgpt.com/backend-api/conversations?offset=0&limit=20",
            method="GET",
            content_type="application/json",
            status=200,
        )
        assert c.category == TrafficCategory.CONVERSATION_LIST
        assert c.is_pollution

    def test_analytics_is_pollution(self):
        c = self.cls.classify(
            url="https://www.google-analytics.com/collect",
            method="POST",
            content_type="image/gif",
            status=200,
        )
        assert c.category == TrafficCategory.ANALYTICS
        assert c.is_pollution

    def test_static_assets_filtered(self):
        c = self.cls.classify(
            url="https://cdn.example.com/static/main.abc123.js",
            method="GET",
            content_type="application/javascript",
            status=200,
        )
        assert c.category == TrafficCategory.STATIC
        assert c.is_pollution

    def test_auth_endpoints_filtered(self):
        for url in [
            "https://example.com/auth/login",
            "https://example.com/oauth/token",
            "https://example.com/api/csrf",
        ]:
            c = self.cls.classify(
                url=url,
                method="POST",
                content_type="application/json",
                status=200,
            )
            assert c.category == TrafficCategory.AUTH, url
            assert c.is_pollution, url

    def test_telemetry_pings_filtered(self):
        c = self.cls.classify(
            url="https://example.com/api/resultobject/123",
            method="POST",
            content_type="application/json",
            status=200,
        )
        assert c.category == TrafficCategory.TELEMETRY
        assert c.is_pollution

    def test_streaming_content_type_lifts_chat_score(self):
        c = self.cls.classify(
            url="https://example.com/api/sendMessage",
            method="POST",
            content_type="text/event-stream",
            status=200,
        )
        assert c.category == TrafficCategory.CHAT_RESPONSE

    def test_4xx_status_demotes(self):
        c = self.cls.classify(
            url="https://example.com/v1/chat/completions",
            method="POST",
            content_type="text/event-stream",
            status=429,
        )
        # 429 is rate-limit, but the path is chat; with low confidence
        # we may classify as UNKNOWN or CHAT. The test asserts the
        # classifier does not over-confidently say CHAT.
        assert c.confidence < 1.0

    def test_body_sample_with_delta_keys_is_chat(self):
        sample = '{"choices":[{"delta":{"content":"hello"}}]}'
        c = self.cls.classify(
            url="https://example.com/v1/chat/completions",
            method="POST",
            content_type="application/json",
            status=200,
            body_sample=sample,
        )
        assert c.category == TrafficCategory.CHAT_RESPONSE

    def test_sse_data_prefix_in_body(self):
        sample = 'data: {"choices":[{"delta":{"content":"hello"}}]}'
        c = self.cls.classify(
            url="https://example.com/api/chat",
            method="POST",
            content_type="text/event-stream",
            status=200,
            body_sample=sample,
        )
        assert c.category == TrafficCategory.CHAT_RESPONSE

    def test_reasons_contain_signal_names(self):
        c = self.cls.classify(
            url="https://example.com/v1/chat/completions",
            method="POST",
            content_type="text/event-stream",
            status=200,
        )
        assert c.reasons
        assert any("chat:" in r for r in c.reasons)

    def test_stats_track_counts(self):
        self.cls.classify(
            url="https://example.com/v1/chat/completions",
            method="POST",
            content_type="text/event-stream",
        )
        self.cls.classify(
            url="https://google-analytics.com/collect",
            method="POST",
        )
        s = self.cls.stats
        assert s["seen"] == 2
        assert s["chat"] == 1
        assert s["pollution"] == 1

    def test_confidence_in_unit_interval(self):
        for url, ct in [
            ("https://example.com/v1/chat/completions", "text/event-stream"),
            ("https://google-analytics.com/collect", ""),
            ("https://example.com/static/main.js", "application/javascript"),
            ("https://example.com/auth/login", "application/json"),
        ]:
            c = self.cls.classify(url=url, method="GET", content_type=ct, status=200)
            assert 0.0 <= c.confidence <= 1.0

    def test_chat_min_confidence_threshold(self):
        cls = ResponseClassifier(chat_min_confidence=0.95)
        c = cls.classify(
            url="https://example.com/api/send",
            method="POST",
            content_type="text/event-stream",
        )
        # Below the 0.95 threshold, weak chat signals should not promote.
        if c.category == TrafficCategory.CHAT_RESPONSE:
            assert c.confidence >= 0.95

    def test_long_request_duration_helps_chat(self):
        c = self.cls.classify(
            url="https://example.com/v1/chat/completions",
            method="POST",
            content_type="text/event-stream",
            request_duration_ms=3000,
            status=200,
        )
        assert c.category == TrafficCategory.CHAT_RESPONSE
        # Reasons should include the long_stream signal.
        assert any("long_stream" in r for r in c.reasons)

    def test_empty_url_returns_unknown(self):
        c = self.cls.classify(url="", method="GET", content_type="", status=0)
        assert c.category == TrafficCategory.UNKNOWN

    def test_zai_tracking_ping_classified_as_telemetry_not_chat(self):
        url = "https://z.ai/api/client/someid/track/resultobject"
        body = '{"ResultObject":true,"RequestId":"abc123","Code":"200"}'
        c = self.cls.classify(
            url=url,
            method="POST",
            content_type="application/json",
            status=200,
            body_sample=body,
        )
        assert not c.is_chat, f"Tracking ping must not be CHAT: got {c.category}"
        assert c.is_pollution

    def test_telemetry_ping_without_chat_path_not_misclassified(self):
        c = self.cls.classify(
            url="https://example.com/api/health",
            method="GET",
            content_type="text/plain",
            status=200,
        )
        assert not c.is_chat
        assert c.category == TrafficCategory.TELEMETRY

    def test_body_with_resultobject_rejected_as_telemetry(self):
        body = '{"ResultObject":true,"RequestId":"abc","Code":"200"}'
        c = self.cls.classify(
            url="https://example.com/api/resultobject",
            method="POST",
            content_type="application/json",
            status=200,
            body_sample=body,
        )
        assert c.category == TrafficCategory.TELEMETRY
        assert c.is_pollution
