"""Tests for Provider Adapters — base protocol and mock stubs (UI + local only)."""

from __future__ import annotations

import pytest

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.adapters.chatgpt_ui import ChatGPTUIAdapter
from ai_orchestrator.adapters.deepseek_ui import DeepSeekUIAdapter
from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter
from ai_orchestrator.adapters.kimi_ui import KimiUIAdapter
from ai_orchestrator.adapters.local_llm import LocalLLMAdapter
from ai_orchestrator.adapters.minimax_ui import MiniMaxUIAdapter
from ai_orchestrator.adapters.qwen_ui import QwenUIAdapter
from ai_orchestrator.adapters.xiaomimimo_ui import XiaomiMiMoUIAdapter
from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter


class TestProviderResponse:
    """ProviderResponse model validation."""

    def test_default_values(self):
        resp = ProviderResponse()
        assert resp.content == ""
        assert resp.model == "unknown"
        assert resp.success is True
        assert resp.error is None

    def test_full_response(self):
        resp = ProviderResponse(
            content="Hello world",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=150.0,
        )
        assert resp.content == "Hello world"
        assert resp.usage["total_tokens"] == 30

    def test_error_response(self):
        resp = ProviderResponse(success=False, error="rate limited")
        assert resp.success is False
        assert resp.error == "rate limited"


class TestProviderAdapterBase:
    """ProviderAdapter ABC enforces interface."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ProviderAdapter()  # type: ignore


class TestChatGPTUIAdapter:
    """ChatGPT UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = ChatGPTUIAdapter()
        resp = await adapter.send("Hello")
        assert isinstance(resp, ProviderResponse)
        assert "ChatGPT UI" in resp.content

    @pytest.mark.asyncio
    async def test_health_check(self):
        adapter = ChatGPTUIAdapter()
        assert await adapter.health_check() is True

    def test_context_limit(self):
        adapter = ChatGPTUIAdapter()
        assert adapter.get_context_limit() == 131072

    def test_provider_name(self):
        adapter = ChatGPTUIAdapter()
        assert adapter.provider_name == "chatgpt_ui"
        assert adapter.supports_streaming is False
        assert adapter.supports_tools is False


class TestQwenUIAdapter:
    """Qwen UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = QwenUIAdapter()
        resp = await adapter.send("Hello")
        assert isinstance(resp, ProviderResponse)
        assert "Qwen UI" in resp.content

    @pytest.mark.asyncio
    async def test_health_check(self):
        adapter = QwenUIAdapter()
        assert await adapter.health_check() is True

    def test_context_limit(self):
        adapter = QwenUIAdapter()
        assert adapter.get_context_limit() == 131072

    def test_provider_name(self):
        adapter = QwenUIAdapter()
        assert adapter.provider_name == "qwen_ui"
        assert adapter.supports_streaming is False
        assert adapter.supports_tools is False


class TestDeepSeekUIAdapter:
    """DeepSeek UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = DeepSeekUIAdapter()
        resp = await adapter.send("Code review")
        assert "DeepSeek UI" in resp.content

    def test_context_limit(self):
        adapter = DeepSeekUIAdapter()
        assert adapter.get_context_limit() > 0

    def test_provider_name(self):
        adapter = DeepSeekUIAdapter()
        assert adapter.provider_name == "deepseek_ui"


class TestKimiUIAdapter:
    """Kimi UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = KimiUIAdapter()
        resp = await adapter.send("Hello")
        assert "Kimi" in resp.content

    def test_context_limit(self):
        adapter = KimiUIAdapter()
        assert adapter.get_context_limit() == 131_072

    def test_supports_tools_false(self):
        adapter = KimiUIAdapter()
        assert adapter.supports_tools is False

    def test_provider_name(self):
        adapter = KimiUIAdapter()
        assert adapter.provider_name == "kimi_ui"


class TestZAIUIAdapter:
    """Z.ai UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = ZAIUIAdapter()
        resp = await adapter.send("Hello")
        assert "Z.ai" in resp.content

    def test_provider_name(self):
        adapter = ZAIUIAdapter()
        assert adapter.provider_name == "z_ai_ui"


class TestMiniMaxUIAdapter:
    """MiniMax UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = MiniMaxUIAdapter()
        resp = await adapter.send("Hello")
        assert isinstance(resp, ProviderResponse)
        assert resp.success

    def test_provider_name(self):
        adapter = MiniMaxUIAdapter()
        assert adapter.provider_name == "minimax_ui"


class TestXiaomiMiMoUIAdapter:
    """Xiaomi MiMo UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = XiaomiMiMoUIAdapter()
        resp = await adapter.send("Hello")
        assert "XiaomiMiMo" in resp.content

    def test_provider_name(self):
        adapter = XiaomiMiMoUIAdapter()
        assert adapter.provider_name == "xiaomimimo_ui"


class TestLocalLLMAdapter:
    """Local LLM adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = LocalLLMAdapter()
        resp = await adapter.send("Hello")
        assert "Local LLM" in resp.content

    def test_context_limit(self):
        adapter = LocalLLMAdapter()
        assert adapter.get_context_limit() == 256_000

    def test_provider_name(self):
        adapter = LocalLLMAdapter()
        assert adapter.provider_name == "local_llm"


class TestNetworkObserverTextBuffer:
    """The network observers must buffer the actual response text so the
    engine can return the assistant's reply without falling back to DOM."""

    def test_sse_observer_buffers_data_field(self):
        from ai_orchestrator.browser_intelligence.sensors.network.sse_observer import (
            SSEObserver,
        )
        obs = SSEObserver()
        obs._active_stream_ids.add("r1")
        obs.on_event_source_message_received({
            "requestId": "r1",
            "eventName": "message",
            "data": '{"type":"chat:completion","data":{"delta_content":"hello "}}',
        })
        obs.on_event_source_message_received({
            "requestId": "r1",
            "eventName": "message",
            "data": '{"type":"chat:completion","data":{"delta_content":"world"}}',
        })
        obs.on_event_source_message_received({
            "requestId": "r1",
            "eventName": "message",
            "data": "[DONE]",
        })
        text = obs.get_response_text()
        # Coerce to clean text via the engine's helper.
        from ai_orchestrator.browser_intelligence.engine import _coerce_sse_text
        out = _coerce_sse_text(text)
        assert "hello " in out
        assert "world" in out
        assert "[DONE]" not in text

    def test_sse_observer_reset_clears_text(self):
        from ai_orchestrator.browser_intelligence.sensors.network.sse_observer import (
            SSEObserver,
        )
        obs = SSEObserver()
        obs._active_stream_ids.add("r1")
        obs.on_event_source_message_received({
            "requestId": "r1",
            "data": "data: chunk1\n",
        })
        assert obs.get_response_text()
        obs.reset()
        assert obs.get_response_text() == ""

    def test_ws_observer_buffers_payload(self):
        from ai_orchestrator.browser_intelligence.sensors.network.ws_observer import (
            WSObserver,
        )
        obs = WSObserver()
        obs.on_ws_frame_received({
            "response": {"payloadData": '{"type":"delta","content":"hi "}'}
        })
        obs.on_ws_frame_received({
            "response": {"payloadData": '{"type":"delta","content":"there"}'}
        })
        obs.on_ws_frame_received({
            "response": {"payloadData": "[DONE]"}
        })
        text = obs.get_response_text()
        from ai_orchestrator.browser_intelligence.engine import _coerce_sse_text
        out = _coerce_sse_text(text)
        assert "hi " in out
        assert "there" in out
        assert "[DONE]" not in text

    def test_fetch_observer_starts_empty(self):
        from ai_orchestrator.browser_intelligence.sensors.network.fetch_observer import (
            FetchObserver,
        )
        obs = FetchObserver()
        assert obs.get_response_text() == ""

    def test_engine_get_response_text_picks_sse(self):
        from ai_orchestrator.browser_intelligence.engine import (
            BrowserIntelligenceEngine,
            _coerce_sse_text,
        )
        # z_ai-shaped stream
        raw = (
            'data: {"type":"chat:completion","data":{"delta_content":"a"}}\n'
            'data: {"type":"chat:completion","data":{"delta_content":"b"}}\n'
            'data: [DONE]\n'
        )
        out = _coerce_sse_text(raw)
        assert out == "ab"
        # OpenAI-shaped
        raw2 = (
            'data: {"choices":[{"delta":{"content":"x"}}]}\n'
            'data: {"choices":[{"delta":{"content":"y"}}]}\n'
        )
        assert _coerce_sse_text(raw2) == "xy"
        # Non-SSE passes through
        assert _coerce_sse_text("plain text body") == "plain text body"

    def test_engine_get_response_text_sse_accessor(self):
        from ai_orchestrator.browser_intelligence.engine import (
            BrowserIntelligenceEngine,
        )
        eng = BrowserIntelligenceEngine()
        eng._composer._network._sse_observer._response_text = "raw"
        assert eng.get_response_text_sse() == "raw"


class TestSseDeltaExtraction:
    """EngineUIAdapter._sse_delta_to_text must handle every provider's SSE shape."""

    def test_zai_delta_content(self):
        raw = (
            'data: {"type":"chat:completion","data":{"delta_content":"1.  **An","phase":"thinking"}}\n\n'
            'data: {"type":"chat:completion","data":{"delta_content":"alyze","phase":"thinking"}}\n\n'
        )
        assert EngineUIAdapter._sse_delta_to_text(raw) == "1.  **Analyze"

    def test_openai_choices_delta(self):
        raw = (
            'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
            'data: {"id":"x","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
        )
        assert EngineUIAdapter._sse_delta_to_text(raw) == "Hello world"

    def test_qwen_response_created(self):
        raw = (
            'data: {"response.created":{"chat_id":"abc","response_id":"r1"}}\n\n'
            'data: {"response.choices":[{"index":0,"message":{"content":"Four"}}]}\n\n'
        )
        assert "Four" in EngineUIAdapter._sse_delta_to_text(raw)

    def test_done_marker_ignored(self):
        raw = 'data: {"content":"hi"}\n\ndata: [DONE]\n\n'
        assert EngineUIAdapter._sse_delta_to_text(raw) == "hi"

    def test_malformed_lines_skipped(self):
        raw = 'not data\ndata: not json\ndata: {"content":"ok"}\n\n'
        assert EngineUIAdapter._sse_delta_to_text(raw) == "ok"

    def test_empty_payloads(self):
        assert EngineUIAdapter._sse_delta_to_text("") == ""
        assert EngineUIAdapter._sse_delta_to_text("data:\n\n") == ""
        assert EngineUIAdapter._sse_delta_to_text("event: ping\n\n") == ""


class TestChatUrlDetection:
    """EngineUIAdapter._looks_like_chat_response URL pattern matching."""

    def test_chat_completions_url(self):
        assert EngineUIAdapter._looks_like_chat_response(
            {"url": "/api/v2/chat/completions?chat_id=abc", "transport": "fetch"}
        )

    def test_openai_messages_url(self):
        assert EngineUIAdapter._looks_like_chat_response(
            {"url": "/v1/messages", "transport": "fetch"}
        )

    def test_sse_transport(self):
        assert EngineUIAdapter._looks_like_chat_response(
            {"url": "/random/path", "transport": "sse"}
        )

    def test_conversation_list_url_rejected(self):
        assert not EngineUIAdapter._looks_like_chat_response(
            {"url": "/api/v1/chats/?page=1&type=default", "transport": "fetch"}
        )

    def test_models_endpoint_rejected(self):
        assert not EngineUIAdapter._looks_like_chat_response(
            {"url": "/api/v1/models", "transport": "fetch"}
        )

    def test_third_party_auth_rejected(self):
        assert not EngineUIAdapter._looks_like_chat_response(
            {"url": "https://cloudauth-device.aliyuncs.com", "transport": "fetch"}
        )


class TestResponseExtractScript:
    """DOM extractor must skip copy / like / regenerate buttons inside the bubble."""

    def test_contains_response_text_only(self):
        from ai_orchestrator.adapters.engine_adapter import RESPONSE_EXTRACT_SCRIPT
        # Just verify the script source has the key selectors
        assert "data-message-author-role=\"assistant\"" in RESPONSE_EXTRACT_SCRIPT
        assert "ACTION_RE" in RESPONSE_EXTRACT_SCRIPT
        assert "copy" in RESPONSE_EXTRACT_SCRIPT.lower()
        assert "regenerate" in RESPONSE_EXTRACT_SCRIPT.lower()

    def test_skip_action_button_text(self):
        from ai_orchestrator.adapters.engine_adapter import RESPONSE_EXTRACT_SCRIPT
        import re
        # The JS regex literal is `/^(...)/i` — extract the inner pattern.
        m = re.search(r"ACTION_RE\s*=\s*/(\^[^/]+)/[a-z]*", RESPONSE_EXTRACT_SCRIPT)
        assert m is not None, "could not find ACTION_RE literal"
        # Python equivalent: strip the JS leading ^ since Python re.match anchors at start.
        inner = m.group(1).lstrip("^")
        regex = re.compile(inner, re.IGNORECASE)
        for label in ["Copy", "copy", "Copied", "Regenerate", "Retry", "Edit",
                      "Share", "Like", "Dislike", "Thumbs up", "Thumbs down",
                      "Good response", "Bad response", "Speak", "Read aloud"]:
            assert regex.match(label), f"regex must match action label {label!r}"
