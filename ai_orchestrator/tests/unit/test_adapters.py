"""Tests for Provider Adapters — base protocol and mock stubs."""

from __future__ import annotations

import pytest

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.adapters.chatgpt_api import ChatGPTAPIAdapter
from ai_orchestrator.adapters.chatgpt_ui import ChatGPTUIAdapter
from ai_orchestrator.adapters.qwen_api import QwenAPIAdapter
from ai_orchestrator.adapters.deepseek_api import DeepSeekAPIAdapter
from ai_orchestrator.adapters.kimi_api import KimiAPIAdapter
from ai_orchestrator.adapters.local_llm import LocalLLMAdapter


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


class TestChatGPTAPIAdapter:
    """ChatGPT API adapter mock stubs."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = ChatGPTAPIAdapter()
        resp = await adapter.send("Hello")
        assert isinstance(resp, ProviderResponse)
        assert resp.success is True
        assert "ChatGPT API" in resp.content

    @pytest.mark.asyncio
    async def test_health_check(self):
        adapter = ChatGPTAPIAdapter()
        assert await adapter.health_check() is True

    def test_context_limit(self):
        adapter = ChatGPTAPIAdapter()
        assert adapter.get_context_limit() > 0

    @pytest.mark.asyncio
    async def test_rate_limited_after_many_calls(self):
        adapter = ChatGPTAPIAdapter()
        assert await adapter.is_rate_limited() is False
        for _ in range(51):
            await adapter.send("test")
        assert await adapter.is_rate_limited() is True

    def test_provider_name(self):
        adapter = ChatGPTAPIAdapter()
        assert adapter.provider_name == "chatgpt"
        assert adapter.supports_streaming is True
        assert adapter.supports_tools is True


class TestChatGPTUIAdapter:
    """ChatGPT UI (browser) adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = ChatGPTUIAdapter()
        resp = await adapter.send("Hello")
        assert isinstance(resp, ProviderResponse)

    @pytest.mark.asyncio
    async def test_health_check(self):
        adapter = ChatGPTUIAdapter()
        assert await adapter.health_check() is True

    def test_context_limit(self):
        adapter = ChatGPTUIAdapter()
        assert adapter.get_context_limit() == 32768

    def test_provider_name(self):
        adapter = ChatGPTUIAdapter()
        assert adapter.provider_name == "chatgpt"
        assert adapter.supports_streaming is False
        assert adapter.supports_tools is False


class TestQwenAPIAdapter:
    """Qwen API adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = QwenAPIAdapter()
        resp = await adapter.send("Translate this")
        assert "Qwen API" in resp.content

    def test_context_limit(self):
        adapter = QwenAPIAdapter()
        assert adapter.get_context_limit() == 131072

    def test_provider_name(self):
        adapter = QwenAPIAdapter()
        assert adapter.provider_name == "qwen"


class TestDeepSeekAPIAdapter:
    """DeepSeek API adapter (1M context)."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = DeepSeekAPIAdapter()
        resp = await adapter.send("Code review")
        assert "DeepSeek API" in resp.content

    def test_context_limit(self):
        adapter = DeepSeekAPIAdapter()
        assert adapter.get_context_limit() == 1_000_000

    def test_provider_name(self):
        adapter = DeepSeekAPIAdapter()
        assert adapter.provider_name == "deepseek"


class TestKimiAPIAdapter:
    """Kimi API adapter."""

    @pytest.mark.asyncio
    async def test_send_returns_response(self):
        adapter = KimiAPIAdapter()
        resp = await adapter.send("Hello")
        assert "Kimi API" in resp.content

    def test_context_limit(self):
        adapter = KimiAPIAdapter()
        assert adapter.get_context_limit() == 128_000

    def test_supports_tools_false(self):
        adapter = KimiAPIAdapter()
        assert adapter.supports_tools is False


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
