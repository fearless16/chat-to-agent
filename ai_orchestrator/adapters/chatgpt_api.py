"""ChatGPT API adapter — mock stub for testing."""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES


class ChatGPTAPIAdapter(ProviderAdapter):
    """Mock adapter for the OpenAI ChatGPT API."""

    provider_name = "chatgpt"
    supports_streaming = True
    supports_tools = True

    def __init__(self, api_key: str = "test-key", model: str = "gpt-4o") -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self._max_healthy_calls = 5

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"ChatGPT API response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            latency_ms=120.0,
        )

    async def health_check(self) -> bool:
        return self._call_count < self._max_healthy_calls

    def get_context_limit(self) -> int:
        return PROVIDER_PROFILES.get("chatgpt_api", {}).context_limit if hasattr(PROVIDER_PROFILES.get("chatgpt_api", {}), "context_limit") else 32768

    async def is_rate_limited(self) -> bool:
        return self._call_count > 50

    async def refresh_session(self) -> bool:
        self._call_count = 0
        return True
