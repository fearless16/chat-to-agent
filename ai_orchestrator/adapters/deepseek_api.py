"""DeepSeek API adapter — mock stub for testing."""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES


class DeepSeekAPIAdapter(ProviderAdapter):
    """Mock adapter for DeepSeek API (1M context window)."""

    provider_name = "deepseek"
    supports_streaming = True
    supports_tools = True

    def __init__(self, api_key: str = "test-key", model: str = "deepseek-v4") -> None:
        self.api_key = api_key
        self.model = model
        self._call_count = 0

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        self._call_count += 1
        return ProviderResponse(
            content=f"DeepSeek API response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 60, "completion_tokens": 120, "total_tokens": 180},
            latency_ms=100.0,
        )

    async def health_check(self) -> bool:
        return True

    def get_context_limit(self) -> int:
        return 1_000_000

    async def is_rate_limited(self) -> bool:
        return False

    async def refresh_session(self) -> bool:
        return True
