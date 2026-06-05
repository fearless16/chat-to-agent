"""Kimi API adapter — mock stub for testing."""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES


class KimiAPIAdapter(ProviderAdapter):
    """Mock adapter for Moonshot AI's Kimi API."""

    provider_name = "kimi"
    supports_streaming = True
    supports_tools = False

    def __init__(self, api_key: str = "test-key", model: str = "kimi-latest") -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"Kimi API response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 40, "completion_tokens": 80, "total_tokens": 120},
            latency_ms=200.0,
        )

    async def health_check(self) -> bool:
        return True

    def get_context_limit(self) -> int:
        return 128_000

    async def is_rate_limited(self) -> bool:
        return self._call_count > 60

    async def refresh_session(self) -> bool:
        return True
