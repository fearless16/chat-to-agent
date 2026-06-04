"""Qwen API adapter — mock stub for testing."""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES


class QwenAPIAdapter(ProviderAdapter):
    """Mock adapter for Alibaba Cloud's Qwen API."""

    provider_name = "qwen"
    supports_streaming = True
    supports_tools = True

    def __init__(self, api_key: str = "test-key", model: str = "qwen3.5-128k") -> None:
        self.api_key = api_key
        self.model = model
        self._call_count = 0

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        self._call_count += 1
        return ProviderResponse(
            content=f"Qwen API response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 80, "completion_tokens": 60, "total_tokens": 140},
            latency_ms=180.0,
        )

    async def health_check(self) -> bool:
        return True

    def get_context_limit(self) -> int:
        return 131072

    async def is_rate_limited(self) -> bool:
        return self._call_count > 100

    async def refresh_session(self) -> bool:
        return True
