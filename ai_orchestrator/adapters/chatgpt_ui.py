"""ChatGPT UI (browser) adapter — mock stub for testing."""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES


class ChatGPTUIAdapter(ProviderAdapter):
    """Mock adapter for ChatGPT accessed via Playwright browser."""

    provider_name = "chatgpt"
    supports_streaming = False
    supports_tools = False

    def __init__(self, headless: bool = True) -> None:
        super().__init__()
        self.headless = headless

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"ChatGPT UI (browser) response to: {prompt[:50]}",
            model="gpt-4o",
            latency_ms=2500.0,
        )

    async def health_check(self) -> bool:
        return True

    def get_context_limit(self) -> int:
        return 32768

    async def is_rate_limited(self) -> bool:
        return False

    async def refresh_session(self) -> bool:
        return True
