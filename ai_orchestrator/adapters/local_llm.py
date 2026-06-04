"""Local LLM adapter — mock stub for Ollama / llama.cpp."""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES


class LocalLLMAdapter(ProviderAdapter):
    """Mock adapter for local LLM inference (Ollama / llama.cpp)."""

    provider_name = "local_llm"
    supports_streaming = True
    supports_tools = False

    def __init__(
        self, model: str = "qwen3.5", endpoint: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self._call_count = 0

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        self._call_count += 1
        return ProviderResponse(
            content=f"Local LLM ({self.model}) response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=500.0,
        )

    async def health_check(self) -> bool:
        return True

    def get_context_limit(self) -> int:
        return 256_000

    async def is_rate_limited(self) -> bool:
        return False

    async def refresh_session(self) -> bool:
        return True
