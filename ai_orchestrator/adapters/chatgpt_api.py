"""ChatGPT API adapter — OpenAI-compatible with optional mock mode for testing."""

from __future__ import annotations

import time
from typing import Optional

import httpx

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.models.capabilities import PROVIDER_PROFILES

_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class ChatGPTAPIAdapter(ProviderAdapter):
    """OpenAI ChatGPT API adapter.

    Set ``mock_mode=False`` for live HTTP calls; defaults to mock for
    backward-compatible testing.
    """

    provider_name = "chatgpt"
    supports_streaming = True
    supports_tools = True

    def __init__(
        self,
        api_key: str = "test-key",
        model: str = "gpt-4o",
        mock_mode: bool = True,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self._mock_mode = mock_mode
        self._client: Optional[httpx.AsyncClient] = None
        self._max_healthy_calls = 5

    # ── public interface ─────────────────────────────────────────────

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        if self._mock_mode:
            return self._mock_send(prompt, context)
        return await self._real_send(prompt, context)

    async def health_check(self) -> bool:
        if self._mock_mode:
            return self._call_count < self._max_healthy_calls
        try:
            client = await self._get_client()
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_context_limit(self) -> int:
        profile = PROVIDER_PROFILES.get("chatgpt_api")
        if profile is not None and hasattr(profile, "context_limit"):
            return profile.context_limit
        return 32768

    async def is_rate_limited(self) -> bool:
        return self._call_count > 50

    async def refresh_session(self) -> bool:
        self._call_count = 0
        return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── internal ─────────────────────────────────────────────────────

    def _mock_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"ChatGPT API response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            latency_ms=120.0,
        )

    async def _real_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        t0 = time.monotonic()
        client = await self._get_client()
        messages = (context or []) + [{"role": "user", "content": prompt}]
        try:
            resp = await client.post(
                _OPENAI_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                },
                timeout=60.0,
            )
            data = resp.json()
            if resp.status_code >= 400:
                return ProviderResponse(
                    success=False,
                    error=data.get("error", {}).get("message", resp.text),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            return ProviderResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.model),
                usage=data.get("usage"),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ProviderResponse(
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client
