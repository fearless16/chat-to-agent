"""Kimi API adapter — Moonshot AI (OpenAI-compatible)."""

from __future__ import annotations

import time
from typing import Optional

import httpx

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse

_KIMI_ENDPOINT = "https://api.moonshot.cn/v1/chat/completions"


class KimiAPIAdapter(ProviderAdapter):
    """Moonshot AI Kimi API adapter (OpenAI-compatible)."""

    provider_name = "kimi"
    supports_streaming = True
    supports_tools = False

    def __init__(
        self,
        api_key: str = "test-key",
        model: str = "moonshot-v1-128k",
        mock_mode: bool = True,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self._mock_mode = mock_mode
        self._client: Optional[httpx.AsyncClient] = None

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        if self._mock_mode:
            return self._mock_send(prompt, context)
        return await self._real_send(prompt, context)

    async def health_check(self) -> bool:
        if self._mock_mode:
            return True
        try:
            client = await self._get_client()
            resp = await client.post(
                _KIMI_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=10.0,
            )
            return resp.status_code < 500
        except Exception:
            return False

    def get_context_limit(self) -> int:
        return 128_000

    async def is_rate_limited(self) -> bool:
        return self._call_count > 60

    async def refresh_session(self) -> bool:
        return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _mock_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"Kimi API response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 40, "completion_tokens": 80, "total_tokens": 120},
            latency_ms=200.0,
        )

    async def _real_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        t0 = time.monotonic()
        client = await self._get_client()
        messages = (context or []) + [{"role": "user", "content": prompt}]
        try:
            resp = await client.post(
                _KIMI_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages},
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
