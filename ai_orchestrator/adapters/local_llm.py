"""Local LLM adapter — Ollama / llama.cpp with optional mock mode for testing."""

from __future__ import annotations

import time

import httpx

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse


class LocalLLMAdapter(ProviderAdapter):
    """Local LLM adapter for Ollama-compatible endpoints.

    Set ``mock_mode=False`` to talk to a running Ollama instance.
    """

    provider_name = "local_llm"
    supports_streaming = True
    supports_tools = False

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        endpoint: str = "http://localhost:11434",
        mock_mode: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self._mock_mode = mock_mode
        self._client: httpx.AsyncClient | None = None

    async def send(
        self, prompt: str, context: list[dict] | None = None
    ) -> ProviderResponse:
        if self._mock_mode:
            return self._mock_send(prompt, context)
        return await self._real_send(prompt, context)

    async def health_check(self) -> bool:
        if self._mock_mode:
            return True
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.endpoint}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def get_context_limit(self) -> int:
        return 256_000

    async def is_rate_limited(self) -> bool:
        return False

    async def refresh_session(self) -> bool:
        return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _mock_send(
        self, prompt: str, _context: list[dict] | None = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"Local LLM ({self.model}) response to: {prompt[:50]}",
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=500.0,
        )

    async def _real_send(
        self, prompt: str, _context: list[dict] | None = None
    ) -> ProviderResponse:
        t0 = time.monotonic()
        client = await self._get_client()
        messages = (_context or []) + [{"role": "user", "content": prompt}]
        try:
            resp = await client.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=120.0,
            )
            data = resp.json()
            if resp.status_code >= 400:
                return ProviderResponse(
                    success=False,
                    error=data.get("error", resp.text),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            return ProviderResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0)
                    + data.get("eval_count", 0),
                },
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
