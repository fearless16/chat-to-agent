"""ProviderAdapter base protocol and response model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from pydantic import BaseModel, Field


class ProviderResponse(BaseModel):
    """Standard response from any provider adapter."""

    content: str = Field(default="")
    model: str = Field(default="unknown")
    usage: Optional[dict] = Field(default=None)
    latency_ms: float = Field(default=0.0)
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)


class ProviderAdapter(ABC):
    """Abstract interface all provider adapters must implement."""

    provider_name: str = ""
    supports_streaming: bool = False
    supports_tools: bool = False

    @abstractmethod
    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        """Send a prompt (with optional context) and return the response."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and healthy."""
        ...

    @abstractmethod
    def get_context_limit(self) -> int:
        """Return the maximum context window size in tokens."""
        ...

    @abstractmethod
    async def is_rate_limited(self) -> bool:
        """Return True if the provider is currently rate-limiting us."""
        ...

    @abstractmethod
    async def refresh_session(self) -> bool:
        """Refresh the session (login, token rotation). Return True on success."""
        ...

    async def send_stream(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> AsyncIterator[Optional[ProviderResponse]]:
        """Optional async generator for streaming responses."""
        yield None

    async def close(self) -> None:
        """Clean up any resources (browser, connections)."""
