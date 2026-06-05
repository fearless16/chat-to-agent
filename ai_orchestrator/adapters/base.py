"""ProviderAdapter base protocol with circuit breaker integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from pydantic import BaseModel, Field

from ai_orchestrator.utils.backoff import CircuitBreaker


class ProviderResponse(BaseModel):
    """Standard response from any provider adapter."""

    content: str = Field(default="")
    model: str = Field(default="unknown")
    usage: Optional[dict] = Field(default=None)
    latency_ms: float = Field(default=0.0)
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)


class ProviderAdapter(ABC):
    """Abstract interface with circuit breaker integration."""

    provider_name: str = ""
    supports_streaming: bool = False
    supports_tools: bool = False

    def __init__(self) -> None:
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout_ms=60_000,
            half_open_max_calls=1,
        )
        self._call_count = 0

    @abstractmethod
    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        ...

    async def protected_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        """Send with circuit breaker — opens after 3 failures.

        Success and failure accounting is owned by
        :meth:`CircuitBreaker.call`, which records exactly one success or
        failure per call.  ``protected_send`` therefore does NOT call
        ``record_success`` / ``record_failure`` itself; doing so would
        double-count every failure and trip the breaker after a single
        real failure.
        """
        self._call_count += 1
        try:
            return await self._circuit_breaker.call(
                self.send, prompt, context=context
            )
        except Exception as e:
            # ``CircuitBreaker.call`` already recorded the failure; we
            # only need to convert the exception into a normalised
            # response shape.
            return ProviderResponse(success=False, error=str(e))

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    async def safe_health_check(self) -> bool:
        """Health check that returns False when circuit is open."""
        if self._circuit_breaker.is_open:
            return False
        try:
            return await self.health_check()
        except Exception:
            return False

    @abstractmethod
    def get_context_limit(self) -> int:
        ...

    @abstractmethod
    async def is_rate_limited(self) -> bool:
        ...

    @abstractmethod
    async def refresh_session(self) -> bool:
        ...

    async def send_stream(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> AsyncIterator[Optional[ProviderResponse]]:
        """Optional async generator for streaming responses."""
        yield None

    async def close(self) -> None:
        """Clean up any resources (browser, connections)."""
