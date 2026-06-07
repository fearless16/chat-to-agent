"""ProviderAdapter base protocol with circuit breaker integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from ai_orchestrator.utils.backoff import CircuitBreaker


class ProviderResponse(BaseModel):
    """Standard response from any provider adapter.

    ``reasoning_content`` captures the model's internal chain-of-thought
    (e.g. DeepSeek-R1 reasoning, o1/o3 thinking, Qwen thinking phase).
    Set to ``None`` when the provider does not expose reasoning or the
    adapter cannot separate it from the final answer.
    """

    content: str = Field(default="")
    reasoning_content: str | None = Field(default=None)
    model: str = Field(default="unknown")
    usage: dict | None = Field(default=None)
    latency_ms: float = Field(default=0.0)
    success: bool = Field(default=True)
    error: str | None = Field(default=None)

    @property
    def is_valid(self) -> bool:
        """Quick validity check — non-empty content and no error."""
        return self.success and bool(self.content and self.content.strip())


class ProviderAdapter(ABC):
    """Abstract interface with circuit breaker integration.

    Supports optional **response validation** (per V6 architecture
    Response Validation pipeline).  When ``validate_responses`` is
    ``True``, ``protected_send`` passes the response through a
    ``ResponseValidator`` before returning it.
    """

    provider_name: str = ""
    supports_streaming: bool = False
    supports_tools: bool = False

    def __init__(self, validate_responses: bool = False) -> None:
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout_ms=60_000,
            half_open_max_calls=1,
        )
        self._call_count = 0
        self._validate_responses = validate_responses
        self._validator: ResponseValidator | None = None  # noqa: F821

    def _lazy_init_validator(self) -> None:
        if self._validator is None and self._validate_responses:
            from ai_orchestrator.validation.validator import ResponseValidator
            self._validator = ResponseValidator()

    @abstractmethod
    async def send(
        self, prompt: str, context: list[dict] | None = None
    ) -> ProviderResponse:
        ...

    async def protected_send(
        self, prompt: str, context: list[dict] | None = None
    ) -> ProviderResponse:
        """Send with circuit breaker + optional response validation.

        Success and failure accounting is owned by
        :meth:`CircuitBreaker.call`, which records exactly one success or
        failure per call.  ``protected_send`` therefore does NOT call
        ``record_success`` / ``record_failure`` itself; doing so would
        double-count every failure and trip the breaker after a single
        real failure.

        When ``validate_responses`` is enabled, the response is checked
        by the ResponseValidator pipeline (Level 1 deterministic checks
        always; Level 2 DeepSeek review if score < 1.0).
        """
        self._call_count += 1
        self._lazy_init_validator()
        try:
            response = await self._circuit_breaker.call(
                self.send, prompt, context=context
            )
            if self._validator and response.success:
                validation = await self._validator.validate(
                    response, prompt=prompt,
                )
                if not validation.passed:
                    err_msg = validation.errors[0].message if validation.errors else "unknown"
                    return ProviderResponse(
                        success=False,
                        error=f"Validation failed: {err_msg}",
                        latency_ms=response.latency_ms,
                    )
            return response
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
        self, _prompt: str, _context: list[dict] | None = None
    ) -> AsyncIterator[ProviderResponse | None]:
        """Optional async generator for streaming responses."""
        yield None

    async def close(self) -> None:  # noqa: B027
        """Clean up any resources (browser, connections)."""
