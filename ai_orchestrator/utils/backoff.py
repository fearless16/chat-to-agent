"""Retry with exponential back-off and a circuit breaker."""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════════════════
# RetryConfig
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetryConfig:
    """Configuration for exponential-backoff retry logic.

    Attributes
    ----------
    max_retries:
        Maximum number of *retries* (not including the initial call).
    base_delay_ms:
        Base delay in milliseconds for the first retry.
    max_delay_ms:
        Upper bound on each individual delay.
    jitter:
        When ``True``, a random fraction [0, delay) is added to each sleep.
    multiplier:
        Factor applied to the delay after each retry (exponential step).
    """

    max_retries: int = 3
    base_delay_ms: float = 1_000.0
    max_delay_ms: float = 60_000.0
    jitter: bool = True
    multiplier: float = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# retry_with_backoff
# ══════════════════════════════════════════════════════════════════════════════

OnRetryCallable = Callable[[Exception, int, float], Awaitable[None]] | None


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    config: RetryConfig | None = None,
    on_retry: OnRetryCallable = None,
) -> T:
    """Execute *func* with exponential-backoff retry logic.

    Parameters
    ----------
    func:
        An async callable to invoke (and possibly retry).
    config:
        Retry configuration; uses ``RetryConfig()`` defaults when ``None``.
    on_retry:
        Optional async callback invoked before each retry sleep.  Receives
        ``(exception, attempt_number, delay_ms)`` where *attempt_number*
        is 1-based.

    Raises
    ------
    Exception:
        The last exception raised by *func* when all retries are exhausted.

    Returns
    -------
    The return value of *func* on success.
    """
    cfg = config or RetryConfig()
    last_exc: Exception | None = None

    for attempt in range(cfg.max_retries + 1):  # initial call + retries
        try:
            return await func()
        except Exception as exc:
            last_exc = exc
            if attempt == cfg.max_retries:
                raise

            delay_ms = min(
                cfg.base_delay_ms * (cfg.multiplier**attempt),
                cfg.max_delay_ms,
            )
            if cfg.jitter:
                delay_ms = random.uniform(0, delay_ms)

            if on_retry is not None:
                await on_retry(exc, attempt + 1, delay_ms)

            await asyncio.sleep(delay_ms / 1000.0)

    # This line is only reachable if func raised every time (caught by raise above)
    assert last_exc is not None
    raise last_exc  # pragma: no cover


# ══════════════════════════════════════════════════════════════════════════════
# CircuitBreaker
# ══════════════════════════════════════════════════════════════════════════════


class CircuitBreaker:
    """Circuit breaker for graceful degradation under failure.

    State machine: ``CLOSED`` → ``OPEN`` → ``HALF_OPEN`` → ``CLOSED`` (or
    back to ``OPEN``).

    *   **CLOSED** — normal operation; calls pass through.
    *   **OPEN** — calls are rejected immediately with
        :class:`CircuitBreakerOpenError`.
    *   **HALF_OPEN** — after *recovery_timeout_ms*; a limited number of probe
        calls are allowed to test if the downstream has recovered.
    """

    class CircuitBreakerOpenError(Exception):
        """Raised when a call is rejected because the circuit is OPEN."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_ms: float = 60_000.0,
        half_open_max_calls: int = 1,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout_ms < 0:
            raise ValueError("recovery_timeout_ms must be >= 0")
        if half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")

        self._failure_threshold = failure_threshold
        self._recovery_timeout_ms = recovery_timeout_ms
        self._half_open_max_calls = half_open_max_calls

        # -- internal state --
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_used = 0
        self._half_open_lock = asyncio.Lock()

    # -- properties -----------------------------------------------------------

    @property
    def state(self) -> str:
        """Current circuit-breaker state: ``CLOSED``, ``OPEN``, or ``HALF_OPEN``."""
        self._maybe_transition_to_half_open()
        return self._state

    @property
    def is_open(self) -> bool:
        """Convenience: ``True`` when the circuit is in the OPEN state."""
        return self.state == "OPEN"

    # -- public mutation ------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful call.

        Resets the consecutive-failure counter and transitions to CLOSED if
        currently HALF_OPEN.
        """
        self._failure_count = 0
        if self._state == "HALF_OPEN":
            self._state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed call.

        Increments the consecutive-failure counter.  When the counter reaches
        *failure_threshold* the circuit transitions to OPEN.
        """
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = "OPEN"

    # -- async call -----------------------------------------------------------

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute *func* through the circuit breaker.

        In the OPEN state the call is rejected immediately.  In the HALF_OPEN
        state only a limited number of probe calls are allowed; other callers
        are rejected.
        """
        await self._sync_state()

        if self._state == "OPEN":
            raise self.CircuitBreakerOpenError("Circuit breaker is OPEN")

        if self._state == "HALF_OPEN":
            async with self._half_open_lock:
                if self._half_open_used >= self._half_open_max_calls:
                    raise self.CircuitBreakerOpenError(
                        "Circuit breaker is HALF_OPEN and probe slot is occupied"
                    )
                self._half_open_used += 1

            # The probe call runs outside the lock so other callers see the slot as taken
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise
            finally:
                async with self._half_open_lock:
                    self._half_open_used -= 1

        # CLOSED state
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    # -- internal helpers -----------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """If the recovery timeout has elapsed, move from OPEN to HALF_OPEN."""
        if (
            self._state == "OPEN"
            and self._last_failure_time is not None
        ):
            elapsed_ms = (time.monotonic() - self._last_failure_time) * 1000.0
            if elapsed_ms >= self._recovery_timeout_ms:
                self._state = "HALF_OPEN"
                self._half_open_used = 0

    async def _sync_state(self) -> None:
        """Evaluate time-based state transitions before dispatching a call."""
        self._maybe_transition_to_half_open()
