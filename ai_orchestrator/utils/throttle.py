"""Rate limiting — token bucket, concurrency limiter, and per-account registry."""

import asyncio
import threading
import time


class TokenBucket:
    """Token-bucket rate limiter.

    Maintains a bucket that fills at *rate* tokens per second, up to *burst*
    capacity.  Use ``acquire`` to block until tokens are available or
    ``try_acquire`` for a non-blocking check.

    The refill + check + decrement critical section is protected by a
    :class:`threading.Lock` so concurrent coroutines on different event
    loops, or threads, cannot both pass the ``self._tokens >= tokens``
    guard and over-draw the bucket.  The lock is released across the
    ``await asyncio.sleep`` in :meth:`acquire` to avoid serialising
    unrelated waiters.
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if burst is not None and burst <= 0:
            raise ValueError("burst must be positive")
        self._rate = rate
        self._burst = float(burst) if burst is not None else rate
        self._tokens = self._burst
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    # -- public API -----------------------------------------------------------

    async def acquire(self, tokens: float = 1.0) -> float:
        """Wait until *tokens* are available, consume them, and return wait time."""
        if tokens <= 0:
            return 0.0

        wait = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return wait
                deficit = tokens - self._tokens
                sleep_for = deficit / self._rate
            # Sleep OUTSIDE the lock so other waiters can keep refilling.
            await asyncio.sleep(sleep_for)
            wait += sleep_for

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking attempt to consume *tokens*."""
        if tokens <= 0:
            return True
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def is_rate_limited(self) -> bool:
        return self.available_tokens < 1.0

    # -- internal helpers -----------------------------------------------------

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now


class ConcurrencyLimiter:
    """Limit the number of concurrently running coroutines.

    Wraps an ``asyncio.Semaphore`` so that callers can use ``await limiter.run(coro)``
    and the slot is automatically released when the coroutine finishes.
    """

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # Tracked counter avoids reading the private ``_value`` attribute
        # of the underlying semaphore, which is not a stable API.
        self._available = max_concurrent

    async def run(self, coro):
        """Acquire a slot, await *coro*, release the slot, and return the result."""
        async with self._semaphore:
            self._available -= 1
            try:
                return await coro
            finally:
                self._available += 1

    @property
    def available(self) -> int:
        return self._available


class RateLimiterRegistry:
    """Per-account :class:`TokenBucket` registry.

    Each account gets its own rate limiter.  The limiter is created lazily on
    first access and cached.  Uses ``setdefault`` so concurrent first-access
    for the same ``account_id`` cannot create two limiters.
    """

    def __init__(self) -> None:
        self._limiters: dict[str, TokenBucket] = {}

    def get_limiter(self, account_id: str, rpm: int = 60) -> TokenBucket:
        """Return (or create) a token-bucket limiter for *account_id*."""
        rate = rpm / 60.0
        return self._limiters.setdefault(
            account_id, TokenBucket(rate=rate, burst=int(rpm))
        )

    def remove_limiter(self, account_id: str) -> None:
        """Remove the limiter for *account_id* if it exists."""
        self._limiters.pop(account_id, None)
