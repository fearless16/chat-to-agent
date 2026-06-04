"""Tests for the Utilities module — throttling, backoff, circuit breaker."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from ai_orchestrator.utils.backoff import CircuitBreaker, RetryConfig, retry_with_backoff
from ai_orchestrator.utils.throttle import ConcurrencyLimiter, RateLimiterRegistry, TokenBucket


# ── TokenBucket ──────────────────────────────────────────────────────────────


class TestTokenBucket:
    """Token bucket rate limiting — acquire, try_acquire, refill, rate limit detection."""

    async def test_burst_defaults_to_rate(self):
        """When burst is None, burst equals rate."""
        bucket = TokenBucket(rate=10.0)
        assert bucket._burst == 10.0

    async def test_available_tokens_starts_at_burst(self):
        """A freshly created bucket has burst tokens available."""
        bucket = TokenBucket(rate=5.0, burst=10)
        assert bucket.available_tokens == 10.0

    async def test_try_acquire_success(self):
        """try_acquire returns True when tokens are available."""
        bucket = TokenBucket(rate=10.0, burst=5)
        assert bucket.try_acquire(1.0) is True

    async def test_try_acquire_failure(self):
        """try_acquire returns False when insufficient tokens."""
        bucket = TokenBucket(rate=10.0, burst=1)
        assert bucket.try_acquire(1.0) is True  # consumes the 1 token
        # No tokens left, and with rate=10, refill is slow
        assert bucket.try_acquire(1.0) is False

    async def test_try_acquire_partial_tokens(self):
        """try_acquire works with fractional token amounts."""
        bucket = TokenBucket(rate=10.0, burst=5)
        assert bucket.try_acquire(0.5) is True
        assert bucket.available_tokens == pytest.approx(4.5, rel=0.01)

    async def test_acquire_waits_when_rate_limited(self):
        """acquire blocks until tokens are available, returning wait time."""
        bucket = TokenBucket(rate=100.0, burst=1)
        bucket.try_acquire(1.0)  # exhaust tokens
        wait = await bucket.acquire(1.0)
        assert wait >= 0.0
        assert bucket.available_tokens < 1.0  # just consumed

    async def test_acquire_returns_zero_when_tokens_available(self):
        """acquire returns 0 when tokens are already available."""
        bucket = TokenBucket(rate=100.0, burst=10)
        wait = await bucket.acquire(1.0)
        assert wait == 0.0
        assert bucket.available_tokens == pytest.approx(9.0, rel=0.01)

    async def test_is_rate_limited_true_when_exhausted(self):
        """is_rate_limited is True when available tokens < 1.0."""
        bucket = TokenBucket(rate=100.0, burst=0.5)
        assert bucket.is_rate_limited is True

    async def test_is_rate_limited_false_when_sufficient(self):
        """is_rate_limited is False when >= 1.0 tokens available."""
        bucket = TokenBucket(rate=100.0, burst=10)
        assert bucket.is_rate_limited is False

    async def test_available_tokens_refills_over_time(self):
        """available_tokens increases as time passes (refill)."""
        bucket = TokenBucket(rate=100.0, burst=10)
        bucket.try_acquire(10.0)  # exhaust
        assert bucket.available_tokens < 1.0
        await asyncio.sleep(0.05)  # ~5 tokens should refill
        assert bucket.available_tokens > 2.0

    async def test_available_tokens_caps_at_burst(self):
        """available_tokens never exceeds burst capacity."""
        bucket = TokenBucket(rate=100.0, burst=5)
        await asyncio.sleep(0.1)  # would refill 10 tokens, but burst is 5
        assert bucket.available_tokens <= 5.0


# ── ConcurrencyLimiter ──────────────────────────────────────────────────────


class TestConcurrencyLimiter:
    """Concurrency limiter — semaphore-based coroutine throttling."""

    async def test_available_starts_at_max(self):
        """Available slots equal max_concurrent on init."""
        limiter = ConcurrencyLimiter(3)
        assert limiter.available == 3

    async def test_run_reduces_and_restores_available(self):
        """run acquires a slot then releases it after coroutine completes."""
        limiter = ConcurrencyLimiter(2)
        assert limiter.available == 2

        async def dummy():
            return 42

        result = await limiter.run(dummy())
        assert result == 42
        assert limiter.available == 2

    async def test_run_blocks_when_full(self):
        """run waits when all slots are taken."""
        limiter = ConcurrencyLimiter(1)
        started = asyncio.Event()
        can_finish = asyncio.Event()

        async def slow_task():
            started.set()
            await can_finish.wait()
            return "done"

        # Start one task that holds the slot
        t1 = asyncio.create_task(limiter.run(slow_task()))
        await started.wait()
        assert limiter.available == 0

        # Trying another should block — verify with a short timeout
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.05):
                await limiter.run(asyncio.sleep(0))

        # Release the slot
        can_finish.set()
        await t1
        assert limiter.available == 1


# ── RateLimiterRegistry ─────────────────────────────────────────────────────


class TestRateLimiterRegistry:
    """RateLimiterRegistry — account-scoped token bucket management."""

    async def test_get_limiter_creates_new(self):
        """get_limiter creates a new TokenBucket for an unknown account."""
        registry = RateLimiterRegistry()
        limiter = registry.get_limiter("acct_1", rpm=60)
        assert isinstance(limiter, TokenBucket)
        assert limiter.available_tokens == 60

    async def test_get_limiter_reuses_existing(self):
        """get_limiter returns the same instance for the same account."""
        registry = RateLimiterRegistry()
        limiter1 = registry.get_limiter("acct_1")
        limiter2 = registry.get_limiter("acct_1")
        assert limiter1 is limiter2

    async def test_get_limiter_respects_rpm_conversion(self):
        """rpm is converted to rate (tokens per second)."""
        registry = RateLimiterRegistry()
        limiter = registry.get_limiter("acct_2", rpm=120)
        # 120 RPM = 2 tokens/second, burst = 120 tokens (RPM = max tokens)
        assert limiter._rate == pytest.approx(2.0, rel=0.01)
        assert limiter.available_tokens == pytest.approx(120.0, rel=0.01)

    async def test_get_limiter_default_rpm(self):
        """Default RPM is 60 — rate of 1 token/s."""
        registry = RateLimiterRegistry()
        limiter = registry.get_limiter("acct_3")
        assert limiter._rate == 1.0

    async def test_remove_limiter(self):
        """remove_limiter deletes the limiter for an account."""
        registry = RateLimiterRegistry()
        registry.get_limiter("acct_1")
        registry.remove_limiter("acct_1")
        # After removal, get_limiter creates fresh
        limiter = registry.get_limiter("acct_1")
        assert limiter.available_tokens == 60

    async def test_remove_limiter_nonexistent_no_error(self):
        """remove_limiter does not error on unknown account."""
        registry = RateLimiterRegistry()
        registry.remove_limiter("does_not_exist")  # should not raise


# ── RetryConfig ──────────────────────────────────────────────────────────────


class TestRetryConfig:
    """RetryConfig dataclass defaults and overrides."""

    async def test_defaults(self):
        """RetryConfig has sensible defaults."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_ms == 1_000
        assert config.max_delay_ms == 60_000
        assert config.jitter is True
        assert config.multiplier == 2.0

    async def test_custom_values(self):
        """RetryConfig accepts custom values."""
        config = RetryConfig(max_retries=5, base_delay_ms=500, multiplier=3.0)
        assert config.max_retries == 5
        assert config.base_delay_ms == 500
        assert config.multiplier == 3.0


# ── retry_with_backoff ───────────────────────────────────────────────────────


class TestRetryWithBackoff:
    """Retry logic — exponential backoff with jitter, on_retry callback."""

    async def test_success_no_retry(self):
        """A function that succeeds on first try is not retried."""
        func = AsyncMock(return_value="ok")
        result = await retry_with_backoff(func)
        assert result == "ok"
        func.assert_awaited_once()

    async def test_retries_on_failure_then_succeeds(self):
        """A function that fails twice then succeeds triggers retries."""
        func = AsyncMock(side_effect=[ValueError("fail1"), ValueError("fail2"), "ok"])
        result = await retry_with_backoff(func)
        assert result == "ok"
        assert func.await_count == 3

    async def test_exhausts_retries_and_raises(self):
        """When max_retries is exhausted, the last exception propagates."""
        func = AsyncMock(side_effect=ValueError("always fail"))
        with pytest.raises(ValueError, match="always fail"):
            await retry_with_backoff(func, config=RetryConfig(max_retries=2))
        assert func.await_count == 3  # initial + 2 retries

    async def test_on_retry_callback_invoked(self):
        """on_retry is called with exception, attempt number, and delay."""
        func = AsyncMock(side_effect=[ValueError("fail"), "ok"])
        callback = AsyncMock()
        await retry_with_backoff(func, config=RetryConfig(max_retries=1), on_retry=callback)
        callback.assert_awaited_once()
        args = callback.await_args
        assert isinstance(args[0][0], ValueError)
        assert args[0][1] == 1  # first retry attempt

    async def test_delay_respects_max_delay(self):
        """Delay is capped at max_delay_ms."""
        # base=1000, multiplier=10, max=1500
        config = RetryConfig(base_delay_ms=1000, multiplier=10.0, max_delay_ms=1500)
        func = AsyncMock(side_effect=[ValueError("fail"), "ok"])
        start = time.monotonic()
        await retry_with_backoff(func, config=config)
        elapsed = time.monotonic() - start
        # With jitter, the delay could be up to max, but should be at most ~max_delay_ms-ish
        assert elapsed < 5.0  # comfortably above 1.5s

    async def test_no_jitter_produces_exponential_delay(self):
        """Without jitter, delay = base * multiplier ** (attempt - 1)."""
        config = RetryConfig(jitter=False, base_delay_ms=100, multiplier=2.0, max_delay_ms=10_000)
        func = AsyncMock(side_effect=[ValueError("fail"), ValueError("fail"), "ok"])
        start = time.monotonic()
        await retry_with_backoff(func, config=config)
        elapsed = time.monotonic() - start
        # attempt 1: 100ms, attempt 2: 200ms  => total ~300ms
        assert 0.2 < elapsed < 0.8


# ── CircuitBreaker ──────────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Circuit breaker — state transitions: CLOSED → OPEN → HALF_OPEN → CLOSED."""

    async def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.is_open is False

    async def test_closed_to_open_after_failure_threshold(self):
        """After failure_threshold consecutive failures, state is OPEN."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.is_open is True

    async def test_open_rejects_calls_immediately(self):
        """When OPEN, call raises CircuitBreakerOpenError."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_ms=60_000)
        cb.record_failure()
        assert cb.state == "OPEN"

        with pytest.raises(CircuitBreaker.CircuitBreakerOpenError):
            await cb.call(AsyncMock(return_value="ok"))

    async def test_open_to_half_open_after_timeout(self):
        """After recovery_timeout_ms, OPEN transitions to HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_ms=50)
        cb.record_failure()  # now OPEN
        await asyncio.sleep(0.06)
        assert cb.state == "HALF_OPEN"

    async def test_half_open_success_transitions_to_closed(self):
        """A successful call in HALF_OPEN transitions back to CLOSED."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_ms=50)
        cb.record_failure()
        await asyncio.sleep(0.06)
        assert cb.state == "HALF_OPEN"

        func = AsyncMock(return_value="recovered")
        result = await cb.call(func)
        assert result == "recovered"
        assert cb.state == "CLOSED"

    async def test_half_open_failure_transitions_to_open(self):
        """A failed call in HALF_OPEN transitions back to OPEN."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_ms=50)
        # Record 2 failures → state should be "OPEN"
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"
        # Wait for recovery timeout to transition to half-open
        await asyncio.sleep(0.06)
        assert cb.state == "HALF_OPEN"

        func = AsyncMock(side_effect=ValueError("nope"))
        with pytest.raises(ValueError):
            await cb.call(func)
        assert cb.state == "OPEN"

    async def test_record_success_resets_failure_count(self):
        """record_success resets the consecutive failure counter."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == "CLOSED"  # only 1 consecutive failure since reset

    async def test_successful_call_in_closed_state(self):
        """A successful call in CLOSED state returns the result."""
        cb = CircuitBreaker()
        func = AsyncMock(return_value=42)
        result = await cb.call(func)
        assert result == 42

    async def test_half_open_limits_concurrent_calls(self):
        """HALF_OPEN only allows half_open_max_calls calls."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_ms=50, half_open_max_calls=1)
        cb.record_failure()
        await asyncio.sleep(0.06)

        # One call goes through (half-open probe)
        func1 = AsyncMock(return_value="ok")
        result1 = await cb.call(func1)
        assert result1 == "ok"

        # After success, circuit is CLOSED again, so subsequent calls work
        assert cb.state == "CLOSED"
