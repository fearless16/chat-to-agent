"""Tests for the Account model and state machine."""

from datetime import datetime, timedelta, timezone

import pytest

from ai_orchestrator.models.account import Account, AccountState, ProviderKind


class TestAccountStateMachine:
    """Account state transitions and health tracking."""

    def test_initial_state(self):
        """Account starts IDLE with full health."""
        acct = Account(id="test:acct-1", provider="openai")
        assert acct.state == AccountState.IDLE
        assert acct.health_score == 1.0
        assert acct.is_available is True

    def test_idle_account_is_available(self):
        """IDLE accounts can accept work."""
        acct = Account(id="test:acct-1", provider="openai")
        assert acct.is_available is True

    def test_warmup_account_is_available(self):
        """WARMUP accounts can accept work (trust-building phase)."""
        acct = Account(id="test:acct-1", provider="openai", state=AccountState.WARMUP)
        assert acct.is_available is True

    def test_active_account_is_not_available(self):
        """ACTIVE means already in use by another agent."""
        acct = Account(id="test:acct-1", provider="openai", state=AccountState.ACTIVE)
        assert acct.is_available is False

    def test_jail_account_is_not_available(self):
        """JAIL means permanently disabled."""
        acct = Account(id="test:acct-1", provider="openai", state=AccountState.JAIL)
        assert acct.is_available is False

    def test_cooldown_account_unavailable_until_time_passes(self):
        """COOLDOWN accounts are unavailable until cooldown expires."""
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        acct = Account(
            id="test:acct-1",
            provider="openai",
            state=AccountState.COOLDOWN,
            cooldown_until=future,
        )
        assert acct.is_available is False

    def test_cooldown_account_available_after_expiry(self):
        """COOLDOWN account returns to available once timeout passes."""
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        acct = Account(
            id="test:acct-1",
            provider="openai",
            state=AccountState.COOLDOWN,
            cooldown_until=past,
        )
        assert acct.is_available is True

    def test_successful_call_improves_health(self):
        """record_success increments warmup steps, resets failures, boosts health."""
        acct = Account(id="test:acct-1", provider="openai", state=AccountState.WARMUP)
        acct.record_success(latency_ms=100.0)
        assert acct.consecutive_failures == 0
        assert acct.warmup_steps_completed == 1
        assert acct.total_calls == 1

    def test_warmup_graduation(self):
        """After enough successful warmup calls, account becomes ACTIVE."""
        acct = Account(id="test:acct-1", provider="openai", state=AccountState.WARMUP)
        for _ in range(acct.total_warmup_steps):
            acct.record_success(latency_ms=50.0)
        assert acct.state == AccountState.ACTIVE

    def test_three_consecutive_failures_triggers_cooldown(self):
        """After 3 failures, account enters COOLDOWN for 5 minutes."""
        acct = Account(id="test:acct-1", provider="openai", state=AccountState.ACTIVE)
        for _ in range(3):
            acct.record_failure(latency_ms=200.0)
        assert acct.state == AccountState.COOLDOWN
        assert acct.cooldown_until is not None

    def test_rate_limit_triggers_cooldown_on_repeat(self):
        """Two rate-limit events in a row trigger COOLDOWN."""
        acct = Account(id="test:acct-1", provider="openai")
        acct.record_rate_limit()
        assert acct.state != AccountState.COOLDOWN  # first is still ok
        acct.record_rate_limit()
        assert acct.state == AccountState.COOLDOWN

    def test_mark_idle_resets_cooldown(self):
        """mark_idle returns to IDLE and clears cooldown."""
        acct = Account(id="test:acct-1", provider="openai", state=AccountState.COOLDOWN,
                       cooldown_until=datetime.now(timezone.utc) + timedelta(minutes=5))
        acct.mark_idle()
        assert acct.state == AccountState.IDLE
        assert acct.cooldown_until is None

    def test_avg_latency_tracking(self):
        """Latency running average computes correctly over multiple calls."""
        acct = Account(id="test:acct-1", provider="openai")
        acct.record_success(latency_ms=100.0)
        assert acct.avg_latency_ms == 100.0
        acct.record_success(latency_ms=200.0)
        assert acct.avg_latency_ms == 150.0
        acct.record_success(latency_ms=300.0)
        assert acct.avg_latency_ms == 200.0

    def test_health_score_decreases_on_failure(self):
        """Health score drops on failure, recovers on success."""
        acct = Account(id="test:acct-1", provider="openai")
        acct.record_failure()
        assert acct.health_score < 1.0
        score_after_failure = acct.health_score
        for _ in range(5):
            acct.record_success()
        assert acct.health_score > score_after_failure

    def test_jail_after_many_failures(self):
        """5+ consecutive failures escalate to JAIL."""
        acct = Account(id="test:acct-1", provider="openai")
        for _ in range(6):
            acct.record_failure()
        assert acct.state == AccountState.JAIL

    def test_mark_jail(self):
        """mark_jail explicitly disables an account."""
        acct = Account(id="test:acct-1", provider="openai")
        acct.mark_jail()
        assert acct.state == AccountState.JAIL
        assert acct.is_available is False

    def test_mark_active(self):
        """mark_active sets state to ACTIVE."""
        acct = Account(id="test:acct-1", provider="openai")
        acct.mark_active()
        assert acct.state == AccountState.ACTIVE

    def test_provider_kind_default(self):
        """Default provider kind is API."""
        acct = Account(id="test:acct-1", provider="openai")
        assert acct.provider_kind == ProviderKind.API

    def test_account_serialization(self):
        """Account model serializes to dict and back."""
        acct = Account(id="test:acct-1", provider="openai", context_limit=16384, rate_limit_rpm=120)
        data = acct.model_dump()
        restored = Account.model_validate(data)
        assert restored.id == acct.id
        assert restored.context_limit == 16384
        assert restored.rate_limit_rpm == 120
