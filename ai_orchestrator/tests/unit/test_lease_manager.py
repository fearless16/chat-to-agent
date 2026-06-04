"""Tests for LeaseManager — account pool management and lease lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ai_orchestrator.models.account import Account, AccountState
from ai_orchestrator.models.lease import Lease, LeaseState
from ai_orchestrator.orchestrator.lease_manager import LeaseManager, NoAvailableAccount


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager() -> LeaseManager:
    return LeaseManager()


@pytest.fixture
def idle_account() -> Account:
    return Account(id="test:acct-1", provider="openai", health_score=0.9)


@pytest.fixture
def warmup_account() -> Account:
    return Account(
        id="test:acct-warmup", provider="openai",
        state=AccountState.WARMUP, health_score=0.8,
    )


@pytest.fixture
def two_provider_accounts() -> list[Account]:
    return [
        Account(id="openai:acct-1", provider="openai", health_score=0.95),
        Account(id="openai:acct-2", provider="openai", health_score=0.85),
        Account(id="deepseek:acct-1", provider="deepseek", health_score=0.90),
        Account(id="deepseek:acct-2", provider="deepseek", health_score=0.70),
        Account(id="qwen:acct-1", provider="qwen", health_score=0.80),
    ]


# ===================================================================
# Registration
# ===================================================================

class TestRegistration:
    """register_account / register_accounts / get_account."""

    def test_register_and_retrieve(self, manager: LeaseManager, idle_account: Account):
        manager.register_account(idle_account)
        retrieved = manager.get_account("test:acct-1")
        assert retrieved is not None
        assert retrieved.id == "test:acct-1"
        assert retrieved.provider == "openai"

    def test_register_nonexistent_returns_none(self, manager: LeaseManager):
        assert manager.get_account("does-not-exist") is None

    def test_register_overwrites_duplicate(self, manager: LeaseManager):
        a1 = Account(id="acct-1", provider="openai", health_score=0.5)
        a2 = Account(id="acct-1", provider="deepseek", health_score=0.9)
        manager.register_account(a1)
        manager.register_account(a2)
        retrieved = manager.get_account("acct-1")
        assert retrieved is not None
        assert retrieved.provider == "deepseek"
        assert retrieved.health_score == 0.9

    def test_register_multiple(self, manager: LeaseManager):
        accounts = [
            Account(id="a", provider="openai"),
            Account(id="b", provider="deepseek"),
            Account(id="c", provider="qwen"),
        ]
        manager.register_accounts(accounts)
        assert manager.get_account("a") is not None
        assert manager.get_account("b") is not None
        assert manager.get_account("c") is not None

    def test_register_empty_list(self, manager: LeaseManager):
        manager.register_accounts([])
        assert manager.list_accounts() == []


# ===================================================================
# Listing and filtering
# ===================================================================

class TestListAccounts:
    """list_accounts with optional provider/state filters."""

    def setup_method(self):
        self.manager = LeaseManager()
        self.accounts = [
            Account(id="openai:1", provider="openai", state=AccountState.IDLE),
            Account(id="openai:2", provider="openai", state=AccountState.ACTIVE),
            Account(id="openai:3", provider="openai", state=AccountState.JAIL),
            Account(id="deepseek:1", provider="deepseek", state=AccountState.IDLE),
            Account(id="deepseek:2", provider="deepseek", state=AccountState.COOLDOWN),
        ]
        self.manager.register_accounts(self.accounts)

    def test_list_all(self):
        assert len(self.manager.list_accounts()) == 5

    def test_filter_by_provider(self):
        result = self.manager.list_accounts(provider="openai")
        assert len(result) == 3
        assert all(a.provider == "openai" for a in result)

    def test_filter_by_state(self):
        result = self.manager.list_accounts(state=AccountState.IDLE)
        assert len(result) == 2
        assert all(a.state == AccountState.IDLE for a in result)

    def test_filter_by_provider_and_state(self):
        result = self.manager.list_accounts(
            provider="openai", state=AccountState.ACTIVE
        )
        assert len(result) == 1
        assert result[0].id == "openai:2"

    def test_filter_no_match(self):
        assert self.manager.list_accounts(provider="nonexistent") == []
        assert self.manager.list_accounts(state=AccountState.WARMUP) == []

    def test_list_empty_manager(self, manager: LeaseManager):
        assert manager.list_accounts() == []


# ===================================================================
# Lease lifecycle: request → activate → heartbeat → release
# ===================================================================

class TestLeaseLifecycle:
    """request_lease, release_lease, heartbeat — the happy path."""

    def setup_method(self):
        self.manager = LeaseManager()
        self.manager.register_accounts([
            Account(id="openai:1", provider="openai", health_score=0.9),
            Account(id="openai:2", provider="openai", health_score=0.7),
            Account(id="deepseek:1", provider="deepseek", health_score=0.8),
        ])

    def test_request_lease_selects_highest_health(self):
        """Best available account (highest health_score) is selected."""
        lease = self.manager.request_lease(
            task_id="task-1", agent_id="agent-1"
        )
        assert lease.state == LeaseState.ACTIVE
        assert lease.task_id == "task-1"
        assert lease.agent_id == "agent-1"
        # openai:1 has health 0.9 > deepseek:1 health 0.8
        assert lease.account_id == "openai:1"

    def test_request_lease_marks_account_active(self):
        lease = self.manager.request_lease(
            task_id="task-1", agent_id="agent-1"
        )
        account = self.manager.get_account(lease.account_id)
        assert account is not None
        assert account.state == AccountState.ACTIVE

    def test_request_lease_preferred_provider(self):
        """When preferred_provider is given, select best from that provider."""
        lease = self.manager.request_lease(
            task_id="task-2", agent_id="agent-2",
            preferred_provider="deepseek",
        )
        assert lease.account_id == "deepseek:1"

    def test_request_lease_preferred_fallback(self):
        """If preferred_provider has no available, fall back to any provider."""
        # Mark all deepseek accounts unavailable
        ds = self.manager.get_account("deepseek:1")
        ds.mark_jail()

        lease = self.manager.request_lease(
            task_id="task-3", agent_id="agent-3",
            preferred_provider="deepseek",
        )
        # Should fall back to openai:1 (highest health overall)
        assert lease.account_id == "openai:1"

    def test_release_lease_returns_account_idle(self):
        lease = self.manager.request_lease(
            task_id="task-1", agent_id="agent-1"
        )
        account = self.manager.release_lease(lease.id)
        assert account is not None
        assert account.id == lease.account_id
        assert account.state == AccountState.IDLE

        # Lease state should be RELEASED
        stored = self.manager.get_lease(lease.id)
        assert stored is not None
        assert stored.state == LeaseState.RELEASED

    def test_release_lease_unknown_returns_none(self, manager: LeaseManager):
        assert manager.release_lease("nonexistent") is None

    def test_heartbeat_alive_lease(self):
        lease = self.manager.request_lease(
            task_id="task-1", agent_id="agent-1"
        )
        original_expiry = lease.expires_at
        # Move expiry close so heartbeat extends
        lease.expires_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        result = self.manager.heartbeat(lease.id)
        assert result is True
        assert lease.heartbeat_at is not None
        assert lease.expires_at > original_expiry

    def test_heartbeat_unknown_lease(self, manager: LeaseManager):
        assert manager.heartbeat("nonexistent") is False

    def test_heartbeat_expired_lease(self):
        lease = self.manager.request_lease(
            task_id="task-1", agent_id="agent-1"
        )
        # Force expiry
        lease.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        assert self.manager.heartbeat(lease.id) is False

    def test_get_active_leases(self):
        lease1 = self.manager.request_lease(task_id="t1", agent_id="a1")
        lease2 = self.manager.request_lease(task_id="t2", agent_id="a2")
        active = self.manager.get_active_leases()
        assert len(active) == 2
        assert all(l.state == LeaseState.ACTIVE for l in active)

    def test_get_lease_unknown(self, manager: LeaseManager):
        assert manager.get_lease("nonexistent") is None


# ===================================================================
# Expiry and reclamation
# ===================================================================

class TestReclaimExpired:
    """reclaim_expired detects and handles expired leases."""

    def setup_method(self):
        self.manager = LeaseManager()
        self.manager.register_accounts([
            Account(id="openai:1", provider="openai", health_score=0.9),
            Account(id="openai:2", provider="openai", health_score=0.8),
        ])

    def test_reclaim_expired_leases(self):
        lease1 = self.manager.request_lease(task_id="t1", agent_id="a1")
        lease2 = self.manager.request_lease(task_id="t2", agent_id="a2")

        # Manually expire lease1
        lease1.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)

        reclaimed = self.manager.reclaim_expired()
        assert len(reclaimed) == 1
        assert reclaimed[0] == lease1.id

        # lease1 should be EXPIRED
        assert lease1.state == LeaseState.EXPIRED
        # Account should be in COOLDOWN
        acct1 = self.manager.get_account("openai:1")
        assert acct1 is not None
        assert acct1.state == AccountState.COOLDOWN

        # lease2 should still be ACTIVE
        assert lease2.state == LeaseState.ACTIVE

    def test_reclaim_no_expired(self):
        self.manager.request_lease(task_id="t1", agent_id="a1")
        self.manager.request_lease(task_id="t2", agent_id="a2")
        reclaimed = self.manager.reclaim_expired()
        assert reclaimed == []

    def test_reclaim_empty_manager(self, manager: LeaseManager):
        assert manager.reclaim_expired() == []

    def test_reclaim_after_release_ignores_released(self):
        lease = self.manager.request_lease(task_id="t1", agent_id="a1")
        self.manager.release_lease(lease.id)
        reclaimed = self.manager.reclaim_expired()
        assert reclaimed == []


# ===================================================================
# mark_account_unavailable
# ===================================================================

class TestMarkUnavailable:
    """mark_account_unavailable sets account state."""

    def setup_method(self):
        self.manager = LeaseManager()
        self.manager.register_account(
            Account(id="acct-1", provider="openai")
        )

    def test_mark_jail_default(self):
        self.manager.mark_account_unavailable("acct-1")
        acct = self.manager.get_account("acct-1")
        assert acct is not None
        assert acct.state == AccountState.JAIL

    def test_mark_with_custom_state(self):
        self.manager.mark_account_unavailable(
            "acct-1", state=AccountState.COOLDOWN
        )
        acct = self.manager.get_account("acct-1")
        assert acct is not None
        assert acct.state == AccountState.COOLDOWN

    def test_mark_nonexistent_account(self):
        # Should not raise
        self.manager.mark_account_unavailable("does-not-exist")


# ===================================================================
# Pool stats
# ===================================================================

class TestPoolStats:
    """get_pool_stats returns correct counts."""

    def test_empty_stats(self, manager: LeaseManager):
        assert manager.get_pool_stats() == {}

    def test_stats_counts(self):
        manager = LeaseManager()
        manager.register_accounts([
            Account(id="o1", provider="openai", state=AccountState.IDLE),
            Account(id="o2", provider="openai", state=AccountState.ACTIVE),
            Account(id="o3", provider="openai", state=AccountState.JAIL),
            Account(id="d1", provider="deepseek", state=AccountState.IDLE),
            Account(id="d2", provider="deepseek", state=AccountState.WARMUP),
            Account(id="d3", provider="deepseek", state=AccountState.COOLDOWN),
        ])
        stats = manager.get_pool_stats()
        assert stats == {
            "openai": {
                "total": 3, "idle": 1, "warmup": 0,
                "active": 1, "cooldown": 0, "jail": 1,
            },
            "deepseek": {
                "total": 3, "idle": 1, "warmup": 1,
                "active": 0, "cooldown": 1, "jail": 0,
            },
        }


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """No accounts, all busy, threading basics."""

    def test_request_lease_no_accounts(self, manager: LeaseManager):
        with pytest.raises(NoAvailableAccount):
            manager.request_lease(task_id="t1", agent_id="a1")

    def test_request_lease_all_busy(self, manager: LeaseManager):
        manager.register_accounts([
            Account(id="o1", provider="openai"),
            Account(id="o2", provider="deepseek"),
        ])
        # Use both accounts
        manager.request_lease(task_id="t1", agent_id="a1")
        manager.request_lease(task_id="t2", agent_id="a2")

        with pytest.raises(NoAvailableAccount):
            manager.request_lease(task_id="t3", agent_id="a3")

    def test_request_lease_all_jailed(self, manager: LeaseManager):
        manager.register_accounts([
            Account(id="o1", provider="openai", state=AccountState.JAIL),
        ])
        with pytest.raises(NoAvailableAccount):
            manager.request_lease(task_id="t1", agent_id="a1")

    def test_request_lease_all_in_cooldown(self, manager: LeaseManager):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        manager.register_accounts([
            Account(
                id="o1", provider="openai",
                state=AccountState.COOLDOWN, cooldown_until=future,
            ),
        ])
        with pytest.raises(NoAvailableAccount):
            manager.request_lease(task_id="t1", agent_id="a1")

    def test_full_lifecycle(self, manager: LeaseManager):
        """End-to-end: register → request → heartbeat → release → reclaim empty."""
        manager.register_account(
            Account(id="acct-1", provider="openai", health_score=0.9)
        )

        # Request
        lease = manager.request_lease(task_id="task-1", agent_id="agent-1")
        assert lease.state == LeaseState.ACTIVE

        # Heartbeat
        assert manager.heartbeat(lease.id) is True

        # Release
        acct = manager.release_lease(lease.id)
        assert acct is not None
        assert acct.state == AccountState.IDLE

        # Reclaim should find nothing (already released)
        assert manager.reclaim_expired() == []

        # Can request again
        lease2 = manager.request_lease(task_id="task-2", agent_id="agent-2")
        assert lease2.account_id == "acct-1"

    def test_preferred_provider_no_accounts(self, manager: LeaseManager):
        manager.register_account(
            Account(id="o1", provider="openai")
        )
        with pytest.raises(NoAvailableAccount):
            manager.request_lease(
                task_id="t1", agent_id="a1",
                preferred_provider="nonexistent",
            )

    def test_warmup_account_is_selectable(self, manager: LeaseManager, warmup_account: Account):
        """WARMUP accounts with is_available=True can be selected."""
        manager.register_account(warmup_account)
        lease = manager.request_lease(task_id="t1", agent_id="a1")
        assert lease.account_id == "test:acct-warmup"
        acct = manager.get_account("test:acct-warmup")
        assert acct is not None
        assert acct.state == AccountState.ACTIVE  # promoted on lease
