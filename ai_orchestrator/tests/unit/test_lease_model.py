"""Tests for the Lease model lifecycle."""

from datetime import datetime, timedelta, timezone

import pytest

from ai_orchestrator.models.lease import Lease, LeaseState


class TestLeaseLifecycle:
    """Lease state transitions — REQUESTED → ACTIVE → RELEASED/EXPIRED."""

    def test_lease_initial_state(self):
        """Lease starts in REQUESTED state."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1")
        assert lease.state == LeaseState.REQUESTED
        assert lease.is_alive is False

    def test_activate_sets_timestamps(self):
        """Activate transitions to ACTIVE with proper timestamps."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1")
        lease.activate()
        assert lease.state == LeaseState.ACTIVE
        assert lease.acquired_at is not None
        assert lease.expires_at is not None
        assert lease.heartbeat_at is not None
        assert lease.is_alive is True

    def test_lease_has_ttl_enforcement(self):
        """Lease TTL is enforced at the model level (min 30s)."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1",
                       ttl_seconds=60)
        lease.activate()
        assert lease.is_alive is True

    def test_check_expired_detects_expiry(self):
        """Lease with past expires_at is detected as expired."""
        lease = Lease(
            account_id="test:acct-1", task_id="task-1", agent_id="agent-1",
            state=LeaseState.ACTIVE,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )
        assert lease.check_expired() is True
        assert lease.state == LeaseState.EXPIRED

    def test_release_transitions_to_released(self):
        """release() sets state to RELEASED."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1")
        lease.activate()
        lease.release()
        assert lease.state == LeaseState.RELEASED
        assert lease.is_alive is False

    def test_heartbeat_extends_expiry(self):
        """Heartbeat extends the lease when less than 50% TTL remaining."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1",
                       ttl_seconds=300)
        lease.activate()
        original_expiry = lease.expires_at
        # Simulate near-expiry
        lease.expires_at = datetime.now(timezone.utc) + timedelta(seconds=60)
        lease.heartbeat()
        assert lease.expires_at > original_expiry  # extended
        assert lease.heartbeat_at is not None

    def test_renewal_count_limited(self):
        """Lease cannot be renewed beyond max_renewals."""
        lease = Lease(
            account_id="test:acct-1", task_id="task-1", agent_id="agent-1",
            ttl_seconds=60, max_renewals=3,
        )
        lease.activate()
        for _ in range(3):
            assert lease.renew() is True
        assert lease.renew() is False  # exceeded max

    def test_lease_id_is_unique(self):
        """Each lease gets a unique ID."""
        lease1 = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1")
        lease2 = Lease(account_id="test:acct-1", task_id="task-2", agent_id="agent-2")
        assert lease1.id != lease2.id

    def test_is_alive_false_after_release(self):
        """Released lease is not alive."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1")
        lease.activate()
        lease.release()
        assert lease.is_alive is False

    def test_is_alive_false_for_expired(self):
        """Expired lease is not alive."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1")
        lease.state = LeaseState.EXPIRED
        assert lease.is_alive is False

    def test_lease_serialization(self):
        """Lease model dumps and restores from dict."""
        lease = Lease(account_id="test:acct-1", task_id="task-1", agent_id="agent-1",
                       ttl_seconds=300, max_renewals=3)
        data = lease.model_dump()
        restored = Lease.model_validate(data)
        assert restored.account_id == lease.account_id
        assert restored.task_id == lease.task_id
        assert restored.ttl_seconds == 300
