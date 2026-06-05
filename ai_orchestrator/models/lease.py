"""Lease model — accounts are never assigned directly; leases are granted to agents."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field


class LeaseState(str, enum.Enum):
    """States in the lease lifecycle."""
    REQUESTED = "REQUESTED"
    ACTIVE = "ACTIVE"
    RENEWING = "RENEWING"
    EXPIRED = "EXPIRED"
    RELEASED = "RELEASED"


class Lease(BaseModel):
    """A lease grants an agent exclusive access to an account for a bounded period.

    Every agent must acquire a lease before making provider calls. This prevents
    two agents from using the same account concurrently.
    """
    id: str = Field(default_factory=lambda: f"lease-{uuid.uuid4().hex[:12]}")
    account_id: str
    task_id: str
    agent_id: str
    state: LeaseState = Field(default=LeaseState.REQUESTED)
    acquired_at: Optional[datetime] = Field(default=None)
    expires_at: Optional[datetime] = Field(default=None)
    heartbeat_at: Optional[datetime] = Field(default=None)
    ttl_seconds: int = Field(default=300, ge=30, le=3600)
    renewal_count: int = Field(default=0, ge=0)
    max_renewals: int = Field(default=5)

    model_config = {"frozen": False, "use_enum_values": True}

    def activate(self) -> None:
        """Activate a requested lease."""
        now = datetime.now(timezone.utc)
        self.state = LeaseState.ACTIVE
        self.acquired_at = now
        self.expires_at = now + timedelta(seconds=self.ttl_seconds)
        self.heartbeat_at = now

    def heartbeat(self) -> None:
        """Record a heartbeat, extending lease by half TTL."""
        now = datetime.now(timezone.utc)
        self.heartbeat_at = now
        remaining = (self.expires_at - now).total_seconds() if self.expires_at else 0
        if remaining < self.ttl_seconds * 0.5:
            self.expires_at = now + timedelta(seconds=self.ttl_seconds)

    def renew(self) -> bool:
        """Attempt to renew the lease. Returns True if renewal succeeded."""
        if self.renewal_count >= self.max_renewals:
            return False
        now = datetime.now(timezone.utc)
        self.state = LeaseState.RENEWING
        self.renewal_count += 1
        self.expires_at = now + timedelta(seconds=self.ttl_seconds)
        self.heartbeat_at = now
        self.state = LeaseState.ACTIVE
        return True

    def release(self) -> None:
        """Release the lease back to the pool."""
        self.state = LeaseState.RELEASED

    def check_expired(self) -> bool:
        """Check and mark as expired if past deadline."""
        if self.state == LeaseState.ACTIVE and self.expires_at:
            if datetime.now(timezone.utc) > self.expires_at:
                self.state = LeaseState.EXPIRED
                return True
        return False

    def expire(self) -> None:
        """Mark this lease as EXPIRED (idempotent).

        Use this method to transition a lease to the EXPIRED state via
        the lease's own API rather than mutating ``self.state`` directly,
        which would bypass any future invariant added here.
        """
        if self.state == LeaseState.ACTIVE:
            self.state = LeaseState.EXPIRED

    @property
    def is_alive(self) -> bool:
        """True if the lease is currently valid for use."""
        if self.state != LeaseState.ACTIVE:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True
