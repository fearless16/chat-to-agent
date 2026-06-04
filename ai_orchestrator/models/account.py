"""Account model and account state machine."""

from __future__ import annotations

import enum
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field


class AccountState(str, enum.Enum):
    """States in the account lifecycle."""
    IDLE = "IDLE"
    WARMUP = "WARMUP"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"
    JAIL = "JAIL"


class ProviderKind(str, enum.Enum):
    """Provider transport type."""
    API = "API"
    BROWSER = "BROWSER"
    LOCAL = "LOCAL"


class Account(BaseModel):
    """An account entry in the provider account pool.

    Tracks state, health metrics, rate limits, and usage statistics
    for a single credential / session across a provider.
    """
    id: str = Field(description="Unique account identifier (e.g. 'openai:acct-001')")
    provider: str = Field(description="Provider name (e.g. 'chatgpt', 'qwen', 'deepseek')")
    provider_kind: ProviderKind = Field(default=ProviderKind.API)
    state: AccountState = Field(default=AccountState.IDLE)
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    consecutive_failures: int = Field(default=0, ge=0)
    total_calls: int = Field(default=0, ge=0)
    total_errors: int = Field(default=0, ge=0)
    rate_limit_rpm: int = Field(default=60, description="Max requests per minute")
    rate_limit_tpm: int = Field(default=100_000, description="Max tokens per minute")
    current_rate_usage: float = Field(default=0.0, ge=0.0, le=1.0)
    context_limit: int = Field(default=8_192, description="Max context window in tokens")
    avg_latency_ms: float = Field(default=0.0, ge=0.0)
    avg_latency_samples: int = Field(default=0, ge=0)
    last_used: Optional[datetime] = Field(default=None)
    cooldown_until: Optional[datetime] = Field(default=None)
    total_warmup_steps: int = Field(default=5, description="Steps needed to graduate WARMUP -> ACTIVE")
    warmup_steps_completed: int = Field(default=0, ge=0)
    proxy: Optional[str] = Field(default=None, description="Proxy address for this account")

    model_config = {"frozen": False, "use_enum_values": True}

    @property
    def is_available(self) -> bool:
        """True if the account can accept work right now."""
        if self.state in (AccountState.JAIL, AccountState.ACTIVE):
            return False
        if self.state == AccountState.COOLDOWN:
            if self.cooldown_until and datetime.now(timezone.utc) < self.cooldown_until:
                return False
            # Cooldown expired — treat as available
            return True
        return self.state in (AccountState.IDLE, AccountState.WARMUP)

    def record_success(self, latency_ms: float = 0.0) -> None:
        """Record a successful call and update health."""
        self.total_calls += 1
        self.consecutive_failures = 0
        self.health_score = min(1.0, self.health_score + 0.05)
        self._update_latency(latency_ms)
        if self.state == AccountState.WARMUP:
            self.warmup_steps_completed += 1
            if self.warmup_steps_completed >= self.total_warmup_steps:
                self.state = AccountState.ACTIVE

    def record_failure(self, latency_ms: float = 0.0) -> None:
        """Record a failure and potentially escalate state."""
        self.total_calls += 1
        self.total_errors += 1
        self.consecutive_failures += 1
        self.health_score = max(0.0, self.health_score - 0.15)
        self._update_latency(latency_ms)

        if self.consecutive_failures >= 5:
            self.state = AccountState.JAIL
        elif self.consecutive_failures >= 3:
            self._enter_cooldown(timedelta(minutes=5))

    def record_rate_limit(self) -> None:
        """Handle rate-limit event."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= 2:
            self._enter_cooldown(timedelta(minutes=2))

    def _enter_cooldown(self, duration: timedelta) -> None:
        """Move to COOLDOWN state for a specified duration."""
        self.state = AccountState.COOLDOWN
        self.cooldown_until = datetime.now(timezone.utc) + duration

    def _update_latency(self, latency_ms: float) -> None:
        """Running average for latency."""
        if latency_ms > 0:
            n = self.avg_latency_samples
            self.avg_latency_ms = (self.avg_latency_ms * n + latency_ms) / (n + 1)
            self.avg_latency_samples = n + 1

    def mark_idle(self) -> None:
        """Return account to idle pool."""
        self.state = AccountState.IDLE
        self.cooldown_until = None

    def mark_active(self) -> None:
        """Mark account as in-use."""
        self.state = AccountState.ACTIVE

    def mark_jail(self) -> None:
        """Permanently disable account."""
        self.state = AccountState.JAIL
