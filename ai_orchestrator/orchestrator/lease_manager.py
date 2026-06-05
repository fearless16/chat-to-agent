"""LeaseManager — account pool management and lease lifecycle.

Manages a pool of provider accounts, grants leases to agents for exclusive
account access, handles heartbeat-based expiry, reclamation of expired
leases, and **reactive account events** that propagate account state
changes (e.g. JAIL) to force-expire leases and trigger workflow replan.

Reactive event flow (per V6 architecture):

    Account → JAIL → Lease Manager → Force Expire Lease → Workflow → REPLAN
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Optional

from ai_orchestrator.models.account import Account, AccountState
from ai_orchestrator.models.lease import Lease, LeaseState

log = logging.getLogger(__name__)


class NoAvailableAccount(Exception):
    """Raised when no account is available to grant a lease."""


@dataclass
class AccountEvent:
    """An event emitted when an account changes state."""
    account_id: str
    provider: str
    old_state: Optional[AccountState]
    new_state: AccountState
    lease_id: Optional[str] = None


AccountEventHandler = Callable[[AccountEvent], None]


class LeaseManager:
    """Manages account registration and the lease lifecycle.

    Thread-safe: all public mutating methods acquire ``self._lock``.
    Supports reactive event handlers that fire on account state changes.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._leases: dict[str, Lease] = {}
        self._lock = threading.Lock()
        self._event_handlers: list[AccountEventHandler] = []

    # ------------------------------------------------------------------
    # Account registration
    # ------------------------------------------------------------------

    def register_account(self, account: Account) -> None:
        """Register a single account into the pool.

        If an account with the same ``id`` already exists it is overwritten.
        """
        with self._lock:
            self._accounts[account.id] = account

    def register_accounts(self, accounts: list[Account]) -> None:
        """Register multiple accounts in a single batch."""
        with self._lock:
            for account in accounts:
                self._accounts[account.id] = account

    # ------------------------------------------------------------------
    # Account lookup / filtering
    # ------------------------------------------------------------------

    def get_account(self, account_id: str) -> Account | None:
        """Look up an account by id. Returns ``None`` if not found."""
        return self._accounts.get(account_id)

    def list_accounts(
        self,
        provider: str | None = None,
        state: AccountState | None = None,
    ) -> list[Account]:
        """List registered accounts, optionally filtered by provider and/or state.

        When both filters are given they are AND-ed.
        """
        accounts = list(self._accounts.values())

        if provider is not None:
            accounts = [a for a in accounts if a.provider == provider]
        if state is not None:
            accounts = [a for a in accounts if a.state == state]

        return accounts

    # ------------------------------------------------------------------
    # Lease lifecycle
    # ------------------------------------------------------------------

    def request_lease(
        self,
        task_id: str,
        agent_id: str,
        preferred_provider: str | None = None,
    ) -> Lease:
        """Request a lease for *task_id* / *agent_id*.

        The best available account (highest ``health_score`` among
        ``is_available`` accounts) is selected.  When *preferred_provider*
        is given the manager first tries accounts from that provider; if
        none are available it falls back to any provider.

        The selected account is marked ``ACTIVE`` and the lease transitions
        to ``ACTIVE`` immediately.

        Raises
        ------
        NoAvailableAccount
            If no account in the pool satisfies ``is_available``.
        """
        with self._lock:
            candidates = [
                a for a in self._accounts.values()
                if a.is_available
            ]

            if preferred_provider is not None:
                preferred = [a for a in candidates if a.provider == preferred_provider]
                if preferred:
                    candidates = preferred
                # else fall through to all-candidates

            if not candidates:
                raise NoAvailableAccount(
                    f"No available account for task={task_id} agent={agent_id}"
                )

            # Pick the healthiest candidate
            best = max(candidates, key=lambda a: a.health_score)

            # Create lease
            lease = Lease(
                account_id=best.id,
                task_id=task_id,
                agent_id=agent_id,
            )
            lease.activate()

            # Mark account in-use
            best.mark_active()

            self._leases[lease.id] = lease
            return lease

    def release_lease(self, lease_id: str) -> Account | None:
        """Release a lease and return the associated account to the idle pool.

        Returns the released ``Account``, or ``None`` if *lease_id* is unknown.
        """
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return None

            lease.release()

            # Return account to IDLE (or WARMUP if it was originally WARMUP;
            # we default to IDLE since the account was promoted to ACTIVE).
            account = self._accounts.get(lease.account_id)
            if account is not None:
                account.mark_idle()

            return account

    def heartbeat(self, lease_id: str) -> bool:
        """Record a heartbeat on the lease.

        Returns ``True`` if the lease was found and is still alive, ``False``
        if the lease does not exist or has already expired.

        The whole lookup + liveness + mutation sequence is performed under
        ``self._lock`` so that a concurrent ``reclaim_expired`` or
        ``release_lease`` cannot observe or modify the lease in an
        inconsistent state.
        """
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            if not lease.is_alive:
                return False
            lease.heartbeat()
            return True

    def reclaim_expired(self) -> list[str]:
        """Check all active leases, expire those past deadline.

        Expired leases transition to ``EXPIRED`` and their accounts are sent
        to ``COOLDOWN``.

        Returns the list of reclaimed (expired) lease ids.
        """
        reclaimed: list[str] = []
        with self._lock:
            for lease in list(self._leases.values()):
                if lease.state != LeaseState.ACTIVE:
                    continue
                if lease.check_expired():
                    reclaimed.append(lease.id)
                    # Move the account to COOLDOWN
                    account = self._accounts.get(lease.account_id)
                    if account is not None:
                        account._enter_cooldown(timedelta(minutes=5))
        return reclaimed

    # ------------------------------------------------------------------
    # Lease introspection
    # ------------------------------------------------------------------

    def get_lease(self, lease_id: str) -> Lease | None:
        """Look up a lease by id.  Returns ``None`` if not found."""
        return self._leases.get(lease_id)

    def get_active_leases(self) -> list[Lease]:
        """Return all leases currently in the ``ACTIVE`` state."""
        return [
            l for l in self._leases.values()
            if l.state == LeaseState.ACTIVE
        ]

    # ------------------------------------------------------------------
    # Account management
    # ------------------------------------------------------------------

    def mark_account_unavailable(
        self,
        account_id: str,
        state: AccountState = AccountState.JAIL,
    ) -> None:
        """Mark an account as unavailable by setting its state.

        If the account has an active lease that lease is expired as well
        (via the lease's own ``expire()`` API, not by mutating
        ``lease.state`` directly).  Emits an ``AccountEvent`` so the
        workflow engine can react (e.g. REPLAN).

        Silently returns if *account_id* is not found.
        """
        with self._lock:
            account = self._accounts.get(account_id)
            if account is None:
                return
            old_state = account.state
            account.state = state

            expired_leases = []
            for lease in self._leases.values():
                if lease.account_id == account_id and lease.state == LeaseState.ACTIVE:
                    lease.expire()
                    expired_leases.append(lease.id)

        # Emit outside the lock
        if old_state != state:
            event = AccountEvent(
                account_id=account_id,
                provider=account.provider,
                old_state=old_state,
                new_state=state,
                lease_id=expired_leases[0] if expired_leases else None,
            )
            self._emit(event)

    # ------------------------------------------------------------------
    # Reactive account events (V6 architecture)
    # ------------------------------------------------------------------

    def on_account_event(self, handler: AccountEventHandler) -> None:
        """Register a callback that fires on every account state change.

        Handlers receive an ``AccountEvent`` and are called *outside*
        the lock so they can safely call back into the manager.
        """
        self._event_handlers.append(handler)

    def _emit(self, event: AccountEvent) -> None:
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception:
                log.exception("Account event handler failed for %s", event.account_id)

    def force_expire_leases_for_account(self, account_id: str) -> list[str]:
        """Force-expire all active leases for a given account.

        Returns the list of expired lease ids.  Used by the reactive
        event system when an account enters JAIL.
        """
        expired: list[str] = []
        with self._lock:
            for lease in list(self._leases.values()):
                if lease.account_id == account_id and lease.state == LeaseState.ACTIVE:
                    lease.expire()
                    expired.append(lease.id)
                    log.info("Force-expired lease %s for account %s", lease.id, account_id)
        return expired

    def account_jailed(self, account_id: str) -> list[str]:
        """Handle an account entering JAIL state (reactive).

        Force-expires all active leases for the account so the workflow
        engine can replan.  Returns force-expired lease ids.
        """
        expired = self.force_expire_leases_for_account(account_id)
        if expired:
            log.warning(
                "Account %s entered JAIL — force-expired %d lease(s)",
                account_id, len(expired),
            )
        return expired

    # ------------------------------------------------------------------
    # Pool statistics
    # ------------------------------------------------------------------

    def get_pool_stats(self) -> dict[str, dict[str, int]]:
        """Return per-provider pool statistics.

        Returns
        -------
        dict
            ``{provider_name: {total, idle, warmup, active, cooldown, jail}}``
        """
        stats: dict[str, dict[str, int]] = {}
        for account in self._accounts.values():
            provider = account.provider
            if provider not in stats:
                stats[provider] = {
                    "total": 0,
                    "idle": 0,
                    "warmup": 0,
                    "active": 0,
                    "cooldown": 0,
                    "jail": 0,
                }
            stats[provider]["total"] += 1
            state = account.state
            if hasattr(state, "value"):
                state = state.value
            state_key = state.lower()
            if state_key in stats[provider]:
                stats[provider][state_key] += 1
        return stats
