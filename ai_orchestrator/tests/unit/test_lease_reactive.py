"""Tests for the Reactive Lease Manager (V6 architecture).

Tests account event emission, force-expire, and workflow integration.
"""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest

from ai_orchestrator.models.account import Account, AccountState
from ai_orchestrator.models.lease import Lease, LeaseState
from ai_orchestrator.orchestrator.lease_manager import AccountEvent, LeaseManager, NoAvailableAccount
from ai_orchestrator.orchestrator.workflow_engine import WorkflowEngine, WorkflowState
from ai_orchestrator.models.task import Task, TaskStatus


class TestReactiveLeaseManager:
    def test_mark_account_unavailable_emits_event(self):
        """When an account is marked JAIL, an event is emitted."""
        lm = LeaseManager()
        lm.register_account(Account(id="acc-1", provider="test"))

        events = []
        lm.on_account_event(lambda e: events.append(e))

        # Acquire a lease first
        lease = lm.request_lease(task_id="task-1", agent_id="agent-1")
        assert lease is not None

        # Jail the account
        lm.mark_account_unavailable("acc-1", AccountState.JAIL)

        assert len(events) == 1
        assert events[0].account_id == "acc-1"
        assert events[0].new_state == AccountState.JAIL
        assert events[0].lease_id == lease.id

    def test_force_expire_leases(self):
        lm = LeaseManager()
        lm.register_account(Account(id="acc-1", provider="test"))
        lm.register_account(Account(id="acc-2", provider="test"))

        lease1 = lm.request_lease(task_id="task-1", agent_id="agent-1")
        lease2 = lm.request_lease(task_id="task-2", agent_id="agent-2", preferred_provider="test")

        # Force-expire all leases for acc-1
        expired = lm.force_expire_leases_for_account("acc-1")
        assert len(expired) == 1
        assert expired[0] == lease1.id

        # Verify lease1 is expired
        assert lm.get_lease(lease1.id).state == LeaseState.EXPIRED

    def test_account_jailed_force_expires(self):
        lm = LeaseManager()
        lm.register_account(Account(id="acc-1", provider="test"))
        lease = lm.request_lease(task_id="task-1", agent_id="agent-1")

        expired = lm.account_jailed("acc-1")
        assert len(expired) == 1
        assert expired[0] == lease.id
        assert lm.get_lease(lease.id).state == LeaseState.EXPIRED

    def test_multiple_event_handlers(self):
        lm = LeaseManager()
        lm.register_account(Account(id="acc-1", provider="test"))

        events = []
        lm.on_account_event(lambda e: events.append(f"handler1:{e.account_id}"))
        lm.on_account_event(lambda e: events.append(f"handler2:{e.account_id}"))

        lm.mark_account_unavailable("acc-1", AccountState.JAIL)

        assert len(events) == 2
        assert "handler1:acc-1" in events
        assert "handler2:acc-1" in events

    def test_no_event_for_same_state(self):
        """No event emitted when state doesn't change."""
        lm = LeaseManager()
        lm.register_account(Account(id="acc-1", provider="test"))

        events = []
        lm.on_account_event(lambda e: events.append(e))

        # Mark IDLE -> IDLE should not emit
        lm.mark_account_unavailable("acc-1", AccountState.IDLE)
        assert len(events) == 0

    def test_event_handler_exception_does_not_crash(self):
        lm = LeaseManager()
        lm.register_account(Account(id="acc-1", provider="test"))

        def crashing_handler(e):
            raise RuntimeError("handler crashed")

        lm.on_account_event(crashing_handler)

        # Should not raise
        lm.mark_account_unavailable("acc-1", AccountState.JAIL)


class TestWorkflowReplanFromLease:
    def test_workflow_replans_on_jail(self):
        engine = WorkflowEngine()
        lm = LeaseManager()
        lm.register_account(Account(id="acc-1", provider="test"))

        # Wire the engine to react to lease events
        event_holder = []

        def handle_event(event):
            event_holder.append(event)
            engine.handle_account_event(event)

        lm.on_account_event(handle_event)

        # Start a task
        task = Task(id="task-1", prompt="test", status=TaskStatus.EXECUTING)
        engine._task_states[task.id] = WorkflowState.EXECUTING

        # Jail the account
        lm.mark_account_unavailable("acc-1", AccountState.JAIL)

        # Engine should have replanned the task
        assert engine._task_states[task.id] == WorkflowState.PLANNING

    def test_workflow_does_not_replan_for_non_jail(self):
        engine = WorkflowEngine()
        task = Task(id="task-1", prompt="test", status=TaskStatus.EXECUTING)
        engine._task_states[task.id] = WorkflowState.EXECUTING

        # Non-JAIL event should not trigger replan
        event = AccountEvent(
            account_id="acc-1",
            provider="test",
            old_state=None,
            new_state=AccountState.COOLDOWN,
        )
        engine.handle_account_event(event)
        assert engine._task_states[task.id] == WorkflowState.EXECUTING
