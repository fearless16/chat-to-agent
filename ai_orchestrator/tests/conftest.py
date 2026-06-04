"""Shared pytest fixtures for the AI Orchestrator test suite."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ai_orchestrator.models.account import Account, AccountState, ProviderKind
from ai_orchestrator.models.lease import Lease
from ai_orchestrator.models.task import Task, TaskPriority, TaskStatus, TaskType
from ai_orchestrator.models.capabilities import TaskRequirements


@pytest.fixture
def sample_account() -> Account:
    """A standard IDLE account for testing."""
    return Account(
        id="openai:acct-001",
        provider="chatgpt",
        provider_kind=ProviderKind.API,
        context_limit=32_768,
        rate_limit_rpm=60,
        rate_limit_tpm=100_000,
        health_score=1.0,
    )


@pytest.fixture
def sample_warmup_account() -> Account:
    """An account in WARMUP state."""
    return Account(
        id="openai:acct-002",
        provider="chatgpt",
        state=AccountState.WARMUP,
        warmup_steps_completed=2,
        total_warmup_steps=5,
    )


@pytest.fixture
def sample_active_account() -> Account:
    """An account currently in use."""
    return Account(
        id="openai:acct-003",
        provider="chatgpt",
        state=AccountState.ACTIVE,
    )


@pytest.fixture
def sample_cooldown_account() -> Account:
    """An account in COOLDOWN."""
    future = datetime.now(timezone.utc).replace(year=2099)
    return Account(
        id="openai:acct-004",
        provider="chatgpt",
        state=AccountState.COOLDOWN,
        cooldown_until=future,
    )


@pytest.fixture
def sample_jail_account() -> Account:
    """A banned account."""
    return Account(
        id="openai:acct-005",
        provider="chatgpt",
        state=AccountState.JAIL,
    )


@pytest.fixture
def sample_lease() -> Lease:
    """A lease ready to be activated."""
    return Lease(
        account_id="openai:acct-001",
        task_id="task-001",
        agent_id="agent-001",
        ttl_seconds=300,
    )


@pytest.fixture
def sample_task() -> Task:
    """A standard interactive task."""
    return Task(
        id="task-001",
        prompt="Write a Python function to sort a list",
        priority=TaskPriority.NORMAL,
        type=TaskType.INTERACTIVE,
    )


@pytest.fixture
def sample_task_requirements() -> TaskRequirements:
    """Standard task requirements for provider matching."""
    return TaskRequirements(
        context_length=4_096,
        requires_reasoning=True,
        requires_coding=True,
        priority={"reasoning": 0.5, "coding": 0.5, "translation": 0.0, "multimodality": 0.0},
    )


@pytest.fixture
def account_pool() -> list[Account]:
    """A mixed pool of accounts for router/lease tests."""
    now = datetime.now(timezone.utc)
    return [
        Account(id="chatgpt:acct-01", provider="chatgpt", health_score=0.95,
                avg_latency_ms=120, current_rate_usage=0.3, context_limit=32_768),
        Account(id="chatgpt:acct-02", provider="chatgpt", health_score=0.70,
                avg_latency_ms=250, current_rate_usage=0.8, context_limit=32_768),
        Account(id="qwen:acct-01", provider="qwen", health_score=0.90,
                avg_latency_ms=180, current_rate_usage=0.2, context_limit=131_072),
        Account(id="deepseek:acct-01", provider="deepseek", health_score=0.99,
                avg_latency_ms=100, current_rate_usage=0.1, context_limit=1_000_000),
        Account(id="local:acct-01", provider="local_llm", health_score=0.80,
                avg_latency_ms=500, current_rate_usage=0.0, context_limit=256_000),
        Account(id="kimi:acct-01", provider="kimi", health_score=0.75,
                avg_latency_ms=200, current_rate_usage=0.4, context_limit=128_000),
    ]
