"""SQLAlchemy async models for the AI orchestration platform.

Provides declarative models (AccountModel, TaskModel, JournalEntry)
and CRUD helper functions that convert between Pydantic domain models
and SQLAlchemy ORM rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base

from ai_orchestrator.models.account import Account, AccountState, ProviderKind
from ai_orchestrator.models.task import Task, TaskPriority, TaskStatus, TaskType

Base = declarative_base()


# ── SQLAlchemy models ────────────────────────────────────────────────


class AccountModel(Base):
    """SQLAlchemy model mirroring :class:`ai_orchestrator.models.account.Account`."""

    __tablename__ = "accounts"

    id = Column(String(255), primary_key=True)
    provider = Column(String(128), nullable=False, default="")
    provider_kind = Column(String(32), nullable=False, default="API")
    state = Column(String(32), nullable=False, default="IDLE")
    health_score = Column(Float, nullable=False, default=1.0)
    consecutive_failures = Column(Integer, nullable=False, default=0)
    total_calls = Column(Integer, nullable=False, default=0)
    total_errors = Column(Integer, nullable=False, default=0)
    rate_limit_rpm = Column(Integer, nullable=False, default=60)
    rate_limit_tpm = Column(Integer, nullable=False, default=100_000)
    current_rate_usage = Column(Float, nullable=False, default=0.0)
    context_limit = Column(Integer, nullable=False, default=8_192)
    avg_latency_ms = Column(Float, nullable=False, default=0.0)
    avg_latency_samples = Column(Integer, nullable=False, default=0)
    last_used = Column(DateTime(timezone=True), nullable=True, default=None)
    cooldown_until = Column(DateTime(timezone=True), nullable=True, default=None)
    total_warmup_steps = Column(Integer, nullable=False, default=5)
    warmup_steps_completed = Column(Integer, nullable=False, default=0)
    proxy = Column(String(512), nullable=True, default=None)

    def __repr__(self) -> str:
        return f"<AccountModel id={self.id!r} provider={self.provider!r} state={self.state!r}>"


class TaskModel(Base):
    """SQLAlchemy model mirroring :class:`ai_orchestrator.models.task.Task`."""

    __tablename__ = "tasks"

    id = Column(String(255), primary_key=True)
    status = Column(String(32), nullable=False, default="IDLE")
    type = Column(String(32), nullable=False, default="interactive")
    priority = Column(Integer, nullable=False, default=2)
    user_id = Column(String(255), nullable=True, default=None)
    prompt = Column(Text, nullable=False, default="")
    current_step = Column(String(255), nullable=False, default="")
    assigned_account_id = Column(String(255), nullable=True, default=None)
    assigned_agent = Column(String(255), nullable=True, default=None)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True, default=None)
    error_message = Column(Text, nullable=True, default=None)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    metadata_ = Column("metadata", Text, nullable=True, default=None)

    def __repr__(self) -> str:
        return f"<TaskModel id={self.id!r} status={self.status!r}>"


class JournalEntry(Base):
    """SQLAlchemy model for execution journal entries."""

    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), nullable=False, index=True)
    agent = Column(String(255), nullable=False, default="")
    action = Column(String(255), nullable=False, default="")
    input = Column(Text, nullable=True, default=None)
    output = Column(Text, nullable=True, default=None)
    status = Column(String(32), nullable=False, default="")
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<JournalEntry id={self.id} task_id={self.task_id!r} action={self.action!r}>"


# ── Table management ─────────────────────────────────────────────────


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables defined on ``Base``.

    This is a synchronous operation run via the engine's ``run_sync``
    method so it works with async engines.
    """
    def _sync_create(conn):
        Base.metadata.create_all(conn)

    await engine.run_sync(_sync_create)


# ── Account CRUD ─────────────────────────────────────────────────────


async def save_account(session: AsyncSession, account: Account) -> bool:
    """Persist an :class:`Account` domain model to the database.

    Creates or updates the corresponding ``AccountModel`` row and
    commits the transaction. Returns ``True`` on success.

    Uses :meth:`AsyncSession.merge` so the second call for the same
    primary key is an UPSERT, not an ``IntegrityError`` from the
    primary-key collision on a plain ``session.add``.
    """
    try:
        model = AccountModel(
            id=account.id,
            provider=account.provider,
            provider_kind=account.provider_kind.value
            if isinstance(account.provider_kind, ProviderKind)
            else account.provider_kind,
            state=account.state.value
            if isinstance(account.state, AccountState)
            else account.state,
            health_score=account.health_score,
            consecutive_failures=account.consecutive_failures,
            total_calls=account.total_calls,
            total_errors=account.total_errors,
            rate_limit_rpm=account.rate_limit_rpm,
            rate_limit_tpm=account.rate_limit_tpm,
            current_rate_usage=account.current_rate_usage,
            context_limit=account.context_limit,
            avg_latency_ms=account.avg_latency_ms,
            avg_latency_samples=account.avg_latency_samples,
            last_used=account.last_used,
            cooldown_until=account.cooldown_until,
            total_warmup_steps=account.total_warmup_steps,
            warmup_steps_completed=account.warmup_steps_completed,
            proxy=account.proxy,
        )
        await session.merge(model)
        await session.commit()
        return True
    except Exception:
        await session.rollback()
        raise


async def load_account(session: AsyncSession, account_id: str) -> Account | None:
    """Load an :class:`Account` domain model by its *account_id*.

    Returns ``None`` if the account does not exist.
    """
    result = await session.execute(
        select(AccountModel).where(AccountModel.id == account_id)
    )
    row: AccountModel | None = result.scalar_one_or_none()
    if row is None:
        return None

    return Account(
        id=row.id,
        provider=row.provider,
        provider_kind=ProviderKind(row.provider_kind) if row.provider_kind else ProviderKind.API,
        state=AccountState(row.state) if row.state else AccountState.IDLE,
        health_score=row.health_score,
        consecutive_failures=row.consecutive_failures,
        total_calls=row.total_calls,
        total_errors=row.total_errors,
        rate_limit_rpm=row.rate_limit_rpm,
        rate_limit_tpm=row.rate_limit_tpm,
        current_rate_usage=row.current_rate_usage,
        context_limit=row.context_limit,
        avg_latency_ms=row.avg_latency_ms,
        avg_latency_samples=row.avg_latency_samples,
        last_used=row.last_used,
        cooldown_until=row.cooldown_until,
        total_warmup_steps=row.total_warmup_steps,
        warmup_steps_completed=row.warmup_steps_completed,
        proxy=row.proxy,
    )


# ── Task CRUD ────────────────────────────────────────────────────────


async def save_task(session: AsyncSession, task: Task) -> bool:
    """Persist a :class:`Task` domain model to the database.

    Creates or updates the corresponding ``TaskModel`` row and commits.
    Returns ``True`` on success.  Uses :meth:`AsyncSession.merge` for
    idempotent UPSERT semantics.
    """
    import json as _json

    try:
        # ``default=str`` keeps NaN / datetime values from blowing up
        # the JSON encoder (Postgres rejects ``NaN`` in JSONB columns).
        metadata_json = (
            _json.dumps(task.metadata, default=str) if task.metadata else None
        )
        model = TaskModel(
            id=task.id,
            status=task.status.value if isinstance(task.status, TaskStatus) else task.status,
            type=task.type.value if isinstance(task.type, TaskType) else task.type,
            priority=task.priority.value if isinstance(task.priority, TaskPriority) else task.priority,
            user_id=task.user_id,
            prompt=task.prompt,
            current_step=task.current_step,
            assigned_account_id=task.assigned_account_id,
            assigned_agent=task.assigned_agent,
            created_at=task.created_at,
            updated_at=task.updated_at,
            completed_at=task.completed_at,
            error_message=task.error_message,
            retry_count=task.retry_count,
            max_retries=task.max_retries,
            metadata_=metadata_json,
        )
        await session.merge(model)
        await session.commit()
        return True
    except Exception:
        await session.rollback()
        raise


async def load_task(session: AsyncSession, task_id: str) -> Task | None:
    """Load a :class:`Task` domain model by its *task_id*.

    Returns ``None`` if the task does not exist.
    """
    result = await session.execute(
        select(TaskModel).where(TaskModel.id == task_id)
    )
    row: TaskModel | None = result.scalar_one_or_none()
    if row is None:
        return None

    return Task(
        id=row.id,
        status=TaskStatus(row.status) if row.status else TaskStatus.IDLE,
        type=TaskType(row.type) if row.type else TaskType.INTERACTIVE,
        priority=TaskPriority(row.priority) if row.priority is not None else TaskPriority.NORMAL,
        user_id=row.user_id,
        prompt=row.prompt or "",
        current_step=row.current_step or "",
        assigned_account_id=row.assigned_account_id,
        assigned_agent=row.assigned_agent,
        created_at=row.created_at or datetime.now(timezone.utc),
        updated_at=row.updated_at or datetime.now(timezone.utc),
        completed_at=row.completed_at,
        error_message=row.error_message,
        retry_count=row.retry_count or 0,
        max_retries=row.max_retries or 3,
        metadata=__import__("json").loads(row.metadata_) if row.metadata_ else {},
    )


# ── Journal CRUD ─────────────────────────────────────────────────────


async def save_journal_entry(
    session: AsyncSession,
    task_id: str,
    agent: str,
    action: str,
    input: Any = None,
    output: Any = None,
    status: str = "",
) -> bool:
    """Persist a journal entry to the database.

    Parameters
    ----------
    session:
        Active async database session.
    task_id:
        The task this entry belongs to.
    agent:
        Agent identifier that performed the action.
    action:
        Action name (e.g. ``"llm_call"``, ``"tool_call"``).
    input:
        Serialized input payload (stored as JSON string).
    output:
        Serialized output payload (stored as JSON string).
    status:
        Entry status (e.g. ``"completed"``, ``"checkpoint"``).

    Returns ``True`` on success.
    """
    import json as _json

    try:
        serialized_input = _json.dumps(input) if not isinstance(input, (str, type(None))) else input
        serialized_output = _json.dumps(output) if not isinstance(output, (str, type(None))) else output

        entry = JournalEntry(
            task_id=task_id,
            agent=agent,
            action=action,
            input=serialized_input,
            output=serialized_output,
            status=status,
        )
        session.add(entry)
        await session.commit()
        return True
    except Exception:
        await session.rollback()
        raise
