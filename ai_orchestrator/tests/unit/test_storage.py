"""Tests for the Storage layer — Redis, Postgres, and Execution Journal."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from ai_orchestrator.storage.redis_client import RedisClient
from ai_orchestrator.storage.journal import ExecutionJournal
from ai_orchestrator.models.account import Account, AccountState, ProviderKind
from ai_orchestrator.models.task import Task, TaskStatus, TaskType, TaskPriority


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def redis_client():
    """Return a fresh in-memory RedisClient for each test."""
    return RedisClient()


@pytest.fixture
def sample_account():
    """A standard account fixture."""
    return Account(
        id="test:acct-001",
        provider="openai",
        provider_kind=ProviderKind.API,
        state=AccountState.IDLE,
    )


@pytest.fixture
def sample_task():
    """A standard task fixture."""
    return Task(
        id="test:task-001",
        prompt="Write a poem about AI orchestration",
        type=TaskType.INTERACTIVE,
        priority=TaskPriority.NORMAL,
    )


# ═══════════════════════════════════════════════════════════════════════
# RedisClient Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRedisClientKeyOps:
    """Key-value operations: set, get, delete, TTL."""

    async def test_set_and_get_key(self, redis_client):
        """Setting a key then getting it returns the value."""
        await redis_client.set_key("greeting", "hello")
        assert await redis_client.get_key("greeting") == "hello"

    async def test_get_nonexistent_key(self, redis_client):
        """Getting a missing key returns None."""
        assert await redis_client.get_key("nope") is None

    async def test_delete_existing_key(self, redis_client):
        """Deleting an existing key returns True."""
        await redis_client.set_key("temp", "value")
        assert await redis_client.delete_key("temp") is True
        assert await redis_client.get_key("temp") is None

    async def test_delete_nonexistent_key(self, redis_client):
        """Deleting a missing key returns False."""
        assert await redis_client.delete_key("ghost") is False

    async def test_set_key_overwrites(self, redis_client):
        """Setting the same key twice overwrites the value."""
        await redis_client.set_key("key", "first")
        await redis_client.set_key("key", "second")
        assert await redis_client.get_key("key") == "second"

    async def test_set_key_with_ttl(self, redis_client):
        """A key set with TTL expires after the TTL passes."""
        await redis_client.set_key("ephemeral", "gone_soon", ttl=0.01)
        assert await redis_client.get_key("ephemeral") == "gone_soon"
        import asyncio
        await asyncio.sleep(0.02)
        assert await redis_client.get_key("ephemeral") is None

    async def test_delete_returns_false_for_expired_key(self, redis_client):
        """Deleting an expired key returns False."""
        await redis_client.set_key("temp", "value", ttl=0.01)
        import asyncio
        await asyncio.sleep(0.02)
        assert await redis_client.delete_key("temp") is False

    async def test_get_key_returns_none_after_delete(self, redis_client):
        """After deletion, the key is gone."""
        await redis_client.set_key("delete_me", "data")
        await redis_client.delete_key("delete_me")
        assert await redis_client.get_key("delete_me") is None

    async def test_set_key_with_none_ttl_is_persistent(self, redis_client):
        """Setting a key with ttl=None keeps it indefinitely."""
        await redis_client.set_key("persistent", "stays")
        assert await redis_client.get_key("persistent") == "stays"

    async def test_set_key_with_integer_value(self, redis_client):
        """Setting a non-string value serializes to string."""
        await redis_client.set_key("number", 42)
        assert await redis_client.get_key("number") == "42"

    async def test_set_key_with_dict_value(self, redis_client):
        """Setting a dict value serializes to JSON string."""
        data = {"key": "value", "num": 1}
        await redis_client.set_key("dict", data)
        result = await redis_client.get_key("dict")
        assert json.loads(result) == data

    async def test_close(self, redis_client):
        """Calling close is a no-op and does not raise."""
        await redis_client.close()


class TestRedisClientStreamOps:
    """Stream operations: push, read."""

    async def test_push_to_stream_returns_entry_id(self, redis_client):
        """Pushing to a stream returns a string entry ID."""
        entry_id = await redis_client.push_to_stream("mystream", {"event": "test"})
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    async def test_read_stream_returns_pushed_data(self, redis_client):
        """Reading a stream returns the data that was pushed."""
        data = {"event": "test", "value": 42}
        await redis_client.push_to_stream("mystream", data)
        entries = await redis_client.read_stream("mystream")
        assert len(entries) == 1
        assert entries[0]["event"] == "test"
        assert entries[0]["value"] == 42

    async def test_read_stream_multiple_entries(self, redis_client):
        """Multiple pushes return in FIFO order."""
        for i in range(5):
            await redis_client.push_to_stream("s", {"idx": i})
        entries = await redis_client.read_stream("s", count=10)
        assert len(entries) == 5
        assert [e["idx"] for e in entries] == [0, 1, 2, 3, 4]

    async def test_read_stream_respects_count(self, redis_client):
        """The count parameter limits how many entries are returned."""
        for i in range(10):
            await redis_client.push_to_stream("s", {"idx": i})
        entries = await redis_client.read_stream("s", count=3)
        assert len(entries) == 3

    async def test_read_stream_empty(self, redis_client):
        """Reading an empty stream returns an empty list."""
        entries = await redis_client.read_stream("new_stream")
        assert entries == []

    async def test_read_stream_separate_streams(self, redis_client):
        """Different streams are isolated."""
        await redis_client.push_to_stream("a", {"key": "a"})
        await redis_client.push_to_stream("b", {"key": "b"})
        a_entries = await redis_client.read_stream("a")
        b_entries = await redis_client.read_stream("b")
        assert len(a_entries) == 1
        assert len(b_entries) == 1
        assert a_entries[0]["key"] == "a"
        assert b_entries[0]["key"] == "b"


class TestRedisClientListOps:
    """List operations: push, pop, length."""

    async def test_push_and_pop_from_list(self, redis_client):
        """Pushing then popping returns the value in FIFO order."""
        await redis_client.push_to_list("mylist", "first")
        await redis_client.push_to_list("mylist", "second")
        assert await redis_client.pop_from_list("mylist") == "first"
        assert await redis_client.pop_from_list("mylist") == "second"

    async def test_pop_from_empty_list(self, redis_client):
        """Popping from an empty list returns None."""
        assert await redis_client.pop_from_list("empty") is None

    async def test_get_list_length(self, redis_client):
        """get_list_length reports the correct count."""
        assert await redis_client.get_list_length("lst") == 0
        await redis_client.push_to_list("lst", "a")
        await redis_client.push_to_list("lst", "b")
        assert await redis_client.get_list_length("lst") == 2
        await redis_client.pop_from_list("lst")
        assert await redis_client.get_list_length("lst") == 1

    async def test_push_to_list_serializes_dict(self, redis_client):
        """Pushing a dict serializes it to JSON string."""
        await redis_client.push_to_list("lst", {"nested": "data"})
        item = await redis_client.pop_from_list("lst")
        assert json.loads(item) == {"nested": "data"}

    async def test_pop_returns_string(self, redis_client):
        """pop_from_list always returns a string or None."""
        await redis_client.push_to_list("lst", "hello")
        result = await redis_client.pop_from_list("lst")
        assert isinstance(result, str)

    async def test_lists_are_independent(self, redis_client):
        """Different list keys are independent."""
        await redis_client.push_to_list("a", "from_a")
        await redis_client.push_to_list("b", "from_b")
        assert await redis_client.pop_from_list("a") == "from_a"
        assert await redis_client.pop_from_list("b") == "from_b"


# ═══════════════════════════════════════════════════════════════════════
# Postgres Model CRUD Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_session():
    """Return a mock AsyncSession for testing."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    session.rollback = AsyncMock()
    return session


class TestPostgresModels:
    """CRUD operations for AccountModel, TaskModel, JournalEntry."""

    async def test_create_tables(self, mocker):
        """create_tables calls Base.metadata.create_all."""
        mock_run_sync = AsyncMock()
        mocker.patch(
            "ai_orchestrator.storage.postgres_models.create_async_engine",
            return_value=MagicMock(run_sync=mock_run_sync),
        )
        from ai_orchestrator.storage.postgres_models import create_tables
        engine = MagicMock()
        await create_tables(engine)
        mock_run_sync.assert_called_once()

    async def test_save_account(self, mock_session, sample_account):
        """save_account adds the account to the session and commits."""
        from ai_orchestrator.storage.postgres_models import save_account
        result = await save_account(mock_session, sample_account)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert result is True

    async def test_save_account_adds_account_model(self, mock_session, sample_account):
        """save_account creates an AccountModel from the Account pydantic model."""
        from ai_orchestrator.storage.postgres_models import save_account, AccountModel
        await save_account(mock_session, sample_account)
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, AccountModel)
        assert added.id == "test:acct-001"
        assert added.provider == "openai"

    async def test_load_account_found(self, mock_session, sample_account):
        """load_account returns an Account when the row exists."""
        from ai_orchestrator.storage.postgres_models import load_account, AccountModel
        # Mock the scalar result
        mock_scalar = AsyncMock(return_value=AccountModel(id="test:acct-001", provider="openai"))
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = mock_scalar
        mock_session.execute = AsyncMock(return_value=mock_result)

        account = await load_account(mock_session, "test:acct-001")
        assert account is not None
        assert account.id == "test:acct-001"
        assert account.provider == "openai"
        mock_session.execute.assert_called_once()

    async def test_load_account_not_found(self, mock_session):
        """load_account returns None when the row does not exist."""
        from ai_orchestrator.storage.postgres_models import load_account
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        account = await load_account(mock_session, "nonexistent")
        assert account is None

    async def test_save_task(self, mock_session, sample_task):
        """save_task adds the task to the session and commits."""
        from ai_orchestrator.storage.postgres_models import save_task, TaskModel
        result = await save_task(mock_session, sample_task)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, TaskModel)
        assert added.id == "test:task-001"
        assert result is True

    async def test_load_task_found(self, mock_session, sample_task):
        """load_task returns a Task when the row exists."""
        from ai_orchestrator.storage.postgres_models import load_task, TaskModel
        mock_row = TaskModel(
            id="test:task-001",
            prompt="Test prompt",
            status="IDLE",
            type="interactive",
            priority=2,
        )
        mock_scalar = AsyncMock(return_value=mock_row)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = mock_scalar
        mock_session.execute = AsyncMock(return_value=mock_result)

        task = await load_task(mock_session, "test:task-001")
        assert task is not None
        assert task.id == "test:task-001"
        assert task.prompt == "Test prompt"

    async def test_load_task_not_found(self, mock_session):
        """load_task returns None when the row does not exist."""
        from ai_orchestrator.storage.postgres_models import load_task
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        task = await load_task(mock_session, "nonexistent")
        assert task is None

    async def test_save_journal_entry(self, mock_session):
        """save_journal_entry adds a JournalEntry and commits."""
        from ai_orchestrator.storage.postgres_models import save_journal_entry, JournalEntry
        result = await save_journal_entry(
            mock_session,
            task_id="test:task-001",
            agent="agent-1",
            action="llm_call",
            input="Hello",
            output="Hi there",
            status="completed",
        )
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, JournalEntry)
        assert added.task_id == "test:task-001"
        assert added.agent == "agent-1"
        assert result is True

    async def test_save_journal_entry_rollback_on_error(self, mock_session):
        """save_journal_entry rolls back on database error."""
        from ai_orchestrator.storage.postgres_models import save_journal_entry
        mock_session.commit.side_effect = Exception("DB error")

        with pytest.raises(Exception, match="DB error"):
            await save_journal_entry(
                mock_session,
                task_id="test:task-001",
                agent="agent-1",
                action="llm_call",
                input="Hello",
                output="Hi there",
                status="completed",
            )
        mock_session.rollback.assert_called_once()

    async def test_save_account_rollback_on_error(self, mock_session, sample_account):
        """save_account rolls back on database error."""
        from ai_orchestrator.storage.postgres_models import save_account
        mock_session.commit.side_effect = Exception("DB error")

        with pytest.raises(Exception, match="DB error"):
            await save_account(mock_session, sample_account)
        mock_session.rollback.assert_called_once()

    async def test_load_account_with_full_fields(self, mock_session):
        """load_account returns an Account with all fields populated."""
        from ai_orchestrator.storage.postgres_models import load_account, AccountModel
        now = datetime.now(timezone.utc)
        mock_row = AccountModel(
            id="test:acct-full",
            provider="deepseek",
            provider_kind="API",
            state="ACTIVE",
            health_score=0.85,
            consecutive_failures=2,
            total_calls=100,
            total_errors=5,
            rate_limit_rpm=120,
            rate_limit_tpm=200000,
            current_rate_usage=0.3,
            context_limit=16384,
            avg_latency_ms=250.0,
            avg_latency_samples=50,
            last_used=now,
            cooldown_until=None,
            total_warmup_steps=5,
            warmup_steps_completed=3,
            proxy="http://proxy:8080",
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_row)
        mock_session.execute = AsyncMock(return_value=mock_result)

        account = await load_account(mock_session, "test:acct-full")
        assert account.id == "test:acct-full"
        assert account.provider == "deepseek"
        assert account.state.value == "ACTIVE"
        assert account.health_score == 0.85
        assert account.rate_limit_rpm == 120
        assert account.avg_latency_ms == 250.0
        assert account.last_used == now
        assert account.warmup_steps_completed == 3


# ═══════════════════════════════════════════════════════════════════════
# ExecutionJournal Tests
# ═══════════════════════════════════════════════════════════════════════

class TestExecutionJournal:
    """Execution journal: log, query, checkpoint, replay."""

    async def test_log_step_returns_entry_id(self, redis_client):
        """log_step returns a string entry ID."""
        journal = ExecutionJournal(redis_client)
        entry_id = await journal.log_step(
            task_id="task-1",
            agent="agent-1",
            action="llm_call",
            input="Hello",
            output="Hi",
            status="completed",
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    async def test_log_step_appends_to_stream(self, redis_client):
        """log_step appends an entry to the task journal stream."""
        journal = ExecutionJournal(redis_client)
        await journal.log_step("task-1", "agent-1", "llm_call", "in", "out", "completed")

        entries = await journal.get_task_journal("task-1")
        assert len(entries) == 1
        assert entries[0]["task_id"] == "task-1"
        assert entries[0]["agent"] == "agent-1"
        assert entries[0]["action"] == "llm_call"
        assert entries[0]["input"] == "in"
        assert entries[0]["output"] == "out"
        assert entries[0]["status"] == "completed"

    async def test_get_task_journal_multiple_entries(self, redis_client):
        """get_task_journal returns multiple entries in order."""
        journal = ExecutionJournal(redis_client)
        for i in range(5):
            await journal.log_step("task-1", "agent-1", f"step_{i}", f"in_{i}", f"out_{i}", "completed")

        entries = await journal.get_task_journal("task-1")
        assert len(entries) == 5
        for i, entry in enumerate(entries):
            assert entry["action"] == f"step_{i}"

    async def test_get_task_journal_empty(self, redis_client):
        """get_task_journal returns empty list for unknown task."""
        journal = ExecutionJournal(redis_client)
        entries = await journal.get_task_journal("nonexistent")
        assert entries == []

    async def test_get_task_journal_respects_limit(self, redis_client):
        """get_task_journal respects the limit parameter."""
        journal = ExecutionJournal(redis_client)
        for i in range(20):
            await journal.log_step("task-1", "agent-1", f"step_{i}", "", "", "completed")

        entries = await journal.get_task_journal("task-1", limit=5)
        assert len(entries) == 5

    async def test_get_last_checkpoint_returns_none_when_no_checkpoint(self, redis_client):
        """get_last_checkpoint returns None if no checkpoint exists."""
        journal = ExecutionJournal(redis_client)
        await journal.log_step("task-1", "agent-1", "llm_call", "in", "out", "completed")

        checkpoint = await journal.get_last_checkpoint("task-1")
        assert checkpoint is None

    async def test_get_last_checkpoint_finds_checkpoint(self, redis_client):
        """get_last_checkpoint finds the most recent 'checkpoint' status entry."""
        journal = ExecutionJournal(redis_client)
        await journal.log_step("task-1", "agent-1", "step_1", "in1", "out1", "running")
        await journal.log_step("task-1", "agent-1", "step_2", "in2", "out2", "checkpoint")
        await journal.log_step("task-1", "agent-1", "step_3", "in3", "out3", "running")
        await journal.log_step("task-1", "agent-1", "step_4", "in4", "out4", "checkpoint")

        checkpoint = await journal.get_last_checkpoint("task-1")
        assert checkpoint is not None
        assert checkpoint["action"] == "step_4"
        assert checkpoint["status"] == "checkpoint"

    async def test_replay_task_returns_all_entries(self, redis_client):
        """replay_task returns all journal entries for a task."""
        journal = ExecutionJournal(redis_client)
        for i in range(3):
            await journal.log_step("task-1", "agent-1", f"step_{i}", f"in_{i}", f"out_{i}", "completed")

        replay = await journal.replay_task("task-1")
        assert len(replay) == 3
        assert [e["action"] for e in replay] == ["step_0", "step_1", "step_2"]

    async def test_replay_task_empty(self, redis_client):
        """replay_task returns empty list for unknown task."""
        journal = ExecutionJournal(redis_client)
        replay = await journal.replay_task("unknown")
        assert replay == []

    async def test_log_step_with_timestamp(self, redis_client):
        """log_step includes an ISO timestamp in the entry."""
        journal = ExecutionJournal(redis_client)
        entry_id = await journal.log_step("task-1", "agent-1", "call", "in", "out", "completed")
        entries = await journal.get_task_journal("task-1")
        assert "timestamp" in entries[0]

    async def test_journal_streams_are_independent(self, redis_client):
        """Different tasks have independent journal streams."""
        journal = ExecutionJournal(redis_client)
        await journal.log_step("task-a", "agent-1", "action", "in", "out", "completed")
        await journal.log_step("task-b", "agent-2", "action", "in", "out", "completed")

        entries_a = await journal.get_task_journal("task-a")
        entries_b = await journal.get_task_journal("task-b")
        assert len(entries_a) == 1
        assert len(entries_b) == 1

    async def test_log_step_with_dict_input_output(self, redis_client):
        """log_step accepts dict input/output and serializes them."""
        journal = ExecutionJournal(redis_client)
        await journal.log_step(
            "task-1", "agent-1", "analyze",
            {"prompt": "hello"},
            {"response": "world"},
            "completed",
        )
        entries = await journal.get_task_journal("task-1")
        assert entries[0]["input"] == {"prompt": "hello"}
        assert entries[0]["output"] == {"response": "world"}

    async def test_log_step_records_entry_on_global_stream(self, redis_client):
        """log_step also pushes to the global 'journal:all' stream."""
        journal = ExecutionJournal(redis_client)
        await journal.log_step("task-1", "agent-1", "call", "in", "out", "completed")
        global_entries = await redis_client.read_stream("journal:all")
        assert len(global_entries) == 1
        assert global_entries[0]["task_id"] == "task-1"
