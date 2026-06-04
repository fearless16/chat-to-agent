"""Tests for the Dead Letter Queue service."""

from ai_orchestrator.models.task import Task, TaskStatus
from ai_orchestrator.orchestrator.dlq import DeadLetterQueue, DLQEntry


class TestDLQEntry:
    def test_entry_from_failed_task(self):
        task = Task(prompt="test", max_retries=3)
        task.mark_failed("rate limited")
        entry = DLQEntry(task, "rate limited", provider="openai", account_id="acct-1")
        assert entry.task_id == task.id
        assert entry.provider == "openai"
        assert entry.retry_count == 1

    def test_to_dict_includes_metadata(self):
        task = Task(prompt="hello", max_retries=2)
        task.mark_failed("timeout")
        entry = DLQEntry(task, "timeout")
        d = entry.to_dict()
        assert d["task_id"] == task.id
        assert d["error"] == "timeout"
        assert "timestamp" in d

    def test_logs_are_stored(self):
        task = Task(prompt="test")
        logs = [{"step": "research", "status": "ok"}]
        entry = DLQEntry(task, "error", logs=logs)
        assert len(entry.logs) == 1


class TestDeadLetterQueue:
    def test_push_adds_entry(self):
        dlq = DeadLetterQueue()
        task = Task(prompt="test", max_retries=1)
        task.mark_failed("error")
        entry = dlq.push(task, "error")
        assert dlq.count() == 1
        assert entry.task_id == task.id

    def test_pop_removes_entry(self):
        dlq = DeadLetterQueue()
        task = Task(prompt="test", max_retries=1)
        task.mark_failed("error")
        dlq.push(task, "error")
        popped = dlq.pop(task.id)
        assert popped is not None
        assert dlq.count() == 0

    def test_pop_nonexistent_returns_none(self):
        dlq = DeadLetterQueue()
        assert dlq.pop("nonexistent") is None

    def test_list_entries(self):
        dlq = DeadLetterQueue()
        for i in range(3):
            task = Task(prompt=f"task {i}", max_retries=1)
            task.mark_failed(f"error {i}")
            dlq.push(task, f"error {i}")
        entries = dlq.list_entries()
        assert len(entries) == 3

    def test_alert_callback_fired(self):
        dlq = DeadLetterQueue()
        alerts = []
        dlq.register_alert(lambda e: alerts.append(e.task_id))
        task = Task(prompt="alert test")
        dlq.push(task, "boom")
        assert len(alerts) == 1

    def test_clear_empties_queue(self):
        dlq = DeadLetterQueue()
        task = Task(prompt="test")
        dlq.push(task, "gone")
        dlq.clear()
        assert dlq.count() == 0

    def test_max_entries_trims_old(self):
        dlq = DeadLetterQueue(max_entries=2)
        for i in range(3):
            task = Task(prompt=f"task {i}", max_retries=1)
            task.mark_failed(f"err {i}")
            dlq.push(task, f"err {i}")
        assert dlq.count() == 2
