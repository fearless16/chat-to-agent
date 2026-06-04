"""Tests for the Task model."""

from datetime import datetime, timezone

import pytest

from ai_orchestrator.models.task import Task, TaskPriority, TaskStatus, TaskType


class TestTaskModel:
    """Task creation, transitions, and failure handling."""

    def test_initial_state(self):
        """Task starts IDLE with NORMAL priority."""
        task = Task(prompt="test the system")
        assert task.status == TaskStatus.IDLE
        assert task.priority == TaskPriority.NORMAL
        assert task.type == TaskType.INTERACTIVE
        assert task.is_terminal is False

    def test_transition_to_updates_status_and_timestamp(self):
        """transition_to changes status and updates timestamp."""
        task = Task(prompt="hello")
        before = task.updated_at
        task.transition_to(TaskStatus.PLANNING)
        assert task.status == TaskStatus.PLANNING
        assert task.updated_at >= before

    def test_transition_to_terminal_sets_completed_at(self):
        """DONE state sets completed_at."""
        task = Task(prompt="hello")
        task.transition_to(TaskStatus.DONE)
        assert task.completed_at is not None
        assert task.is_terminal is True

    def test_failed_task_goes_to_failed_on_first_retry(self):
        """First failure moves to FAILED, not DLQ."""
        task = Task(prompt="hello", max_retries=3)
        task.mark_failed("rate limited")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 1
        assert task.is_terminal is False

    def test_failed_task_goes_to_dlq_after_max_retries(self):
        """After max retries, task goes to DLQ."""
        task = Task(prompt="hello", max_retries=2)
        task.mark_failed("error 1")
        task.mark_failed("error 2")
        task.mark_failed("error 3")
        assert task.status == TaskStatus.DLQ
        assert task.retry_count == 3
        assert task.is_terminal is True

    def test_task_priority_values(self):
        """TaskPriority enum has correct ordering."""
        assert TaskPriority.CRITICAL.value < TaskPriority.NORMAL.value
        assert TaskPriority.NORMAL.value < TaskPriority.LOW.value
        assert TaskPriority.LOW.value < TaskPriority.BACKGROUND.value

    def test_task_halted_is_terminal(self):
        """HALTED tasks are terminal."""
        task = Task(prompt="hello")
        task.transition_to(TaskStatus.HALTED)
        assert task.is_terminal is True

    def test_task_id_is_unique(self):
        """Each task gets a unique ID."""
        t1 = Task(prompt="task 1")
        t2 = Task(prompt="task 2")
        assert t1.id != t2.id

    def test_task_serialization(self):
        """Task model dumps and restores from dict."""
        task = Task(prompt="test", priority=TaskPriority.HIGH, type=TaskType.BATCH)
        data = task.model_dump()
        restored = Task.model_validate(data)
        assert restored.prompt == "test"
        assert restored.priority == TaskPriority.HIGH
        assert restored.type == TaskType.BATCH
