"""Task model — a unit of work flowing through the orchestration system."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, enum.Enum):
    """All states a task transitions through."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFICATION = "VERIFICATION"
    DONE = "DONE"
    FAILED = "FAILED"
    HALTED = "HALTED"
    DLQ = "DLQ"


class TaskPriority(int, enum.Enum):
    """Task priority levels for scheduling."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskType(str, enum.Enum):
    """Categories of tasks the orchestrator handles."""
    INTERACTIVE = "interactive"
    BATCH = "batch"
    BACKGROUND = "background"
    MAINTENANCE = "maintenance"


class Task(BaseModel):
    """A discrete unit of work flowing through the orchestration pipeline."""
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    status: TaskStatus = Field(default=TaskStatus.IDLE)
    type: TaskType = Field(default=TaskType.INTERACTIVE)
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    user_id: Optional[str] = Field(default=None)
    prompt: str = Field(default="")
    current_step: str = Field(default="")
    assigned_account_id: Optional[str] = Field(default=None)
    assigned_agent: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "use_enum_values": True}

    def transition_to(self, new_status: TaskStatus) -> None:
        """Transition task to a new state, updating timestamps.

        Truly terminal states (``DONE``, ``DLQ``) are absorbing: once a
        task reaches one, no further transitions are accepted.  HALTED is
        *paused* (not terminal in the strict sense) and the orchestrator
        resumes out of it via :meth:`WorkflowEngine.resume_task`.  Same-
        state transitions are idempotent (used by the retry loop when a
        task is already in DLQ).
        """
        if new_status == self.status:
            return
        if self.status in (TaskStatus.DONE, TaskStatus.DLQ):
            raise ValueError(
                f"task {self.id} is in terminal state {self.status}; "
                f"cannot transition to {new_status}"
            )
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
        if new_status in (TaskStatus.DONE, TaskStatus.DLQ):
            self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        """Record a failure, optionally routing to DLQ after max retries."""
        self.retry_count += 1
        self.error_message = error
        if self.retry_count >= self.max_retries:
            self.transition_to(TaskStatus.DLQ)
        else:
            self.transition_to(TaskStatus.FAILED)

    @property
    def is_terminal(self) -> bool:
        """True if the task has reached a terminal state."""
        return self.status in (TaskStatus.DONE, TaskStatus.DLQ, TaskStatus.HALTED)
