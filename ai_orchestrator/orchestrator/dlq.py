"""Dead Letter Queue — handles persistently failed tasks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ai_orchestrator.models.task import Task, TaskStatus


class DLQEntry:
    """An entry in the dead letter queue."""

    def __init__(
        self,
        task: Task,
        error: str,
        provider: str = "",
        account_id: str = "",
        logs: Optional[list[dict]] = None,
    ) -> None:
        self.task_id = task.id
        self.prompt = task.prompt
        self.error = error
        self.provider = provider
        self.account_id = account_id
        self.retry_count = task.retry_count
        self.max_retries = task.max_retries
        self.logs = logs or []
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt[:200],
            "error": self.error,
            "provider": self.provider,
            "account_id": self.account_id,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "log_count": len(self.logs),
            "timestamp": self.timestamp.isoformat(),
        }


class DeadLetterQueue:
    """Manages the dead letter queue — stores, alerts, retries, archives.

    Tasks land here after exhausting their max retries (Task.mark_failed → DLQ).
    """

    def __init__(self, max_entries: int = 1_000) -> None:
        self._entries: list[DLQEntry] = []
        self._max_entries = max_entries
        self._alert_callbacks: list[callable] = []

    def push(self, task: Task, error: str, provider: str = "", account_id: str = "", logs: Optional[list[dict]] = None) -> DLQEntry:
        """Add a failed task to the DLQ."""
        entry = DLQEntry(task, error, provider, account_id, logs)
        self._entries.append(entry)
        # Trim oldest if over limit
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)
        # Fire alert callbacks
        for cb in self._alert_callbacks:
            try:
                cb(entry)
            except Exception:
                pass
        return entry

    def pop(self, task_id: str) -> Optional[DLQEntry]:
        """Remove and return a DLQ entry (for retry)."""
        for i, entry in enumerate(self._entries):
            if entry.task_id == task_id:
                return self._entries.pop(i)
        return None

    def list_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent DLQ entries."""
        return [e.to_dict() for e in self._entries[-limit:]]

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def register_alert(self, callback: callable) -> None:
        """Register a callback for DLQ alerts (e.g. to Slack, email)."""
        self._alert_callbacks.append(callback)
