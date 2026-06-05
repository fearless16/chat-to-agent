"""Execution journal backed by Redis streams.

Tracks every step of task execution in a Redis stream per task
(``journal:{task_id}``) plus a global stream (``journal:all``)
for cross-task observability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ai_orchestrator.storage.redis_client import RedisClient


class ExecutionJournal:
    """Records and retrieves execution steps for tasks using Redis streams.

    Each task gets its own stream (``journal:{task_id}``) so journal
    entries can be read per-task.  Every logged step is also pushed to
    a global stream (``journal:all``) for cross-task observability.

    Parameters
    ----------
    redis:
        An active :class:`RedisClient` instance.
    """

    #: Hard cap on entries read by bulk operations.  Replay and
    #: checkpoint-scan use this so a runaway stream cannot OOM the
    #: process.  Tweak via the constructor in tests if needed.
    _MAX_BULK_FETCH: int = 1_000_000

    def __init__(self, redis: RedisClient, max_bulk_fetch: int | None = None) -> None:
        self._redis = redis
        if max_bulk_fetch is not None:
            if max_bulk_fetch < 1:
                raise ValueError("max_bulk_fetch must be >= 1")
            self._MAX_BULK_FETCH = max_bulk_fetch

    # ── Public API ───────────────────────────────────────────────────

    async def log_step(
        self,
        task_id: str,
        agent: str,
        action: str,
        input: Any = None,
        output: Any = None,
        status: str = "",
    ) -> str:
        """Record one execution step in the task journal.

        The entry is pushed to ``journal:{task_id}`` (per-task stream)
        and to ``journal:all`` (global stream).

        Parameters
        ----------
        task_id:
            The task this step belongs to.
        agent:
            Identifier of the agent that performed the step.
        action:
            Action name (e.g. ``"llm_call"``, ``"tool_call"``).
        input:
            Input payload (dict or str).
        output:
            Output payload (dict or str).
        status:
            Step status (e.g. ``"completed"``, ``"checkpoint"``).

        Returns
        -------
        str
            The auto-generated Redis stream entry ID.
        """
        # Serialize non-string, non-None values to JSON so the stream
        # entry is always text-safe.
        serialized_input = (
            json.dumps(input) if not isinstance(input, (str, type(None))) else input
        )
        serialized_output = (
            json.dumps(output) if not isinstance(output, (str, type(None))) else output
        )

        entry: dict[str, Any] = {
            "task_id": task_id,
            "agent": agent,
            "action": action,
            "input": serialized_input,
            "output": serialized_output,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Push to both the per-task stream and the global stream.
        entry_id = await self._redis.push_to_stream(f"journal:{task_id}", entry)
        await self._redis.push_to_stream("journal:all", entry)
        return entry_id

    async def get_task_journal(
        self,
        task_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the most recent journal entries for *task_id*.

        Parameters
        ----------
        task_id:
            The task to fetch entries for.
        limit:
            Maximum number of entries to return (default 10).

        Returns
        -------
        list[dict]
            Each entry dict.  ``input`` and ``output`` fields that were
            JSON-serialised strings are deserialised back to Python
            objects.
        """
        entries = await self._redis.read_stream(f"journal:{task_id}", count=limit)
        return [_deserialize_entry(e) for e in entries]

    async def get_last_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """Return the most recent checkpoint entry for *task_id*.

        Scans the task stream for entries whose ``status`` is
        ``"checkpoint"`` and returns the last one (most recently
        pushed), or ``None`` if no checkpoint exists.

        Parameters
        ----------
        task_id:
            The task to search.
        """
        # Fetch all entries so we can find the last checkpoint.
        entries = await self._redis.read_stream(
            f"journal:{task_id}", count=self._MAX_BULK_FETCH
        )
        checkpoint: dict[str, Any] | None = None
        for e in entries:
            if e.get("status") == "checkpoint":
                checkpoint = e
        if checkpoint is not None:
            return _deserialize_entry(checkpoint)
        return None

    async def replay_task(self, task_id: str) -> list[dict[str, Any]]:
        """Return **all** journal entries for *task_id* in order.

        This is equivalent to ``get_task_journal`` without a limit.
        """
        entries = await self._redis.read_stream(
            f"journal:{task_id}", count=self._MAX_BULK_FETCH
        )
        return [_deserialize_entry(e) for e in entries]


# ── Helpers ──────────────────────────────────────────────────────────


def _deserialize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Deserialize JSON-string fields in a journal entry back to Python objects.

    The stream stores everything as strings; this helper attempts to
    parse ``input`` and ``output`` as JSON if they look like JSON
    strings.
    """
    result = dict(entry)
    for field in ("input", "output"):
        raw = result.get(field)
        if isinstance(raw, str) and raw:
            try:
                result[field] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass  # keep as-is if not valid JSON
    return result
