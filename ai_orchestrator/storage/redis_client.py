"""In-memory Redis client for development and testing.

Provides the same async interface as a real Redis client but stores
everything in a plain Python dict. Supports key-value ops, streams,
and list queues.

All public mutating methods take a single :class:`asyncio.Lock` so that
concurrent coroutines see consistent state.  This is a mock — production
Redis operations are atomic on the server — but the API contract is
preserved.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any


class RedisClient:
    """Async-friendly in-memory Redis mock with key, stream, and list operations.

    Features:
    - Key-value store with optional TTL (seconds)
    - Streams with auto-generated entry IDs
    - Consumer groups with message acknowledgment
    - Pub/sub event publishing
    - List (queue) operations with FIFO semantics
    - JSON serialization for dict/list values
    - Coroutine-safe via a single :class:`asyncio.Lock`
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}  # key -> (value, expiry_ts)
        self._streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}  # stream -> [(id, data)]
        self._lists: dict[str, list[str]] = {}  # key -> [value, ...]
        self._consumer_groups: dict[tuple[str, str], set[str]] = {}  # (group, stream) -> {acked_ids}
        self._events: list[dict[str, Any]] = []  # published events log
        self._counter: int = 0
        self._lock = asyncio.Lock()

    # ── Key-value ops ────────────────────────────────────────────────

    async def set_key(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set *key* to *value* with an optional *ttl* in seconds.

        Non-string values are serialized via :func:`json.dumps`.
        """
        serialized = json.dumps(value) if not isinstance(value, str) else value
        expiry = time.time() + ttl if ttl is not None else None
        async with self._lock:
            self._data[key] = (serialized, expiry)

    async def get_key(self, key: str) -> str | None:
        """Return the value for *key*, or ``None`` if missing or expired."""
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if expiry is not None and time.time() > expiry:
                del self._data[key]
                return None
            return value

    async def delete_key(self, key: str) -> bool:
        """Delete *key*. Returns ``True`` if the key existed and was not expired."""
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            value, expiry = entry
            if expiry is not None and time.time() > expiry:
                del self._data[key]
                return False
            del self._data[key]
            return True

    # ── Stream ops ───────────────────────────────────────────────────

    async def push_to_stream(self, stream: str, data: dict[str, Any]) -> str:
        """Push *data* to *stream* and return the auto-generated entry ID."""
        async with self._lock:
            self._counter += 1
            entry_id = f"{int(time.time() * 1000)}-{self._counter}"
            self._streams.setdefault(stream, []).append((entry_id, data))
            return entry_id

    async def read_stream(
        self,
        stream: str,
        count: int = 10,
        block_ms: int = 0,
    ) -> list[dict[str, Any]]:
        """Read up to *count* entries from *stream*.

        Returns the data portion of each entry as a ``dict``.
        The *block_ms* parameter is accepted for interface compatibility
        but is ignored (non-blocking in-memory store always returns
        immediately).
        """
        async with self._lock:
            entries = self._streams.get(stream, [])
            return [data for _, data in entries[:count]]

    # ── List ops ─────────────────────────────────────────────────────

    async def push_to_list(self, key: str, value: Any) -> None:
        """Push *value* to the tail of the list at *key*.

        Non-string values are JSON-serialized.
        """
        serialized = json.dumps(value) if not isinstance(value, str) else value
        async with self._lock:
            self._lists.setdefault(key, []).append(serialized)

    async def pop_from_list(self, key: str) -> str | None:
        """Pop the head of the list at *key* and return it, or ``None``."""
        async with self._lock:
            queue = self._lists.get(key)
            if not queue:
                return None
            return queue.pop(0)

    async def get_list_length(self, key: str) -> int:
        """Return the number of items in the list at *key*."""
        async with self._lock:
            return len(self._lists.get(key, []))

    # ── Consumer group ops ───────────────────────────────────────────

    async def create_consumer_group(self, group: str, stream: str) -> bool:
        """Create a consumer group for *stream*.

        Returns ``True`` if the group was newly created, ``False`` if it
        already existed (idempotent).
        """
        key = (group, stream)
        async with self._lock:
            if key in self._consumer_groups:
                return False
            self._consumer_groups[key] = set()
            return True

    async def ack_message(self, group: str, stream: str, entry_id: str) -> bool:
        """Acknowledge a message in *group* for *stream*.

        Returns ``True`` if the group existed and the ack was recorded.
        """
        key = (group, stream)
        async with self._lock:
            group_set = self._consumer_groups.get(key)
            if group_set is None:
                return False
            group_set.add(entry_id)
            return True

    # ── Pub/sub ops ──────────────────────────────────────────────────

    async def publish_event(self, channel: str, data: dict[str, Any]) -> int:
        """Publish an event on *channel*.

        In this in-memory emulator the event is simply appended to an
        internal log. Returns the number of subscribers (0 in the
        in-memory version since no real subscribers exist).
        """
        async with self._lock:
            self._events.append({"channel": channel, "data": data})
            return 0

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources (no-op for in-memory store)."""
        async with self._lock:
            self._data.clear()
            self._streams.clear()
            self._lists.clear()
            self._consumer_groups.clear()
            self._events.clear()
