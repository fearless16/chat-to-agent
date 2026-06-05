"""Redis client — production Redis with in-memory fallback for development and testing.

When ``url`` is omitted, all data lives in plain Python dicts (ideal for
testing and local dev without a Redis server).  Provide a ``url`` to
connect to a real Redis instance.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.client import Redis as RealRedis


class RedisClient:
    """Async Redis client with transparent in-memory fallback.

    Usage::

        # In-memory mock (default)
        client = RedisClient()

        # Real Redis
        client = RedisClient(url="redis://localhost:6379")

    All operations are coroutine-safe — the in-memory path uses a single
    :class:`asyncio.Lock` and the real-Redis path delegates to
    ``redis.asyncio`` which handles connection pooling and concurrency
    natively.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url
        self._real: RealRedis | None = None
        self._lock = asyncio.Lock()

        # ── in-memory stores (used when url is None) ──────────────
        self._data: dict[str, tuple[str, float | None]] = {}
        self._streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._lists: dict[str, list[str]] = {}
        self._consumer_groups: dict[tuple[str, str], set[str]] = {}
        self._events: list[dict[str, Any]] = []
        self._counter: int = 0

    async def _ensure_real(self) -> RealRedis:
        if self._real is None:
            self._real = aioredis.from_url(self._url, decode_responses=False)
        return self._real

    # ══════════════════════════════════════════════════════════════════
    # Key-value ops
    # ══════════════════════════════════════════════════════════════════

    async def set_key(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set *key* to *value* with an optional *ttl* in seconds."""
        if self._url is not None:
            r = await self._ensure_real()
            serialized = json.dumps(value) if not isinstance(value, str) else value
            if ttl is not None:
                await r.setex(key, int(ttl), serialized)
            else:
                await r.set(key, serialized)
            return

        serialized = json.dumps(value) if not isinstance(value, str) else value
        expiry = time.time() + ttl if ttl is not None else None
        async with self._lock:
            self._data[key] = (serialized, expiry)

    async def get_key(self, key: str) -> str | None:
        """Return the value for *key*, or ``None`` if missing or expired."""
        if self._url is not None:
            r = await self._ensure_real()
            val = await r.get(key)
            return val.decode() if val else None

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
        """Delete *key*. Returns ``True`` if the key existed."""
        if self._url is not None:
            r = await self._ensure_real()
            return bool(await r.delete(key))

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

    # ══════════════════════════════════════════════════════════════════
    # Stream ops
    # ══════════════════════════════════════════════════════════════════

    async def push_to_stream(self, stream: str, data: dict[str, Any]) -> str:
        """Push *data* to *stream* and return the entry ID."""
        if self._url is not None:
            r = await self._ensure_real()
            entry_id = await r.xadd(stream, data)
            return entry_id.decode()

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
        """Read up to *count* entries from *stream*."""
        if self._url is not None:
            r = await self._ensure_real()
            entries = await r.xread({stream: "0"}, count=count, block=block_ms)
            if not entries:
                return []
            stream_name, msgs = entries[0]
            result = []
            for msg_id, data in msgs:
                decoded = {
                    k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                    for k, v in data.items()
                }
                result.append(decoded)
            return result

        async with self._lock:
            entries = self._streams.get(stream, [])
            return [data for _, data in entries[:count]]

    # ══════════════════════════════════════════════════════════════════
    # List ops
    # ══════════════════════════════════════════════════════════════════

    async def push_to_list(self, key: str, value: Any) -> None:
        """Push *value* to the tail of the list at *key*."""
        if self._url is not None:
            r = await self._ensure_real()
            serialized = json.dumps(value) if not isinstance(value, str) else value
            await r.rpush(key, serialized)
            return

        serialized = json.dumps(value) if not isinstance(value, str) else value
        async with self._lock:
            self._lists.setdefault(key, []).append(serialized)

    async def pop_from_list(self, key: str) -> str | None:
        """Pop the head of the list at *key* and return it, or ``None``."""
        if self._url is not None:
            r = await self._ensure_real()
            val = await r.lpop(key)
            return val.decode() if val else None

        async with self._lock:
            queue = self._lists.get(key)
            if not queue:
                return None
            return queue.pop(0)

    async def get_list_length(self, key: str) -> int:
        """Return the number of items in the list at *key*."""
        if self._url is not None:
            r = await self._ensure_real()
            return await r.llen(key)

        async with self._lock:
            return len(self._lists.get(key, []))

    # ══════════════════════════════════════════════════════════════════
    # Consumer group ops
    # ══════════════════════════════════════════════════════════════════

    async def create_consumer_group(self, group: str, stream: str) -> bool:
        """Create a consumer group for *stream*. Returns ``True`` if newly created."""
        if self._url is not None:
            r = await self._ensure_real()
            try:
                await r.xgroup_create(stream, group, "0", mkstream=True)
                return True
            except Exception:
                return False

        key = (group, stream)
        async with self._lock:
            if key in self._consumer_groups:
                return False
            self._consumer_groups[key] = set()
            return True

    async def ack_message(self, group: str, stream: str, entry_id: str) -> bool:
        """Acknowledge a message in *group* for *stream*."""
        if self._url is not None:
            r = await self._ensure_real()
            count = await r.xack(stream, group, entry_id)
            return count > 0

        key = (group, stream)
        async with self._lock:
            group_set = self._consumer_groups.get(key)
            if group_set is None:
                return False
            group_set.add(entry_id)
            return True

    # ══════════════════════════════════════════════════════════════════
    # Pub/sub ops
    # ══════════════════════════════════════════════════════════════════

    async def publish_event(self, channel: str, data: dict[str, Any]) -> int:
        """Publish an event on *channel*. Returns subscriber count."""
        if self._url is not None:
            r = await self._ensure_real()
            payload = json.dumps(data)
            return await r.publish(channel, payload)

        async with self._lock:
            self._events.append({"channel": channel, "data": data})
            return 0

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════

    async def close(self) -> None:
        """Release resources and clear in-memory stores."""
        if self._real is not None:
            await self._real.aclose()
            self._real = None

        async with self._lock:
            self._data.clear()
            self._streams.clear()
            self._lists.clear()
            self._consumer_groups.clear()
            self._events.clear()
