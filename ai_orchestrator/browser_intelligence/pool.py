"""Browser Intelligence — engine reuse pool.

The old pattern was:

    engine = BrowserIntelligenceEngine()        # fresh
    await engine.attach(page)                   # 1× per request
    ...
    await engine.detach()                       # throws away calibration

Phase 7 forbids this. `EnginePool` keeps long-lived engines keyed by
`page` (or by account). The same engine instance is reused for every
send on that page, so emission calibration, recovery history, and the
reliability store all accumulate across calls. When the page closes,
the pool entry is evicted.

The pool also hands out engines that share a single
`ProviderReliabilityStore` so different pages on the same provider
benefit from each other's outcomes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine
from ai_orchestrator.browser_intelligence.learning import ProviderReliabilityStore

log = logging.getLogger(__name__)


@dataclass
class PoolEntry:
    engine: BrowserIntelligenceEngine
    page: Any
    provider_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    uses: int = 0
    attached: bool = False

    def touch(self) -> None:
        self.last_used_at = time.monotonic()
        self.uses += 1


class EnginePool:
    """Long-lived engine cache keyed by page identity.

    Lookup is by `id(page)` for simplicity — page objects in
    Playwright are stable per browser context.

    Concurrency: this pool is single-process and uses an asyncio Lock
    for mutation. Read access (get / stats) does not lock.
    """

    def __init__(
        self,
        *,
        reliability_store: ProviderReliabilityStore | None = None,
        max_pool_size: int = 64,
        idle_ttl_seconds: float = 600.0,
    ):
        self._store = reliability_store
        self._max_size = max_pool_size
        self._idle_ttl = float(idle_ttl_seconds)
        self._entries: dict[int, PoolEntry] = {}
        self._lock = asyncio.Lock()
        self._evictions: int = 0
        self._hits: int = 0
        self._misses: int = 0

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._entries),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    @property
    def reliability_store(self) -> ProviderReliabilityStore | None:
        return self._store

    async def get_or_create(
        self,
        page,
        provider_id: str,
    ) -> BrowserIntelligenceEngine:
        """Return a long-lived engine for this page. Creates a new one
        on first call. Reuses the existing one on subsequent calls.
        """
        key = id(page)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.page is page:
                entry.touch()
                self._hits += 1
                return entry.engine
            if entry is not None:
                # Same id, different page (e.g. page replaced). Evict.
                await self._evict_locked(key)
            self._misses += 1
            engine = BrowserIntelligenceEngine(
                reliability_store=self._store,
            )
            engine.bind_provider(provider_id)
            entry = PoolEntry(
                engine=engine,
                page=page,
                provider_id=provider_id,
            )
            # Attach immediately so the engine is wired to the page.
            try:
                await engine.attach(page)
                entry.attached = True
            except Exception as exc:
                log.warning("EnginePool attach failed: %s", exc)
            self._entries[key] = entry
            self._enforce_capacity_locked()
            return engine

    async def release(self, page) -> None:
        key = id(page)
        async with self._lock:
            await self._evict_locked(key)

    async def release_all(self) -> None:
        async with self._lock:
            keys = list(self._entries.keys())
            for k in keys:
                await self._evict_locked(k)

    async def gc(self) -> int:
        """Drop entries that have been idle longer than `idle_ttl`.
        Returns the number of entries evicted."""
        now = time.monotonic()
        async with self._lock:
            stale: list[int] = [
                k for k, e in self._entries.items()
                if (now - e.last_used_at) > self._idle_ttl
            ]
            for k in stale:
                await self._evict_locked(k)
        return len(stale)

    def snapshot_brains(self) -> list[dict]:
        """Return JSON-serializable snapshots for every pooled engine.
        Useful for periodic persistence into the persistent brain."""
        out: list[dict] = []
        for entry in self._entries.values():
            try:
                out.append(entry.engine.snapshot_brain())
            except Exception as exc:
                log.warning("snapshot_brains error: %s", exc)
        return out

    # ── internals ───────────────────────────────────────────────

    async def _evict_locked(self, key: int) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._evictions += 1
        if entry.attached:
            try:
                await entry.engine.detach()
            except Exception as exc:
                log.debug("EnginePool evict detach error: %s", exc)

    def _enforce_capacity_locked(self) -> None:
        if len(self._entries) <= self._max_size:
            return
        # Evict least-recently-used entries.
        ordered = sorted(
            self._entries.items(),
            key=lambda kv: kv[1].last_used_at,
        )
        excess = len(self._entries) - self._max_size
        for k, _ in ordered[:excess]:
            # Schedule eviction; can't await inside a non-async helper,
            # so we do best-effort sync cleanup of the inner state and
            # leave the async detach for the next pool call.
            self._entries.pop(k, None)
            self._evictions += 1


__all__ = ["EnginePool", "PoolEntry"]
