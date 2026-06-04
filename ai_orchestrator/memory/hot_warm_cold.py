"""Multi-tier memory — HOT (working context), WARM (compressed), COLD (offloaded).

Provides ContextManager for per-session memory lifecycle and MemoryPool for
task-scoped manager isolation.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryTier(str, enum.Enum):
    """Temperature tier for a memory entry."""

    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


class MemoryEntry(BaseModel):
    """A single unit of remembered conversation or system context."""

    tier: MemoryTier = Field(default=MemoryTier.HOT)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: str = Field(description="Message role: user, assistant, system, tool")
    content: str = Field(description="Message body")
    token_count: int = Field(default=0, ge=0, description="Approximate token count")
    summary: Optional[str] = Field(default=None, description="Compressed summary of this entry")
    embedding: Optional[list[float]] = Field(default=None, description="Semantic embedding vector")

    model_config = {"frozen": False, "use_enum_values": True}


def _estimate_token_count(text: str) -> int:
    """Rough token estimate based on character count (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """Manages the three-tier memory lifecycle for a single conversation session.

    HOT entries  — recent, full-fidelity working context.
    WARM entries — compressed summaries of older hot entries.
    COLD entries — fully offloaded (keyed by storage_key), retrievable on demand.
    """

    def __init__(
        self,
        cold_storage: Optional[dict[str, list[MemoryEntry]]] = None,
        max_hot_tokens: int = 4096,
    ) -> None:
        self._cold_storage: dict[str, list[MemoryEntry]] = (
            cold_storage if cold_storage is not None else {}
        )
        self._max_hot_tokens: int = max_hot_tokens
        self._hot: list[MemoryEntry] = []
        self._warm: list[MemoryEntry] = []
        self._cold_keys: list[str] = []

    # ---- Public API -------------------------------------------------------

    def add_entry(
        self,
        role: str,
        content: str,
        token_count: Optional[int] = None,
    ) -> MemoryEntry:
        """Add a new HOT entry. Token count is auto-estimated when not provided."""
        if token_count is None:
            token_count = _estimate_token_count(content)
        entry = MemoryEntry(
            tier=MemoryTier.HOT,
            role=role,
            content=content,
            token_count=token_count,
        )
        self._hot.append(entry)
        return entry

    def get_working_context(self, max_tokens: int = 4096) -> list[MemoryEntry]:
        """Return the most recent entries fitting within *max_tokens*.

        Walks newest-first and stops when adding the next entry would exceed
        the budget.  Always returns at least the newest entry.
        """
        if not self._hot:
            return list(self._warm) if self._warm else []

        # Start with warm entries (they are summaries, so we keep them)
        result: list[MemoryEntry] = list(self._warm)
        warm_tokens = sum(e.token_count for e in result)

        remaining = max_tokens - warm_tokens
        # Walk hot entries newest-first
        for entry in reversed(self._hot):
            if entry.token_count <= remaining:
                result.insert(0, entry)  # keep chronological order
                remaining -= entry.token_count
            else:
                # Always include the newest entry even if it exceeds remaining
                if not result or result[0] is not entry:
                    result.insert(0, entry)
                break

        return result

    def get_full_history(self) -> list[MemoryEntry]:
        """Return all entries across all tiers, oldest-first."""
        result: list[MemoryEntry] = []
        # Warm entries come before hot in chronological order
        result.extend(self._warm)
        result.extend(self._hot)
        return result

    async def compress_to_warm(self, threshold_tokens: int = 4096) -> int:
        """Move oldest HOT entries to WARM when total hot tokens exceed threshold.

        Returns the number of entries moved.
        """
        hot_tokens = sum(e.token_count for e in self._hot)
        if hot_tokens <= threshold_tokens:
            return 0

        moved = 0
        accumulated = 0
        to_move: list[MemoryEntry] = []

        # Move oldest entries until we drop below threshold
        for entry in self._hot:
            if hot_tokens - accumulated <= threshold_tokens:
                break
            to_move.append(entry)
            accumulated += entry.token_count
            moved += 1

        for entry in to_move:
            entry.tier = MemoryTier.WARM
            # Build a simple summary from the content
            entry.summary = self._make_summary(entry.content)
            self._warm.append(entry)

        # Remove moved entries from hot
        self._hot = self._hot[moved:]

        return moved

    async def offload_to_cold(self, storage_key: Optional[str] = None) -> str:
        """Offload all WARM entries to cold storage.

        Returns the storage key used.
        """
        if not self._warm:
            return ""

        key = storage_key or f"cold-{uuid.uuid4().hex[:12]}"
        self._cold_storage[key] = list(self._warm)
        for entry in self._cold_storage[key]:
            entry.tier = MemoryTier.COLD
        self._cold_keys.append(key)
        self._warm.clear()
        return key

    async def load_from_cold(self, storage_key: str) -> int:
        """Restore entries from cold storage back into WARM.

        Returns the number of entries restored, or 0 if the key is unknown.
        """
        entries = self._cold_storage.get(storage_key)
        if not entries:
            return 0
        for entry in entries:
            entry.tier = MemoryTier.WARM
            self._warm.append(entry)
        del self._cold_storage[storage_key]
        return len(entries)

    def get_current_token_count(self) -> int:
        """Sum of token counts across all HOT entries."""
        return sum(e.token_count for e in self._hot)

    def clear_hot(self) -> None:
        """Remove all HOT entries (WARM and COLD are preserved)."""
        self._hot.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of current memory statistics."""
        return {
            "total_entries": len(self._hot) + len(self._warm) + sum(
                len(v) for v in self._cold_storage.values()
            ),
            "hot_entries": len(self._hot),
            "warm_entries": len(self._warm),
            "cold_entries": sum(len(v) for v in self._cold_storage.values()),
            "cold_keys": len(self._cold_keys),
            "total_tokens": self.get_current_token_count() + sum(
                e.token_count for e in self._warm
            ),
            "hot_tokens": self.get_current_token_count(),
            "warm_tokens": sum(e.token_count for e in self._warm),
        }

    # ---- Internal helpers -------------------------------------------------

    @staticmethod
    def _make_summary(content: str, max_chars: int = 200) -> str:
        """Truncate content for WARM storage summary."""
        if not content:
            return ""
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 3] + "..."


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------

class MemoryPool:
    """Task-scoped pool of ContextManagers.

    Each task gets its own isolated ContextManager.  Use *trim_all* to
    reclaim tokens across the entire pool.
    """

    def __init__(self) -> None:
        self._managers: dict[str, ContextManager] = {}

    def get_or_create(self, task_id: str) -> ContextManager:
        """Return the ContextManager for *task_id*, creating one if absent."""
        if task_id not in self._managers:
            self._managers[task_id] = ContextManager()
        return self._managers[task_id]

    async def trim_all(self, target_tokens: int) -> dict[str, int]:
        """Compress every manager's hot memory to *target_tokens*.

        Returns a dict mapping task_id -> number of entries moved.
        """
        results: dict[str, int] = {}
        for task_id, mgr in self._managers.items():
            moved = await mgr.compress_to_warm(threshold_tokens=target_tokens)
            results[task_id] = moved
        return results
