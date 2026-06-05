"""Selector Cache — fast-path, zero-AI selector store.

Stores resolved selectors per provider so subsequent runs skip the
intelligence pipeline entirely when selectors have not changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class SelectorCache:
    """JSON-backed cache of discovered selectors per provider.

    Each entry stores the selector string, how it was discovered, and
    a confidence score.  The cache is persisted to disk so it survives
    restarts.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self._path = Path(path) if path else None
        self._data: dict[str, dict[str, dict]] = {}

        if self._path and self._path.exists():
            self._load()

    # ── public API ──────────────────────────────────────────────────

    def get(
        self, provider: str, role: str
    ) -> Optional[dict]:
        """Look up a cached selector for *provider* / *role*.

        Returns ``None`` on cache miss.
        """
        return self._data.get(provider, {}).get(role)

    def set(
        self,
        provider: str,
        role: str,
        selector: str,
        source: str = "cache",
        confidence: float = 1.0,
    ) -> None:
        """Store a selector in the cache."""
        self._data.setdefault(provider, {})[role] = {
            "value": selector,
            "source": source,
            "confidence": confidence,
        }

    def get_all(self, provider: str) -> dict[str, dict]:
        """Return all cached selectors for a provider."""
        return self._data.get(provider, {})

    def invalidate(self, provider: str, role: Optional[str] = None) -> None:
        """Remove cached selectors for a provider (or a single role)."""
        if role:
            self._data.get(provider, {}).pop(role, None)
        else:
            self._data.pop(provider, None)

    def persist(self, path: Optional[str | Path] = None) -> None:
        """Write cache to disk as JSON."""
        p = Path(path) if path else self._path
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._data, indent=2))

    # ── internal ────────────────────────────────────────────────────

    def _load(self) -> None:
        raw = self._path.read_text()
        self._data = json.loads(raw) if raw else {}
