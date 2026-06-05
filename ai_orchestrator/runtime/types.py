"""Runtime type definitions shared across the runtime package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FixResult:
    """Return value from a fix callback indicating what was done.

    Attributes
    ----------
    fixes_applied:
        Number of distinct fixes applied (0 means nothing changed).
    message:
        Human-readable summary of the fix.
    abort:
        If ``True``, the runtime loop should stop immediately (e.g. the
        fixer determines the task is impossible or requires human help).
    """

    fixes_applied: int = 0
    message: str = ""
    abort: bool = False
