"""Token budget management — estimation, budgeting, trimming, and compression.

Provides a TokenBudget class that approximates token counts using a simple
character-based heuristic (~4 characters per token) and supports context
trimming and compression for memory management.
"""

from __future__ import annotations

from typing import Optional

from ai_orchestrator.memory.hot_warm_cold import MemoryEntry


_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token estimate based on character count."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


class TokenBudget:
    """Token budget tracker for a conversation context.

    Works with MemoryEntry objects to estimate, trim, and compress context
    within a configurable token limit.  All token counts are approximations
    based on character length; no external tokenizer is used.
    """

    def __init__(self, max_context: int = 8192) -> None:
        self._max_context = max_context

    # ---- Token estimation ------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Return an approximate token count for *text*.

        Uses a simple heuristic of 4 characters per token.
        """
        return _estimate_tokens(text)

    def count_tokens(self, context: list[MemoryEntry], prompt: str) -> int:
        """Return total tokens for *context* entries plus *prompt*."""
        context_tokens = sum(e.token_count for e in context)
        prompt_tokens = _estimate_tokens(prompt)
        return context_tokens + prompt_tokens

    # ---- Budget checks ---------------------------------------------------

    def would_exceed_limit(
        self,
        context: list[MemoryEntry],
        prompt: str,
        limit: Optional[int] = None,
    ) -> bool:
        """Return True if adding *prompt* to *context* would exceed *limit*."""
        budget = limit if limit is not None else self._max_context
        total = self.count_tokens(context, prompt)
        return total > budget

    # ---- Trimming --------------------------------------------------------

    def trim_to_limit(
        self,
        context: list[MemoryEntry],
        limit: Optional[int] = None,
    ) -> list[MemoryEntry]:
        """Remove oldest entries until the total is within *limit*.

        Returns the trimmed list (newest entries are preserved).
        The original list is not mutated.
        """
        budget = limit if limit is not None else self._max_context
        if not context:
            return []

        total = sum(e.token_count for e in context)
        if total <= budget:
            return list(context)

        # Walk oldest-first, dropping entries until we fit
        trimmed: list[MemoryEntry] = list(context)
        while trimmed and sum(e.token_count for e in trimmed) > budget:
            trimmed.pop(0)

        # Always keep at least the newest entry
        if not trimmed:
            trimmed = [context[-1]]

        return trimmed

    # ---- Compression -----------------------------------------------------

    def compress_context(
        self,
        context: list[MemoryEntry],
        target_tokens: int,
    ) -> tuple[list[MemoryEntry], str]:
        """Trim *context* to *target_tokens* and return a text summary of removed entries.

        Returns a tuple of (trimmed_context, summary_of_removed).
        The summary is built from the content of the oldest entries that were
        dropped, joined and truncated.
        """
        if not context:
            return [], ""

        total = sum(e.token_count for e in context)
        if total <= target_tokens:
            return list(context), ""

        removed_entries: list[MemoryEntry] = []
        trimmed: list[MemoryEntry] = list(context)

        while trimmed and sum(e.token_count for e in trimmed) > target_tokens:
            removed_entries.append(trimmed.pop(0))

        # Always keep at least the newest entry
        if not trimmed:
            trimmed = [context[-1]]
            if removed_entries and removed_entries[-1] is context[-1]:
                removed_entries.pop()

        # Build summary from removed content
        summary_parts: list[str] = []
        for entry in removed_entries:
            text = entry.content.strip()
            if text:
                summary_parts.append(text)

        summary = " ... ".join(summary_parts) if summary_parts else ""
        # Truncate summary to roughly target_tokens
        max_summary_chars = target_tokens * _CHARS_PER_TOKEN
        if len(summary) > max_summary_chars:
            summary = summary[: max_summary_chars - 3] + "..."

        return trimmed, summary

    # ---- Reporting -------------------------------------------------------

    def get_usage_percentage(
        self,
        used: int,
        limit: Optional[int] = None,
    ) -> float:
        """Return the percentage (0–100) of the token budget used."""
        budget = limit if limit is not None else self._max_context
        if budget <= 0:
            return 0.0
        return (used / budget) * 100.0
