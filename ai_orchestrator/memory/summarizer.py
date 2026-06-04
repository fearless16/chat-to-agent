"""Extractive summarization — no LLM calls.

Provides ContextSummarizer with three strategies:
1.  summarize_turns       — compress a list of MemoryEntry objects into a short string.
2.  summarize_conversation — compress a list of message dicts into a short string.
3.  extract_key_points     — pull key sentences from a text.

All methods use text-based heuristics: sentence splitting, position-biased
extraction, and character-length truncation.
"""

from __future__ import annotations

import re
from typing import Any

from ai_orchestrator.memory.hot_warm_cold import MemoryEntry


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CHARS_PER_TOKEN = 4


class ContextSummarizer:
    """Extractive summarizer that never calls an LLM.

    All operations are deterministic text transformations: sentence
    extraction, truncation, and concatenation.
    """

    # ---- Turn-level summarization ----------------------------------------

    def summarize_turns(
        self,
        entries: list[MemoryEntry],
        max_tokens: int = 512,
    ) -> str:
        """Compress a list of MemoryEntry objects into a single summary string.

        Each entry is formatted as ``role: content``.  The result is
        truncated to approximately *max_tokens* tokens.
        """
        if not entries:
            return ""

        parts: list[str] = []
        for entry in entries:
            label = entry.role.capitalize()
            parts.append(f"{label}: {entry.content}")

        combined = "\n".join(parts)
        return self._truncate_to_tokens(combined, max_tokens)

    # ---- Conversation-level summarization --------------------------------

    def summarize_conversation(
        self,
        messages: list[dict[str, Any]],
        target_length_chars: int = 500,
    ) -> str:
        """Compress a list of message dicts into a summary.

        Each dict should have at least ``role`` and ``content`` keys.
        The method extracts the first and last few messages (with simple
        position-biased selection) and concatenates them, then optionally
        appends key points extracted from the middle.

        The result is capped at *target_length_chars* characters.
        """
        if not messages:
            return ""

        # For very short conversations, return everything
        char_total = sum(len(m.get("content", "")) for m in messages)
        if char_total <= target_length_chars:
            parts: list[str] = []
            for m in messages:
                role = m.get("role", "unknown").capitalize()
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n".join(parts)

        # Position-biased extraction: keep first 2 and last 2 messages
        # as a representative sample, then add key points from the middle
        selected: list[dict[str, Any]] = []
        if len(messages) <= 4:
            selected = list(messages)
        else:
            selected.extend(messages[:2])
            selected.extend(messages[-2:])

        parts = []
        for m in selected:
            role = m.get("role", "unknown").capitalize()
            content = m.get("content", "")
            # Truncate each message individually
            max_msg_chars = target_length_chars // len(selected)
            if len(content) > max_msg_chars:
                content = content[: max_msg_chars - 3] + "..."
            parts.append(f"{role}: {content}")

        summary = "\n".join(parts)

        # If there's room and enough middle content, add extracted key points
        middle_content = "\n".join(
            m.get("content", "")
            for m in messages[2:-2]
            if m.get("content")
        )
        if middle_content and len(summary) < target_length_chars:
            points = self.extract_key_points(
                middle_content,
                max_points=2,
            )
            if points:
                point_text = "Key points: " + " | ".join(points)
                if len(summary) + len(point_text) + 2 <= target_length_chars:
                    summary += "\n" + point_text
                else:
                    # Fit what we can
                    remaining = target_length_chars - len(summary) - 2
                    if remaining > 10:
                        summary += "\n" + point_text[:remaining]

        return summary[:target_length_chars]

    # ---- Key-point extraction -------------------------------------------

    def extract_key_points(
        self,
        text: str,
        max_points: int = 5,
    ) -> list[str]:
        """Extract important sentences from *text* as key points.

        Uses simple heuristics:
        - Split text into sentences.
        - Score sentences by length (moderate sentences score highest).
        - Return the top-scoring sentences up to *max_points*.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return [text.strip()]

        # Single sentence — return the whole text
        if len(sentences) <= 1:
            return [sentences[0].strip()]

        # Score each sentence, keeping original index
        scored: list[tuple[float, str, int]] = []
        for i, sentence in enumerate(sentences):
            stripped = sentence.strip()
            if not stripped:
                continue
            score = self._score_sentence(stripped, i, len(sentences))
            scored.append((score, stripped, i))

        # Sort by score descending, pick top max_points
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_points]

        # Restore original position order for readability
        top.sort(key=lambda x: x[2])  # sort by original index
        return [t[1] for t in top]

    # ---- Internal helpers -------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        raw = _SENTENCE_SPLIT.split(text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _score_sentence(sentence: str, index: int, total: int) -> float:
        """Score a sentence for key-point extraction (higher = more important).

        Factors:
        - Length score: moderate length sentences (30–150 chars) score best.
        - Position score: first and last sentences get a boost.
        - Content clues: sentences with question words or action verbs score
          higher.
        """
        length = len(sentence)
        # Ideal length: 30–150 characters
        if 30 <= length <= 150:
            length_score = 1.0
        elif length < 30:
            length_score = 0.5
        else:
            length_score = max(0.1, 1.0 - (length - 150) / 500.0)

        # Position: boost first (index 0) and last sentences
        position_score = 1.0
        if index == 0 or index == total - 1:
            position_score = 1.3

        # Content clues
        content_score = 1.0
        lower = sentence.lower()
        clue_words = [
            "important", "key", "critical", "essential", "significant",
            "therefore", "conclusion", "result", "because", "however",
            "first", "finally", "overall", "summary",
        ]
        for word in clue_words:
            if word in lower:
                content_score += 0.2

        # Questions are often important
        if "?" in sentence:
            content_score += 0.3

        return length_score * position_score * content_score

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate *text* to approximately *max_tokens* tokens."""
        if max_tokens <= 0:
            return ""
        max_chars = max_tokens * _CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
