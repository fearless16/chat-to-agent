"""Tests for the Memory Management module — hot/warm/cold tiers, token budget, summarization."""

from __future__ import annotations

import pytest

from ai_orchestrator.memory.hot_warm_cold import (
    ContextManager,
    MemoryEntry,
    MemoryPool,
    MemoryTier,
)
from ai_orchestrator.memory.summarizer import ContextSummarizer
from ai_orchestrator.memory.token_budget import TokenBudget


# ---------------------------------------------------------------------------
# MemoryTier & MemoryEntry
# ---------------------------------------------------------------------------

class TestMemoryTierAndEntry:
    """MemoryTier enum and MemoryEntry model."""

    def test_memory_tier_values(self):
        assert MemoryTier.HOT.value == "HOT"
        assert MemoryTier.WARM.value == "WARM"
        assert MemoryTier.COLD.value == "COLD"

    def test_memory_entry_creation_minimal(self):
        entry = MemoryEntry(role="user", content="Hello")
        assert entry.role == "user"
        assert entry.content == "Hello"
        assert entry.tier == MemoryTier.HOT
        assert entry.token_count == 0
        assert entry.summary is None
        assert entry.embedding is None

    def test_memory_entry_creation_full(self):
        entry = MemoryEntry(
            tier=MemoryTier.WARM,
            role="assistant",
            content="Hi there!",
            token_count=4,
            summary="Greeting",
            embedding=[0.1, 0.2, 0.3],
        )
        assert entry.tier == MemoryTier.WARM
        assert entry.role == "assistant"
        assert entry.token_count == 4
        assert entry.summary == "Greeting"
        assert entry.embedding == [0.1, 0.2, 0.3]

    def test_memory_entry_serialization(self):
        entry = MemoryEntry(role="user", content="test", token_count=2)
        data = entry.model_dump()
        restored = MemoryEntry.model_validate(data)
        assert restored.role == "user"
        assert restored.content == "test"
        assert restored.token_count == 2
        assert restored.tier == MemoryTier.HOT


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class TestContextManagerAddAndGet:
    """Adding entries and retrieving context."""

    def test_add_entry_returns_entry(self):
        mgr = ContextManager()
        entry = mgr.add_entry("user", "Hello")
        assert isinstance(entry, MemoryEntry)
        assert entry.role == "user"
        assert entry.content == "Hello"
        assert entry.tier == MemoryTier.HOT

    def test_add_entry_auto_counts_tokens(self):
        mgr = ContextManager()
        entry = mgr.add_entry("user", "Hello world")
        assert entry.token_count > 0

    def test_add_entry_with_explicit_token_count(self):
        mgr = ContextManager()
        entry = mgr.add_entry("assistant", "Hi", token_count=5)
        assert entry.token_count == 5

    def test_get_full_history_returns_all_entries(self):
        mgr = ContextManager()
        mgr.add_entry("user", "Hello")
        mgr.add_entry("assistant", "Hi")
        mgr.add_entry("user", "How are you?")
        history = mgr.get_full_history()
        assert len(history) == 3

    def test_get_working_context_respects_max_tokens(self):
        mgr = ContextManager()
        mgr.add_entry("user", "A" * 200)  # ~50 tokens at 4 chars/token
        mgr.add_entry("assistant", "B" * 200)
        mgr.add_entry("user", "C" * 200)
        # Limit to 20 tokens — should only return the newest entries
        context = mgr.get_working_context(max_tokens=20)
        assert len(context) < 3
        # The last entry should always be included
        assert context[-1].content == "C" * 200

    def test_get_current_token_count(self):
        mgr = ContextManager()
        assert mgr.get_current_token_count() == 0
        mgr.add_entry("user", "Hello world")  # ~3 tokens
        mgr.add_entry("assistant", "Hi there!")  # ~3 tokens
        assert mgr.get_current_token_count() > 0


class TestContextManagerCompress:
    """Compression and offloading."""

    @pytest.mark.asyncio
    async def test_compress_to_warm_moves_entries(self):
        mgr = ContextManager(max_hot_tokens=100)
        for i in range(10):
            # ~100 chars each → ~25 tokens; 10 entries → ~250 tokens total
            mgr.add_entry("user", f"Message number {i} " + "x" * 80)
        assert len(mgr.get_full_history()) == 10

        moved = await mgr.compress_to_warm(threshold_tokens=50)
        # At least some entries should have moved to warm
        history = mgr.get_full_history()
        warm_count = sum(1 for e in history if e.tier == MemoryTier.WARM)
        assert warm_count > 0
        assert moved > 0

    @pytest.mark.asyncio
    async def test_compress_to_warm_noop_when_under_threshold(self):
        mgr = ContextManager(max_hot_tokens=100)
        mgr.add_entry("user", "Small message")
        moved = await mgr.compress_to_warm(threshold_tokens=4096)
        assert moved == 0

    @pytest.mark.asyncio
    async def test_offload_and_load_cold(self):
        mgr = ContextManager()
        mgr.add_entry("user", "Hello")
        mgr.add_entry("assistant", "World")
        # Manually move to warm so we have something to offload
        await mgr.compress_to_warm(threshold_tokens=1)

        key = await mgr.offload_to_cold()
        assert isinstance(key, str)
        assert len(key) > 0

        count = await mgr.load_from_cold(key)
        assert count > 0
        # After loading, entries should be back
        history = mgr.get_full_history()
        assert len(history) > 0


class TestContextManagerClearAndStats:
    """Clearing and statistics."""

    def test_clear_hot_removes_hot_entries(self):
        mgr = ContextManager()
        mgr.add_entry("user", "Hello")
        mgr.add_entry("assistant", "Hi")
        mgr.clear_hot()
        assert mgr.get_current_token_count() == 0
        # warm/cold entries may still exist
        history = mgr.get_full_history()
        hot_count = sum(1 for e in history if e.tier == MemoryTier.HOT)
        assert hot_count == 0

    def test_get_stats_returns_dict(self):
        mgr = ContextManager()
        mgr.add_entry("user", "Hello")
        mgr.add_entry("assistant", "Hi")
        stats = mgr.get_stats()
        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "hot_entries" in stats
        assert "warm_entries" in stats
        assert "cold_entries" in stats
        assert "total_tokens" in stats
        assert stats["total_entries"] == 2
        assert stats["hot_entries"] == 2

    def test_get_stats_after_compress(self):
        mgr = ContextManager(max_hot_tokens=100)
        mgr.add_entry("user", "A")
        mgr.add_entry("assistant", "B")
        stats = mgr.get_stats()
        assert stats["hot_entries"] == 2


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------

class TestMemoryPool:
    """Task-scoped memory pools."""

    def test_get_or_create_returns_same_instance(self):
        pool = MemoryPool()
        mgr1 = pool.get_or_create("task-1")
        mgr2 = pool.get_or_create("task-1")
        assert mgr1 is mgr2

    def test_get_or_create_different_tasks(self):
        pool = MemoryPool()
        mgr1 = pool.get_or_create("task-1")
        mgr2 = pool.get_or_create("task-2")
        assert mgr1 is not mgr2

    @pytest.mark.asyncio
    async def test_trim_all(self):
        pool = MemoryPool()
        mgr1 = pool.get_or_create("task-1")
        mgr2 = pool.get_or_create("task-2")
        for _ in range(5):
            mgr1.add_entry("user", "Hello from task 1")
            mgr2.add_entry("user", "Hello from task 2")

        results = await pool.trim_all(target_tokens=50)
        assert isinstance(results, dict)
        assert "task-1" in results or "task-2" in results


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    """Token estimation, trimming, and compression."""

    def test_estimate_tokens_empty_string(self):
        tb = TokenBudget()
        assert tb.estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        tb = TokenBudget()
        count = tb.estimate_tokens("Hello world")
        assert count > 0
        # ~4 chars per token, 12 chars -> ~3 tokens
        assert count <= 6

    def test_estimate_tokens_long_text(self):
        tb = TokenBudget()
        text = "word " * 1000  # ~5000 chars
        count = tb.estimate_tokens(text)
        assert count > 0

    def test_count_tokens_context_and_prompt(self):
        tb = TokenBudget(max_context=8192)
        context = [MemoryEntry(role="user", content="Hello"), MemoryEntry(role="assistant", content="Hi")]
        total = tb.count_tokens(context, "New prompt")
        assert total > 0

    def test_would_exceed_limit_true(self):
        tb = TokenBudget(max_context=10)
        context = [MemoryEntry(role="user", content="A" * 100, token_count=25)]
        assert tb.would_exceed_limit(context, "More text", limit=20) is True

    def test_would_exceed_limit_false(self):
        tb = TokenBudget(max_context=100)
        context = [MemoryEntry(role="user", content="Hi", token_count=1)]
        assert tb.would_exceed_limit(context, "Hello", limit=100) is False

    def test_would_exceed_limit_default_limit(self):
        tb = TokenBudget(max_context=50)
        context = [MemoryEntry(role="user", content="A" * 1000, token_count=200)]
        # Should exceed default max_context
        assert tb.would_exceed_limit(context, "more") is True

    def test_trim_to_limit_removes_oldest(self):
        tb = TokenBudget(max_context=100)
        context = [
            MemoryEntry(role="user", content="First", token_count=10),
            MemoryEntry(role="assistant", content="Second", token_count=10),
            MemoryEntry(role="user", content="Third", token_count=10),
        ]
        trimmed = tb.trim_to_limit(context, limit=15)
        assert len(trimmed) < 3
        # The last entry(s) should remain
        assert trimmed[-1].content == "Third"

    def test_trim_to_limit_default_limit(self):
        tb = TokenBudget(max_context=50)
        context = [
            MemoryEntry(role="user", content="A" * 100, token_count=30),
            MemoryEntry(role="user", content="B" * 100, token_count=30),
        ]
        trimmed = tb.trim_to_limit(context)
        assert len(trimmed) <= 1

    def test_compress_context_returns_trimmed_and_summary(self):
        tb = TokenBudget()
        context = [
            MemoryEntry(role="user", content="This is the first long message.", token_count=10),
            MemoryEntry(role="assistant", content="This is the second long message.", token_count=10),
            MemoryEntry(role="user", content="This is the third message.", token_count=10),
        ]
        trimmed, summary = tb.compress_context(context, target_tokens=15)
        assert len(trimmed) < len(context)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_usage_percentage(self):
        tb = TokenBudget(max_context=100)
        pct = tb.get_usage_percentage(used=25, limit=100)
        assert pct == 25.0

    def test_get_usage_percentage_default_limit(self):
        tb = TokenBudget(max_context=200)
        pct = tb.get_usage_percentage(used=50)
        assert pct == 25.0

    def test_get_usage_percentage_zero_limit(self):
        tb = TokenBudget()
        pct = tb.get_usage_percentage(used=0, limit=0)
        assert pct == 0.0


# ---------------------------------------------------------------------------
# ContextSummarizer
# ---------------------------------------------------------------------------

class TestContextSummarizer:
    """Extractive text-based summarization."""

    def test_summarize_turns_empty(self):
        cs = ContextSummarizer()
        result = cs.summarize_turns([])
        assert result == ""

    def test_summarize_turns_single_entry(self):
        cs = ContextSummarizer()
        entry = MemoryEntry(role="user", content="Hello world")
        result = cs.summarize_turns([entry])
        assert "Hello world" in result
        assert "user" in result or "User" in result

    def test_summarize_turns_multiple_entries(self):
        cs = ContextSummarizer()
        entries = [
            MemoryEntry(role="user", content="What is AI?"),
            MemoryEntry(role="assistant", content="AI stands for Artificial Intelligence."),
            MemoryEntry(role="user", content="Tell me more."),
        ]
        result = cs.summarize_turns(entries, max_tokens=100)
        assert "What is AI?" in result
        assert "Artificial Intelligence" in result

    def test_summarize_turns_respects_max_tokens(self):
        cs = ContextSummarizer()
        long_content = "word " * 500
        entries = [MemoryEntry(role="user", content=long_content)]
        result = cs.summarize_turns(entries, max_tokens=20)
        # Should be truncated
        token_count = len(result) // 4  # rough estimate
        assert token_count <= 25  # allow some slop

    def test_summarize_conversation_empty(self):
        cs = ContextSummarizer()
        result = cs.summarize_conversation([])
        assert result == ""

    def test_summarize_conversation_basic(self):
        cs = ContextSummarizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How does this work?"},
        ]
        result = cs.summarize_conversation(messages, target_length_chars=200)
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) <= 250  # approximate

    def test_summarize_conversation_respects_length(self):
        cs = ContextSummarizer()
        messages = [
            {"role": "user", "content": "A" * 500},
            {"role": "assistant", "content": "B" * 500},
        ]
        result = cs.summarize_conversation(messages, target_length_chars=100)
        assert len(result) <= 150  # allow some buffer

    def test_extract_key_points_empty(self):
        cs = ContextSummarizer()
        result = cs.extract_key_points("")
        assert result == []

    def test_extract_key_points_simple(self):
        cs = ContextSummarizer()
        text = (
            "Python is a programming language. "
            "It is widely used for data science. "
            "Many companies use Python for web development. "
            "The language is easy to learn."
        )
        points = cs.extract_key_points(text, max_points=3)
        assert len(points) <= 3
        assert all(isinstance(p, str) for p in points)
        assert all(len(p) > 0 for p in points)

    def test_extract_key_points_respects_max_points(self):
        cs = ContextSummarizer()
        text = "First. Second. Third. Fourth. Fifth. Sixth."
        points = cs.extract_key_points(text, max_points=4)
        assert len(points) <= 4

    def test_extract_key_points_no_sentences(self):
        cs = ContextSummarizer()
        text = "just one sentence"
        points = cs.extract_key_points(text, max_points=3)
        assert len(points) >= 1  # at least the whole text
