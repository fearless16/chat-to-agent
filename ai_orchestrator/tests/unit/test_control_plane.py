"""Tests for the DeepSeek Control Plane (Tier 0/1/2 routing)."""

from __future__ import annotations

import pytest

from ai_orchestrator.models.capabilities import TaskRequirements
from ai_orchestrator.orchestrator.control_plane import (
    ControlPlane,
    IntelligenceTier,
    Tier0Router,
)


class TestTier0Classification:
    def test_coding_task(self):
        result = Tier0Router.classify_task_type("Write a Python function to sort a list using quicksort")
        assert result.requires_coding
        assert result.task_type == "coding"

    def test_research_task(self):
        result = Tier0Router.classify_task_type("Explain the theory of relativity")
        assert result.requires_reasoning
        assert result.task_type == "research"

    def test_translation_task(self):
        result = Tier0Router.classify_task_type("Translate this to French")
        assert result.requires_translation
        assert result.task_type == "translation"

    def test_general_task(self):
        result = Tier0Router.classify_task_type("Hello, how are you?")
        assert result.task_type == "general"
        assert not result.requires_coding
        assert not result.requires_reasoning

    def test_empty_prompt(self):
        result = Tier0Router.classify_task_type("")
        assert result.task_type == "general"
        assert result.confidence == 0.5

    def test_mixed_signals(self):
        """When both coding and reasoning keywords appear, coding wins if more frequent."""
        result = Tier0Router.classify_task_type(
            "Write a function, implement an algorithm, and explain why it works"
        )
        assert result.requires_coding
        assert result.requires_reasoning
        # coding has more keyword matches (write a function, implement, algorithm)
        assert result.task_type == "coding"


class TestControlPlane:
    async def test_classify_escalation(self):
        cp = ControlPlane()
        # High confidence — stays Tier 0
        result = await cp.classify("Implement a binary search tree in Python")
        assert result.tier == IntelligenceTier.TIER_0
        assert result.requires_coding

    async def test_classify_low_confidence(self):
        cp = ControlPlane()
        # Low confidence — escalates to Tier 1 (stub returns same)
        result = await cp.classify("Do something")
        assert result.task_type == "general"

    async def test_plan(self):
        cp = ControlPlane()
        plan = await cp.plan("Implement a feature", "coding")
        assert isinstance(plan, list)
        assert len(plan) > 0

    async def test_replan(self):
        cp = ControlPlane()
        plan = await cp.replan(
            "task-1", "Test failed: AssertionError",
            [{"step": "implement", "agent": "coder"}],
        )
        assert isinstance(plan, list)
        assert len(plan) > 0
