"""DeepSeek Control Plane — Tier 0/1/2 intelligence for task routing and planning.

Per V6 architecture:

  Tier 0 (No AI):
    - Selector Cache
    - Rule Engine
    - Heuristics
    - Known Patterns

  Tier 1 (Cheap Intelligence):
    - classification
    - basic validation
    - dom labeling
    - light summarization

  Tier 2 (DeepSeek):
    - planning
    - replanning
    - code review
    - architecture review
    - complex dom analysis
    - workflow repair

DeepSeek should never be invoked for routine operations that Tier 0
or Tier 1 can handle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from ai_orchestrator.models.capabilities import PROVIDER_PROFILES, TaskRequirements
from ai_orchestrator.orchestrator.provider_router import ProviderRouter, ScoredAccount

log = logging.getLogger(__name__)


class IntelligenceTier(IntEnum):
    TIER_0 = 0  # No AI — rules, cache, heuristics
    TIER_1 = 1  # Cheap AI — classification, basic validation
    TIER_2 = 2  # DeepSeek — planning, review, complex analysis


@dataclass
class RoutingDecision:
    """Result of the control plane's routing decision."""
    provider: str
    account_id: str
    score: float
    tier: IntelligenceTier
    reason: str


@dataclass
class TaskClassification:
    """Classification of an incoming task."""
    task_type: str = "general"  # general | coding | translation | research
    requires_reasoning: bool = False
    requires_coding: bool = False
    requires_translation: bool = False
    requires_multimodality: bool = False
    confidence: float = 1.0
    tier: IntelligenceTier = IntelligenceTier.TIER_0


class Tier0Router:
    """Tier 0 — deterministic routing rules.

    No AI.  Uses provider capability profiles, account health scores,
    and task requirements.
    """

    def __init__(self) -> None:
        self._router = ProviderRouter()

    def route(
        self,
        requirements: TaskRequirements,
        provider_pool: dict[str, list],
    ) -> Optional[RoutingDecision]:
        """Route using capability scoring only (no AI)."""
        result = self._router.select_provider(requirements, provider_pool)
        if result is None:
            return None

        provider_name, account = result
        scored = self._router.score_account(account, requirements)

        return RoutingDecision(
            provider=provider_name,
            account_id=account.id,
            score=scored.score,
            tier=IntelligenceTier.TIER_0,
            reason=scored.reason,
        )

    @staticmethod
    def classify_task_type(prompt: str) -> TaskClassification:
        """Tier 0 — keyword-based heuristic classification.

        No AI involved.  Quick regex matching.
        """
        prompt_lower = prompt.lower()

        coding_keywords = [
            "code", "function", "class", "implement", "write a program",
            "debug", "fix", "refactor", "algorithm", "sort", "api",
            "endpoint", "route", "database", "sql", "async",
        ]
        reasoning_keywords = [
            "explain", "why", "how does", "compare", "contrast",
            "analyse", "analyze", "evaluate", "what is the difference",
            "math", "proof", "theorem", "logic",
        ]
        translation_keywords = [
            "translate", "translation", "convert language",
        ]

        has_coding = any(kw in prompt_lower for kw in coding_keywords)
        has_reasoning = any(kw in prompt_lower for kw in reasoning_keywords)
        has_translation = any(kw in prompt_lower for kw in translation_keywords)

        counts = {
            "has_coding": sum(1 for kw in coding_keywords if kw in prompt_lower),
            "has_reasoning": sum(1 for kw in reasoning_keywords if kw in prompt_lower),
            "has_translation": sum(1 for kw in translation_keywords if kw in prompt_lower),
        }

        # Determine primary type
        if counts["has_coding"] >= 2 and counts["has_coding"] >= counts["has_reasoning"]:
            task_type = "coding"
        elif counts["has_translation"] >= 1:
            task_type = "translation"
        elif counts["has_reasoning"] >= 1:
            task_type = "research"
        else:
            task_type = "general"

        return TaskClassification(
            task_type=task_type,
            requires_coding=has_coding,
            requires_reasoning=has_reasoning,
            requires_translation=has_translation,
            confidence=0.8 if task_type != "general" else 0.5,
            tier=IntelligenceTier.TIER_0,
        )


class Tier1Router:
    """Tier 1 — cheap intelligence for classification and validation.

    Uses the cheapest available LLM (e.g. a local model or a fast/cheap
    API provider) for tasks that Tier 0 cannot resolve confidently.
    """

    async def classify_task_type(
        self, prompt: str, tier0_result: TaskClassification,
    ) -> TaskClassification:
        """Escalate to Tier 1 when Tier 0 confidence is low.

        This is a stub — the real implementation sends a focused prompt
        to a cheap LLM for classification.
        """
        return tier0_result

    async def resolve_ambigious_routing(
        self, prompt: str, candidates: list[ScoredAccount],
    ) -> Optional[str]:
        """Resolve ties or low-confidence routing via Tier 1.

        Stub — real implementation asks a cheap LLM.
        """
        return None


class Tier2Router:
    """Tier 2 — DeepSeek for planning, review, and complex analysis.

    DeepSeek should only be used when Tier 0 and Tier 1 cannot resolve
    the problem.
    """

    async def create_plan(self, prompt: str, task_type: str) -> list[dict]:
        """Decompose a task into executable steps via DeepSeek.

        Stub — real implementation sends a planning prompt to DeepSeek.
        """
        return [{"step": "implement", "agent": "coder"}]

    async def replan(
        self, task_id: str, error: str, previous_steps: list[dict],
    ) -> list[dict]:
        """Re-plan a failed task via DeepSeek.

        Stub — real implementation sends failure context to DeepSeek.
        """
        return [{"step": "fix", "agent": "fixer"}]

    async def analyze_error(self, error: str, context: str) -> tuple[str, float]:
        """Analyse a workflow error via DeepSeek.

        Returns (analysis, confidence).
        """
        return (f"DeepSeek analysis: {error[:100]}", 0.8)


class ControlPlane:
    """The system's decision-making layer.

    Implements the Tier 0 → Tier 1 → Tier 2 escalation pattern.

    Usage::

        cp = ControlPlane()
        classification = await cp.classify("Write a Python function to sort a list")
         decision = cp.route(requirements, provider_pool)
    """

    def __init__(self) -> None:
        self._tier0 = Tier0Router()
        self._tier1 = Tier1Router()
        self._tier2 = Tier2Router()

    async def classify(self, prompt: str) -> TaskClassification:
        """Classify a task using Tier 0 → Tier 1 escalation.

        Tier 0 handles the common case (keyword matching).  If
        confidence is low, Tier 1 (cheap LLM) is consulted.
        """
        classification = self._tier0.classify_task_type(prompt)

        # Escalate to Tier 1 if Tier 0 is unsure
        if classification.confidence < 0.6:
            log.debug("Tier 0 confidence low (%.2f) — escalating to Tier 1", classification.confidence)
            classification = await self._tier1.classify_task_type(prompt, classification)

        return classification

    def route(
        self,
        requirements: TaskRequirements,
        provider_pool: dict[str, list],
    ) -> Optional[RoutingDecision]:
        """Route a task using Tier 0 routing (capability scoring).

        Tier 0 is always sufficient for routing because the capability
        profiles are pre-defined.  Escalation to Tier 1/2 is only
        needed when scores are tied or all profiles fail.
        """
        return self._tier0.route(requirements, provider_pool)

    async def plan(self, prompt: str, task_type: str) -> list[dict]:
        """Create an execution plan using DeepSeek (Tier 2).

        Only called when the task is complex enough to need planning.
        """
        return await self._tier2.create_plan(prompt, task_type)

    async def replan(
        self, task_id: str, error: str, previous_steps: list[dict],
    ) -> list[dict]:
        """Re-plan a failed task via DeepSeek (Tier 2)."""
        return await self._tier2.replan(task_id, error, previous_steps)
