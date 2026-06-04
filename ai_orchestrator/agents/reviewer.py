"""Reviewer agent — reviews code and provides feedback."""

from __future__ import annotations

from ai_orchestrator.agents.base import AgentResult, BaseAgent


class ReviewerAgent(BaseAgent):
    """Agent responsible for reviewing code and providing feedback."""

    agent_type = "reviewer"

    async def execute(self, context: dict) -> AgentResult:
        """Review the work described in context."""
        step = context.get("step", "")
        return AgentResult(success=True, output=f"reviewer executed: {step}")
