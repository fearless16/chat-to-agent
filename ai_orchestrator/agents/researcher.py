"""Researcher agent — gathers information and context."""

from __future__ import annotations

from ai_orchestrator.agents.base import AgentResult, BaseAgent


class ResearcherAgent(BaseAgent):
    """Agent responsible for researching and gathering information."""

    agent_type = "researcher"

    async def execute(self, context: dict) -> AgentResult:
        """Research the topic described in context."""
        step = context.get("step", "")
        return AgentResult(success=True, output=f"researcher executed: {step}")
