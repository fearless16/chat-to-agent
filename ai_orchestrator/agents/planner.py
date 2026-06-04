"""Planner agent — decomposes tasks into executable steps."""

from __future__ import annotations

from ai_orchestrator.agents.base import AgentResult, BaseAgent


class PlannerAgent(BaseAgent):
    """Agent responsible for planning and decomposing work."""

    agent_type = "planner"

    async def execute(self, context: dict) -> AgentResult:
        """Plan the next step based on context."""
        step = context.get("step", "")
        return AgentResult(success=True, output=f"planner executed: {step}")
