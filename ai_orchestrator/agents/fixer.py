"""Fixer agent — applies fixes based on review feedback."""

from __future__ import annotations

from ai_orchestrator.agents.base import AgentResult, BaseAgent


class FixerAgent(BaseAgent):
    """Agent responsible for applying fixes to identified issues."""

    agent_type = "fixer"

    async def execute(self, context: dict) -> AgentResult:
        """Apply fixes described in context."""
        step = context.get("step", "")
        return AgentResult(success=True, output=f"fixer executed: {step}")
