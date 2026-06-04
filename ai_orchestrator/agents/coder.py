"""Coder agent — writes and modifies code."""

from __future__ import annotations

from ai_orchestrator.agents.base import AgentResult, BaseAgent


class CoderAgent(BaseAgent):
    """Agent responsible for writing and modifying source code."""

    agent_type = "coder"

    async def execute(self, context: dict) -> AgentResult:
        """Write or modify code as described in context."""
        step = context.get("step", "")
        return AgentResult(success=True, output=f"coder executed: {step}")
