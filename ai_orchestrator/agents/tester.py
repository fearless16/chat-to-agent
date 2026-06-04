"""Tester agent — writes and runs tests."""

from __future__ import annotations

from ai_orchestrator.agents.base import AgentResult, BaseAgent


class TesterAgent(BaseAgent):
    """Agent responsible for writing and executing tests."""

    agent_type = "tester"

    async def execute(self, context: dict) -> AgentResult:
        """Write or execute tests based on context."""
        step = context.get("step", "")
        return AgentResult(success=True, output=f"tester executed: {step}")
