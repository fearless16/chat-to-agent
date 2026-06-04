"""Executor agent — executes approved actions and commands."""

from __future__ import annotations

from ai_orchestrator.agents.base import AgentResult, BaseAgent


class ExecutorAgent(BaseAgent):
    """Agent responsible for executing approved actions and commands."""

    agent_type = "executor"

    async def execute(self, context: dict) -> AgentResult:
        """Execute the action described in context."""
        step = context.get("step", "")
        return AgentResult(success=True, output=f"executor executed: {step}")
