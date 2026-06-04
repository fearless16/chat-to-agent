"""Tests for the Agent Framework — AgentResult, BaseAgent, and all role agents."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ai_orchestrator.agents.base import AgentResult, BaseAgent
from ai_orchestrator.agents.coder import CoderAgent
from ai_orchestrator.agents.executor import ExecutorAgent
from ai_orchestrator.agents.fixer import FixerAgent
from ai_orchestrator.agents.planner import PlannerAgent
from ai_orchestrator.agents.researcher import ResearcherAgent
from ai_orchestrator.agents.reviewer import ReviewerAgent
from ai_orchestrator.agents.tester import TesterAgent


# =============================================================================
# AgentResult model tests
# =============================================================================

class TestAgentResult:
    """AgentResult field validation and defaults."""

    def test_minimal_creation(self):
        """AgentResult can be created with just required fields."""
        result = AgentResult(success=True, output="done")
        assert result.success is True
        assert result.output == "done"
        assert result.actions_taken == []
        assert result.duration_ms == 0.0
        assert result.error is None

    def test_all_fields_provided(self):
        """AgentResult accepts all fields."""
        result = AgentResult(
            success=False,
            output="failed",
            actions_taken=[{"step": 1}],
            duration_ms=150.5,
            error="something went wrong",
        )
        assert result.success is False
        assert result.output == "failed"
        assert result.actions_taken == [{"step": 1}]
        assert result.duration_ms == 150.5
        assert result.error == "something went wrong"

    def test_actions_taken_defaults_to_empty_list(self):
        """actions_taken defaults to [] when not provided."""
        result = AgentResult(success=True, output="ok")
        assert result.actions_taken == []

    def test_duration_ms_defaults_to_zero(self):
        """duration_ms defaults to 0.0."""
        result = AgentResult(success=True, output="ok")
        assert result.duration_ms == 0.0

    def test_error_defaults_to_none(self):
        """error defaults to None."""
        result = AgentResult(success=True, output="ok")
        assert result.error is None

    def test_error_can_be_set_to_string(self):
        """error can be a string."""
        result = AgentResult(success=False, output="", error="timeout")
        assert result.error == "timeout"

    def test_duration_ms_accepts_float(self):
        """duration_ms handles float values."""
        result = AgentResult(success=True, output="ok", duration_ms=123.456)
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms == 123.456

    def test_missing_required_fields_raises(self):
        """ValidationError raised when required fields are missing."""
        with pytest.raises(ValidationError):
            AgentResult()  # type: ignore[call-arg]

    def test_missing_output_raises(self):
        """ValidationError raised when output is missing."""
        with pytest.raises(ValidationError):
            AgentResult(success=True)  # type: ignore[call-arg]

    def test_missing_success_raises(self):
        """ValidationError raised when success is missing."""
        with pytest.raises(ValidationError):
            AgentResult(output="oops")  # type: ignore[call-arg]

    def test_model_dump_roundtrip(self):
        """AgentResult serialises and deserialises correctly."""
        original = AgentResult(
            success=True,
            output="hello",
            actions_taken=[{"a": 1}],
            duration_ms=10.0,
            error=None,
        )
        data = original.model_dump()
        restored = AgentResult.model_validate(data)
        assert restored == original


# =============================================================================
# BaseAgent tests (using a concrete subclass)
# =============================================================================

class _ConcreteAgent(BaseAgent):
    """Minimal concrete subclass for testing BaseAgent functionality."""
    agent_type = "concrete"

    async def execute(self, context: dict) -> AgentResult:
        return AgentResult(success=True, output=f"concrete: {context.get('step', 'no-step')}")


class TestBaseAgent:
    """BaseAgent initialisation, limits, action recording, and run_step."""

    def test_init_sets_attributes(self):
        """Constructor stores provided identifiers and limits."""
        agent = _ConcreteAgent(agent_id="a1", task_id="t1", max_steps=10, max_runtime_ms=5000)
        assert agent.agent_id == "a1"
        assert agent.task_id == "t1"
        assert agent.max_steps == 10
        assert agent.max_runtime_ms == 5000

    def test_init_defaults(self):
        """Constructor applies default max_steps and max_runtime_ms."""
        agent = _ConcreteAgent(agent_id="a2", task_id="t2")
        assert agent.max_steps == 25
        assert agent.max_runtime_ms == 300000

    def test_agent_type_class_attr(self):
        """agent_type is a class-level attribute."""
        assert _ConcreteAgent.agent_type == "concrete"

    def test_record_action_appends(self):
        """record_action adds entries to the internal list."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        agent.record_action("run", {"step": 1})
        agent.record_action("check", {"step": 2})
        assert len(agent._actions) == 2
        assert agent._actions[0] == {"action": "run", "details": {"step": 1}}
        assert agent._actions[1] == {"action": "check", "details": {"step": 2}}

    def test_record_action_without_details(self):
        """record_action works when details is None."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        agent.record_action("ping")
        assert agent._actions == [{"action": "ping"}]

    def test_check_limits_allows_execution_initially(self):
        """check_limits returns True when no steps have run."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        assert agent.check_limits() is True

    def test_check_limits_false_when_steps_exceeded(self):
        """check_limits returns False when step count >= max_steps."""
        agent = _ConcreteAgent(agent_id="a", task_id="t", max_steps=2)
        agent._step_count = 2
        assert agent.check_limits() is False

    def test_check_limits_false_when_runtime_exceeded(self):
        """check_limits returns False when runtime >= max_runtime_ms."""
        agent = _ConcreteAgent(agent_id="a", task_id="t", max_runtime_ms=100)
        agent._start_time = 0.0  # way in the past
        # time.monotonic() will be >> 0.1s, so limit is exceeded
        with patch("ai_orchestrator.agents.base.time.monotonic", return_value=10.0):
            assert agent.check_limits() is False

    @pytest.mark.asyncio
    async def test_run_step_returns_agent_result(self):
        """run_step returns an AgentResult on success."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        result = await agent.run_step({"step": "hello"})
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "concrete: hello" in result.output

    @pytest.mark.asyncio
    async def test_run_step_increments_step_count(self):
        """run_step increments the internal step counter."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        assert agent._step_count == 0
        await agent.run_step({"step": "x"})
        assert agent._step_count == 1

    @pytest.mark.asyncio
    async def test_run_step_records_action(self):
        """run_step automatically records an action via record_action."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        await agent.run_step({"step": "test"})
        assert len(agent._actions) == 1
        assert agent._actions[0]["action"] == "concrete"
        assert agent._actions[0]["details"]["step"] == 1

    @pytest.mark.asyncio
    async def test_run_step_sets_duration_ms(self):
        """run_step sets duration_ms on the result."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        result = await agent.run_step({"step": "timed"})
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_run_step_returns_limit_failure_when_blocked(self):
        """run_step returns failure AgentResult when limits are exceeded."""
        agent = _ConcreteAgent(agent_id="a", task_id="t", max_steps=0)
        result = await agent.run_step({"step": "nope"})
        assert result.success is False
        assert "Limits exceeded" in result.output or "limits" in result.output.lower()

    @pytest.mark.asyncio
    async def test_run_step_sets_start_time_on_first_call(self):
        """_start_time is initialised on the first run_step call."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        assert agent._start_time is None
        await agent.run_step({"step": "first"})
        assert agent._start_time is not None

    @pytest.mark.asyncio
    async def test_run_step_preserves_start_time_across_calls(self):
        """_start_time is set only on the first run_step call."""
        agent = _ConcreteAgent(agent_id="a", task_id="t")
        await agent.run_step({"step": "first"})
        t1 = agent._start_time
        await agent.run_step({"step": "second"})
        assert agent._start_time == t1

    def test_agent_type_not_overridden(self):
        """Subclasses that don't override agent_type keep 'base'."""
        class UnsetAgent(BaseAgent):
            async def execute(self, context: dict) -> AgentResult:
                return AgentResult(success=True, output="")

        assert UnsetAgent.agent_type == "base"


# =============================================================================
# Role agent tests
# =============================================================================

class _RoleAgentTestBase:
    """Mixin with shared role-agent test helpers."""
    agent_class: type[BaseAgent] | None = None
    expected_type: str = ""

    @pytest.mark.asyncio
    async def test_execute_returns_agent_result(self):
        """execute() returns an AgentResult with success=True."""
        agent = self.agent_class(agent_id="r1", task_id="t1")  # type: ignore[union-attr]
        result = await agent.execute({"step": "go"})
        assert isinstance(result, AgentResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_output_contains_agent_type(self):
        """execute() output includes the agent_type string."""
        agent = self.agent_class(agent_id="r1", task_id="t1")  # type: ignore[union-attr]
        result = await agent.execute({"step": "plan"})
        assert self.expected_type in result.output

    @pytest.mark.asyncio
    async def test_execute_output_includes_step(self):
        """execute() output includes the step from context."""
        agent = self.agent_class(agent_id="r1", task_id="t1")  # type: ignore[union-attr]
        result = await agent.execute({"step": "my-step"})
        assert "my-step" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_empty_context(self):
        """execute() handles empty context gracefully."""
        agent = self.agent_class(agent_id="r1", task_id="t1")  # type: ignore[union-attr]
        result = await agent.execute({})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_agent_type_class_attr(self):
        """Each role agent has a distinct agent_type."""
        assert self.agent_class.agent_type == self.expected_type  # type: ignore[union-attr]


class TestPlannerAgent(_RoleAgentTestBase):
    agent_class = PlannerAgent
    expected_type = "planner"


class TestResearcherAgent(_RoleAgentTestBase):
    agent_class = ResearcherAgent
    expected_type = "researcher"


class TestCoderAgent(_RoleAgentTestBase):
    agent_class = CoderAgent
    expected_type = "coder"


class TestTesterAgent(_RoleAgentTestBase):
    agent_class = TesterAgent
    expected_type = "tester"


class TestReviewerAgent(_RoleAgentTestBase):
    agent_class = ReviewerAgent
    expected_type = "reviewer"


class TestFixerAgent(_RoleAgentTestBase):
    agent_class = FixerAgent
    expected_type = "fixer"


class TestExecutorAgent(_RoleAgentTestBase):
    agent_class = ExecutorAgent
    expected_type = "executor"
