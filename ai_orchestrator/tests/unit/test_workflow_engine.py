"""Tests for the WorkflowEngine — FSM transitions, lifecycle, loop detection, halt/resume."""

import pytest

from ai_orchestrator.models.task import Task, TaskStatus
from ai_orchestrator.orchestrator.workflow_engine import (
    WorkflowEngine,
    WorkflowState,
    TaskPlan,
)


class TestWorkflowState:
    """WorkflowState enum values and semantics."""

    def test_all_states_present(self):
        assert WorkflowState.IDLE.value == "IDLE"
        assert WorkflowState.PLANNING.value == "PLANNING"
        assert WorkflowState.EXECUTING.value == "EXECUTING"
        assert WorkflowState.TESTING.value == "TESTING"
        assert WorkflowState.REVIEW.value == "REVIEW"
        assert WorkflowState.FIX.value == "FIX"
        assert WorkflowState.DONE.value == "DONE"
        assert WorkflowState.HALTED.value == "HALTED"

    def test_states_are_string_enum(self):
        assert isinstance(WorkflowState.IDLE, str)


class TestValidTransitions:
    """get_valid_transitions returns the correct set of next states."""

    def test_idle_transitions(self):
        engine = WorkflowEngine()
        assert engine.get_valid_transitions(WorkflowState.IDLE) == [WorkflowState.PLANNING]

    def test_planning_transitions(self):
        engine = WorkflowEngine()
        assert engine.get_valid_transitions(WorkflowState.PLANNING) == [WorkflowState.EXECUTING]

    def test_executing_transitions(self):
        engine = WorkflowEngine()
        transitions = engine.get_valid_transitions(WorkflowState.EXECUTING)
        assert WorkflowState.TESTING in transitions
        assert WorkflowState.HALTED in transitions
        assert len(transitions) == 2

    def test_testing_transitions(self):
        engine = WorkflowEngine()
        transitions = engine.get_valid_transitions(WorkflowState.TESTING)
        assert WorkflowState.REVIEW in transitions
        assert WorkflowState.FIX in transitions
        assert len(transitions) == 2

    def test_fix_transitions(self):
        engine = WorkflowEngine()
        assert engine.get_valid_transitions(WorkflowState.FIX) == [WorkflowState.EXECUTING]

    def test_review_transitions(self):
        engine = WorkflowEngine()
        transitions = engine.get_valid_transitions(WorkflowState.REVIEW)
        assert WorkflowState.DONE in transitions
        assert WorkflowState.FIX in transitions
        assert len(transitions) == 2

    def test_done_is_terminal(self):
        engine = WorkflowEngine()
        assert engine.get_valid_transitions(WorkflowState.DONE) == []

    def test_halted_is_terminal(self):
        engine = WorkflowEngine()
        assert engine.get_valid_transitions(WorkflowState.HALTED) == []


class TestCanProceed:
    """can_proceed validates allowed state transitions."""

    def test_valid_transition(self):
        engine = WorkflowEngine()
        task = Task(prompt="hello", status=TaskStatus.IDLE)
        assert engine.can_proceed(task, WorkflowState.PLANNING) is True

    def test_invalid_transition(self):
        engine = WorkflowEngine()
        task = Task(prompt="hello", status=TaskStatus.IDLE)
        assert engine.can_proceed(task, WorkflowState.EXECUTING) is False

    def test_halted_task_cannot_proceed(self):
        engine = WorkflowEngine()
        task = Task(prompt="hello", status=TaskStatus.HALTED)
        assert engine.can_proceed(task, WorkflowState.PLANNING) is False

    def test_done_task_cannot_proceed(self):
        engine = WorkflowEngine()
        task = Task(prompt="hello", status=TaskStatus.DONE)
        assert engine.can_proceed(task, WorkflowState.PLANNING) is False

    def test_dlq_task_cannot_proceed(self):
        engine = WorkflowEngine()
        task = Task(prompt="hello", status=TaskStatus.DLQ)
        assert engine.can_proceed(task, WorkflowState.PLANNING) is False


class TestStartTask:
    """start_task assigns a planner and moves to PLANNING."""

    @pytest.mark.asyncio
    async def test_start_task_transitions_to_planning(self):
        engine = WorkflowEngine()
        task = Task(prompt="build a feature")
        result = await engine.start_task(task)
        assert result.status == TaskStatus.PLANNING
        assert result.assigned_agent == "planner"

    @pytest.mark.asyncio
    async def test_start_task_already_running_raises(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", status=TaskStatus.EXECUTING)
        with pytest.raises(ValueError, match="can only start"):
            await engine.start_task(task)


class TestPlanTask:
    """plan_task generates a TaskPlan from a prompt."""

    @pytest.mark.asyncio
    async def test_plan_task_returns_taskplan(self):
        engine = WorkflowEngine()
        task = Task(prompt="build a login feature")
        plan = await engine.plan_task(task, "implement user login")
        assert isinstance(plan, TaskPlan)
        assert len(plan.steps) > 0
        assert "planner" in plan.required_agents

    @pytest.mark.asyncio
    async def test_plan_contains_relevant_steps(self):
        engine = WorkflowEngine()
        task = Task(prompt="add auth")
        plan = await engine.plan_task(task, "add authentication")
        assert len(plan.steps) > 0

    @pytest.mark.asyncio
    async def test_default_max_parallel_steps(self):
        engine = WorkflowEngine()
        task = Task(prompt="hello")
        plan = await engine.plan_task(task, "do something")
        assert plan.max_parallel_steps == 2


class TestExecuteStep:
    """execute_step progresses through the FSM."""

    @pytest.mark.asyncio
    async def test_execute_step_transitions_through_fsm(self):
        engine = WorkflowEngine()
        task = Task(prompt="test task")
        await engine.start_task(task)
        # Execute a step — should advance from PLANNING to EXECUTING
        result = await engine.execute_step(task, "implement", "coder")
        assert result is True
        assert task.status == TaskStatus.EXECUTING

    @pytest.mark.asyncio
    async def test_execute_step_with_loop_halts(self):
        engine = WorkflowEngine()
        task = Task(prompt="loopy task")
        await engine.start_task(task)

        # Simulate loop: same action hash 4 times
        action_hash = "hash-loop-123"
        for _ in range(3):
            engine.handle_loop_detection(task, action_hash)

        # 4th time should detect loop
        assert engine.handle_loop_detection(task, action_hash) is True

        # Now execute should return False and halt
        result = await engine.execute_step(task, "flaky_step", "coder")
        assert result is False
        assert task.status == TaskStatus.HALTED

    @pytest.mark.asyncio
    async def test_execute_step_from_wrong_state(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", status=TaskStatus.DONE)
        with pytest.raises(ValueError, match="cannot execute"):
            await engine.execute_step(task, "step1", "coder")


class TestGetNextStep:
    """get_next_step returns the correct next step from plan."""

    def test_first_step_when_no_current_step(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        plan = TaskPlan(steps=["design", "implement", "test"], required_agents=["coder"])
        next_step = engine.get_next_step(task, plan)
        assert next_step == "design"

    def test_advances_to_next_step(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", current_step="design")
        plan = TaskPlan(steps=["design", "implement", "test"], required_agents=["coder"])
        next_step = engine.get_next_step(task, plan)
        assert next_step == "implement"

    def test_last_step_returns_none(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", current_step="test")
        plan = TaskPlan(steps=["design", "implement", "test"], required_agents=["coder"])
        next_step = engine.get_next_step(task, plan)
        assert next_step is None

    def test_empty_plan_returns_none(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        plan = TaskPlan(steps=[], required_agents=[])
        next_step = engine.get_next_step(task, plan)
        assert next_step is None

    def test_step_not_in_plan_returns_first(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", current_step="unknown")
        plan = TaskPlan(steps=["design", "implement"], required_agents=["coder"])
        next_step = engine.get_next_step(task, plan)
        assert next_step == "design"


class TestLoopDetection:
    """handle_loop_detection tracks action hashes and detects loops."""

    def test_first_action_no_loop(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        assert engine.handle_loop_detection(task, "hash-1") is False

    def test_three_repeats_no_loop(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        engine.handle_loop_detection(task, "hash-x")
        engine.handle_loop_detection(task, "hash-x")
        assert engine.handle_loop_detection(task, "hash-x") is False

    def test_fourth_repeat_detects_loop(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        for _ in range(3):
            engine.handle_loop_detection(task, "hash-loop")
        assert engine.handle_loop_detection(task, "hash-loop") is True

    def test_different_actions_do_not_cross_contaminate(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        for _ in range(5):
            engine.handle_loop_detection(task, "hash-a")
        # hash-b should be clean
        assert engine.handle_loop_detection(task, "hash-b") is False

    def test_loop_per_task_isolation(self):
        engine = WorkflowEngine()
        task_a = Task(prompt="a")
        task_b = Task(prompt="b")

        for _ in range(4):
            engine.handle_loop_detection(task_a, "same-hash")

        # task_b should have its own counter
        assert engine.handle_loop_detection(task_b, "same-hash") is False

    def test_many_repeats_still_detected(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        for _ in range(10):
            engine.handle_loop_detection(task, "hash-z")
        assert engine.handle_loop_detection(task, "hash-z") is True


class TestHaltAndResume:
    """halt_task and resume_task pause and continue execution."""

    @pytest.mark.asyncio
    async def test_halt_task_sets_halted_and_error(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        await engine.start_task(task)
        await engine.halt_task(task, "something went wrong")
        assert task.status == TaskStatus.HALTED
        assert task.error_message == "something went wrong"
        # Workflow state should be HALTED
        wf_state = engine._task_states.get(task.id)
        assert wf_state == WorkflowState.HALTED

    @pytest.mark.asyncio
    async def test_resume_halved_task(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        await engine.start_task(task)
        await engine.halt_task(task, "oops")
        result = await engine.resume_task(task)
        assert result is True
        assert task.status == TaskStatus.PLANNING
        assert task.error_message is None

    @pytest.mark.asyncio
    async def test_resume_non_halted_task_returns_false(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        result = await engine.resume_task(task)
        assert result is False

    @pytest.mark.asyncio
    async def test_done_task_cannot_be_halted(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        await engine.start_task(task)
        # Execute through the full lifecycle to reach DONE
        plan = await engine.plan_task(task, "do something")
        for step in plan.steps:
            await engine.execute_step(task, step, "coder")
            task.current_step = step
        # Task should now be DONE — halting should raise
        with pytest.raises(ValueError, match="cannot halt"):
            await engine.halt_task(task, "too late")


class TestFullLifecycle:
    """Complete task lifecycle: create → plan → execute → test → review → done."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_happy_path(self):
        engine = WorkflowEngine()
        task = Task(prompt="implement feature X")

        # 1. Start task -> PLANNING
        task = await engine.start_task(task)
        assert task.status == TaskStatus.PLANNING
        assert task.assigned_agent == "planner"

        # 2. Plan -> get TaskPlan
        plan = await engine.plan_task(task, "implement feature X")
        assert len(plan.steps) >= 1
        assert "planner" in plan.required_agents

        # 3. Execute steps sequentially
        step = engine.get_next_step(task, plan)
        while step is not None:
            result = await engine.execute_step(task, step, "coder")
            assert result is True
            task.current_step = step
            step = engine.get_next_step(task, plan)

        # 4. Task should be DONE
        assert task.status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_lifecycle_with_loop_halts(self):
        engine = WorkflowEngine()
        task = Task(prompt="flaky feature")
        await engine.start_task(task)
        plan = await engine.plan_task(task, "flaky feature")

        action_hash = "stuck-step"
        for _ in range(4):
            engine.handle_loop_detection(task, action_hash)

        step = engine.get_next_step(task, plan)
        result = await engine.execute_step(task, step, "coder")
        assert result is False
        assert task.status == TaskStatus.HALTED


class TestEdgeCases:
    """Edge cases: empty plans, invalid transitions, max loops."""

    def test_empty_plan_get_next_step_none(self):
        engine = WorkflowEngine()
        task = Task(prompt="test")
        plan = TaskPlan(steps=[], required_agents=[])
        assert engine.get_next_step(task, plan) is None

    @pytest.mark.asyncio
    async def test_plan_with_empty_prompt_returns_default(self):
        engine = WorkflowEngine()
        task = Task(prompt="")
        plan = await engine.plan_task(task, "")
        assert isinstance(plan, TaskPlan)
        # Even empty prompt should produce a minimal plan

    def test_can_proceed_with_task_not_yet_tracked(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", status=TaskStatus.IDLE)
        assert engine.can_proceed(task, WorkflowState.PLANNING) is True

    def test_can_proceed_from_planning_to_executing(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", status=TaskStatus.PLANNING)
        assert engine.can_proceed(task, WorkflowState.EXECUTING) is True

    def test_cannot_proceed_from_idle_to_done(self):
        engine = WorkflowEngine()
        task = Task(prompt="test", status=TaskStatus.IDLE)
        assert engine.can_proceed(task, WorkflowState.DONE) is False

    def test_workflow_state_tracking(self):
        engine = WorkflowEngine()
        task = Task(prompt="track me")
        # Initially not tracked
        assert task.id not in engine._task_states
        engine._task_states[task.id] = WorkflowState.IDLE
        assert engine._task_states[task.id] == WorkflowState.IDLE

    @pytest.mark.asyncio
    async def test_resume_clears_error_and_restores_state(self):
        engine = WorkflowEngine()
        task = Task(prompt="resume me")
        await engine.start_task(task)
        await engine.halt_task(task, "temp failure")
        assert task.error_message == "temp failure"

        result = await engine.resume_task(task)
        assert result is True
        assert task.error_message is None
        assert task.status == TaskStatus.PLANNING
