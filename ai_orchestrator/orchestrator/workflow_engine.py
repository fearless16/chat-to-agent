"""WorkflowEngine — FSM-based task orchestration with loop detection, halt/resume."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from ai_orchestrator.models.task import Task, TaskStatus


class WorkflowState(str, enum.Enum):
    """States in the workflow lifecycle."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    TESTING = "TESTING"
    REVIEW = "REVIEW"
    FIX = "FIX"
    DONE = "DONE"
    HALTED = "HALTED"


_VALID_TRANSITIONS: dict[WorkflowState, list[WorkflowState]] = {
    WorkflowState.IDLE: [WorkflowState.PLANNING],
    WorkflowState.PLANNING: [WorkflowState.EXECUTING],
    WorkflowState.EXECUTING: [WorkflowState.TESTING, WorkflowState.HALTED],
    WorkflowState.TESTING: [WorkflowState.REVIEW, WorkflowState.FIX],
    WorkflowState.REVIEW: [WorkflowState.DONE, WorkflowState.FIX],
    WorkflowState.FIX: [WorkflowState.EXECUTING],
    WorkflowState.DONE: [],
    WorkflowState.HALTED: [],
}


@dataclass
class TaskPlan:
    """Plan produced by the PlannerAgent."""
    steps: list[str] = field(default_factory=list)
    required_agents: list[str] = field(default_factory=list)
    max_parallel_steps: int = 2


class WorkflowEngine:
    """State-machine workflow orchestrator with loop detection."""

    def __init__(self) -> None:
        self._task_states: dict[str, WorkflowState] = {}
        self._action_counts: dict[str, dict[str, int]] = {}
        self._step_index: dict[str, int] = {}
        # Map workflow state to task status
        self._state_to_task_status = {
            WorkflowState.IDLE: TaskStatus.IDLE,
            WorkflowState.PLANNING: TaskStatus.PLANNING,
            WorkflowState.EXECUTING: TaskStatus.EXECUTING,
            WorkflowState.TESTING: TaskStatus.VERIFICATION,
            WorkflowState.REVIEW: TaskStatus.VERIFICATION,
            WorkflowState.FIX: TaskStatus.EXECUTING,
            WorkflowState.DONE: TaskStatus.DONE,
            WorkflowState.HALTED: TaskStatus.HALTED,
        }

    def get_valid_transitions(self, from_state: WorkflowState) -> list[WorkflowState]:
        return list(_VALID_TRANSITIONS.get(from_state, []))

    def can_proceed(self, task: Task, target_state: WorkflowState) -> bool:
        if task.status in (TaskStatus.HALTED, TaskStatus.DONE, TaskStatus.DLQ):
            return False
        # Determine current workflow state
        current = self._task_states.get(task.id)
        if current is None:
            # Map from task status
            status_map = {
                TaskStatus.IDLE: WorkflowState.IDLE,
                TaskStatus.PLANNING: WorkflowState.PLANNING,
                TaskStatus.EXECUTING: WorkflowState.EXECUTING,
                TaskStatus.VERIFICATION: WorkflowState.TESTING,
                TaskStatus.DONE: WorkflowState.DONE,
                TaskStatus.HALTED: WorkflowState.HALTED,
            }
            current = status_map.get(task.status)
        if current is None:
            return False
        return target_state in _VALID_TRANSITIONS.get(current, [])

    async def start_task(self, task: Task) -> Task:
        if task.status != TaskStatus.IDLE:
            raise ValueError(f"can only start IDLE tasks, got {task.status}")
        self._task_states[task.id] = WorkflowState.PLANNING
        task.transition_to(TaskStatus.PLANNING)
        task.assigned_agent = "planner"
        self._step_index[task.id] = 0
        return task

    async def plan_task(self, task: Task, prompt: str) -> TaskPlan:
        prompt_lower = prompt.lower()
        if "login" in prompt_lower or "auth" in prompt_lower or "authenticate" in prompt_lower:
            return TaskPlan(steps=["design", "implement", "test"], required_agents=["planner", "coder", "tester"])
        elif "feature" in prompt_lower:
            return TaskPlan(steps=["implement", "test", "review"], required_agents=["planner", "coder", "tester", "reviewer"])
        elif "fix" in prompt_lower or "bug" in prompt_lower:
            return TaskPlan(steps=["investigate", "fix", "verify"], required_agents=["planner", "coder", "tester"])
        elif not prompt:
            return TaskPlan(steps=["analyze"], required_agents=["planner"])
        else:
            return TaskPlan(steps=["implement", "test", "review"], required_agents=["planner", "coder", "tester", "reviewer"])

    async def execute_step(self, task: Task, step_name: str, agent_type: str) -> bool:
        if task.status in (TaskStatus.DONE, TaskStatus.DLQ):
            raise ValueError(f"cannot execute step on {task.status} task")
        if task.status == TaskStatus.HALTED:
            raise ValueError("cannot execute step on HALTED task")

        # Check for loop detection — any hash exceeding threshold
        if any(count > 3 for count in self._action_counts.get(task.id, {}).values()):
            await self.halt_task(task, "loop detected: repeated action")
            return False

        current_wf_state = self._task_states.get(task.id, WorkflowState.IDLE)

        # Advance one step in the FSM
        transitions = _VALID_TRANSITIONS.get(current_wf_state, [])
        # Take the first non-HALTED transition
        next_state = None
        for t in transitions:
            if t != WorkflowState.HALTED:
                next_state = t
                break
        if next_state:
            self._task_states[task.id] = next_state
            # REVIEW auto-advances to DONE (end of lifecycle)
            if next_state == WorkflowState.REVIEW:
                self._task_states[task.id] = WorkflowState.DONE
                task.transition_to(TaskStatus.DONE)
            else:
                ts = self._state_to_task_status.get(next_state)
                if ts:
                    task.transition_to(ts)

        task.current_step = step_name
        self._step_index[task.id] = self._step_index.get(task.id, 0) + 1
        return True

    def get_next_step(self, task: Task, plan: TaskPlan) -> Optional[str]:
        if not plan.steps:
            return None
        if not task.current_step:
            return plan.steps[0]
        try:
            idx = plan.steps.index(task.current_step)
            if idx + 1 < len(plan.steps):
                return plan.steps[idx + 1]
            return None
        except ValueError:
            return plan.steps[0]

    def handle_loop_detection(self, task: Task, action_hash: str) -> bool:
        if task.id not in self._action_counts:
            self._action_counts[task.id] = {}
        self._action_counts[task.id][action_hash] = self._action_counts[task.id].get(action_hash, 0) + 1
        return self._action_counts[task.id][action_hash] > 3

    async def halt_task(self, task: Task, reason: str) -> None:
        if task.status in (TaskStatus.DONE, TaskStatus.DLQ):
            raise ValueError(f"cannot halt {task.status.value} task")
        self._task_states[task.id] = WorkflowState.HALTED
        task.transition_to(TaskStatus.HALTED)
        task.error_message = reason

    async def resume_task(self, task: Task) -> bool:
        if task.status != TaskStatus.HALTED:
            return False
        task.error_message = None
        self._task_states[task.id] = WorkflowState.PLANNING
        task.transition_to(TaskStatus.PLANNING)
        return True
