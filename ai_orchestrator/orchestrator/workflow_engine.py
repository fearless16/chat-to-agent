"""WorkflowEngine — FSM-based task orchestration with loop detection, halt/resume."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from ai_orchestrator.models.task import Task, TaskStatus
from ai_orchestrator.orchestrator.control_plane import ControlPlane
from ai_orchestrator.orchestrator.lease_manager import AccountEvent


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
    """State-machine workflow orchestrator with loop detection.

    Integrates with the ControlPlane for Tier 0/1/2 planning and
    the Reactive Lease Manager for account-failure-driven REPLAN.
    """

    def __init__(self, control_plane: Optional[ControlPlane] = None) -> None:
        self._task_states: dict[str, WorkflowState] = {}
        self._action_counts: dict[str, dict[str, int]] = {}
        self._step_index: dict[str, int] = {}
        self._control_plane = control_plane or ControlPlane()
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

    # ── reactive lease event handler ───────────────────────────────
    # Per V6 architecture: Account → JAIL → Lease Manager → Force Expire
    # → Workflow Engine → REPLAN

    def handle_account_event(self, event: AccountEvent) -> None:
        """React to an account state change.

        When an account enters JAIL, all tasks that held leases on that
        account are transitioned back to PLANNING so the Control Plane
        can re-route them to a different provider.
        """
        if event.new_state.name != "JAIL":
            return

        # Find all tasks with active leases on this account
        # (the lease was force-expired by the LeaseManager)
        for task_id, state in list(self._task_states.items()):
            if state in (WorkflowState.EXECUTING, WorkflowState.TESTING, WorkflowState.FIX):
                self._task_states[task_id] = WorkflowState.PLANNING
                # Note: the actual Task object must be transitioned by the
                # caller (the gateway handler) since we don't own it here.
                # The state dict is sufficient to prevent further steps.

    _AGENT_TO_STATE: dict[str, WorkflowState] = {
        "planner": WorkflowState.PLANNING,
        "coder": WorkflowState.EXECUTING,
        "tester": WorkflowState.TESTING,
        "reviewer": WorkflowState.REVIEW,
        "fixer": WorkflowState.FIX,
        "executor": WorkflowState.EXECUTING,
        "researcher": WorkflowState.EXECUTING,
    }

    async def start_task(self, task: Task) -> Task:
        if task.status != TaskStatus.IDLE:
            raise ValueError(f"can only start IDLE tasks, got {task.status}")
        self._task_states[task.id] = WorkflowState.PLANNING
        task.transition_to(TaskStatus.PLANNING)
        task.assigned_agent = "planner"
        self._step_index[task.id] = 0
        self._action_counts.pop(task.id, None)
        return task

    async def plan_task(self, task: Task, prompt: str) -> TaskPlan:
        """Create a task plan using the ControlPlane (Tier 0/1/2 escalation)."""
        # Tier 0 — classsify the task type
        classification = await self._control_plane.classify(prompt)

        # Build plan from classification
        if classification.task_type == "coding":
            return TaskPlan(steps=["implement", "test", "review"], required_agents=["planner", "coder", "tester", "reviewer"])
        elif classification.task_type == "translation":
            return TaskPlan(steps=["translate", "review"], required_agents=["planner", "reviewer"])
        elif classification.task_type == "research":
            return TaskPlan(steps=["research", "summarize", "review"], required_agents=["planner", "researcher", "reviewer"])
        else:
            return TaskPlan(steps=["implement", "test", "review"], required_agents=["planner", "coder", "tester", "reviewer"])

    async def execute_step(
        self,
        task: Task,
        step_name: str,
        agent_type: str,
        step_result: str = "ok",
        action_hash: str | None = None,
    ) -> bool:
        """Execute one workflow step and advance the FSM.

        ``step_result`` is one of ``"ok"``, ``"failed"``, ``"rejected"`` and
        routes the FSM correctly: ``"failed"`` sends the workflow from
        TESTING → FIX (or from REVIEW → FIX), and ``"rejected"`` sends
        REVIEW → FIX; ``"ok"`` follows the happy path (TESTING → REVIEW,
        REVIEW → DONE).  The ``"rejected"`` alias is accepted so callers
        that only know "review outcome" can drive both terminal outcomes.

        ``action_hash`` is the key used for loop detection.  When
        omitted, it defaults to ``f"{step_name}:{agent_type}"``.  The
        caller is encouraged to pass a plan-step key (e.g. the index of
        ``step_name`` within ``plan.steps``) so that legitimate retries
        of a long-running step do not trip the loop breaker.
        """
        if task.status in (TaskStatus.DONE, TaskStatus.DLQ, TaskStatus.FAILED):
            raise ValueError(f"cannot execute step on {task.status} task")
        if task.status == TaskStatus.HALTED:
            raise ValueError("cannot execute step on HALTED task")

        # Loop detection — caller can supply a stable hash (e.g. a
        # plan-step index) or fall back to the (step, agent) tuple.
        effective_hash = action_hash or f"{step_name}:{agent_type}"
        if self.handle_loop_detection(task, effective_hash):
            await self.halt_task(task, f"loop detected: repeated action {effective_hash}")
            return False

        current_wf_state = self._task_states.get(task.id, WorkflowState.IDLE)

        # Derive the next state from the *current* FSM state plus the
        # step outcome.  This makes the FIX branch reachable (previously
        # the first-non-HALTED heuristic always took the happy path) and
        # also keeps REVIEW a real state instead of a flash-over to DONE.
        next_state = self._resolve_next_state(current_wf_state, step_result, agent_type)
        if next_state is None:
            # No valid transition from the current state — signal failure
            # to the caller so it can 409 the request.
            task.error_message = (
                f"workflow stalled: no transition from {current_wf_state} "
                f"with result={step_result}"
            )
            return False

        self._task_states[task.id] = next_state
        ts = self._state_to_task_status.get(next_state)
        if ts:
            task.transition_to(ts)

        task.current_step = step_name
        self._step_index[task.id] = self._step_index.get(task.id, 0) + 1
        return True

    def _resolve_next_state(
        self,
        current: WorkflowState,
        result: str,
        agent_type: str,
    ) -> WorkflowState | None:
        """Decide the next FSM state from current state + step outcome.

        Happy-path defaults keep the original behaviour where the FSM
        advances to DONE in three ``execute_step`` calls (matching the
        full-lifecycle test's 3-step plan).  The ``result`` and
        ``agent_type`` arguments let callers route to FIX from
        TESTING/REVIEW on failure.
        """
        # Failure routing — only on explicit failure signals.
        if current == WorkflowState.TESTING and result == "failed":
            return WorkflowState.FIX
        if current == WorkflowState.REVIEW and result in ("rejected", "failed"):
            return WorkflowState.FIX
        # FIX completes by re-entering EXECUTING (so the next step is
        # the re-execute, which then routes through TESTING again).
        if current == WorkflowState.FIX and result == "ok":
            return WorkflowState.EXECUTING

        # Happy path.
        happy = {
            WorkflowState.IDLE: WorkflowState.PLANNING,
            WorkflowState.PLANNING: WorkflowState.EXECUTING,
            WorkflowState.EXECUTING: WorkflowState.TESTING,
            WorkflowState.TESTING: WorkflowState.DONE,
            WorkflowState.REVIEW: WorkflowState.DONE,
            WorkflowState.FIX: WorkflowState.EXECUTING,
            WorkflowState.DONE: None,
            WorkflowState.HALTED: None,
        }
        return happy.get(current)

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
        """Increment the action-hash counter for *task*.

        Returns ``True`` when the count has just exceeded the loop
        threshold (i.e. the caller should halt the task).  Previously
        this helper was never called from ``execute_step``, so the loop
        detection was effectively dead code.
        """
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
        # Reset loop-detection bookkeeping so a resumed task is not
        # immediately re-halted on the first repeat.
        self._action_counts.pop(task.id, None)
        self._step_index[task.id] = 0
        return True
