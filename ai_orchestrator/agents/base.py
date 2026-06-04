"""Base agent class and result model."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class AgentResult(BaseModel):
    """Result payload returned by every agent execution."""

    success: bool
    output: str
    actions_taken: list[dict] = []
    duration_ms: float = 0.0
    error: str | None = None


class BaseAgent(ABC):
    """Abstract base for all agent role implementations.

    Provides limit-checking, action recording, and a common run_step
    wrapper that concrete subclasses extend via ``execute``.
    """

    agent_type: str = "base"

    def __init__(
        self,
        agent_id: str,
        task_id: str,
        max_steps: int = 25,
        max_runtime_ms: int = 300000,
    ) -> None:
        self.agent_id = agent_id
        self.task_id = task_id
        self.max_steps = max_steps
        self.max_runtime_ms = max_runtime_ms
        self._step_count: int = 0
        self._start_time: float | None = None
        self._actions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> AgentResult:
        """Execute the agent's core logic.

        Every concrete role agent must implement this method.  The
        *context* dict carries task-level data such as the current
        step description.
        """
        ...

    async def run_step(self, context: dict[str, Any]) -> AgentResult:
        """Run a single step and record the action.

    1.  Checks limits via :meth:`check_limits`.
    2.  Initialises the runtime clock on first call.
    3.  Delegates to :meth:`execute`.
    4.  Records the action and returns the result.

        If limits have been exceeded a failure :class:`AgentResult`
        is returned immediately without calling ``execute``.
        """
        if not self.check_limits():
            return AgentResult(
                success=False,
                output="Limits exceeded",
                error="Max steps or runtime exceeded",
            )

        if self._start_time is None:
            self._start_time = time.monotonic()

        self._step_count += 1
        result = await self.execute(context)
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        result.duration_ms = elapsed_ms
        self.record_action(self.agent_type, {"step": self._step_count, "result": result.output})
        return result

    def check_limits(self) -> bool:
        """Return ``True`` if the agent may continue executing.

        Checks both the step count and the wall-clock runtime
        (measured from the first ``run_step`` call).  Returns
        ``False`` when either limit has been reached.
        """
        if self._step_count >= self.max_steps:
            return False
        if self._start_time is not None:
            elapsed_ms = (time.monotonic() - self._start_time) * 1000
            if elapsed_ms >= self.max_runtime_ms:
                return False
        return True

    def record_action(self, action: str, details: dict[str, Any] | None = None) -> None:
        """Append an action to the agent's internal action log."""
        entry: dict[str, Any] = {"action": action}
        if details is not None:
            entry["details"] = details
        self._actions.append(entry)
