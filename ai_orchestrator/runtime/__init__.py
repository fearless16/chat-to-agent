"""Runtime orchestration — fix-analysis-test cycle for task workspaces.

The :class:`RuntimeLoop` coordinates :class:`Sandbox`, :class:`TestRunner`,
and :class:`GitWorkspace` to drive the build-test-fix loop that every
code-generation agent follows.
"""

from ai_orchestrator.runtime.loop import CycleRecord, LoopResult, RuntimeLoop
from ai_orchestrator.runtime.types import FixResult

__all__ = [
    "RuntimeLoop",
    "LoopResult",
    "CycleRecord",
    "FixResult",
]
