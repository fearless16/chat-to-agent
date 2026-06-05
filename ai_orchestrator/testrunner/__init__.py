"""Test runner — detect, execute, and parse test results in a workspace.

Bridges :class:`Sandbox` (secure subprocess execution) and
:class:`FileWorkspace` (agent workspace) to provide a high-level test-run
loop for the fix-analysis-retry cycle.
"""

from ai_orchestrator.testrunner.runner import (
    TestFramework,
    TestResult,
    TestRun,
    TestRunner,
)

__all__ = [
    "TestFramework",
    "TestResult",
    "TestRun",
    "TestRunner",
]
