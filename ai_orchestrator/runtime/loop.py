"""Runtime loop — fix-analysis-test cycle for a task workspace.

The :class:`RuntimeLoop` orchestrates the end-to-end cycle that every code-
generation agent goes through:

1. **Build** — the workspace is populated with code
2. **Test** — :class:`TestRunner` detects and runs the test suite
3. **Analyze** — failing tests are identified and reported to a fix callback
4. **Fix** — the callback updates workspace files; the cycle repeats
5. **Conclude** — all tests pass or the iteration budget is exhausted

Git is layered over the workspace so every successful test-run is a commit
and every fix cycle is visible in the history.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Coroutine

from ai_orchestrator.runtime.types import FixResult
from ai_orchestrator.workspace.git import GitWorkspace

if TYPE_CHECKING:
    from ai_orchestrator.security.sandbox import Sandbox
    from ai_orchestrator.testrunner.runner import TestFramework, TestRunner, TestRun
    from ai_orchestrator.workspace.manager import FileWorkspace


FixCallback = Callable[
    ["TestRun", "FileWorkspace", int],
    Coroutine[None, None, "FixResult"],
]


@dataclass
class CycleRecord:
    """Record of one test-run cycle."""

    iteration: int
    test_run: TestRun
    fix_applied: bool = False
    committed: bool = False
    duration_ms: float = 0.0


@dataclass
class LoopResult:
    """Final result of a :meth:`RuntimeLoop.run` call."""

    success: bool
    test_run: TestRun
    cycles: list[CycleRecord] = field(default_factory=list)
    iterations_used: int = 0
    total_duration_ms: float = 0.0


class RuntimeLoop:
    """Orchestrate the build-test-fix cycle for one task workspace.

    Parameters
    ----------
    sandbox:
        Sandbox used for test execution (wraps subprocess with resource
        limits).
    test_runner:
        TestRunner instance (auto-created from *sandbox* if omitted).
    fix_callback:
        Async callable invoked on each test failure with
        ``(test_run, workspace, iteration)``.  Should update workspace
        files and return a :class:`FixResult`.  When ``None`` the loop
        stops after the first test run (effectively a single-shot test).
    author_name, author_email:
        Identity stamped on git commits within each cycle.
    """

    MAX_ITERATIONS: int = 5

    def __init__(
        self,
        sandbox: Sandbox,
        test_runner: TestRunner | None = None,
        fix_callback: FixCallback | None = None,
        *,
        author_name: str = "ai-orchestrator",
        author_email: str = "ai@orchestrator.local",
    ) -> None:
        self._sandbox = sandbox
        self._test_runner = test_runner
        self._fix_callback = fix_callback
        self._author_name = author_name
        self._author_email = author_email

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def fix_callback(self) -> FixCallback | None:
        return self._fix_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        workspace: FileWorkspace,
        *,
        framework: TestFramework | None = None,
        max_iterations: int | None = None,
    ) -> LoopResult:
        """Execute the full build-test-fix cycle in *workspace*.

        Parameters
        ----------
        workspace:
            The task workspace to operate on.  Must already contain code.
        framework:
            Test framework override.  Auto-detected when ``None``.
        max_iterations:
            Maximum number of test-fix cycles (defaults to
            :attr:`MAX_ITERATIONS`).

        Returns
        -------
        LoopResult
            A summary of all cycles and the final test run.
        """
        max_iterations = max_iterations or self.MAX_ITERATIONS
        t0 = time.monotonic()

        # Ensure git is initialised so we can track history
        git_ws = GitWorkspace(
            workspace,
            author_name=self._author_name,
            author_email=self._author_email,
        )
        await git_ws.init()

        # Resolve a test runner (create one from sandbox if not given)
        test_runner = self._test_runner or _default_runner(self._sandbox)

        cycles: list[CycleRecord] = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            cycle_t0 = time.monotonic()

            # Auto-detect framework on first iteration
            fw = framework or await test_runner.detect_framework(workspace)

            # Purge stale Python bytecode so fixes take effect
            self._clean_pycache(workspace.workspace_root)

            test_run = await test_runner.run_tests(workspace, framework=fw)

            # Commit the test run state (even on failure, so we can roll
            # back to the last known-good point).
            msg = f"cycle {iteration}: {test_run.passed}/{test_run.total} passed"
            try:
                await git_ws.commit(msg)
                committed = True
            except Exception:
                committed = False

            passed = test_run.timed_out or (
                test_run.return_code == 0
                and test_run.failed == 0
                and test_run.error == 0
            )

            cycle_duration = (time.monotonic() - cycle_t0) * 1000.0

            if passed or self._fix_callback is None:
                # Tests pass or there's no fixer — we're done
                cycles.append(CycleRecord(
                    iteration=iteration,
                    test_run=test_run,
                    fix_applied=False,
                    committed=committed,
                    duration_ms=round(cycle_duration, 2),
                ))
                total_duration = (time.monotonic() - t0) * 1000.0
                return LoopResult(
                    success=passed,
                    test_run=test_run,
                    cycles=cycles,
                    iterations_used=iteration,
                    total_duration_ms=round(total_duration, 2),
                )

            # Tests failed — attempt a fix
            fix_result = await self._fix_callback(test_run, workspace, iteration)
            cycles.append(CycleRecord(
                iteration=iteration,
                test_run=test_run,
                fix_applied=fix_result.fixes_applied > 0,
                committed=committed,
                duration_ms=round(cycle_duration, 2),
            ))

            if fix_result.abort:
                total_duration = (time.monotonic() - t0) * 1000.0
                return LoopResult(
                    success=False,
                    test_run=test_run,
                    cycles=cycles,
                    iterations_used=iteration,
                    total_duration_ms=round(total_duration, 2),
                )

        # Exhausted iterations without passing
        total_duration = (time.monotonic() - t0) * 1000.0
        return LoopResult(
            success=False,
            test_run=test_run,
            cycles=cycles,
            iterations_used=iteration,
            total_duration_ms=round(total_duration, 2),
        )


    @staticmethod
    def _clean_pycache(root: Path) -> None:
        """Recursively delete ``__pycache__/`` directories under *root*.

        Prevents stale bytecode from masking applied fixes between cycles.
        """
        for pycache in root.rglob("__pycache__"):
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)


def _default_runner(sandbox: Sandbox) -> TestRunner:
    """Lazy-import to avoid circular dependencies."""
    from ai_orchestrator.testrunner.runner import TestRunner
    return TestRunner(sandbox)
