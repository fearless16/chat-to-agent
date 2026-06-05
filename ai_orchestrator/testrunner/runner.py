"""Test runner — detect, execute, and parse test results in a workspace.

This module bridges :class:`Sandbox` (secure subprocess execution) and
:class:`FileWorkspace` (agent workspace) to provide a high-level test-run
loop.  It auto-detects the test framework, runs the appropriate command, and
returns structured :class:`TestRun` results that downstream code (the runtime
loop, reviewer agents) can consume without raw output scraping.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_orchestrator.security.sandbox import Sandbox, SandboxResult
    from ai_orchestrator.workspace.manager import FileWorkspace


class TestFramework(StrEnum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    NPM = "npm"
    GO = "go"
    UNKNOWN = "unknown"


# Maximum stderr size we'll parse for failure messages
_MAX_MESSAGE_CHARS = 10_000

# pytest verbose output line:  test_file.py::test_name PASSED [ 50%]
_PYTEST_LINE_RE = re.compile(
    r"^(?P<name>\S+)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED(?:\s*\(.*?\))?)"
    r"(?:\s+\[.*?\])?\s*$"
)
# pytest short summary failure line: "FAILED test_file.py::test_name - message"
_PYTEST_FAILURE_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+)")
_PYTEST_FAILURE_HEADER = "short test summary info"


@dataclass
class TestResult:
    """A single test case outcome."""

    name: str
    """Fully-qualified test name (e.g. ``tests/test_foo.py::test_bar``)."""

    passed: bool
    """``True`` if the test passed, ``False`` otherwise."""

    message: str = ""
    """Error / failure message for failed tests.  Empty for passes."""

    duration_ms: float = 0.0
    """Test-level duration in milliseconds (best-effort)."""


@dataclass
class TestRun:
    """Complete result of a test-suite execution."""

    framework: TestFramework
    total: int
    passed: int
    failed: int
    skipped: int = 0
    error: int = 0
    results: list[TestResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    timed_out: bool = False
    duration_ms: float = 0.0


class TestRunner:
    """High-level test runner that orchestrates detection, execution, and parsing.

    Usage::

        sandbox = Sandbox()
        runner = TestRunner(sandbox)

        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("test_foo.py", "def test_ok(): pass")

        run = await runner.run_tests(ws)
        print(run.passed, "/", run.total)
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect_framework(
        self,
        workspace: FileWorkspace,
    ) -> TestFramework:
        """Probe *workspace* and return the detected test framework.

        Checks for known config files in order of specificity:

        * ``pytest.ini``, ``setup.cfg`` (sections), ``pyproject.toml``,
          ``conftest.py``
        * ``package.json`` (with a ``test`` script)
        * ``go.mod``
        * ``Makefile`` (with a ``test`` target)

        Returns :attr:`TestFramework.UNKNOWN` when nothing matches.
        """
        # Pytest indicators (most specific first)
        for marker in ("pytest.ini", "conftest.py", "setup.cfg", "pyproject.toml"):
            if workspace.exists(marker):
                if marker in ("setup.cfg", "pyproject.toml"):
                    pkg = workspace.read(marker)
                    if marker == "setup.cfg" and "[tool:pytest]" not in pkg:
                        continue
                    if marker == "pyproject.toml" and "[tool.pytest" not in pkg:
                        continue
                return TestFramework.PYTEST
        # Check for _test.py / test_*.py files as a last pytest resort
        tree = workspace.list_tree()
        for entry in tree:
            if entry.path.startswith("test_") or entry.path.endswith("_test.py"):
                return TestFramework.PYTEST

        # npm / node
        if workspace.exists("package.json"):
            return TestFramework.NPM

        # Go
        if workspace.exists("go.mod"):
            return TestFramework.GO

        return TestFramework.UNKNOWN

    async def run_tests(
        self,
        workspace: FileWorkspace,
        framework: TestFramework | None = None,
        args: list[str] | None = None,
        timeout_ms: int = 120_000,
    ) -> TestRun:
        """Run tests in *workspace* and return structured results.

        Parameters
        ----------
        workspace:
            The workspace whose test suite to run.
        framework:
            Explicit framework.  Auto-detected when ``None``.
        args:
            Extra arguments to pass to the test command.
        timeout_ms:
            Maximum wall-clock time for the full test run.
        """
        if framework is None:
            framework = await self.detect_framework(workspace)
        args = args or []

        t0 = time.monotonic()

        if framework == TestFramework.PYTEST:
            raw = await self._sandbox.execute_python_module(
                "pytest",
                args=[
                    "-v",
                    "--tb=short",
                    "-p", "no:cacheprovider",
                    "--no-header",
                    *args,
                ],
                workdir=str(workspace.workspace_root),
                timeout_ms=timeout_ms,
            )
        elif framework == TestFramework.NPM:
            raw = await self._sandbox.execute_bash(
                f"npm test {' '.join(args)}" if args else "npm test",
                workdir=str(workspace.workspace_root),
                timeout_ms=timeout_ms,
            )
        elif framework == TestFramework.GO:
            raw = await self._sandbox.execute_command(
                "go",
                args=["test", "./...", "-v", *args],
                workdir=str(workspace.workspace_root),
                timeout_ms=timeout_ms,
            )
        else:
            # Unknown framework — try a best-effort bash run
            raw = await self._sandbox.execute_bash(
                "make test" if workspace.exists("Makefile") else "echo 'no test command'",
                workdir=str(workspace.workspace_root),
                timeout_ms=timeout_ms,
            )

        duration = (time.monotonic() - t0) * 1000.0
        return self._parse_raw(framework, raw, duration)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def _parse_raw(
        cls,
        framework: TestFramework,
        raw: SandboxResult,
        duration_ms: float,
    ) -> TestRun:
        if framework == TestFramework.PYTEST:
            return cls._parse_pytest(raw, duration_ms)
        passed = raw.return_code == 0 and not raw.timed_out
        return TestRun(
            framework=framework,
            total=1 if passed else 0,
            passed=1 if passed else 0,
            failed=0 if passed else 1,
            return_code=raw.return_code,
            timed_out=raw.timed_out,
            duration_ms=duration_ms,
            stdout=raw.stdout,
            stderr=raw.stderr,
        )

    @classmethod
    def _parse_pytest(
        cls,
        raw: SandboxResult,
        duration_ms: float,
    ) -> TestRun:
        stdout = raw.stdout

        # Collect failure messages from the short summary section
        failure_messages: dict[str, str] = {}
        in_failures = False
        for line in stdout.splitlines():
            if _PYTEST_FAILURE_HEADER in line:
                in_failures = True
                continue
            if in_failures:
                m = _PYTEST_FAILURE_LINE.match(line)
                if m:
                    test_name = m.group(2)
                    failure_messages[test_name] = line

        results: list[TestResult] = []
        summary_counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}

        for line in stdout.splitlines():
            m = _PYTEST_LINE_RE.match(line)
            if not m:
                continue

            name = m.group("name")
            status = m.group("status").split()[0]  # "SKIPPED (reason)" → "SKIPPED"

            if status == "PASSED":
                results.append(TestResult(name=name, passed=True))
                summary_counts["passed"] += 1
            elif status == "SKIPPED":
                results.append(TestResult(name=name, passed=True))
                summary_counts["skipped"] += 1
            elif status in ("FAILED", "ERROR"):
                msg = failure_messages.get(name, "")
                results.append(TestResult(name=name, passed=False, message=msg))
                summary_counts["failed" if status == "FAILED" else "error"] += 1

        # If parsing per-test lines yielded nothing, fall back to footer counts
        if not results and not raw.timed_out:
            passed_ft, failed_ft, skipped_ft = cls._extract_footer_counts(stdout)
            total_ft = passed_ft + failed_ft + skipped_ft
            if total_ft > 0:
                if failed_ft == 0:
                    for _ in range(total_ft):
                        results.append(TestResult(name="<unknown>", passed=True))
                else:
                    for f_name, f_msg in failure_messages.items():
                        results.append(TestResult(name=f_name, passed=False, message=f_msg))
                    unknown_passed = total_ft - len(results)
                    for _ in range(unknown_passed):
                        results.append(TestResult(name="<unknown>", passed=True))
                summary_counts["passed"] = passed_ft
                summary_counts["failed"] = failed_ft
                summary_counts["skipped"] = skipped_ft

        return TestRun(
            framework=TestFramework.PYTEST,
            total=sum(summary_counts.values()),
            passed=summary_counts["passed"],
            failed=summary_counts["failed"],
            skipped=summary_counts["skipped"],
            error=summary_counts["error"],
            results=results,
            stdout=stdout,
            stderr=raw.stderr,
            return_code=raw.return_code,
            timed_out=raw.timed_out,
            duration_ms=duration_ms,
        )

    @classmethod
    def _extract_footer_counts(cls, stdout: str) -> tuple[int, int, int]:
        """Extract (passed, failed, skipped) from the pytest footer line.

        Handles any order of counts: "1 failed, 2 passed", "2 passed, 1 failed",
        "1 passed, 1 skipped", etc.  Returns (0, 0, 0) when no footer found.
        """
        passed = failed = skipped = 0
        for line in stdout.splitlines():
            if "passed" in line or "failed" in line or "skipped" in line:
                for token in re.findall(r"(\d+)\s+(passed|failed|skipped)", line):
                    count = int(token[0])
                    kind = token[1]
                    if kind == "passed":
                        passed += count
                    elif kind == "failed":
                        failed += count
                    elif kind == "skipped":
                        skipped += count
        return passed, failed, skipped
