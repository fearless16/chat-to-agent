"""Tests for the TestRunner — detection, execution, and result parsing."""

import textwrap
from pathlib import Path

import pytest

from ai_orchestrator.security.sandbox import Sandbox, SandboxResult
from ai_orchestrator.testrunner.runner import (
    TestFramework,
    TestResult,
    TestRun,
    TestRunner,
)


# ---------------------------------------------------------------------------
# TestResult / TestRun dataclass basics
# ---------------------------------------------------------------------------

class TestTestResult:
    """TestResult dataclass behaviour."""

    def test_defaults(self):
        r = TestResult(name="t", passed=True)
        assert r.name == "t"
        assert r.passed is True
        assert r.message == ""
        assert r.duration_ms == 0.0

    def test_failure_with_message(self):
        r = TestResult(name="t", passed=False, message="assert 1 == 2")
        assert r.passed is False
        assert "assert" in r.message


class TestTestRun:
    """TestRun dataclass behaviour."""

    def test_defaults(self):
        r = TestRun(framework=TestFramework.UNKNOWN, total=0, passed=0, failed=0)
        assert r.skipped == 0
        assert r.error == 0
        assert r.results == []
        assert r.timed_out is False


# ---------------------------------------------------------------------------
# detect_framework
# ---------------------------------------------------------------------------

class TestDetectFramework:
    """Framework detection via workspace probing."""

    @pytest.mark.asyncio
    async def test_pytest_ini_detected(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("pytest.ini", "[pytest]\n")
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.PYTEST
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_conftest_detected(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("conftest.py", "")
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.PYTEST
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_setup_cfg_with_pytest(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("setup.cfg", "[tool:pytest]\n")
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.PYTEST
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_setup_cfg_without_pytest(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("setup.cfg", "[metadata]\nname = foo\n")
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw != TestFramework.PYTEST
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_pyproject_toml_with_pytest(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("pyproject.toml", "[tool.pytest.ini_options]\n")
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.PYTEST
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_test_file_detected(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("test_foo.py", "def test_x(): pass\n")
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.PYTEST
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_npm_detected(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("package.json", '{"scripts": {"test": "echo ok"}}\n')
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.NPM
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_go_detected(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("go.mod", "module test\n")
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.GO
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_empty_workspace_unknown(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            runner = TestRunner(sandbox)
            fw = await runner.detect_framework(ws)
            assert fw == TestFramework.UNKNOWN
        finally:
            await sandbox.close()


# ---------------------------------------------------------------------------
# run_tests — real execution
# ---------------------------------------------------------------------------

class TestRunTests:
    """End-to-end test execution via run_tests."""

    @pytest.mark.asyncio
    async def test_pytest_all_pass(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("test_ok.py", textwrap.dedent("""\
                def test_one():
                    assert 1 == 1
                def test_two():
                    assert 2 == 2
            """))
            runner = TestRunner(sandbox)
            run = await runner.run_tests(ws)
            assert run.framework == TestFramework.PYTEST
            assert run.total == 2
            assert run.passed == 2
            assert run.failed == 0
            assert run.return_code == 0
            assert not run.timed_out
            assert run.duration_ms > 0
            assert len(run.results) == 2
            assert all(r.passed for r in run.results)
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_pytest_one_fail(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("test_fail.py", textwrap.dedent("""\
                def test_pass():
                    assert 1 == 1
                def test_fail():
                    assert 0 == 1
            """))
            runner = TestRunner(sandbox)
            run = await runner.run_tests(ws)
            assert run.framework == TestFramework.PYTEST
            assert run.total == 2
            assert run.passed == 1
            assert run.failed == 1
            assert run.return_code != 0
            assert len(run.results) == 2
            passed = [r for r in run.results if r.passed]
            failed = [r for r in run.results if not r.passed]
            assert len(passed) == 1
            assert len(failed) == 1
            assert "test_fail" in failed[0].name
            assert failed[0].message
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_explicit_framework(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("test_explicit.py", "def test_ok(): pass\n")
            runner = TestRunner(sandbox)
            run = await runner.run_tests(ws, framework=TestFramework.PYTEST)
            assert run.framework == TestFramework.PYTEST
            assert run.total == 1
            assert run.passed == 1
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("test_slow.py", "import time; def test_slow(): time.sleep(10)\n")
            runner = TestRunner(sandbox)
            run = await runner.run_tests(ws, timeout_ms=300)
            assert run.timed_out is True
            assert run.return_code != 0
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_unknown_framework(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("readme.txt", "hello\n")
            runner = TestRunner(sandbox)
            run = await runner.run_tests(ws)
            assert run.framework == TestFramework.UNKNOWN
            # Unknown runs echo "no test command", which succeeds
            assert run.return_code == 0
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_with_extra_args(self, tmp_path: Path):
        from ai_orchestrator.workspace.manager import FileWorkspace
        sandbox = Sandbox()
        try:
            ws = FileWorkspace.for_task("t", root=tmp_path)
            ws.write("test_args.py", "def test_ok(): pass\ndef test_skip(): pass\n")
            runner = TestRunner(sandbox)
            run = await runner.run_tests(
                ws,
                framework=TestFramework.PYTEST,
                args=["-k", "test_ok"],
            )
            assert run.total == 1
            assert run.passed == 1
        finally:
            await sandbox.close()


# ---------------------------------------------------------------------------
# Pytest output parsing
# ---------------------------------------------------------------------------

class TestPytestParsing:
    """Unit tests for _parse_pytest with synthetic output."""

    def _make_raw(self, stdout: str, return_code: int = 0, timed_out: bool = False) -> SandboxResult:
        return SandboxResult(
            stdout=stdout,
            stderr="",
            return_code=return_code,
            timed_out=timed_out,
            duration_ms=100.0,
        )

    def test_parse_all_pass(self):
        stdout = textwrap.dedent("""\
            tests/test_foo.py::test_a PASSED
            tests/test_foo.py::test_b PASSED

            ====== 2 passed in 0.01s ======
        """)
        raw = self._make_raw(stdout, return_code=0)
        run = TestRunner._parse_pytest(raw, 100.0)

        assert run.total == 2
        assert run.passed == 2
        assert run.failed == 0
        assert len(run.results) == 2
        assert all(r.passed for r in run.results)
        assert all(r.name for r in run.results)

    def test_parse_one_failure(self):
        stdout = textwrap.dedent("""\
            tests/test_foo.py::test_a PASSED
            tests/test_foo.py::test_b FAILED

            ====== short test summary info ======
            FAILED tests/test_foo.py::test_b - AssertionError: assert 0 == 1
            ====== 1 passed, 1 failed in 0.02s ======
        """)
        raw = self._make_raw(stdout, return_code=1)
        run = TestRunner._parse_pytest(raw, 100.0)

        assert run.total == 2
        assert run.passed == 1
        assert run.failed == 1
        passed = [r for r in run.results if r.passed]
        failed = [r for r in run.results if not r.passed]
        assert len(passed) == 1
        assert len(failed) == 1
        assert "test_b" in failed[0].name
        assert "AssertionError" in failed[0].message

    def test_parse_with_skip(self):
        stdout = textwrap.dedent("""\
            tests/test_foo.py::test_a PASSED
            tests/test_foo.py::test_b SKIPPED (reason)
            tests/test_foo.py::test_c PASSED

            ====== 2 passed, 1 skipped in 0.01s ======
        """)
        raw = self._make_raw(stdout, return_code=0)
        run = TestRunner._parse_pytest(raw, 100.0)

        assert run.total == 3
        assert run.passed == 2
        assert run.skipped == 1
        assert run.failed == 0
        assert len(run.results) == 3

    def test_parse_empty_suite(self):
        stdout = "====== no tests ran in 0.00s ======\n"
        raw = self._make_raw(stdout, return_code=5)
        run = TestRunner._parse_pytest(raw, 100.0)

        assert run.total == 0
        assert run.passed == 0
        assert run.failed == 0
        assert len(run.results) == 0

    def test_parse_timeout(self):
        stdout = ""
        raw = self._make_raw(stdout, return_code=-1, timed_out=True)
        run = TestRunner._parse_pytest(raw, 100.0)

        assert run.total == 0
        assert run.timed_out is True
        assert run.return_code == -1

    def test_parse_all_fail_no_footer(self):
        """Parser handles output without a footer line."""
        stdout = textwrap.dedent("""\
            tests/test_bad.py::test_x FAILED

            tests/test_bad.py:2: AssertionError
        """)
        raw = self._make_raw(stdout, return_code=1)
        run = TestRunner._parse_pytest(raw, 100.0)

        # Should still capture at least one failure
        assert run.failed >= 1
        assert run.return_code == 1

    def test_parse_with_error(self):
        stdout = textwrap.dedent("""\
            tests/test_err.py::test_a ERROR

            ====== short test summary info ======
            ERROR tests/test_err.py::test_a - RuntimeError: boom
            ====== 1 error in 0.01s ======
        """)
        raw = self._make_raw(stdout, return_code=1)
        run = TestRunner._parse_pytest(raw, 100.0)

        assert run.error >= 1
        assert run.failed == 0
        assert run.total == 1
