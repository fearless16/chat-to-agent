"""Tests for RuntimeLoop — the fix-analysis-test cycle."""

import textwrap
from pathlib import Path

import pytest

from ai_orchestrator.runtime.loop import CycleRecord, LoopResult, RuntimeLoop
from ai_orchestrator.runtime.types import FixResult
from ai_orchestrator.security.sandbox import Sandbox


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------

class TestCycleRecord:
    def test_defaults(self):
        r = CycleRecord(iteration=1, test_run=None)  # type: ignore[arg-type]
        assert r.fix_applied is False
        assert r.committed is False
        assert r.duration_ms == 0.0


class TestLoopResult:
    def test_defaults(self):
        r = LoopResult(success=True, test_run=None)  # type: ignore[arg-type]
        assert r.cycles == []
        assert r.iterations_used == 0
        assert r.total_duration_ms == 0.0


class TestFixResult:
    def test_defaults(self):
        r = FixResult()
        assert r.fixes_applied == 0
        assert r.message == ""
        assert r.abort is False


# ---------------------------------------------------------------------------
# RuntimeLoop — integration tests
# ---------------------------------------------------------------------------

async def _fix_assert(test_run, workspace, iteration: int) -> FixResult:
    """Fix callback: replaces ``assert 0 == 1`` with ``assert 1 == 1``."""
    fixes = 0
    for entry in workspace.list_tree():
        if not entry.path.startswith("test_") and not entry.path.endswith("_test.py"):
            continue
        if not entry.is_file:
            continue
        content = workspace.read(entry.path)
        if "assert 0 == 1" in content:
            content = content.replace("assert 0 == 1", "assert 1 == 1")
            workspace.write(entry.path, content)
            fixes += 1
    return FixResult(fixes_applied=fixes, message=f"fixed {fixes} files")


class TestRuntimeLoop:
    """End-to-end runtime loop tests with real sandbox + workspace."""

    @pytest.mark.asyncio
    async def test_happy_path_all_pass(self, tmp_path: Path):
        """Workspace with all-passing tests succeeds in one cycle."""
        from ai_orchestrator.workspace.manager import FileWorkspace

        ws = FileWorkspace.for_task("happy", root=tmp_path)
        ws.write("test_ok.py", "def test_pass(): assert 1 == 1\n")

        sandbox = Sandbox()
        try:
            loop = RuntimeLoop(sandbox)
            result = await loop.run(ws)
            assert result.success is True
            assert result.iterations_used == 1
            assert len(result.cycles) == 1
            assert result.test_run.total == 1
            assert result.test_run.passed == 1
            assert result.test_run.failed == 0
            assert result.total_duration_ms > 0
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_fix_callback_repairs_failure(self, tmp_path: Path):
        """Fix callback corrects a failing test and loop succeeds."""
        from ai_orchestrator.workspace.manager import FileWorkspace

        ws = FileWorkspace.for_task("fixable", root=tmp_path)
        ws.write("test_fix.py", "def test_bad(): assert 0 == 1\n")

        sandbox = Sandbox()
        try:
            loop = RuntimeLoop(sandbox, fix_callback=_fix_assert)
            result = await loop.run(ws, max_iterations=3)
            assert result.success is True
            # Should take 2 iterations: fail → fix → pass
            assert result.iterations_used == 2
            assert len(result.cycles) == 2
            assert result.cycles[0].test_run.failed == 1
            assert result.cycles[1].test_run.passed == 1
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, tmp_path: Path):
        """Loop stops after max_iterations when fix can't resolve."""
        from ai_orchestrator.workspace.manager import FileWorkspace

        ws = FileWorkspace.for_task("hopeless", root=tmp_path)
        ws.write("test_bad.py", "def test_bad(): assert 0 == 1\n")

        async def noop_fix(tc, ws, it):
            return FixResult(fixes_applied=0, message="can't fix")

        sandbox = Sandbox()
        try:
            loop = RuntimeLoop(sandbox, fix_callback=noop_fix)
            result = await loop.run(ws, max_iterations=3)
            assert result.success is False
            assert result.iterations_used == 3
            assert len(result.cycles) == 3
            for c in result.cycles:
                assert c.test_run.failed == 1
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_fix_callback_abort(self, tmp_path: Path):
        """Fix callback returning abort=True stops the loop."""
        from ai_orchestrator.workspace.manager import FileWorkspace

        ws = FileWorkspace.for_task("abort", root=tmp_path)
        ws.write("test_abort.py", "def test_bad(): assert 0 == 1\n")

        async def abort_fix(tc, ws, it):
            return FixResult(fixes_applied=0, message="need human help", abort=True)

        sandbox = Sandbox()
        try:
            loop = RuntimeLoop(sandbox, fix_callback=abort_fix)
            result = await loop.run(ws, max_iterations=5)
            assert result.success is False
            assert result.iterations_used == 1
            assert len(result.cycles) == 1
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_no_fix_callback_single_shot(self, tmp_path: Path):
        """Without fix_callback, loop runs tests once even on failure."""
        from ai_orchestrator.workspace.manager import FileWorkspace

        ws = FileWorkspace.for_task("single", root=tmp_path)
        ws.write("test_fail.py", "def test_bad(): assert 0 == 1\n")

        sandbox = Sandbox()
        try:
            loop = RuntimeLoop(sandbox)
            result = await loop.run(ws, max_iterations=10)
            assert result.success is False
            assert result.iterations_used == 1
            assert len(result.cycles) == 1
        finally:
            await sandbox.close()

    @pytest.mark.asyncio
    async def test_git_commits_on_each_cycle(self, tmp_path: Path):
        """Each cycle creates a git commit in the workspace."""
        from ai_orchestrator.workspace.manager import FileWorkspace

        ws = FileWorkspace.for_task("gittrack", root=tmp_path)
        ws.write("test_ok.py", "def test_pass(): assert 1 == 1\n")

        sandbox = Sandbox()
        try:
            loop = RuntimeLoop(sandbox)
            result = await loop.run(ws)
            assert result.success is True

            # Verify git is active and tracked the workspace
            from ai_orchestrator.workspace.git import GitWorkspace
            git_ws = GitWorkspace(ws)
            log = await git_ws.log(max_count=5)
            assert len(log) >= 1
            # Status should be clean (all changes committed)
            assert await git_ws.is_clean()
        finally:
            await sandbox.close()
