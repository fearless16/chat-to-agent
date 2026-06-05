"""Tests for GitWorkspace — version control for task workspaces."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from ai_orchestrator.workspace import (
    CommitInfo,
    FileStatus,
    GitConflictError,
    GitError,
    GitNotInstalled,
    GitNotRepoError,
    GitWorkspace,
)
from ai_orchestrator.workspace.exceptions import WorkspaceNotFoundError
from ai_orchestrator.workspace.manager import FileWorkspace


# Skipped if git is not available.
pytestmark = pytest.mark.skipif(
    not any(
        (Path(d) / "git").is_file() and os.access(Path(d) / "git", os.X_OK)
        for d in os.environ.get("PATH", "").split(os.pathsep)
    ),
    reason="git executable not found on PATH",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> FileWorkspace:
    return FileWorkspace.for_task("git-test", root=tmp_path)


@pytest.fixture
async def gw(workspace: FileWorkspace) -> GitWorkspace:
    g = GitWorkspace(workspace)
    await g.init()
    return g


# ---------------------------------------------------------------------------
# Construction & init
# ---------------------------------------------------------------------------


class TestInit:
    async def test_init_creates_repo(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        assert await g.is_initialized() is False
        await g.init()
        assert await g.is_initialized() is True

    async def test_init_is_idempotent(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        await g.init()
        await g.init()
        assert await g.is_initialized() is True

    async def test_init_creates_branch(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace, author_name="tester", author_email="t@t.com")
        await g.init()
        assert await g.current_branch() == "main"

    async def test_init_with_custom_default_branch(
        self, workspace: FileWorkspace
    ) -> None:
        g = GitWorkspace(workspace)
        await g.init(default_branch="trunk")
        assert await g.current_branch() == "trunk"

    async def test_current_branch_detached(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        await g.init()
        sha = await g.current_sha()
        await g.checkout(sha)
        assert await g.current_branch() is None

    async def test_current_sha_returns_hex(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        await g.init()
        sha = await g.current_sha()
        assert re.fullmatch(r"[0-9a-f]{40}", sha)

    async def test_init_creates_initial_commit(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        await g.init(initial_message="bootstrap")
        log = await g.log()
        assert len(log) >= 1
        assert log[0].message == "bootstrap"

    async def test_init_stages_existing_files(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("prepop", root=tmp_path)
        ws.write("a.txt", "hello")
        g = GitWorkspace(ws)
        await g.init()
        assert await g.file_count() == 1

    async def test_not_a_repo_raises(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        with pytest.raises(GitNotRepoError):
            await g.current_branch()
        with pytest.raises(GitNotRepoError):
            await g.commit("x")


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------


class TestBranch:
    async def test_create_and_switch(self, gw: GitWorkspace) -> None:
        await gw.branch("task/my-feature")
        assert await gw.current_branch() == "task/my-feature"

    async def test_duplicate_branch_raises(self, gw: GitWorkspace) -> None:
        await gw.branch("task/a")
        with pytest.raises(GitError):
            await gw.branch("task/a")

    async def test_branch_from_custom_ref(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("on-main.txt", "main content")
        await gw.commit("feat: on main")
        sha_before = await gw.current_sha()

        await gw.branch("task/feature", from_ref="main")
        assert await gw.current_branch() == "task/feature"
        content = ws.read("on-main.txt")
        assert content == "main content"

    async def test_branch_without_checkout(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        await g.init()
        await g.branch("task/b1")
        # Now on task/b1
        assert await g.current_branch() == "task/b1"


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------


class TestCheckout:
    async def test_checkout_branch(self, gw: GitWorkspace) -> None:
        await gw.branch("task/feat")
        await gw.checkout("main")
        assert await gw.current_branch() == "main"

    async def test_checkout_commit(self, gw: GitWorkspace) -> None:
        sha = await gw.current_sha()
        await gw.checkout(sha)
        assert await gw.current_branch() is None

    async def test_checkout_nonexistent_raises(self, gw: GitWorkspace) -> None:
        with pytest.raises(GitError):
            await gw.checkout("no-such-branch")


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


class TestCommit:
    async def test_commit_creates(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "content")
        c = await gw.commit("feat: a")
        assert len(c.sha) == 40
        assert c.message == "feat: a"
        assert c.author_name == "ai-orchestrator"

    async def test_commit_clean_tree_raises(self, gw: GitWorkspace) -> None:
        with pytest.raises(GitError, match="nothing to commit"):
            await gw.commit("nothing")

    async def test_commit_stages_automatically(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "v1")
        c1 = await gw.commit("first")
        ws.write("a.txt", "v2")
        ws.write("b.txt", "new")
        c2 = await gw.commit("second")
        assert c1.sha != c2.sha

    async def test_commit_adds_parents(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "1")
        c1 = await gw.commit("c1")
        ws.write("b.txt", "2")
        c2 = await gw.commit("c2")
        assert c1.sha == c2.parents[0]

    async def test_commit_info_timestamp(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "1")
        c = await gw.commit("c1")
        assert c.timestamp.year >= 2024


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


class TestDiff:
    async def test_diff_shows_new_file(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("base.txt", "base")
        await gw.commit("c1 on main")
        await gw.branch("task/add")
        ws.write("new.txt", "new content")
        await gw.commit("c2 on branch")
        diff = await gw.diff("main")
        assert "new.txt" in diff
        assert "new content" in diff

    async def test_diff_shows_modified(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "old")
        await gw.commit("c1")
        await gw.branch("task/change")
        ws.write("a.txt", "new content")
        await gw.commit("c2")
        diff = await gw.diff("main")
        assert "-old" in diff
        assert "+new content" in diff

    async def test_diff_empty_for_same(self, gw: GitWorkspace) -> None:
        diff = await gw.diff("main")
        assert diff.strip() == ""

    async def test_diff_uncommitted(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "original")
        await gw.commit("c1")
        ws.write("a.txt", "modified")
        diff = await gw.diff_uncommitted()
        assert "a.txt" in diff
        assert "+modified" in diff or "modified" in diff

    async def test_diff_staged(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "v1")
        await gw.commit("initial")
        ws.write("a.txt", "v2")
        # Now stage it
        from subprocess import run
        run(["git", "add", "a.txt"], cwd=gw.root, check=True)
        staged = await gw.diff_staged()
        assert "v2" in staged
        # Working tree matches index → no unstaged diff
        working = await gw.diff_working_tree()
        assert working.strip() == ""
        # diff_uncommitted still shows v1→v2 (all vs HEAD)
        uncommitted = await gw.diff_uncommitted()
        assert "v2" in uncommitted

    async def test_diff_with_context_lines(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "\n".join(f"line{i}" for i in range(50)))
        await gw.commit("c1 on main")
        await gw.branch("task/ctx")
        ws.write("a.txt", "\n".join(f"line{i}" for i in range(50)) + "\nnew line")
        await gw.commit("c2 on branch")
        diff = await gw.diff("main", context=0)
        assert "a.txt" in diff
        assert "new line" in diff


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    async def test_rollback_removes_changes(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "new file")
        await gw.commit("c1")
        ws.write("a.txt", "modified")
        await gw.commit("c2")
        await gw.rollback("HEAD~1")
        assert ws.read("a.txt") == "new file"

    async def test_rollback_to_initial_state(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "temp")
        await gw.commit("c1")
        await gw.rollback("HEAD~1")  # back to initial empty commit
        with pytest.raises(WorkspaceNotFoundError):
            ws.read("a.txt")

    async def test_rollback_with_uncommitted(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "v1")
        await gw.commit("c1")
        ws.write("a.txt", "uncommitted")
        await gw.rollback("HEAD")  # discard uncommitted
        assert ws.read("a.txt") == "v1"


# ---------------------------------------------------------------------------
# Soft reset & restore
# ---------------------------------------------------------------------------


class TestSoftReset:
    async def test_soft_reset_keeps_changes(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "v1")
        await gw.commit("c1")
        ws.write("a.txt", "v2")
        ws.write("b.txt", "new")
        await gw.commit("c2")
        await gw.reset_soft("HEAD~1")
        # The files should still be there, but not committed
        assert ws.read("a.txt") == "v2"
        assert ws.read("b.txt") == "new"
        log = await gw.log()
        assert len(log) >= 1  # c1 still there

    async def test_restore_discards_unstaged(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "committed")
        await gw.commit("c1")
        ws.write("a.txt", "dirty")
        await gw.restore("a.txt")
        assert ws.read("a.txt") == "committed"

    async def test_restore_all_by_default(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "committed")
        await gw.commit("c1")
        ws.write("a.txt", "dirty1")
        ws.write("b.txt", "dirty2")  # untracked
        await gw.restore()
        assert ws.read("a.txt") == "committed"
        assert ws.exists("b.txt")  # restore doesn't delete untracked files


# ---------------------------------------------------------------------------
# Stash
# ---------------------------------------------------------------------------


class TestStash:
    async def test_stash_and_unstash(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "original")
        await gw.commit("c1")
        ws.write("a.txt", "modified")
        await gw.stash()
        # After stash, file reverts to the committed version
        assert ws.read("a.txt") == "original"
        await gw.unstash()
        assert ws.read("a.txt") == "modified"

    async def test_stash_empty_raises(self, gw: GitWorkspace) -> None:
        with pytest.raises(GitError):
            await gw.unstash()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    async def test_merge_branch(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        # On main, create a file and commit.
        ws.write("base.txt", "base")
        await gw.commit("c1: base")
        # Branch off to feature
        await gw.branch("task/feat")
        ws.write("feat.txt", "feature")
        await gw.commit("c2: feature")
        # Back to main
        await gw.checkout("main")
        ws.write("main-only.txt", "main-only")
        await gw.commit("c3: main-only")
        # Merge feature into main
        merge_sha = await gw.merge("task/feat")
        assert await gw.current_branch() == "main"
        assert ws.read("feat.txt") == "feature"
        assert ws.read("base.txt") == "base"
        assert ws.read("main-only.txt") == "main-only"

    async def test_merge_fast_forward_no_divergence(
        self, gw: GitWorkspace
    ) -> None:
        ws = gw.workspace
        ws.write("a.txt", "a")
        await gw.commit("c1")
        await gw.branch("task/ff")
        ws.write("b.txt", "b")
        await gw.commit("c2")
        await gw.checkout("main")
        result = await gw.merge("task/ff")
        # No divergence, so fast-forward
        assert ws.read("b.txt") == "b"

    async def test_merge_conflict_raises(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("shared.txt", "base")
        await gw.commit("c1")
        await gw.branch("task/feat")
        ws.write("shared.txt", "feature change")
        await gw.commit("c2: feat")
        await gw.checkout("main")
        ws.write("shared.txt", "main change")
        await gw.commit("c3: main change")
        with pytest.raises(GitConflictError):
            await gw.merge("task/feat")
        # Clean up
        await gw.abort_merge()
        assert await gw.current_branch() == "main"

    async def test_merge_log_includes_merge_commit(
        self, gw: GitWorkspace
    ) -> None:
        ws = gw.workspace
        ws.write("a.txt", "a")
        await gw.commit("c1")
        await gw.branch("task/feat")
        ws.write("feat.txt", "f")
        await gw.commit("c2")
        await gw.checkout("main")
        ws.write("main.txt", "m")
        await gw.commit("c3")
        merge_sha = await gw.merge("task/feat")
        log = await gw.log()
        # At least the merge commit should be in the log
        shas = [c.sha for c in log]
        assert merge_sha is None or merge_sha in shas


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


class TestLog:
    async def test_log_returns_commits(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "1")
        c1 = await gw.commit("c1")
        ws.write("b.txt", "2")
        c2 = await gw.commit("c2")
        log = await gw.log()
        assert len(log) >= 2
        assert log[0].sha == c2.sha  # newest first
        assert log[1].sha == c1.sha

    async def test_log_formats_commit_info(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "1")
        c = await gw.commit("my message\n\nbody here")
        log = await gw.log(max_count=1)
        assert len(log) == 1
        assert log[0].sha == c.sha
        assert log[0].message == "my message\n\nbody here"
        assert log[0].author_name == "ai-orchestrator"
        assert isinstance(log[0].parents, tuple)

    async def test_log_respects_max_count(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        for i in range(5):
            ws.write(f"f{i}.txt", str(i))
            await gw.commit(f"c{i}")
        log = await gw.log(max_count=3)
        assert len(log) <= 3


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    async def test_status_clean(self, gw: GitWorkspace) -> None:
        st = await gw.status()
        assert st == []

    async def test_status_modified(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "v1")
        await gw.commit("c1")
        ws.write("a.txt", "v2")
        st = await gw.status()
        assert any(s.path == "a.txt" and s.status == "M" and not s.staged for s in st)

    async def test_status_staged(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("a.txt", "v1")
        await gw.commit("c1")
        ws.write("a.txt", "v2")

        # Stage via git add
        from subprocess import run
        run(["git", "add", "a.txt"], cwd=gw.root, check=True)
        st = await gw.status()
        # Should show staged "M" in index
        staged = [s for s in st if s.staged]
        assert any(s.path == "a.txt" for s in staged)

    async def test_status_untracked(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("new.txt", "new")
        st = await gw.status()
        assert any(s.path == "new.txt" and s.status == "?" and not s.staged for s in st)

    async def test_is_clean(self, gw: GitWorkspace) -> None:
        assert await gw.is_clean() is True
        ws = gw.workspace
        ws.write("dirty.txt", "x")
        assert await gw.is_clean() is False


# ---------------------------------------------------------------------------
# from_existing (bootstrapping from a real project)
# ---------------------------------------------------------------------------


class TestFromExisting:
    async def test_from_existing_basic(self, tmp_path: Path) -> None:
        # Create a small source project
        source = tmp_path / "source-project"
        source.mkdir(parents=True)
        (source / "README.md").write_text("# My Project\n")
        (source / "src").mkdir()
        (source / "src" / "main.py").write_text("print('hello')\n")

        gw = await GitWorkspace.from_existing(source, "task-1", root=tmp_path)
        assert await gw.is_initialized()
        assert await gw.current_branch() == "main"
        assert await gw.file_count() >= 2
        # Files are present
        ws = gw.workspace
        assert ws.read("README.md") == "# My Project\n"
        assert ws.read("src/main.py") == "print('hello')\n"

    async def test_from_existing_with_exclude(self, tmp_path: Path) -> None:
        source = tmp_path / "source-project"
        source.mkdir()
        (source / "keep.py").write_text("keep")
        (source / "node_modules").mkdir()
        (source / "node_modules" / "dep").write_text("should exclude")
        (source / ".venv").mkdir()
        (source / ".venv" / "lib").mkdir(parents=True)

        gw = await GitWorkspace.from_existing(
            source, "task-2", root=tmp_path,
            exclude=("node_modules*", ".venv*", "*.pyc"),
        )
        ws = gw.workspace
        assert ws.exists("keep.py")
        assert not ws.exists("node_modules/dep")
        assert not ws.exists(".venv/lib")

    async def test_from_existing_missing_source(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await GitWorkspace.from_existing(
                tmp_path / "nope", "task-3", root=tmp_path
            )

    async def test_from_existing_initial_message(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.txt").write_text("content")

        gw = await GitWorkspace.from_existing(
            source, "task-4", root=tmp_path,
        )
        log = await gw.log()
        assert "initialise workspace from" in log[0].message
        assert "src" in log[0].message


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_directory(self, workspace: FileWorkspace) -> None:
        g = GitWorkspace(workspace)
        await g.init()
        assert await g.is_initialized()
        await g.branch("task/empty")
        with pytest.raises(GitError, match="nothing to commit"):
            await g.commit("empty commit")

    async def test_multiple_branches_independent(self, gw: GitWorkspace) -> None:
        ws = gw.workspace
        ws.write("shared.txt", "main ver")
        await gw.commit("c1 on main")

        await gw.branch("task/a")
        ws.write("a.txt", "a content")
        await gw.commit("c2 on a")

        await gw.branch("task/b", from_ref="main")
        ws.write("b.txt", "b content")
        await gw.commit("c3 on b")

        await gw.checkout("main")
        assert not ws.exists("a.txt")
        assert not ws.exists("b.txt")
        await gw.checkout("task/a")
        assert ws.exists("a.txt")
        assert not ws.exists("b.txt")
        await gw.checkout("task/b")
        assert ws.exists("b.txt")
        assert not ws.exists("a.txt")
