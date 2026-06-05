"""Tests for FileWorkspace — the core safe file operations module."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import pytest

from ai_orchestrator.workspace import (
    AtomicWriteError,
    FileEntry,
    FileWorkspace,
    PatchConflictError,
    PathTraversalError,
    SearchMatch,
    SnapshotChange,
    WorkspaceAlreadyExistsError,
    WorkspaceError,
    WorkspaceNotFoundError,
    WorkspaceSnapshot,
)


# ---------------------------------------------------------------------------
# Construction & path-traversal guard
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_for_task_creates_directory(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("abc", root=tmp_path)
        assert ws.workspace_root.is_dir()
        assert ws.workspace_root == tmp_path / "abc"
        assert ws.task_id == "abc"
        assert ws.root == tmp_path

    def test_create_false_on_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceNotFoundError):
            FileWorkspace(tmp_path, "abc", create=False)

    def test_create_false_on_existing_ok(self, tmp_path: Path) -> None:
        (tmp_path / "abc").mkdir()
        ws = FileWorkspace(tmp_path, "abc", create=False)
        assert ws.workspace_root.is_dir()

    @pytest.mark.parametrize(
        "bad",
        ["", ".", "..", "a/b", "a\\b", "a\x00b"],
    )
    def test_invalid_task_id_rejected(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(ValueError):
            FileWorkspace(tmp_path, bad)

    def test_repr_contains_task_id(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("demo", root=tmp_path)
        assert "demo" in repr(ws)
        assert str(ws.workspace_root) in repr(ws)

    def test_default_root_is_workspaces(self) -> None:
        ws = FileWorkspace.for_task("x")
        # Will be <cwd>/workspaces/x — we don't assume cwd, just check it ends with x
        assert ws.workspace_root.name == "x"
        assert ws.workspace_root.parent.name == "workspaces"


class TestPathTraversalGuard:
    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(PathTraversalError):
            ws.read("/etc/passwd")
        with pytest.raises(PathTraversalError):
            ws.write("/etc/passwd", "x")
        with pytest.raises(PathTraversalError):
            ws.exists("/etc/passwd")

    def test_dotdot_rejected(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(PathTraversalError):
            ws.read("../escape.txt")
        with pytest.raises(PathTraversalError):
            ws.write("../escape.txt", "x")
        with pytest.raises(PathTraversalError):
            ws.exists("../sibling")

    def test_nul_byte_rejected(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(PathTraversalError):
            ws.write("ok\x00.txt", "x")

    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        # Create a symlink inside the workspace that points outside
        (ws.workspace_root / "link").symlink_to(tmp_path.parent)
        with pytest.raises(PathTraversalError):
            ws.read("link/anything")

    def test_non_string_path_rejected(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(TypeError):
            ws.read(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------


class TestExistenceChecks:
    def test_exists_true(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "x")
        assert ws.exists("a.txt")
        assert ws.is_file("a.txt")
        assert not ws.is_dir("a.txt")

    def test_exists_false(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        assert not ws.exists("nope.txt")

    def test_is_dir(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.mkdir("sub")
        assert ws.is_dir("sub")

    def test_contains_dunder(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "x")
        assert "a.txt" in ws
        assert "nope.txt" not in ws
        assert "../escape" not in ws  # traversal returns False, not raises


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


class TestReadWrite:
    def test_write_creates_parents(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        target = ws.write("a/b/c.txt", "hello")
        assert target.is_file()
        assert target.read_text() == "hello"

    def test_read_round_trip(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "hello world")
        assert ws.read("a.txt") == "hello world"

    def test_read_missing_raises(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(WorkspaceNotFoundError):
            ws.read("nope.txt")

    def test_write_overwrites(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "v1")
        ws.write("a.txt", "v2")
        assert ws.read("a.txt") == "v2"

    def test_write_exclusive(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "v1")
        with pytest.raises(WorkspaceAlreadyExistsError):
            ws.write("a.txt", "v2", exclusive=True)
        assert ws.read("a.txt") == "v1"

    def test_write_bytes(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write_bytes("a.bin", b"\x00\x01\x02")
        assert ws.read_bytes("a.bin") == b"\x00\x01\x02"

    def test_append_to_new(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.append("a.txt", "line1\n")
        assert ws.read("a.txt") == "line1\n"

    def test_append_to_existing(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "line1\n")
        ws.append("a.txt", "line2\n")
        assert ws.read("a.txt") == "line1\nline2\n"

    def test_atomic_write_cleans_temp_on_failure(self, tmp_path: Path, monkeypatch) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        # Make os.replace raise to simulate mid-write crash
        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            ws.write("a.txt", "hello")
        # No leftover .tmp files
        leftover = [p for p in ws.workspace_root.iterdir() if p.suffix == ".tmp"]
        assert leftover == []
        # No half-written file
        assert not (ws.workspace_root / "a.txt").exists()

    def test_atomic_write_error_wrapping(self, tmp_path: Path, monkeypatch) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        monkeypatch.setattr(
            "tempfile.mkstemp", lambda *a, **kw: (_ for _ in ()).throw(OSError("no tmp"))
        )
        with pytest.raises(AtomicWriteError):
            ws.write("a.txt", "x")


# ---------------------------------------------------------------------------
# Delete / rmtree / mkdir
# ---------------------------------------------------------------------------


class TestDeleteAndMkdir:
    def test_delete_file(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "x")
        ws.delete("a.txt")
        assert not ws.exists("a.txt")

    def test_delete_missing_raises(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(WorkspaceNotFoundError):
            ws.delete("nope.txt")

    def test_delete_missing_ok(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.delete("nope.txt", missing_ok=True)

    def test_delete_directory_rejected(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.mkdir("d")
        with pytest.raises(WorkspaceError):
            ws.delete("d")

    def test_rmtree_removes_directory(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("d/a.txt", "x")
        ws.write("d/b.txt", "y")
        ws.rmtree("d")
        assert not ws.exists("d")

    def test_rmtree_missing_is_noop(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.rmtree("d")  # missing → no-op

    def test_rmtree_on_file_raises(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "x")
        with pytest.raises(WorkspaceError):
            ws.rmtree("a.txt")

    def test_rmtree_default_removes_workspace(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "x")
        ws.rmtree()
        assert not ws.workspace_root.exists()

    def test_mkdir_creates_parents(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.mkdir("a/b/c")
        assert (ws.workspace_root / "a" / "b" / "c").is_dir()


# ---------------------------------------------------------------------------
# Patch (string replace)
# ---------------------------------------------------------------------------


class TestPatch:
    def test_patch_single_occurrence(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "hello world")
        new = ws.patch("a.txt", "hello", "goodbye")
        assert new == "goodbye world"
        assert ws.read("a.txt") == "goodbye world"

    def test_patch_expected_count_mismatch_raises(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "foo foo foo")
        with pytest.raises(PatchConflictError) as exc_info:
            ws.patch("a.txt", "foo", "bar", expected_count=2)
        assert exc_info.value.path == "a.txt"
        # File unchanged
        assert ws.read("a.txt") == "foo foo foo"

    def test_patch_count_zero_means_substring_absent(self, tmp_path: Path) -> None:
        # expected_count=0 means "I expect zero occurrences"; matching zero is success.
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "hello")
        new = ws.patch("a.txt", "xyz", "abc", expected_count=0)
        assert new == "hello"  # unchanged
        assert ws.read("a.txt") == "hello"

    def test_patch_count_zero_with_substring_present_raises(self, tmp_path: Path) -> None:
        # expected_count=0 conflicts if the substring is actually present.
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "hello xyz")
        with pytest.raises(PatchConflictError):
            ws.patch("a.txt", "xyz", "abc", expected_count=0)

    def test_patch_missing_file_raises(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(WorkspaceNotFoundError):
            ws.patch("nope.txt", "a", "b")


# ---------------------------------------------------------------------------
# list_tree & search
# ---------------------------------------------------------------------------


class TestListTree:
    def test_empty_workspace(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        assert ws.list_tree() == []

    def test_nested_files(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "1")
        ws.write("sub/b.txt", "2")
        ws.write("sub/inner/c.txt", "3")
        entries = ws.list_tree()
        paths = [e.path for e in entries]
        assert "a.txt" in paths
        assert "sub/b.txt" in paths
        assert "sub/inner/c.txt" in paths
        assert "sub" in paths  # dir entry
        assert "sub/inner" in paths  # dir entry

    def test_directories_before_files(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "1")
        ws.write("z.txt", "1")
        ws.mkdir("m")
        entries = ws.list_tree()
        # First entries should be the directories
        first_dir_idx = next(i for i, e in enumerate(entries) if e.is_dir)
        first_file_idx = next(i for i, e in enumerate(entries) if not e.is_dir)
        assert first_dir_idx < first_file_idx

    def test_glob_filter(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.py", "x")
        ws.write("b.txt", "x")
        ws.write("sub/c.py", "x")
        entries = ws.list_tree(glob="**/*.py")
        paths = [e.path for e in entries if not e.is_dir]
        assert "a.py" in paths
        assert "sub/c.py" in paths
        assert "b.txt" not in paths

    def test_prefix_restricts_listing(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "1")
        ws.write("sub/b.txt", "2")
        entries = ws.list_tree("sub")
        paths = [e.path for e in entries]
        assert "sub/b.txt" in paths
        assert "a.txt" not in paths

    def test_file_entry_size_and_mtime(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "hello")
        entries = ws.list_tree()
        a = next(e for e in entries if e.path == "a.txt")
        assert a.size == 5
        assert a.mtime > 0
        assert a.is_file
        assert not a.is_dir


class TestSearch:
    def test_substring_match(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.py", "def foo():\n    return 42\n")
        ws.write("b.py", "def bar():\n    return foo()\n")
        results = ws.search("foo")
        paths = {r.path for r in results}
        assert paths == {"a.py", "b.py"}

    def test_line_numbers_accurate(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.py", "line1\nline2 foo\nline3\nfoo line4\n")
        results = ws.search("foo")
        lines = [r.line_number for r in results]
        assert lines == [2, 4]

    def test_regex_match(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.py", "foo123\nbar\nbaz456\n")
        results = ws.search(r"[a-z]+\d+", regex=True)
        assert {r.line_number for r in results} == {1, 3}

    def test_case_insensitive(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.py", "FOO\nfoo\nFoO\n")
        assert len(ws.search("foo")) == 1
        assert len(ws.search("foo", case_sensitive=False)) == 3

    def test_glob_filter(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.py", "def foo(): pass\n")
        ws.write("b.txt", "def foo(): pass\n")
        results = ws.search("foo", glob="**/*.py")
        paths = {r.path for r in results}
        assert paths == {"a.py"}

    def test_skips_binary_files(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write_bytes("a.bin", b"\x00\x01\x02\xff")
        ws.write("b.py", "foo")
        results = ws.search("foo")
        assert {r.path for r in results} == {"b.py"}

    def test_skips_large_files(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("big.txt", "a" * 2_000_000)
        ws.write("small.txt", "needle")
        results = ws.search("needle", max_file_size=100)
        assert {r.path for r in results} == {"small.txt"}

    def test_invalid_regex_raises(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(ValueError):
            ws.search("[unclosed", regex=True)

    def test_empty_pattern_raises(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        with pytest.raises(ValueError):
            ws.search("")


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_empty_workspace_snapshot(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        snap = ws.snapshot()
        assert snap.file_count == 0
        assert snap.total_size == 0
        assert snap.manifest == {}
        # Hash is deterministic for empty workspace
        assert snap.hash == ws.snapshot().hash

    def test_snapshot_detects_changes(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "v1")
        snap1 = ws.snapshot()
        ws.write("a.txt", "v2")
        snap2 = ws.snapshot()
        assert snap1.hash != snap2.hash
        assert snap1.manifest["a.txt"] != snap2.manifest["a.txt"]

    def test_snapshot_deterministic_for_same_content(self, tmp_path: Path) -> None:
        ws1 = FileWorkspace.for_task("t1", root=tmp_path)
        ws2 = FileWorkspace.for_task("t2", root=tmp_path)
        ws1.write("a.txt", "x")
        ws2.write("a.txt", "x")
        # Same content → same per-file hash
        assert ws1.snapshot().manifest == ws2.snapshot().manifest

    def test_diff_snapshots_added_removed_modified(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "1")
        ws.write("b.txt", "2")
        old = ws.snapshot()
        ws.write("c.txt", "3")  # added
        ws.delete("a.txt")  # removed
        ws.write("b.txt", "22")  # modified
        changes = ws.diff_snapshots(old)
        actions = {c.action for c in changes}
        assert actions == {"added", "removed", "modified"}
        # Paths match
        added = next(c for c in changes if c.action == "added")
        removed = next(c for c in changes if c.action == "removed")
        modified = next(c for c in changes if c.action == "modified")
        assert added.path == "c.txt"
        assert removed.path == "a.txt"
        assert modified.path == "b.txt"
        assert modified.old_hash is not None
        assert modified.new_hash is not None

    def test_diff_snapshots_no_changes(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        ws.write("a.txt", "x")
        snap = ws.snapshot()
        assert ws.diff_snapshots(snap) == []


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_writes_dont_corrupt(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                for j in range(50):
                    ws.write(f"f{i}.txt", f"iter {j}\n", exclusive=True)
            except WorkspaceAlreadyExistsError:
                # Some threads will race on the first write — exclusive catches that
                pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # Each file should exist exactly once
        for i in range(8):
            assert ws.exists(f"f{i}.txt")

    def test_atomic_write_visible_to_concurrent_readers(self, tmp_path: Path) -> None:
        ws = FileWorkspace.for_task("t", root=tmp_path)
        # No initial write — let the writer loop be the only source of truth.
        # The reader catches WorkspaceNotFoundError if it reads before any write.
        seen: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    seen.append(ws.read("a.txt"))
                except WorkspaceNotFoundError:
                    pass

        t = threading.Thread(target=reader)
        t.start()
        # Hammer writes
        for i in range(100):
            ws.write("a.txt", f"v{i}\n")
        # Brief settle
        import time
        time.sleep(0.05)
        stop.set()
        t.join()
        # Every observed content must be a complete line, never partial
        for content in seen:
            assert content.startswith("v") and content.endswith("\n")
