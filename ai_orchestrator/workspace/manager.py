"""File workspace — safe, audited file operations for one task.

:class:`FileWorkspace` gives agents a place to read and write code without
ever escaping the workspace root. All operations are confined to
``<root>/<task_id>/``; any attempt to escape (via ``../``, absolute paths,
symlinks pointing outside) raises :class:`PathTraversalError`.

**Atomicity.** Writes go through a temp file in the same directory followed
by :func:`os.replace`, so a crash mid-write cannot leave a half-written
file. Readers will see either the old content or the new content, never a
mix.

**Thread-safety.** All mutating methods hold a per-instance
:class:`threading.Lock`, so concurrent agents editing the same workspace
cannot corrupt files. Read-only methods do not lock; combined with atomic
writes, this gives safe concurrent access.

**Discovery.** :meth:`list_tree` and :meth:`search` give agents a way to
enumerate and grep the workspace. :meth:`snapshot` produces a
content-addressed fingerprint that downstream code (Git layer, reviewer
agents) can use as a diff base.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field

from ai_orchestrator.workspace.exceptions import (
    AtomicWriteError,
    PatchConflictError,
    PathTraversalError,
    WorkspaceAlreadyExistsError,
    WorkspaceError,
    WorkspaceNotFoundError,
)


# Default root location, resolved relative to CWD at construction time.
DEFAULT_WORKSPACE_ROOT: Path = Path("workspaces")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileEntry:
    """One entry returned by :meth:`FileWorkspace.list_tree`."""

    path: str
    """Path relative to the workspace root, using forward slashes."""
    size: int
    """File size in bytes; ``0`` for directories."""
    mtime: float
    """POSIX mtime (seconds since epoch)."""
    is_dir: bool

    @property
    def is_file(self) -> bool:
        return not self.is_dir


@dataclass(frozen=True)
class SearchMatch:
    """One match returned by :meth:`FileWorkspace.search`."""

    path: str
    """Path relative to the workspace root."""
    line_number: int
    """1-indexed line number."""
    line: str
    """Full text of the matched line, without trailing newline."""
    match_start: int
    """0-indexed column where the match starts."""
    match_end: int
    """0-indexed column (exclusive) where the match ends."""


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


class WorkspaceSnapshot(BaseModel):
    """Content-addressed snapshot of a workspace at a point in time.

    Snapshots store per-file SHA-256 hashes, not file content. They are
    cheap to compute and compare; restoring a file from a snapshot is
    impossible without the workspace itself (use a Git layer for that).
    """

    hash: str
    """Top-level SHA-256 over the sorted manifest."""
    file_count: int
    total_size: int
    captured_at: datetime
    manifest: dict[str, str] = Field(default_factory=dict)
    """Map of relative path → SHA-256 hex digest."""

    model_config = {"frozen": True}


class SnapshotChange(BaseModel):
    """One change between two snapshots."""

    action: str
    """One of ``"added"``, ``"removed"``, ``"modified"``."""
    path: str
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# FileWorkspace
# ---------------------------------------------------------------------------


class FileWorkspace:
    """A safe, audited file workspace scoped to a single task_id.

    Parameters
    ----------
    root:
        Parent directory that contains all task workspaces. Defaults to
        ``./workspaces`` (relative to CWD). Resolved to an absolute path on
        construction.
    task_id:
        Unique task identifier; this workspace's files live at
        ``<root>/<task_id>/``. Must be a single path segment (no ``/``,
        no ``\\``, not ``.`` or ``..``).
    create:
        If ``True`` (default), create the workspace directory on
        construction. Set to ``False`` to require it already exists (raises
        :class:`WorkspaceNotFoundError` otherwise).
    """

    _MAX_FILE_SIZE_FOR_SEARCH = 1_048_576  # 1 MiB

    def __init__(
        self,
        root: Union[Path, str],
        task_id: str,
        *,
        create: bool = True,
    ) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"task_id must be a non-empty str, got {task_id!r}")
        if task_id in (".", "..") or "/" in task_id or "\\" in task_id or "\x00" in task_id:
            raise ValueError(
                f"task_id must be a single path segment, got {task_id!r}"
            )

        self._root = Path(root).resolve()
        self._task_id = task_id
        self._workspace_root = (self._root / task_id).resolve()
        # Defense in depth: the resolved workspace must remain inside _root.
        try:
            self._workspace_root.relative_to(self._root)
        except ValueError as e:
            raise PathTraversalError(task_id, str(self._root)) from e

        self._lock = threading.Lock()
        self._created_at = datetime.now(timezone.utc)

        if create:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
        elif not self._workspace_root.exists():
            raise WorkspaceNotFoundError(
                f"workspace {task_id!r} does not exist under {self._root}"
            )

    # ----- Factories & properties ----------------------------------------

    @classmethod
    def for_task(
        cls,
        task_id: str,
        *,
        root: Union[Path, str] = DEFAULT_WORKSPACE_ROOT,
        create: bool = True,
    ) -> "FileWorkspace":
        """Convenience constructor using the default ``workspaces/`` root."""
        return cls(root=root, task_id=task_id, create=create)

    @property
    def root(self) -> Path:
        """Parent directory that contains this and other task workspaces."""
        return self._root

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def workspace_root(self) -> Path:
        """Absolute path of this workspace's directory."""
        return self._workspace_root

    @property
    def created_at(self) -> datetime:
        return self._created_at

    def __repr__(self) -> str:
        return f"FileWorkspace(task_id={self._task_id!r}, root={self._workspace_root})"

    def __contains__(self, relative: str) -> bool:  # pragma: no cover - trivial
        try:
            return self.exists(relative)
        except PathTraversalError:
            return False

    # ----- Path resolution (private) -------------------------------------

    def _resolve(self, relative: str) -> Path:
        """Resolve a relative path to an absolute Path inside the workspace.

        Raises :class:`PathTraversalError` if the path attempts to escape.
        """
        if not isinstance(relative, str):
            raise TypeError(f"path must be str, got {type(relative).__name__}")
        if os.path.isabs(relative):
            raise PathTraversalError(relative, str(self._workspace_root))
        if "\x00" in relative:
            raise PathTraversalError(relative, str(self._workspace_root))
        candidate = (self._workspace_root / relative).resolve()
        try:
            candidate.relative_to(self._workspace_root)
        except ValueError as e:
            raise PathTraversalError(relative, str(self._workspace_root)) from e
        return candidate

    # ----- Existence checks ----------------------------------------------

    def exists(self, relative: str) -> bool:
        return self._resolve(relative).exists()

    def is_file(self, relative: str) -> bool:
        return self._resolve(relative).is_file()

    def is_dir(self, relative: str) -> bool:
        return self._resolve(relative).is_dir()

    # ----- Read operations -----------------------------------------------

    def read(self, relative: str, *, encoding: str = "utf-8") -> str:
        path = self._resolve(relative)
        if not path.is_file():
            raise WorkspaceNotFoundError(f"file not found: {relative}")
        return path.read_text(encoding=encoding)

    def read_bytes(self, relative: str) -> bytes:
        path = self._resolve(relative)
        if not path.is_file():
            raise WorkspaceNotFoundError(f"file not found: {relative}")
        return path.read_bytes()

    # ----- Write operations ----------------------------------------------

    def _atomic_write(self, target: Path, content: bytes) -> None:
        """Write ``content`` to ``target`` atomically.

        Writes to a temp file in the same directory, fsyncs, then
        :func:`os.replace` into place. If anything fails, the temp file is
        cleaned up and the original file is untouched.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=str(target.parent),
            )
        except OSError as e:
            raise AtomicWriteError(
                f"failed to create temp file for {target}: {e}"
            ) from e
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def write(
        self,
        relative: str,
        content: str,
        *,
        encoding: str = "utf-8",
        exclusive: bool = False,
    ) -> Path:
        """Write ``content`` to ``relative``, creating parents as needed.

        :raises WorkspaceAlreadyExistsError: if ``exclusive`` and the file
            already exists.
        :returns: the absolute path of the written file.
        """
        target = self._resolve(relative)
        if exclusive and target.exists():
            raise WorkspaceAlreadyExistsError(f"file already exists: {relative}")
        with self._lock:
            self._atomic_write(target, content.encode(encoding))
        return target

    def write_bytes(
        self,
        relative: str,
        content: bytes,
        *,
        exclusive: bool = False,
    ) -> Path:
        target = self._resolve(relative)
        if exclusive and target.exists():
            raise WorkspaceAlreadyExistsError(f"file already exists: {relative}")
        with self._lock:
            self._atomic_write(target, content)
        return target

    def append(self, relative: str, content: str, *, encoding: str = "utf-8") -> Path:
        target = self._resolve(relative)
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if target.exists() else "wb"
            with target.open(mode) as f:
                f.write(content.encode(encoding))
        return target

    # ----- Delete & directory ops ----------------------------------------

    def delete(self, relative: str, *, missing_ok: bool = False) -> None:
        target = self._resolve(relative)
        with self._lock:
            if not target.exists():
                if missing_ok:
                    return
                raise WorkspaceNotFoundError(f"path not found: {relative}")
            if target.is_dir():
                raise WorkspaceError(
                    f"cannot delete directory with delete(); use rmtree(): {relative}"
                )
            target.unlink()

    def rmtree(self, relative: str = ".") -> None:
        """Recursively remove ``relative`` (default: the entire workspace)."""
        target = self._workspace_root if relative == "." else self._resolve(relative)
        with self._lock:
            if not target.exists():
                return
            if not target.is_dir():
                raise WorkspaceError(f"rmtree target is not a directory: {relative}")
            shutil.rmtree(target)

    def mkdir(
        self,
        relative: str,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> Path:
        target = self._resolve(relative)
        target.mkdir(parents=parents, exist_ok=exist_ok)
        return target

    # ----- Patch (string replace) ----------------------------------------

    def patch(
        self,
        relative: str,
        old: str,
        new: str,
        *,
        expected_count: int = 1,
        encoding: str = "utf-8",
    ) -> str:
        """Replace ``old`` with ``new`` in the file.

        This is the OpenCode / Aider-style "find/replace" patch: the
        caller is expected to know the file's current content, and the
        patch fails loudly if the expected substring is not found exactly
        ``expected_count`` times.

        :raises WorkspaceNotFoundError: if the file does not exist.
        :raises PatchConflictError: if ``old`` does not appear exactly
            ``expected_count`` times in the file.
        :returns: the new file content.
        """
        target = self._resolve(relative)
        with self._lock:
            if not target.is_file():
                raise WorkspaceNotFoundError(f"file not found: {relative}")
            content = target.read_text(encoding=encoding)
            actual = content.count(old)
            if actual != expected_count:
                excerpt = content[:200] if content else ""
                raise PatchConflictError(
                    path=relative,
                    expected=f"{old!r} appearing {expected_count} time(s)",
                    actual_excerpt=excerpt,
                )
            new_content = content.replace(old, new)
            self._atomic_write(target, new_content.encode(encoding))
        return new_content

    # ----- Discovery -----------------------------------------------------

    def list_tree(
        self,
        prefix: str = "",
        *,
        glob: Optional[str] = None,
    ) -> list[FileEntry]:
        """List all entries under ``prefix`` (default: the whole workspace).

        Returns a sorted list of :class:`FileEntry`. Directories are listed
        before files; both are alphabetical.
        """
        base = self._resolve(prefix) if prefix else self._workspace_root
        if not base.exists():
            return []
        if not base.is_dir():
            raise WorkspaceError(f"list_tree prefix is not a directory: {prefix!r}")

        if glob is None:
            file_paths = [p for p in base.rglob("*") if p.is_file()]
        else:
            file_paths = [p for p in base.glob(glob) if p.is_file()]

        dir_paths = sorted({p.parent for p in file_paths if p != base})
        entries: list[FileEntry] = []
        for d in dir_paths:
            try:
                stat = d.stat()
            except OSError:
                continue
            entries.append(
                FileEntry(
                    path=str(d.relative_to(self._workspace_root)),
                    size=0,
                    mtime=stat.st_mtime,
                    is_dir=True,
                )
            )
        for p in file_paths:
            try:
                stat = p.stat()
            except OSError:
                continue
            entries.append(
                FileEntry(
                    path=str(p.relative_to(self._workspace_root)),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    is_dir=False,
                )
            )
        entries.sort(key=lambda e: (not e.is_dir, e.path))
        return entries

    def search(
        self,
        pattern: str,
        *,
        glob: str = "**/*",
        regex: bool = False,
        case_sensitive: bool = True,
        encoding: str = "utf-8",
        max_file_size: int = _MAX_FILE_SIZE_FOR_SEARCH,
    ) -> list[SearchMatch]:
        """Search for ``pattern`` (substring or regex) across all files.

        Skips files larger than ``max_file_size`` (default 1 MiB) and files
        that fail to decode as ``encoding`` (default utf-8). Results are
        returned in (path, line_number) order.
        """
        if not pattern:
            raise ValueError("pattern must be non-empty")
        flags = 0 if case_sensitive else re.IGNORECASE
        if regex:
            try:
                compiled = re.compile(pattern, flags=flags)
            except re.error as e:
                raise ValueError(f"invalid regex: {e}") from e
        else:
            compiled = re.compile(re.escape(pattern), flags=flags)

        base = self._workspace_root
        results: list[SearchMatch] = []
        for path in sorted(base.glob(glob)):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > max_file_size:
                continue
            try:
                text = path.read_text(encoding=encoding)
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                for m in compiled.finditer(line):
                    results.append(
                        SearchMatch(
                            path=str(path.relative_to(self._workspace_root)),
                            line_number=lineno,
                            line=line,
                            match_start=m.start(),
                            match_end=m.end(),
                        )
                    )
        return results

    # ----- Snapshots -----------------------------------------------------

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def snapshot(self) -> WorkspaceSnapshot:
        """Compute a content-addressed snapshot of the workspace.

        Walks every file, hashes its content, and rolls the hashes into a
        single top-level hash. Cheap; safe to call frequently.
        """
        manifest: dict[str, str] = {}
        total_size = 0
        for entry in self.list_tree():
            if entry.is_dir:
                continue
            path = self._workspace_root / entry.path
            try:
                manifest[entry.path] = self._hash_file(path)
                total_size += entry.size
            except OSError:
                continue
        manifest_h = hashlib.sha256()
        for p in sorted(manifest.keys()):
            manifest_h.update(p.encode("utf-8"))
            manifest_h.update(b"\x00")
            manifest_h.update(manifest[p].encode("utf-8"))
            manifest_h.update(b"\n")
        return WorkspaceSnapshot(
            hash=manifest_h.hexdigest(),
            file_count=len(manifest),
            total_size=total_size,
            captured_at=datetime.now(timezone.utc),
            manifest=manifest,
        )

    def diff_snapshots(
        self,
        old: WorkspaceSnapshot,
        new: Optional[WorkspaceSnapshot] = None,
    ) -> list[SnapshotChange]:
        """Return a list of changes between ``old`` and ``new`` (default: current).

        Each change is an :class:`SnapshotChange` with action ``added``,
        ``removed``, or ``modified``.
        """
        if new is None:
            new = self.snapshot()
        old_paths = set(old.manifest.keys())
        new_paths = set(new.manifest.keys())
        changes: list[SnapshotChange] = []
        for p in sorted(new_paths - old_paths):
            changes.append(
                SnapshotChange(action="added", path=p, new_hash=new.manifest[p])
            )
        for p in sorted(old_paths - new_paths):
            changes.append(
                SnapshotChange(action="removed", path=p, old_hash=old.manifest[p])
            )
        for p in sorted(old_paths & new_paths):
            if old.manifest[p] != new.manifest[p]:
                changes.append(
                    SnapshotChange(
                        action="modified",
                        path=p,
                        old_hash=old.manifest[p],
                        new_hash=new.manifest[p],
                    )
                )
        return changes
