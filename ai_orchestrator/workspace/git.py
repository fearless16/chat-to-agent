"""Git version control for task workspaces — branch, commit, diff, rollback.

:class:`GitWorkspace` wraps a :class:`FileWorkspace` and turns its directory
into a self-contained git repository.  Every task gets a branch off ``main``,
so agents can experiment freely and reviewers see exactly what changed.

Workflow
--------

::

    ws = FileWorkspace.for_task(\"task-123\")
    gw = GitWorkspace(ws)
    await gw.init()
    await gw.branch(\"task/add-helper\")

    ws.write(\"src/main.py\", \"...\")
    ws.patch(\"src/main.py\", \"old\", \"new\")

    await gw.commit(\"feat: add helper\")
    diff = await gw.diff(\"main\")

    # Reviewer looks at diff ...
    await gw.rollback(\"HEAD~1\")       # undo the last commit
    await gw.merge(\"task/add-helper\") # merge back into main (via checkout + merge)
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

from ai_orchestrator.workspace.exceptions import (
    GitConflictError,
    GitError,
    GitNotInstalled,
    GitNotRepoError,
)
from ai_orchestrator.workspace.manager import DEFAULT_WORKSPACE_ROOT, FileWorkspace

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitInfo:
    """Metadata for a single commit returned by :meth:`GitWorkspace.log`."""

    sha: str
    """Full SHA-1 (40 hex chars)."""
    message: str
    """Commit message (first line stripped of trailing newline)."""
    author_name: str
    author_email: str
    timestamp: datetime
    parents: tuple[str, ...]
    """Parent SHAs (empty for root commit)."""


@dataclass(frozen=True)
class FileStatus:
    """One file in the working tree or staging area."""

    path: str
    """Path relative to workspace root."""
    status: str
    """One-letter code: ``M`` modified, ``A`` added, ``D`` deleted, ``?`` untracked,
    ``!`` ignored, ``R`` renamed."""
    staged: bool
    """``True`` if the change is staged (in the index); ``False`` for working tree."""


# ---------------------------------------------------------------------------
# GitWorkspace
# ---------------------------------------------------------------------------


class GitWorkspace:
    """Version control for a task workspace backed by git.

    Parameters
    ----------
    workspace:
        The :class:`FileWorkspace` to manage. Must be on a local filesystem
        accessible by the ``git`` command.
    author_name, author_email:
        Identity used for commits.  Passed via ``-c user.name=...`` and
        ``-c user.email=...`` on every command, so never touches global/machine
        git config.
    """

    _COMMIT_FORMAT = "%H%n%an%n%ae%n%ct%n%P%n%B%n===END==="
    _GIT_MIN_VERSION = (2, 20, 0)

    def __init__(
        self,
        workspace: FileWorkspace,
        *,
        author_name: str = "ai-orchestrator",
        author_email: str = "ai@orchestrator.local",
    ) -> None:
        self._ws = workspace
        self._author = author_name
        self._email = author_email
        self._lock = asyncio.Lock()
        self._git: Optional[str] = None  # Resolved git path, cached

    @property
    def workspace(self) -> FileWorkspace:
        return self._ws

    @property
    def root(self) -> Path:
        return self._ws.workspace_root

    # ----- Helpers -------------------------------------------------------

    def _locate_git(self) -> str:
        """Return path to the git binary.  Raises ``GitNotInstalled`` if not
        found.  The result is cached for the lifetime of the instance."""
        if self._git is not None:
            return self._git
        for candidate in ("/usr/bin/git", "/usr/local/bin/git", "git"):
            if candidate == "git":
                # Search PATH
                which = os.environ.get("PATH", "").split(os.pathsep)
                for d in which:
                    p = Path(d) / "git"
                    if p.is_file() and os.access(p, os.X_OK):
                        self._git = str(p.resolve())
                        return self._git
            elif os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                self._git = candidate
                return candidate
        raise GitNotInstalled(
            "git executable not found on PATH.  Install git (brew install git / "
            "apt-get install git / winget install git) and try again."
        )

    async def _run(
        self,
        *args: str,
        check: bool = True,
        timeout: int = 60,
        input_data: Optional[bytes] = None,
    ) -> str:
        """Run a git command inside the workspace root.

        Returns stdout.  Raises ``GitError`` on non-zero exit unless
        ``check=False``.
        """
        git = self._locate_git()
        cmd = [
            git,
            "-c", f"user.name={self._author}",
            "-c", f"user.email={self._email}",
            *args,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.root),
                stdin=asyncio.subprocess.PIPE if input_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_data), timeout=timeout
            )
        except FileNotFoundError as e:
            raise GitNotInstalled(
                f"git binary not found at {git!r}"
            ) from e
        except asyncio.TimeoutError:
            raise GitError(f"git command timed out after {timeout}s: {' '.join(cmd)}")

        out_text = stdout.decode("utf-8", errors="replace")
        err_text = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            combined = (out_text + "\n" + err_text).strip()
            if "conflict" in err_text.lower():
                raise GitConflictError(combined)
            if check:
                raise GitError(combined or f"git exited with code {proc.returncode}")

        return out_text

    async def _ensure_repo(self) -> None:
        """Raise ``GitNotRepoError`` if the workspace is not a git repository."""
        ret = await self._run("rev-parse", "--git-dir", check=False)
        if not ret.strip():
            raise GitNotRepoError(
                f"workspace {self.root!r} is not a git repository. "
                f"Call .init() first."
            )

    def _parse_timestamp(self, unix_ts: str) -> datetime:
        try:
            return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return datetime.now(timezone.utc)

    # ----- Setup ---------------------------------------------------------

    async def is_initialized(self) -> bool:
        """Check whether the workspace has been initialized as a git repo."""
        try:
            ret = await self._run("rev-parse", "--git-dir", check=False)
            return bool(ret.strip())
        except GitNotInstalled:
            return False

    async def init(
        self,
        *,
        default_branch: str = "main",
        initial_message: str = "initial workspace state",
    ) -> None:
        """Initialize the workspace as a git repository (idempotent).

        Creates the initial branch, stages all existing files, and creates
        the first commit.  Safe to call multiple times — subsequent calls
        are no-ops.
        """
        if hasattr(self, "_init_done"):
            return
        if await self.is_initialized():
            self._init_done = True
            return

        await self._run("init", f"--initial-branch={default_branch}")

        # Stage everything already in the workspace.
        await self._run("add", ".", check=False)

        # Create root commit (empty if nothing staged).
        ret = await self._run(
            "commit", "--allow-empty", "-m", initial_message, check=False
        )
        self._init_done = True

    # ----- Branching -----------------------------------------------------

    async def branch(
        self,
        name: str,
        *,
        from_ref: str = "main",
        checkout: bool = True,
    ) -> str:
        """Create branch ``name`` from ``from_ref`` and optionally switch to it.

        Returns the branch name.
        """
        await self._ensure_repo()
        # Guard: refuse to branch onto an already-existing branch.
        existing = await self._run(
            "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False
        )
        if existing.strip():
            raise GitError(f"branch {name!r} already exists")

        await self._run("checkout", "-b", name, from_ref)
        return name

    async def current_branch(self) -> Optional[str]:
        """Return the current branch name, or ``None`` if HEAD is detached."""
        await self._ensure_repo()
        raw = await self._run("rev-parse", "--abbrev-ref", "HEAD", check=False)
        ref = raw.strip()
        if ref == "HEAD":
            return None
        return ref

    async def checkout(self, ref: str) -> None:
        """Switch to an existing branch or a specific commit (detached HEAD)."""
        await self._ensure_repo()
        await self._run("checkout", ref, check=True)

    async def current_sha(self) -> str:
        """Return the full SHA of HEAD."""
        await self._ensure_repo()
        return (await self._run("rev-parse", "HEAD")).strip()

    # ----- Committing ----------------------------------------------------

    async def commit(self, message: str) -> CommitInfo:
        """Stage all changes and create a commit.

        Returns :class:`CommitInfo` with SHA, message, author, and timestamp.
        """
        await self._ensure_repo()
        async with self._lock:
            # Stage everything.
            await self._run("add", "-A")
            # Check whether there is something to commit.
            status_raw = await self._run("status", "--porcelain", check=False)
            if not status_raw.strip():
                raise GitError("nothing to commit — working tree clean")

            await self._run("commit", "-m", message)

        return await self._latest_commit()

    async def _latest_commit(self) -> CommitInfo:
        """Parse the most recent commit from ``git log -1``."""
        raw = await self._run(
            "log",
            "-1",
            f"--format={self._COMMIT_FORMAT}",
        )
        return self._parse_single_commit(raw)

    @staticmethod
    def _parse_single_commit(raw: str) -> CommitInfo:
        """Parse output of ``git log --format=<format> -1``.

        Expected format::

            sha
            author_name
            author_email
            unix_timestamp
            parent_shas (space-separated)
            message lines
            ===END===
        """
        parts = raw.split("\n===END===\n", 1)
        body = parts[0].strip() if parts else raw.strip()
        lines = body.split("\n")
        sha = lines[0] if len(lines) > 0 else ""
        author_name = lines[1] if len(lines) > 1 else ""
        author_email = lines[2] if len(lines) > 2 else ""
        # Parse timestamp
        ts_str = lines[3].strip() if len(lines) > 3 else "0"
        try:
            timestamp = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            timestamp = datetime.now(timezone.utc)
        parents_str = lines[4].strip() if len(lines) > 4 else ""
        parents = tuple(parents_str.split()) if parents_str else ()
        message = "\n".join(lines[5:]) if len(lines) > 5 else ""
        message = message.strip()
        return CommitInfo(
            sha=sha,
            message=message,
            author_name=author_name,
            author_email=author_email,
            timestamp=timestamp,
            parents=parents,
        )

    # ----- Diff ----------------------------------------------------------

    async def diff(
        self,
        ref: str = "main",
        *,
        context: int = 3,
        path: Optional[str] = None,
    ) -> str:
        """Return unified diff of changes between the current branch and ``ref``.

        Uses the three-dot syntax ``ref...HEAD`` so the diff shows only
        changes that happened on the current branch, not changes from
        ``ref`` that are missing.
        """
        await self._ensure_repo()
        args = ["diff", f"--unified={context}", f"{ref}...HEAD"]
        if path:
            # Path must be relative to workspace root; resolve it.
            resolved = self._ws._resolve(path)
            args.append(str(resolved))
        ret = await self._run(*args, check=False)
        return ret

    async def diff_uncommitted(self) -> str:
        """Return unified diff of working tree changes (vs HEAD)."""
        await self._ensure_repo()
        return await self._run("diff", "HEAD", check=False)

    async def diff_working_tree(self) -> str:
        """Diff of uncommitted + unstaged changes only (vs index)."""
        await self._ensure_repo()
        return await self._run("diff", check=False)

    async def diff_staged(self) -> str:
        """Diff of staged changes only (vs HEAD)."""
        await self._ensure_repo()
        return await self._run("diff", "--cached", check=False)

    # ----- Rollback ------------------------------------------------------

    async def rollback(self, ref: str = "HEAD") -> None:
        """Hard-reset the workspace to ``ref``.

        **Destructive.** All uncommitted changes and all commits above
        ``ref`` are lost.  For a safer undo, check :meth:`log` first and
        pick the exact SHA to roll back to.
        """
        await self._ensure_repo()
        async with self._lock:
            await self._run("reset", "--hard", ref)

    async def reset_soft(self, ref: str = "HEAD") -> None:
        """Soft-reset — move HEAD to ``ref`` but keep all changes staged."""
        await self._ensure_repo()
        async with self._lock:
            await self._run("reset", "--soft", ref)

    async def restore(self, *paths: str) -> None:
        """Discard unstaged changes in the given paths (``git restore``)."""
        await self._ensure_repo()
        if not paths:
            await self._run("restore", ".")
        else:
            resolved = [str(self._ws._resolve(p)) for p in paths]
            await self._run("restore", *resolved)

    async def unstage(self, *paths: str) -> None:
        """Unstage paths without discarding working-tree changes."""
        await self._ensure_repo()
        if not paths:
            await self._run("reset", "HEAD", "--", ".")
        else:
            resolved = [str(self._ws._resolve(p)) for p in paths]
            await self._run("reset", "HEAD", "--", *resolved)

    # ----- Stash ---------------------------------------------------------

    async def stash(self, *, message: str = "") -> None:
        """Stash working tree and index changes."""
        await self._ensure_repo()
        args = ["stash"]
        if message:
            args.extend(["-m", message])
        else:
            args.append("--")
        await self._run(*args)

    async def unstash(self) -> None:
        """Apply and drop the most recent stash (``git stash pop``)."""
        await self._ensure_repo()
        try:
            await self._run("stash", "pop")
        except GitError as e:
            msg = str(e)
            if "No stash entries" in msg:
                raise GitError("no stash entries to pop") from e
            raise

    # ----- Merge ---------------------------------------------------------

    async def merge(self, ref: str, *, message: Optional[str] = None) -> Optional[str]:
        """Merge ``ref`` into the current branch.

        Returns the merge-commit SHA (``None`` for fast-forward merges).
        Raises :class:`GitConflictError` on conflicts.

        For the standard ``task/… → main`` workflow::

            await gw.checkout("main")
            await gw.merge("task/add-helper")
        """
        await self._ensure_repo()
        async with self._lock:
            args = ["merge", "--no-ff", ref]
            if message:
                args.extend(["-m", message])
            ret = await self._run(*args, check=False)
            if "conflict" in ret.lower():
                # Abort and let the caller handle the conflict.
                raise GitConflictError(
                    f"merge conflict merging {ref!r} into current branch:\n{ret}"
                )
        # Detect fast-forward: the merge output shows "Fast-forward"
        if "fast-forward" in ret.lower():
            return None
        # Return the new merge commit SHA.
        return await self.current_sha()

    async def abort_merge(self) -> None:
        """Abort a conflicted merge (``git merge --abort``)."""
        await self._ensure_repo()
        await self._run("merge", "--abort")

    # ----- Information ---------------------------------------------------

    async def log(self, max_count: int = 30) -> list[CommitInfo]:
        """Return recent commit history.

        The list is sorted newest-first.
        """
        await self._ensure_repo()
        raw = await self._run(
            "log",
            f"-{max_count}",
            f"--format={self._COMMIT_FORMAT}",
        )
        return self._parse_log(raw)

    @staticmethod
    def _parse_log(raw: str) -> list[CommitInfo]:
        commits: list[CommitInfo] = []
        blocks = raw.split("\n===END===\n")
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            try:
                commit = GitWorkspace._parse_single_commit(block)
                commits.append(commit)
            except Exception:
                continue
        return commits

    async def status(self) -> list[FileStatus]:
        """Return working-tree and staging-area status.

        Uses ``git status --porcelain=v1``, which gives two-character
        codes per line::

            XY path

        X = index status, Y = working-tree status.
        """
        await self._ensure_repo()
        raw = await self._run("status", "--porcelain=v1")
        results: list[FileStatus] = []
        for line in raw.splitlines():
            line = line.rstrip("\n")
            if len(line) < 3:
                continue
            codes = line[:2]
            path = line[3:]
            # Index status
            index_code = codes[0]
            if index_code != " ":
                results.append(
                    FileStatus(path=path, status=index_code, staged=True)
                )
            # Working-tree status
            wt_code = codes[1]
            if wt_code != " ":
                results.append(
                    FileStatus(path=path, status=wt_code, staged=False)
                )
            # If both are ' ' or '?' — check for untracked
            if codes == "??":
                results.append(FileStatus(path=path, status="?", staged=False))
        return results

    async def file_count(self) -> int:
        """Count tracked files in the current commit."""
        await self._ensure_repo()
        raw = await self._run("ls-files", check=False)
        return len([l for l in raw.splitlines() if l.strip()]) if raw.strip() else 0

    async def is_clean(self) -> bool:
        """Return ``True`` if the working tree is clean (no unstaged/staged changes)."""
        raw = await self._run("status", "--porcelain=v1", check=False)
        return not raw.strip()

    # ----- Class methods for bootstrapping -------------------------------

    @classmethod
    async def from_existing(
        cls,
        source: Path,
        task_id: str,
        *,
        root: Union[Path, str] = DEFAULT_WORKSPACE_ROOT,
        author_name: str = "ai-orchestrator",
        author_email: str = "ai@orchestrator.local",
        exclude: tuple[str, ...] = (
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".ruff_cache",
            ".pytest_cache",
            ".mypy_cache",
            ".coverage",
            "htmlcov",
            "*.pyc",
            "*.pyo",
            "__pycache__",
        ),
    ) -> GitWorkspace:
        """Create a workspace pre-populated with an existing project.

        Copies ``source`` into ``workspaces/<task_id>/`` (respecting
        ``.gitignore``-like exclusions), initialises a git repo with a root
        commit, and returns a :class:`GitWorkspace` ready for branch-commit-
        diff-rollback.

        ``exclude`` is a tuple of glob-style patterns to skip.
        """
        import fnmatch
        import shutil

        source = Path(source).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"source is not a directory: {source}")

        # Create the workspace directory.  We use ``with create=True`` which
        # calls ``mkdir(parents=True, exist_ok=True)``.
        ws = FileWorkspace.for_task(task_id, root=root, create=True)
        target = ws.workspace_root

        # Copy files from source to target, excluding patterns.
        def _should_exclude(rel_path: str) -> bool:
            for pat in exclude:
                if fnmatch.fnmatch(rel_path, pat):
                    return True
            return False

        for dirpath, dirnames, filenames in os.walk(source, topdown=True):
            # Prune excluded directories in-place so os.walk doesn't enter them.
            rel_dir = os.path.relpath(dirpath, source)
            dirnames[:] = [
                d
                for d in dirnames
                if not _should_exclude(os.path.join(rel_dir, d))
            ]

            for fn in filenames:
                src_path = os.path.join(dirpath, fn)
                rel_file = os.path.relpath(src_path, source)
                if _should_exclude(rel_file):
                    continue
                dst_path = target / rel_file
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

        gw = cls(ws, author_name=author_name, author_email=author_email)
        await gw.init(initial_message=f"initialise workspace from {source.name}")

        return gw
