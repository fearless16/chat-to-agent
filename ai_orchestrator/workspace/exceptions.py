"""Workspace-specific exceptions.

All errors raised by :mod:`ai_orchestrator.workspace` derive from
:class:`WorkspaceError` so callers can catch the whole family with a single
``except`` clause.
"""

from __future__ import annotations


class WorkspaceError(Exception):
    """Base class for every workspace-related error."""


class PathTraversalError(WorkspaceError, ValueError):
    """A relative path attempted to escape the workspace root.

    Catching this implies a programming error or hostile input — never catch
    it silently in production code.
    """

    def __init__(self, attempted: str, root: str) -> None:
        super().__init__(f"path {attempted!r} escapes workspace root {root!r}")
        self.attempted = attempted
        self.root = root


class WorkspaceNotFoundError(WorkspaceError, FileNotFoundError):
    """The requested file or directory does not exist inside the workspace."""


class WorkspaceAlreadyExistsError(WorkspaceError, FileExistsError):
    """Attempted to create a resource that already exists with exclusive=True."""


class PatchConflictError(WorkspaceError):
    """A patch could not be applied because the file has changed.

    The :attr:`expected` and :attr:`actual` attributes hold the relevant
    substrings to help callers build a rebase strategy.
    """

    def __init__(self, path: str, expected: str, actual_excerpt: str) -> None:
        super().__init__(
            f"patch conflict at {path!r}: expected {expected!r} not found verbatim"
        )
        self.path = path
        self.expected = expected
        self.actual_excerpt = actual_excerpt


class AtomicWriteError(WorkspaceError, OSError):
    """An atomic write failed mid-operation; the workspace may be inconsistent."""


# ---------------------------------------------------------------------------
# Git errors
# ---------------------------------------------------------------------------


class GitError(WorkspaceError):
    """Base class for git-layer errors."""


class GitNotInstalled(GitError, RuntimeError):
    """The ``git`` executable was not found on the system PATH."""


class GitNotRepoError(GitError):
    """The workspace directory has not been initialized as a git repository.

    Call :meth:`GitWorkspace.init` first.
    """


class GitConflictError(GitError):
    """A git operation (merge, rebase) encountered a conflict."""


class GitCanceledError(GitError):
    """A git operation was skipped or canceled mid-flight."""
