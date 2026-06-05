"""Workspace — file operations and artifact storage for agent tasks.

Provides :class:`FileWorkspace` (low-level read/write/patch/list/search) and
:class:`ArtifactStore` (high-level convention layer for plans, code, reports,
screenshots). Both are the foundation for any agent that needs to touch a
codebase, since they give agents a safe, audited place to read and write.

Layout on disk (created on first use)::

    workspaces/
        <task_id>/
            src/                 # code the agents edit
            artifacts/
                plans/           # plan.md, planner output
                code/            # generated code snapshots
                reports/         # test reports, review reports
                diffs/           # unified diffs
                screenshots/     # PNG/JPG visual evidence
                transcripts/     # JSONL conversation transcripts
                logs/            # raw execution logs
"""

from ai_orchestrator.workspace.artifacts import Artifact, ArtifactKind, ArtifactStore
from ai_orchestrator.workspace.ast_patch import ASTPatchEngine, Patch, PatchError, PatchResult
from ai_orchestrator.workspace.exceptions import (
    AtomicWriteError,
    GitConflictError,
    GitError,
    GitNotInstalled,
    GitNotRepoError,
    PatchConflictError,
    PathTraversalError,
    WorkspaceAlreadyExistsError,
    WorkspaceError,
    WorkspaceNotFoundError,
)
from ai_orchestrator.workspace.git import CommitInfo, FileStatus, GitWorkspace
from ai_orchestrator.workspace.manager import (
    DEFAULT_WORKSPACE_ROOT,
    FileEntry,
    FileWorkspace,
    SearchMatch,
    SnapshotChange,
    WorkspaceSnapshot,
)

__all__ = [
    "DEFAULT_WORKSPACE_ROOT",
    "FileWorkspace",
    "FileEntry",
    "SearchMatch",
    "WorkspaceSnapshot",
    "SnapshotChange",
    "ArtifactStore",
    "Artifact",
    "ArtifactKind",
    "ASTPatchEngine",
    "Patch",
    "PatchError",
    "PatchResult",
    "GitWorkspace",
    "CommitInfo",
    "FileStatus",
    "WorkspaceError",
    "WorkspaceNotFoundError",
    "WorkspaceAlreadyExistsError",
    "PathTraversalError",
    "PatchConflictError",
    "AtomicWriteError",
    "GitError",
    "GitNotInstalled",
    "GitNotRepoError",
    "GitConflictError",
]

