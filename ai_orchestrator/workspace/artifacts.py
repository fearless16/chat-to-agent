"""Artifact store — convention layer for plans, code, reports, and media.

:class:`ArtifactStore` wraps a :class:`FileWorkspace` and routes writes to
conventional subdirectories under ``artifacts/``::

    artifacts/
        plans/        # planner output (.md)
        code/         # generated code snapshots
        reports/      # test reports, review reports (.json)
        diffs/        # unified diffs (.patch)
        screenshots/  # PNG/JPG visual evidence
        transcripts/  # conversation transcripts (.jsonl)
        logs/         # raw execution logs (.log)

Each write returns an :class:`Artifact` reference that captures the kind,
id, relative path, size, timestamp, and any caller-supplied metadata.
Reads are uniform: :meth:`ArtifactStore.read` accepts an :class:`Artifact`
or a relative path and returns the file content (bytes for binary kinds,
text otherwise).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from ai_orchestrator.workspace.exceptions import WorkspaceError
from ai_orchestrator.workspace.manager import FileWorkspace


# ---------------------------------------------------------------------------
# ArtifactKind
# ---------------------------------------------------------------------------


class ArtifactKind(str, Enum):
    """Categories of artifacts an agent can produce."""

    PLAN = "plan"
    CODE = "code"
    REPORT = "report"
    DIFF = "diff"
    SCREENSHOT = "screenshot"
    TRANSCRIPT = "transcript"
    LOG = "log"

    @property
    def directory(self) -> str:
        """Subdirectory under ``artifacts/`` where this kind is stored."""
        return self.value + "s"


# Subdirectory mapping; some kinds don't follow the simple +s rule.
_KIND_TO_DIR: dict[ArtifactKind, str] = {
    ArtifactKind.PLAN: "plans",
    ArtifactKind.CODE: "code",
    ArtifactKind.REPORT: "reports",
    ArtifactKind.DIFF: "diffs",
    ArtifactKind.SCREENSHOT: "screenshots",
    ArtifactKind.TRANSCRIPT: "transcripts",
    ArtifactKind.LOG: "logs",
}

_BINARY_KINDS: frozenset[ArtifactKind] = frozenset({ArtifactKind.SCREENSHOT})


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """Reference to a single artifact written to the store."""

    kind: ArtifactKind
    artifact_id: str
    path: str
    """Path relative to the workspace root (always starts with ``artifacts/``)."""
    task_id: str
    size: int
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary(self) -> bool:
        return self.kind in _BINARY_KINDS


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Convention-based artifact storage for one task.

    Wraps a :class:`FileWorkspace` and routes writes to ``artifacts/<dir>/``
    subdirectories based on :class:`ArtifactKind`. Each write returns an
    :class:`Artifact` reference; reads accept either an :class:`Artifact`
    or a relative path.

    The store is safe to use from multiple threads; all mutating methods
    hold the underlying workspace's lock.
    """

    _ARTIFACTS_DIR = "artifacts"

    def __init__(self, workspace: FileWorkspace) -> None:
        self._ws = workspace
        self._counter = 0
        self._lock = threading.Lock()
        # Ensure all kind directories exist up front.
        for subdir in _KIND_TO_DIR.values():
            workspace.mkdir(f"{self._ARTIFACTS_DIR}/{subdir}", exist_ok=True)

    # ----- ID generation -------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        with self._lock:
            self._counter += 1
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            return f"{prefix}-{ts}-{self._counter:04d}-{uuid.uuid4().hex[:6]}"

    def _make_artifact(
        self,
        kind: ArtifactKind,
        artifact_id: str,
        relative_path: str,
        size: int,
        metadata: dict[str, Any],
    ) -> Artifact:
        return Artifact(
            kind=kind,
            artifact_id=artifact_id,
            path=relative_path,
            task_id=self._ws.task_id,
            size=size,
            created_at=datetime.now(timezone.utc),
            metadata=metadata,
        )

    # ----- Text writes ---------------------------------------------------

    def plan(
        self,
        content: str,
        *,
        plan_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        aid = plan_id or self._next_id("plan")
        rel = f"{self._ARTIFACTS_DIR}/plans/{aid}.md"
        self._ws.write(rel, content)
        return self._make_artifact(
            ArtifactKind.PLAN, aid, rel, len(content.encode("utf-8")), metadata or {}
        )

    def code(
        self,
        content: str,
        *,
        path: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        """Snapshot of generated code under ``artifacts/code/``."""
        if not path or path.endswith("/"):
            raise ValueError("path must be a file path, not a directory")
        clean = path.lstrip("/")
        if ".." in clean.split("/"):
            raise ValueError(f"path must not contain '..': {path!r}")
        rel = f"{self._ARTIFACTS_DIR}/code/{clean}"
        self._ws.write(rel, content)
        return self._make_artifact(
            ArtifactKind.CODE, clean, rel, len(content.encode("utf-8")), metadata or {}
        )

    def diff(
        self,
        content: str,
        *,
        diff_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        aid = diff_id or self._next_id("diff")
        rel = f"{self._ARTIFACTS_DIR}/diffs/{aid}.patch"
        self._ws.write(rel, content)
        return self._make_artifact(
            ArtifactKind.DIFF, aid, rel, len(content.encode("utf-8")), metadata or {}
        )

    def log(
        self,
        content: str,
        *,
        log_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        aid = log_id or self._next_id("log")
        rel = f"{self._ARTIFACTS_DIR}/logs/{aid}.log"
        self._ws.write(rel, content)
        return self._make_artifact(
            ArtifactKind.LOG, aid, rel, len(content.encode("utf-8")), metadata or {}
        )

    def report(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        report_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        """Write a structured JSON report.

        ``kind`` is a sub-categorization inside ``reports/`` (e.g. ``test``,
        ``review``, ``lint``). The file name is ``<kind>-<id>.json``.
        """
        if not kind or not kind.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"report kind must be alphanumeric, got {kind!r}")
        aid = report_id or self._next_id("report")
        rel = f"{self._ARTIFACTS_DIR}/reports/{kind}-{aid}.json"
        body = json.dumps(payload, indent=2, sort_keys=True, default=str)
        meta = {"report_kind": kind, **(metadata or {})}
        self._ws.write(rel, body)
        return self._make_artifact(
            ArtifactKind.REPORT, aid, rel, len(body.encode("utf-8")), meta
        )

    def transcript(
        self,
        entries: list[dict[str, Any]],
        *,
        transcript_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        """Write a JSONL transcript (one JSON object per line)."""
        if not isinstance(entries, list):
            raise TypeError(f"entries must be a list, got {type(entries).__name__}")
        aid = transcript_id or self._next_id("transcript")
        rel = f"{self._ARTIFACTS_DIR}/transcripts/{aid}.jsonl"
        body = "\n".join(json.dumps(e, default=str) for e in entries) + "\n"
        self._ws.write(rel, body)
        return self._make_artifact(
            ArtifactKind.TRANSCRIPT, aid, rel, len(body.encode("utf-8")), metadata or {}
        )

    # ----- Binary writes -------------------------------------------------

    def screenshot(
        self,
        png_bytes: bytes,
        *,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        """Write a PNG/JPG screenshot. ``name`` should include the extension."""
        if not isinstance(png_bytes, (bytes, bytearray)):
            raise TypeError(
                f"png_bytes must be bytes, got {type(png_bytes).__name__}"
            )
        if not name:
            raise ValueError("name must be a non-empty file name")
        clean = name.lstrip("/")
        if ".." in clean.split("/"):
            raise ValueError(f"name must not contain '..': {name!r}")
        aid = self._next_id("screenshot")
        rel = f"{self._ARTIFACTS_DIR}/screenshots/{clean}"
        self._ws.write_bytes(rel, bytes(png_bytes))
        return self._make_artifact(
            ArtifactKind.SCREENSHOT, aid, rel, len(png_bytes), metadata or {}
        )

    # ----- Read ----------------------------------------------------------

    def read(
        self,
        target: Union[Artifact, str],
        *,
        encoding: str = "utf-8",
    ) -> Union[str, bytes]:
        """Read artifact content.

        Returns bytes for binary kinds (e.g. screenshots) and text
        otherwise.
        """
        if isinstance(target, Artifact):
            rel = target.path
            is_binary = target.is_binary
        else:
            rel = target
            is_binary = self._is_binary_path(rel)
        if is_binary:
            return self._ws.read_bytes(rel)
        return self._ws.read(rel, encoding=encoding)

    @staticmethod
    def _is_binary_path(rel: str) -> bool:
        ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
        return ext in {"png", "jpg", "jpeg", "gif", "webp", "pdf", "zip", "tar", "gz"}

    # ----- List ----------------------------------------------------------

    def list(
        self,
        kind: Optional[ArtifactKind] = None,
    ) -> list[Artifact]:
        """List artifacts, optionally filtered by kind.

        Each returned :class:`Artifact` is reconstructed from filesystem
        metadata (no in-memory cache), so the list is always current.
        """
        results: list[Artifact] = []
        kinds = [kind] if kind is not None else list(_KIND_TO_DIR.keys())
        for k in kinds:
            subdir = _KIND_TO_DIR[k]
            for entry in self._ws.list_tree(f"{self._ARTIFACTS_DIR}/{subdir}"):
                if entry.is_dir:
                    continue
                rel = entry.path
                # Reconstruct an artifact_id from the filename.
                stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                meta: dict[str, Any] = {}
                if k == ArtifactKind.REPORT:
                    # Reports are named "<report_kind>-<artifact_id>".
                    parts = stem.split("-", 1)
                    if len(parts) == 2:
                        meta["report_kind"] = parts[0]
                        aid = parts[1]
                    else:
                        aid = stem
                else:
                    aid = stem
                from datetime import datetime as _dt
                created = _dt.fromtimestamp(entry.mtime, tz=timezone.utc)
                results.append(
                    Artifact(
                        kind=k,
                        artifact_id=aid,
                        path=rel,
                        task_id=self._ws.task_id,
                        size=entry.size,
                        created_at=created,
                        metadata=meta,
                    )
                )
        results.sort(key=lambda a: a.created_at, reverse=True)
        return results

    def __repr__(self) -> str:
        return f"ArtifactStore(task_id={self._ws.task_id!r})"
