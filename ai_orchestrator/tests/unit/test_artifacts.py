"""Tests for ArtifactStore — convention layer for plans/reports/diffs/etc."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_orchestrator.workspace import (
    Artifact,
    ArtifactKind,
    ArtifactStore,
    FileWorkspace,
)


@pytest.fixture
def workspace(tmp_path: Path) -> FileWorkspace:
    return FileWorkspace.for_task("art-task", root=tmp_path)


@pytest.fixture
def store(workspace: FileWorkspace) -> ArtifactStore:
    return ArtifactStore(workspace)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_kind_directories(self, workspace: FileWorkspace) -> None:
        ArtifactStore(workspace)
        for sub in (
            "plans",
            "code",
            "reports",
            "diffs",
            "screenshots",
            "transcripts",
            "logs",
        ):
            assert (workspace.workspace_root / "artifacts" / sub).is_dir()

    def test_idempotent_construction(self, workspace: FileWorkspace) -> None:
        ArtifactStore(workspace)
        # Second construction must not raise
        ArtifactStore(workspace)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


class TestPlan:
    def test_plan_writes_markdown(self, store: ArtifactStore) -> None:
        art = store.plan("# My Plan\n\n1. Step one\n2. Step two\n")
        assert art.kind == ArtifactKind.PLAN
        assert art.path.startswith("artifacts/plans/")
        assert art.path.endswith(".md")
        assert art.size > 0
        assert store.read(art) == "# My Plan\n\n1. Step one\n2. Step two\n"

    def test_plan_with_explicit_id(self, store: ArtifactStore) -> None:
        art = store.plan("body", plan_id="my-plan-v1")
        assert art.artifact_id == "my-plan-v1"
        assert "my-plan-v1" in art.path

    def test_plan_metadata_attached(self, store: ArtifactStore) -> None:
        art = store.plan("body", metadata={"author": "planner", "version": 2})
        assert art.metadata == {"author": "planner", "version": 2}


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


class TestCode:
    def test_code_snapshot(self, store: ArtifactStore) -> None:
        art = store.code("print('hi')\n", path="hello.py")
        assert art.kind == ArtifactKind.CODE
        assert art.path == "artifacts/code/hello.py"
        assert store.read(art) == "print('hi')\n"

    def test_code_with_nested_path(self, store: ArtifactStore) -> None:
        art = store.code("x = 1\n", path="pkg/sub/mod.py")
        assert art.path == "artifacts/code/pkg/sub/mod.py"
        assert store.read(art) == "x = 1\n"

    def test_code_rejects_dotdot(self, store: ArtifactStore) -> None:
        with pytest.raises(ValueError):
            store.code("x", path="../escape.py")
        with pytest.raises(ValueError):
            store.code("x", path="dir/")

    def test_code_creates_parents(self, store: ArtifactStore) -> None:
        art = store.code("x", path="a/b/c.py")
        assert store.read(art) == "x"


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_writes_patch(self, store: ArtifactStore) -> None:
        art = store.diff("--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n")
        assert art.kind == ArtifactKind.DIFF
        assert art.path.endswith(".patch")
        assert store.read(art).startswith("--- a\n")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_writes_json(self, store: ArtifactStore) -> None:
        art = store.report("test", {"passed": 3, "failed": 0})
        assert art.kind == ArtifactKind.REPORT
        assert art.path.startswith("artifacts/reports/test-")
        assert art.path.endswith(".json")
        # metadata captures report kind
        assert art.metadata.get("report_kind") == "test"
        # body is valid JSON
        body = store.read(art)
        assert isinstance(body, str)
        assert json.loads(body) == {"passed": 3, "failed": 0}

    def test_report_with_metadata(self, store: ArtifactStore) -> None:
        art = store.report(
            "review",
            {"score": 8},
            metadata={"reviewer": "claude", "round": 1},
        )
        assert art.metadata["report_kind"] == "review"
        assert art.metadata["reviewer"] == "claude"

    def test_report_invalid_kind_rejected(self, store: ArtifactStore) -> None:
        with pytest.raises(ValueError):
            store.report("not valid!", {})

    def test_report_sorts_keys(self, store: ArtifactStore) -> None:
        art = store.report("test", {"z": 1, "a": 2})
        body = store.read(art)
        assert body.index('"a"') < body.index('"z"')

    def test_report_serialization_handles_non_json(self, store: ArtifactStore) -> None:
        from datetime import datetime, timezone
        from pathlib import Path

        payload = {
            "when": datetime(2026, 6, 5, tzinfo=timezone.utc),
            "where": Path("/tmp"),
        }
        art = store.report("test", payload)
        body = store.read(art)
        parsed = json.loads(body)
        assert "2026" in parsed["when"]


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


class TestLog:
    def test_log_writes_text(self, store: ArtifactStore) -> None:
        art = store.log("Starting up...\nDone.\n")
        assert art.kind == ArtifactKind.LOG
        assert art.path.endswith(".log")
        assert store.read(art) == "Starting up...\nDone.\n"


# ---------------------------------------------------------------------------
# transcript (JSONL)
# ---------------------------------------------------------------------------


class TestTranscript:
    def test_transcript_writes_jsonl(self, store: ArtifactStore) -> None:
        entries = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        art = store.transcript(entries)
        assert art.kind == ArtifactKind.TRANSCRIPT
        assert art.path.endswith(".jsonl")
        body = store.read(art)
        lines = [line for line in body.split("\n") if line]
        assert len(lines) == 2
        for i, entry in enumerate(entries):
            assert json.loads(lines[i]) == entry

    def test_transcript_requires_list(self, store: ArtifactStore) -> None:
        with pytest.raises(TypeError):
            store.transcript("not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# screenshot
# ---------------------------------------------------------------------------


class TestScreenshot:
    def test_screenshot_writes_bytes(self, store: ArtifactStore) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        art = store.screenshot(png, name="screenshot.png")
        assert art.kind == ArtifactKind.SCREENSHOT
        assert art.path == "artifacts/screenshots/screenshot.png"
        # read returns bytes for binary
        data = store.read(art)
        assert isinstance(data, bytes)
        assert data == png

    def test_screenshot_metadata(self, store: ArtifactStore) -> None:
        art = store.screenshot(b"x", name="a.png", metadata={"url": "https://x"})
        assert art.metadata["url"] == "https://x"

    def test_screenshot_rejects_non_bytes(self, store: ArtifactStore) -> None:
        with pytest.raises(TypeError):
            store.screenshot("not bytes", name="x.png")  # type: ignore[arg-type]

    def test_screenshot_rejects_empty_name(self, store: ArtifactStore) -> None:
        with pytest.raises(ValueError):
            store.screenshot(b"x", name="")

    def test_screenshot_rejects_dotdot(self, store: ArtifactStore) -> None:
        with pytest.raises(ValueError):
            store.screenshot(b"x", name="../escape.png")


# ---------------------------------------------------------------------------
# read by path
# ---------------------------------------------------------------------------


class TestReadByPath:
    def test_read_by_string_path(self, store: ArtifactStore) -> None:
        store.plan("hello")
        body = store.read("artifacts/plans/" + (store.list(ArtifactKind.PLAN)[0].artifact_id) + ".md")
        assert body == "hello"


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


class TestList:
    def test_list_empty(self, store: ArtifactStore) -> None:
        assert store.list() == []

    def test_list_all(self, store: ArtifactStore) -> None:
        store.plan("p1")
        store.plan("p2")
        store.diff("d1")
        store.report("test", {"x": 1})
        store.screenshot(b"x", name="a.png")
        arts = store.list()
        kinds = {a.kind for a in arts}
        assert kinds == {
            ArtifactKind.PLAN,
            ArtifactKind.DIFF,
            ArtifactKind.REPORT,
            ArtifactKind.SCREENSHOT,
        }

    def test_list_filtered_by_kind(self, store: ArtifactStore) -> None:
        store.plan("p1")
        store.plan("p2")
        store.diff("d1")
        plans = store.list(ArtifactKind.PLAN)
        assert len(plans) == 2
        assert all(a.kind == ArtifactKind.PLAN for a in plans)

    def test_list_reports_round_trip_kind(self, store: ArtifactStore) -> None:
        store.report("test", {"a": 1})
        store.report("review", {"b": 2})
        reports = store.list(ArtifactKind.REPORT)
        report_kinds = {a.metadata.get("report_kind") for a in reports}
        assert report_kinds == {"test", "review"}

    def test_list_sorted_by_created_desc(self, store: ArtifactStore) -> None:
        import time
        store.plan("first")
        time.sleep(0.01)
        store.plan("second")
        arts = store.list(ArtifactKind.PLAN)
        # Most recent first
        assert arts[0].size == len("second")
        assert arts[1].size == len("first")


# ---------------------------------------------------------------------------
# Artifact value object
# ---------------------------------------------------------------------------


class TestArtifactValueObject:
    def test_artifact_is_binary(self, store: ArtifactStore) -> None:
        art = store.screenshot(b"x", name="a.png")
        assert art.is_binary is True
        plan = store.plan("x")
        assert plan.is_binary is False

    def test_artifact_task_id_matches_workspace(self, store: ArtifactStore) -> None:
        art = store.plan("x")
        assert art.task_id == store._ws.task_id
