"""Tests for AST Patch Engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ai_orchestrator.workspace.ast_patch import ASTPatchEngine, Patch, PatchError, PatchResult


class TestASTPatchEngine:
    def test_apply_simple(self):
        engine = ASTPatchEngine()
        patch = Patch(old_code="x = 1", new_code="x = 2")
        result = engine.apply("x = 1\ny = 3", patch)
        assert result.applied
        assert result.lines_changed > 0

    def test_apply_file(self):
        engine = ASTPatchEngine()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n")
            path = f.name

        try:
            patch = Patch(old_code="x = 1", new_code="x = 2")
            result = engine.apply_to_file(path, patch)
            assert result.applied

            content = Path(path).read_text()
            assert "x = 2" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_apply_file_not_found(self):
        engine = ASTPatchEngine()
        result = engine.apply_to_file("/nonexistent/file.py", Patch(old_code="", new_code=""))
        assert not result.applied
        assert "File not found" in result.error

    def test_validate_python_syntax(self):
        engine = ASTPatchEngine()
        patch = Patch(old_code="x = 1", new_code="x = :")
        result = engine.apply("x = 1", patch)
        assert not result.applied
        assert "Python syntax error" in result.error

    def test_create_patch(self):
        engine = ASTPatchEngine()
        patch = engine.create_patch("/tmp/test.py", "old", "new")
        assert patch.file_path == "/tmp/test.py"
        assert patch.old_code == "old"
        assert patch.new_code == "new"

    def test_diff_generation(self):
        engine = ASTPatchEngine()
        patch = Patch(old_code="line1", new_code="line1_changed")
        result = engine.apply("line1\nline2\nline3", patch)
        assert result.applied
        assert "line1" in result.diff
        assert "line1_changed" in result.diff

    def test_old_code_not_found(self):
        engine = ASTPatchEngine()
        patch = Patch(old_code="nonexistent", new_code="new")
        result = engine.apply("real content", patch)
        assert not result.applied
        assert "not found" in result.error

    def test_empty_source(self):
        engine = ASTPatchEngine()
        patch = Patch(old_code="", new_code="new")
        result = engine.apply("", patch)
        assert not result.applied

    def test_tree_sitter_check(self):
        engine = ASTPatchEngine()
        # Should not crash, even if tree_sitter is not installed
        assert isinstance(engine._ts_available, bool)
