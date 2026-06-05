"""Tests for UI Schema Engine, Selector Cache, Accessibility Runtime, UI Intelligence."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from ai_orchestrator.browser.accessibility import A11yNode, A11ySnapshot, AccessibilityRuntime
from ai_orchestrator.browser.schema import UIMessage, UIRole, UISchema, UISelector, UISchemaEngine
from ai_orchestrator.browser.selector_cache import SelectorCache


class TestUISchema:
    def test_schema_complete(self):
        schema = UISchema(
            input_box=UISelector(role=UIRole.INPUT, value="textarea"),
            send_button=UISelector(role=UIRole.SEND, value="button"),
        )
        assert schema.is_complete()

    def test_schema_incomplete(self):
        schema = UISchema()
        assert not schema.is_complete()

    def test_schema_engine_register(self):
        engine = UISchemaEngine()
        schema = engine.build_schema(
            messages=[UIMessage(content="hello", role="user")],
            title="Test",
        )
        engine.register("test", schema)
        assert engine.get("test") is schema
        assert engine.get("unknown") is None


class TestSelectorCache:
    def test_cache_miss(self):
        cache = SelectorCache()
        assert cache.get("qwen_ui", "input") is None

    def test_cache_hit(self):
        cache = SelectorCache()
        cache.set("qwen_ui", "input", "textarea", source="dom", confidence=0.9)
        entry = cache.get("qwen_ui", "input")
        assert entry == {"value": "textarea", "source": "dom", "confidence": 0.9}

    def test_get_all(self):
        cache = SelectorCache()
        cache.set("p1", "input", "textarea")
        cache.set("p1", "send", "button")
        all_ = cache.get_all("p1")
        assert "input" in all_
        assert "send" in all_

    def test_invalidate_role(self):
        cache = SelectorCache()
        cache.set("p1", "input", "textarea")
        cache.set("p1", "send", "button")
        cache.invalidate("p1", "input")
        assert cache.get("p1", "input") is None
        assert cache.get("p1", "send") is not None

    def test_invalidate_all(self):
        cache = SelectorCache()
        cache.set("p1", "input", "textarea")
        cache.invalidate("p1")
        assert cache.get_all("p1") == {}

    def test_persist_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            cache = SelectorCache(path)
            cache.set("p1", "input", "textarea")
            cache.persist()

            loaded = SelectorCache(path)
            assert loaded.get("p1", "input") == {"value": "textarea", "source": "cache", "confidence": 1.0}
        finally:
            Path(path).unlink(missing_ok=True)


class TestA11y:
    def test_parse_snapshot(self):
        raw = {
            "role": "RootWebArea",
            "name": "Chat",
            "children": [
                {
                    "role": "textbox",
                    "name": "Message input",
                    "focused": True,
                    "value": "hello",
                },
                {
                    "role": "button",
                    "name": "Send",
                    "disabled": False,
                },
            ],
        }
        rt = AccessibilityRuntime()
        root = rt._parse_node(raw)
        assert root.role == "RootWebArea"
        assert root.name == "Chat"
        assert len(root.children) == 2

        flat = []
        rt._flatten(root, flat)
        assert len(flat) == 3  # root + 2 children

    def test_node_classification(self):
        tb = A11yNode(role="textbox", name="input")
        btn = A11yNode(role="button", name="Send")
        div = A11yNode(role="generic", name="")

        assert tb.is_interactive
        assert tb.is_text_input
        assert not tb.is_button

        assert btn.is_interactive
        assert not btn.is_text_input
        assert btn.is_button

        assert not div.is_interactive

    def test_snapshot_filters(self):
        nodes = [
            A11yNode(role="textbox", name="input1"),
            A11yNode(role="button", name="Send"),
            A11yNode(role="generic", name="container"),
        ]
        snap = A11ySnapshot(root=nodes[0], _flat=nodes)
        assert len(snap.interactive_nodes()) == 2
        assert len(snap.text_inputs()) == 1
        assert len(snap.buttons()) == 1


class TestUIIntelligence:
    def test_a11y_node_to_css(self):
        from ai_orchestrator.browser.intelligence import UIIntelligence
        css = UIIntelligence._a11y_node_to_css(A11yNode(role="button", name='Send "Message"'))
        assert css == '[aria-label="Send \\"Message\\""]'

    def test_known_selectors_registered(self):
        from ai_orchestrator.browser.intelligence import UIIntelligence
        ui = UIIntelligence()
        assert "qwen_ui" in ui._provider_known_selectors
        assert "chatgpt_ui" in ui._provider_known_selectors
        assert UIRole.INPUT in ui._provider_known_selectors["qwen_ui"]
        assert UIRole.SEND in ui._provider_known_selectors["qwen_ui"]
