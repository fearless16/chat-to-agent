"""Tests for accessibility sensor and runtime."""
from __future__ import annotations

import pytest

from ai_orchestrator.browser.accessibility import A11yNode, A11ySnapshot, AccessibilityRuntime
from ai_orchestrator.browser_intelligence.sensors.accessibility_sensor import (
    AccessibilityFeatures,
    AccessibilitySensor,
)


SAMPLE_ARIA_YAML = """\
- banner
- main
  - heading "Welcome"
  - textbox "Type a message..."
  - button "Send"
  - article
    - text "Assistant response here"
    - button "Copy"
    - button "Regenerate"
- contentinfo
"""


class MockPage:
    def __init__(self, yaml_text: str = SAMPLE_ARIA_YAML):
        self._yaml = yaml_text
        self._call_count = 0

    async def aria_snapshot(self):
        self._call_count += 1
        return self._yaml


class FailingMockPage:
    async def aria_snapshot(self):
        raise AttributeError("no such method")

    async def accessibility(self):
        raise AttributeError("no such method")


class TestA11yNode:
    def test_is_interactive(self):
        assert A11yNode(role="button", name="x").is_interactive is True
        assert A11yNode(role="text", name="x").is_interactive is False

    def test_is_text_input(self):
        assert A11yNode(role="textbox", name="x").is_text_input is True

    def test_is_button(self):
        assert A11yNode(role="button", name="x").is_button is True
        assert A11yNode(role="textbox", name="x").is_button is False


class TestAccessibilityRuntime:
    @pytest.mark.asyncio
    async def test_snapshot_parses_aria_yaml(self):
        rt = AccessibilityRuntime()
        page = MockPage()
        snap = await rt.snapshot(page)
        assert snap.root is not None
        assert len(snap.all_nodes()) > 0
        assert len(snap.text_inputs()) >= 1
        assert len(snap.buttons()) >= 1

    @pytest.mark.asyncio
    async def test_find_input(self):
        rt = AccessibilityRuntime()
        page = MockPage()
        inp = await rt.find_input(page, hint="message")
        assert inp is not None
        assert "message" in (inp.name or "").lower()

    @pytest.mark.asyncio
    async def test_find_send_button(self):
        rt = AccessibilityRuntime()
        page = MockPage()
        btn = await rt.find_send_button(page, hint="Send")
        assert btn is not None
        assert btn.role == "button"

    @pytest.mark.asyncio
    async def test_find_message_container(self):
        rt = AccessibilityRuntime()
        page = MockPage("""\
- article "conversation"
  - text "Hello"
""")
        container = await rt.find_message_container(page)
        assert container is not None

    @pytest.mark.asyncio
    async def test_snapshot_on_failing_page_returns_empty(self):
        rt = AccessibilityRuntime()
        page = FailingMockPage()
        snap = await rt.snapshot(page)
        assert snap.root is None
        assert snap.all_nodes() == []


class TestAccessibilitySensor:
    @pytest.mark.asyncio
    async def test_sense_extracts_text_input_count(self):
        sensor = AccessibilitySensor()
        page = MockPage()
        features = await sensor.sense(page)
        assert features.text_input_count >= 1
        assert features.extraction_success is True
        assert features.accessibility_confidence > 0

    @pytest.mark.asyncio
    async def test_sense_extracts_button_count(self):
        sensor = AccessibilitySensor()
        page = MockPage()
        features = await sensor.sense(page)
        assert features.button_count >= 1

    @pytest.mark.asyncio
    async def test_sense_detects_thinking_marker(self):
        sensor = AccessibilitySensor()
        page = MockPage("""\
- text "thinking"
- text "reasoning in progress"
""")
        features = await sensor.sense(page)
        assert features.has_thinking_marker is True

    @pytest.mark.asyncio
    async def test_sense_detects_error_marker(self):
        sensor = AccessibilitySensor()
        page = MockPage("""\
- banner
  - text "something went wrong"
  - button "try again"
""")
        features = await sensor.sense(page)
        assert features.has_error_marker is True

    @pytest.mark.asyncio
    async def test_sense_detects_rate_limit_marker(self):
        sensor = AccessibilitySensor()
        page = MockPage("""\
- dialog
  - text "rate limit exceeded"
  - text "try again later"
""")
        features = await sensor.sense(page)
        assert features.has_rate_limit_marker is True

    @pytest.mark.asyncio
    async def test_sense_detects_streaming_marker(self):
        sensor = AccessibilitySensor()
        page = MockPage("""\
- button "stop generating"
""")
        features = await sensor.sense(page)
        assert features.has_streaming_marker is True

    @pytest.mark.asyncio
    async def test_sense_failure_decreases_confidence(self):
        sensor = AccessibilitySensor()
        page_fail = FailingMockPage()
        features_ok = await sensor.sense(MockPage())
        assert features_ok.accessibility_confidence > 0.5
        assert features_ok.extraction_success is True
        # Trigger failures
        for _ in range(5):
            features_bad = await sensor.sense(page_fail)
        assert features_bad.extraction_success is False
        assert features_bad.accessibility_confidence < 0.5

    @pytest.mark.asyncio
    async def test_consecutive_failures_degrade_reliability(self):
        sensor = AccessibilitySensor()
        page_fail = FailingMockPage()
        for _ in range(3):
            await sensor.sense(page_fail)
        stats = sensor.get_uptime_stats()
        assert stats["consecutive_failures"] >= 3
        assert stats["reliability_factor"] < 1.0

    @pytest.mark.asyncio
    async def test_snapshot_node_count_positive(self):
        sensor = AccessibilitySensor()
        page = MockPage()
        features = await sensor.sense(page)
        assert features.snapshot_node_count > 0

    @pytest.mark.asyncio
    async def test_reset_clears_stats(self):
        sensor = AccessibilitySensor()
        await sensor.sense(MockPage())
        sensor.reset()
        stats = sensor.get_uptime_stats()
        assert stats["total_calls"] == 0
        assert stats["consecutive_failures"] == 0
