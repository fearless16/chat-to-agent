"""AccessibilitySensor — semantic features from the accessibility tree."""

from __future__ import annotations

from dataclasses import dataclass

from ai_orchestrator.browser.accessibility import AccessibilityRuntime
from ai_orchestrator.browser_intelligence.sensors.base import BaseSensor

THINKING_MARKERS = (
    "thinking", "reasoning", "chain of thought",
    "thought process",
)
ERROR_MARKERS = (
    "something went wrong", "error", "try again",
    "unexpected error", "failed",
)
RATE_LIMIT_MARKERS = (
    "rate limit", "too many", "slow down", "try again later",
    "usage limit", "quota", "limit reached",
)
STREAMING_MARKERS = (
    "stop generating", "stop", "pause", "cancel",
)


@dataclass
class AccessibilityFeatures:
    text_input_count: int = 0
    button_count: int = 0
    article_count: int = 0
    has_thinking_marker: bool = False
    has_error_marker: bool = False
    has_rate_limit_marker: bool = False
    has_streaming_marker: bool = False
    semantic_tree_depth: int = 0
    semantic_tree_breadth: int = 0
    accessibility_confidence: float = 1.0


class AccessibilitySensor(BaseSensor):
    """Extracts semantic features from the accessibility tree.

    Uses page.aria_snapshot() — compact, semantic, stable.
    """

    def __init__(self):
        self._a11y = AccessibilityRuntime()

    async def sense(self, page) -> AccessibilityFeatures:
        features = AccessibilityFeatures()
        try:
            snap = await self._a11y.snapshot(page)
            all_nodes = snap.all_nodes()
            features.text_input_count = len(snap.text_inputs())
            features.button_count = len(snap.buttons())
            features.article_count = sum(
                1 for n in all_nodes if n.role == "article"
            )
            features.semantic_tree_breadth = len(all_nodes)
            features.semantic_tree_depth = self._compute_depth(snap.root)

            aria_text = self._collect_text(snap.root).lower()
            features.has_thinking_marker = any(
                m in aria_text for m in THINKING_MARKERS
            )
            features.has_error_marker = any(
                m in aria_text for m in ERROR_MARKERS
            )
            features.has_rate_limit_marker = any(
                m in aria_text for m in RATE_LIMIT_MARKERS
            )
            features.has_streaming_marker = any(
                m in aria_text for m in STREAMING_MARKERS
            )
            features.accessibility_confidence = self._compute_confidence(snap)
        except Exception:
            features.accessibility_confidence = 0.0
        return features

    def _compute_depth(self, node, current_depth: int = 0) -> int:
        if node is None:
            return 0
        max_child = 0
        for child in node.children:
            max_child = max(max_child, self._compute_depth(child, current_depth + 1))
        return max(current_depth, max_child)

    def _collect_text(self, node) -> str:
        if node is None:
            return ""
        parts = [node.name, node.value or "", node.description or ""]
        text = " ".join(p for p in parts if p)
        for child in node.children:
            text += " " + self._collect_text(child)
        return text

    def _compute_confidence(self, snap) -> float:
        all_nodes = snap.all_nodes()
        if not all_nodes:
            return 0.0
        named = sum(1 for n in all_nodes if n.name)
        return min(named / max(len(all_nodes), 1), 1.0)
