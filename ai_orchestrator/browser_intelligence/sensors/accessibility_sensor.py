"""AccessibilitySensor — semantic features from the accessibility tree."""

from __future__ import annotations

import time
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
    extraction_success: bool = False
    snapshot_node_count: int = 0


class AccessibilitySensor(BaseSensor):
    """Extracts semantic features from the accessibility tree.

    Uses page.accessibility.snapshot() — compact, semantic, stable.
    """

    def __init__(self):
        self._a11y = AccessibilityRuntime()
        self._total_calls = 0
        self._successful_calls = 0
        self._consecutive_failures = 0
        self._last_success_time: float = 0.0
        self._last_failure_time: float = 0.0

    async def sense(self, page) -> AccessibilityFeatures:
        self._total_calls += 1
        features = AccessibilityFeatures()
        try:
            snap = await self._a11y.snapshot(page)
            all_nodes = snap.all_nodes()
            
            if not all_nodes:
                raise ValueError("Empty accessibility tree")

            self._record_success()

            features.text_input_count = len(snap.text_inputs())
            features.button_count = len(snap.buttons())
            features.article_count = sum(
                1 for n in all_nodes if n.role == "article"
            )
            features.semantic_tree_breadth = len(all_nodes)
            features.semantic_tree_depth = self._compute_depth(snap.root)
            features.snapshot_node_count = len(all_nodes)

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
            
            features.extraction_success = True
            features.accessibility_confidence = self._compute_confidence(snap)
        except Exception as e:
            features.accessibility_confidence = self._compute_failure_confidence()
            features.extraction_success = False
            self._record_failure()
        return features

    def _record_success(self) -> None:
        self._successful_calls += 1
        self._consecutive_failures = 0
        self._last_success_time = time.monotonic()

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()

    def _compute_confidence(self, snap) -> float:
        all_nodes = snap.all_nodes()
        if not all_nodes:
            return 0.0
        named = sum(1 for n in all_nodes if n.name)
        base_confidence = min(named / max(len(all_nodes), 1), 1.0)
        return base_confidence * self._reliability_factor()

    def _compute_failure_confidence(self) -> float:
        return self._reliability_factor() * 0.1

    def _reliability_factor(self) -> float:
        if self._total_calls == 0:
            return 1.0
        
        success_rate = self._successful_calls / self._total_calls
        
        if self._consecutive_failures >= 5:
            return max(0.05, success_rate * 0.1)
        
        if self._consecutive_failures >= 3:
            return max(0.1, success_rate * 0.3)
        
        now = time.monotonic()
        if self._last_success_time > 0 and (now - self._last_success_time) > 60:
            return max(0.2, success_rate * 0.5)
        
        return success_rate

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

    def get_uptime_stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "consecutive_failures": self._consecutive_failures,
            "success_rate": self._successful_calls / max(self._total_calls, 1),
            "reliability_factor": self._reliability_factor(),
        }

    def reset(self) -> None:
        self._total_calls = 0
        self._successful_calls = 0
        self._consecutive_failures = 0
        self._last_success_time = 0.0
        self._last_failure_time = 0.0
