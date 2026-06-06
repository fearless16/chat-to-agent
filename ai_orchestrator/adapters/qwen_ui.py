"""Qwen UI (browser) adapter — engine-driven for chat.qwen.ai."""

from __future__ import annotations

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter, SiteConfig


class QwenUIAdapter(EngineUIAdapter):
    provider_name = "qwen_ui"
    mock_content_prefix = "Qwen UI"
    mock_model = "qwen3.7-max"
    mock_context_limit = 131_072
    _site = SiteConfig(
        url="https://chat.qwen.ai",
        name="Qwen",
        title_keyword="Qwen",
    )
