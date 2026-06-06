"""MiniMax UI (browser) adapter — engine-driven for agent.minimax.io."""

from __future__ import annotations

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter, SiteConfig


class MiniMaxUIAdapter(EngineUIAdapter):
    provider_name = "minimax_ui"
    mock_content_prefix = "MiniMax"
    mock_model = "minimax-m3"
    mock_context_limit = 1_048_576
    _site = SiteConfig(
        url="https://agent.minimax.io",
        name="MiniMax",
        title_keyword="MiniMax",
    )
