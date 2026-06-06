"""Kimi UI (browser) adapter — engine-driven for kimi.com."""

from __future__ import annotations

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter, SiteConfig


class KimiUIAdapter(EngineUIAdapter):
    provider_name = "kimi_ui"
    mock_content_prefix = "Kimi"
    mock_model = "kimi-k2.6"
    mock_context_limit = 131_072
    _site = SiteConfig(
        url="https://www.kimi.com",
        name="Kimi",
        title_keyword="Kimi",
    )
