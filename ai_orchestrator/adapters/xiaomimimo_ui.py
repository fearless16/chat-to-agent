"""XiaomiMiMo UI (browser) adapter — engine-driven for aistudio.xiaomimimo.com."""

from __future__ import annotations

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter, SiteConfig


class XiaomiMiMoUIAdapter(EngineUIAdapter):
    provider_name = "xiaomimimo_ui"
    mock_content_prefix = "XiaomiMiMo"
    mock_model = "mimo-v2.5-pro"
    mock_context_limit = 200_000
    _site = SiteConfig(
        url="https://aistudio.xiaomimimo.com/#/c",
        name="XiaomiMiMo",
        title_keyword="XiaomiMiMo",
    )
