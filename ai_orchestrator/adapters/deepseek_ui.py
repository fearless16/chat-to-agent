"""DeepSeek UI (browser) adapter — engine-driven for chat.deepseek.com."""

from __future__ import annotations

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter, SiteConfig


class DeepSeekUIAdapter(EngineUIAdapter):
    provider_name = "deepseek_ui"
    mock_content_prefix = "DeepSeek UI"
    mock_model = "deepseek-v4"
    mock_context_limit = 1_048_576
    _site = SiteConfig(
        url="https://chat.deepseek.com",
        name="DeepSeek",
        title_keyword="DeepSeek",
    )
