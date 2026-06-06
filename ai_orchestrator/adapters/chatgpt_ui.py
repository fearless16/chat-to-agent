"""ChatGPT UI (browser) adapter — engine-driven for chatgpt.com."""

from __future__ import annotations

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter, SiteConfig


class ChatGPTUIAdapter(EngineUIAdapter):
    provider_name = "chatgpt_ui"
    mock_content_prefix = "ChatGPT UI"
    mock_model = "gpt-5.5"
    mock_context_limit = 131_072
    _site = SiteConfig(
        url="https://chatgpt.com",
        name="ChatGPT",
        title_keyword="ChatGPT",
    )
