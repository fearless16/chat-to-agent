"""Z.ai UI (browser) adapter — engine-driven for chat.z.ai."""

from __future__ import annotations

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter, SiteConfig


class ZAIUIAdapter(EngineUIAdapter):
    provider_name = "z_ai_ui"
    mock_content_prefix = "Z.ai"
    mock_model = "glm-5.1"
    mock_context_limit = 131_072
    _site = SiteConfig(
        url="https://chat.z.ai",
        name="Z.ai",
        title_keyword="Z.ai",
    )
