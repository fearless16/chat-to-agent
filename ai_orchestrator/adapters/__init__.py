"""Provider adapters — common interface wrapping browser UI and local LLM providers."""

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.adapters.chatgpt_ui import ChatGPTUIAdapter
from ai_orchestrator.adapters.cookie_to_storage_state import netscape_cookies_to_storage_state
from ai_orchestrator.adapters.deepseek_ui import DeepSeekUIAdapter
from ai_orchestrator.adapters.kimi_ui import KimiUIAdapter
from ai_orchestrator.adapters.local_llm import LocalLLMAdapter
from ai_orchestrator.adapters.minimax_ui import MiniMaxUIAdapter
from ai_orchestrator.adapters.qwen_ui import QwenUIAdapter
from ai_orchestrator.adapters.xiaomimimo_ui import XiaomiMiMoUIAdapter
from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter

__all__ = [
    "ChatGPTUIAdapter",
    "DeepSeekUIAdapter",
    "KimiUIAdapter",
    "LocalLLMAdapter",
    "MiniMaxUIAdapter",
    "ProviderAdapter",
    "ProviderResponse",
    "QwenUIAdapter",
    "XiaomiMiMoUIAdapter",
    "ZAIUIAdapter",
    "netscape_cookies_to_storage_state",
]
