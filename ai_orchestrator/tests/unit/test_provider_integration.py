"""Integration tests — all UI providers, capabilities, routing, mock pipeline."""

from __future__ import annotations

import pytest

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.adapters.chatgpt_ui import ChatGPTUIAdapter
from ai_orchestrator.adapters.deepseek_ui import DeepSeekUIAdapter
from ai_orchestrator.adapters.kimi_ui import KimiUIAdapter
from ai_orchestrator.adapters.local_llm import LocalLLMAdapter
from ai_orchestrator.adapters.minimax_ui import MiniMaxUIAdapter
from ai_orchestrator.adapters.qwen_ui import QwenUIAdapter
from ai_orchestrator.adapters.xiaomimimo_ui import XiaomiMiMoUIAdapter
from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter
from ai_orchestrator.models.account import Account, AccountState
from ai_orchestrator.models.capabilities import (
    PROVIDER_PROFILES,
    TaskRequirements,
)
from ai_orchestrator.orchestrator.lease_manager import LeaseManager
from ai_orchestrator.orchestrator.provider_router import ProviderRouter

EXPECTED_PROVIDER_COUNT = 8

CAPABILITY_NAMES = list(PROVIDER_PROFILES.keys())

ADAPTER_PROVIDER_NAMES = {
    "chatgpt_ui": "chatgpt_ui",
    "qwen_ui": "qwen_ui",
    "deepseek_ui": "deepseek_ui",
    "kimi_ui": "kimi_ui",
    "local_llm": "local_llm",
    "z_ai_ui": "z_ai_ui",
    "xiaomimimo_ui": "xiaomimimo_ui",
    "minimax_ui": "minimax_ui",
}


def _make_mock_adapter(name: str) -> ProviderAdapter:
    mapping = {
        "chatgpt_ui": ChatGPTUIAdapter,
        "qwen_ui": QwenUIAdapter,
        "deepseek_ui": DeepSeekUIAdapter,
        "kimi_ui": KimiUIAdapter,
        "local_llm": LocalLLMAdapter,
        "z_ai_ui": ZAIUIAdapter,
        "xiaomimimo_ui": XiaomiMiMoUIAdapter,
        "minimax_ui": MiniMaxUIAdapter,
    }
    cls = mapping[name]
    return cls(mock_mode=True)


# ═══════════════════════════════════════════════════════════
# All providers — mock-mode health check + send
# ═══════════════════════════════════════════════════════════

class TestAllProvidersMockMode:
    """Every adapter returns a valid response in mock mode."""

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_health_check(self, name):
        adapter = _make_mock_adapter(name)
        assert await adapter.health_check()

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_send_returns_valid_response(self, name):
        adapter = _make_mock_adapter(name)
        result = await adapter.send("Integration test prompt")
        assert isinstance(result, ProviderResponse)
        assert result.success
        assert result.content
        assert "Integration test prompt" in result.content

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_provider_name_matches(self, name):
        adapter = _make_mock_adapter(name)
        assert adapter.provider_name == ADAPTER_PROVIDER_NAMES[name]

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    def test_context_limit_positive(self, name):
        adapter = _make_mock_adapter(name)
        limit = adapter.get_context_limit()
        assert limit > 0, f"{name}: context_limit={limit}"

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_is_rate_limited_returns_bool(self, name):
        adapter = _make_mock_adapter(name)
        assert isinstance(await adapter.is_rate_limited(), bool)

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_refresh_session_returns_bool(self, name):
        adapter = _make_mock_adapter(name)
        assert await adapter.refresh_session()

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_protected_send_succeeds(self, name):
        adapter = _make_mock_adapter(name)
        result = await adapter.protected_send("Protected send test")
        assert result.success
        assert result.content

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_close_does_not_raise(self, name):
        adapter = _make_mock_adapter(name)
        await adapter.close()

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_safe_health_check(self, name):
        adapter = _make_mock_adapter(name)
        assert await adapter.safe_health_check()


# ═══════════════════════════════════════════════════════════
# Capability profiles
# ═══════════════════════════════════════════════════════════

class TestProviderProfiles:
    def test_all_providers_registered(self):
        assert len(PROVIDER_PROFILES) == EXPECTED_PROVIDER_COUNT

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    def test_profile_exists(self, name):
        assert name in PROVIDER_PROFILES

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    def test_profile_has_valid_context_limit(self, name):
        profile = PROVIDER_PROFILES[name]
        assert profile.context_limit > 0

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    def test_profile_has_capabilities(self, name):
        profile = PROVIDER_PROFILES[name]
        caps = profile.capabilities
        assert 0.0 <= caps.reasoning <= 1.0
        assert 0.0 <= caps.coding <= 1.0

    def test_transport_types_browser_and_local(self):
        transports = {p.transport for p in PROVIDER_PROFILES.values()}
        assert "BROWSER" in transports
        assert "LOCAL" in transports
        # All API providers removed — no API transport should remain
        assert "API" not in transports

    def test_deepseek_ui_has_1m_context(self):
        profile = PROVIDER_PROFILES["deepseek_ui"]
        assert profile.context_limit == 1_048_576

    def test_browser_providers_no_streaming_no_tools(self):
        for name, p in PROVIDER_PROFILES.items():
            if p.transport == "BROWSER":
                assert not p.supports_streaming, f"{name} should not support streaming"
                assert not p.supports_tools, f"{name} should not support tools"


# ═══════════════════════════════════════════════════════════
# Provider router integration
# ═══════════════════════════════════════════════════════════

class TestRouterWithAllProviders:
    def test_all_profiles_have_account_match(self):
        """Every profile's provider_name must match an adapter's provider_name."""
        router = ProviderRouter()
        requirements = TaskRequirements(context_length=1_000, requires_reasoning=True)

        for name, profile in PROVIDER_PROFILES.items():
            account = Account(
                id=f"{name}:test-01",
                provider=profile.provider_name,
                state=AccountState.IDLE,
                context_limit=profile.context_limit,
            )
            scored = router.score_account(account, requirements)
            assert scored.score > 0, f"{name}: score=0 (likely unknown_provider)"
            assert scored.reason != "unknown_provider"

    def test_rank_accounts_scores_descending(self):
        router = ProviderRouter()
        requirements = TaskRequirements(context_length=4_096, requires_reasoning=True)

        accounts = [
            Account(id=f"{n}:test", provider=ADAPTER_PROVIDER_NAMES[n], state=AccountState.IDLE,
                    context_limit=PROVIDER_PROFILES[n].context_limit)
            for n in CAPABILITY_NAMES
        ]
        ranked = router.rank_accounts(accounts, requirements)
        assert len(ranked) == EXPECTED_PROVIDER_COUNT
        for i in range(len(ranked) - 1):
            assert ranked[i].score >= ranked[i + 1].score

    def test_deepseek_ui_tops_reasoning(self):
        router = ProviderRouter()
        requirements = TaskRequirements(
            context_length=4_096,
            requires_reasoning=True,
            priority={"reasoning": 1.0, "coding": 0.0, "translation": 0.0, "multimodality": 0.0},
        )
        accounts = [
            Account(id=f"{n}:test", provider=ADAPTER_PROVIDER_NAMES[n], state=AccountState.IDLE,
                    context_limit=PROVIDER_PROFILES[n].context_limit)
            for n in CAPABILITY_NAMES
        ]
        ranked = router.rank_accounts(accounts, requirements)
        top = ranked[0]
        assert top.account.provider == "deepseek_ui"


# ═══════════════════════════════════════════════════════════
# Lease manager with all providers
# ═══════════════════════════════════════════════════════════

class TestLeaseManagerAllProviders:
    def test_register_all_providers(self):
        lm = LeaseManager()
        all_names = set(ADAPTER_PROVIDER_NAMES.values())
        accounts = [
            Account(id=f"{n}:test", provider=n, state=AccountState.IDLE,
                    context_limit=131_072)
            for n in all_names
        ]
        lm.register_accounts(accounts)
        assert len(lm.list_accounts()) == len(all_names)

    def test_lease_acquire_release_cycle(self):
        lm = LeaseManager()
        all_names = set(ADAPTER_PROVIDER_NAMES.values())
        accounts = [
            Account(id=f"{n}:test", provider=n, state=AccountState.IDLE,
                    context_limit=131_072)
            for n in all_names
        ]
        lm.register_accounts(accounts)

        for name in all_names:
            lease = lm.request_lease(
                task_id=f"task-{name}", agent_id="agent-1",
                preferred_provider=name,
            )
            assert lease is not None
            assert lease.account_id.startswith(name)

            account = lm.release_lease(lease.id)
            assert account is not None

    def test_list_accounts_by_provider(self):
        lm = LeaseManager()
        all_names = set(ADAPTER_PROVIDER_NAMES.values())
        accounts = [
            Account(id=f"{n}:test", provider=n, state=AccountState.IDLE,
                    context_limit=131_072)
            for n in all_names
        ]
        lm.register_accounts(accounts)

        for name in all_names:
            filtered = lm.list_accounts(provider=name)
            assert len(filtered) == 1
            assert filtered[0].provider == name


# ═══════════════════════════════════════════════════════════
# Full pipeline: adapter → response → validate
# ═══════════════════════════════════════════════════════════

class TestFullPipeline:
    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_pipeline_send_validate(self, name):
        adapter = _make_mock_adapter(name)
        result = await adapter.protected_send(
            "Write a Python function that adds two numbers"
        )
        assert isinstance(result, ProviderResponse)
        assert result.success

    @pytest.mark.parametrize("name", CAPABILITY_NAMES)
    async def test_pipeline_provider_response_shape(self, name):
        adapter = _make_mock_adapter(name)
        result = await adapter.send("Test prompt")
        assert isinstance(result.model, str)
        assert result.model != "unknown"
        assert result.latency_ms > 0
        assert result.is_valid


# ═══════════════════════════════════════════════════════════
# Browser adapter storage_state support
# ═══════════════════════════════════════════════════════════

class TestBrowserAdapterStorageState:
    def test_chatgpt_ui_accepts_storage_state(self):
        adapter = ChatGPTUIAdapter(mock_mode=True, storage_state={"cookies": [], "origins": []})
        assert adapter._storage_state == {"cookies": [], "origins": []}

    def test_deepseek_ui_accepts_storage_state(self):
        adapter = DeepSeekUIAdapter(mock_mode=True, storage_state={"cookies": [], "origins": []})
        assert adapter._storage_state == {"cookies": [], "origins": []}

    def test_zai_ui_accepts_storage_state(self):
        adapter = ZAIUIAdapter(mock_mode=True, storage_state={"cookies": [], "origins": []})
        assert adapter._storage_state == {"cookies": [], "origins": []}

    def test_xiaomimimo_ui_accepts_storage_state(self):
        adapter = XiaomiMiMoUIAdapter(mock_mode=True, storage_state={"cookies": [], "origins": []})
        assert adapter._storage_state == {"cookies": [], "origins": []}

    def test_minimax_ui_accepts_storage_state(self):
        adapter = MiniMaxUIAdapter(mock_mode=True, storage_state={"cookies": [], "origins": []})
        assert adapter._storage_state == {"cookies": [], "origins": []}

    def test_kimi_ui_accepts_storage_state(self):
        adapter = KimiUIAdapter(mock_mode=True, storage_state={"cookies": [], "origins": []})
        assert adapter._storage_state == {"cookies": [], "origins": []}

    def test_qwen_ui_accepts_storage_state(self):
        adapter = QwenUIAdapter(mock_mode=True, storage_state={"cookies": [], "origins": []})
        assert adapter._storage_state == {"cookies": [], "origins": []}

    def test_browser_adapters_accept_persistent_profile(self):
        adapters = [
            ChatGPTUIAdapter(mock_mode=True, persistent_profile="/tmp/cg"),
            DeepSeekUIAdapter(mock_mode=True, persistent_profile="/tmp/ds"),
            ZAIUIAdapter(mock_mode=True, persistent_profile="/tmp/zai"),
            XiaomiMiMoUIAdapter(mock_mode=True, persistent_profile="/tmp/xmm"),
            MiniMaxUIAdapter(mock_mode=True, persistent_profile="/tmp/mm"),
        ]
        for a in adapters:
            assert a._persistent_profile is not None
            assert a._mock_mode is True
