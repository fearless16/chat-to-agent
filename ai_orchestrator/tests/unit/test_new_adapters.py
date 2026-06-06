"""Tests for new provider adapters (DeepSeek, Z.ai, XiaomiMiMo, MiniMax, Kimi)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ai_orchestrator.adapters.base import ProviderResponse
from ai_orchestrator.adapters.cookie_to_storage_state import (
    netscape_cookies_to_storage_state,
)
from ai_orchestrator.adapters.deepseek_ui import DeepSeekUIAdapter
from ai_orchestrator.adapters.kimi_ui import KimiUIAdapter
from ai_orchestrator.adapters.minimax_ui import MiniMaxUIAdapter
from ai_orchestrator.adapters.xiaomimimo_ui import XiaomiMiMoUIAdapter
from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter


class TestCookieToStorageState:
    def _profile_dir(self):
        return Path(__file__).parent.parent.parent.parent / "profiles"

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Just a comment\n\n")
            cookie_file = f.name

        try:
            state = netscape_cookies_to_storage_state(cookie_file)
            assert state["cookies"] == []
            assert state["origins"] == []
        finally:
            Path(cookie_file).unlink()

    def test_single_cookie(self):
        content = (
            ".example.com\tTRUE\t/\tTRUE\t1781864091\tsession\ttest-token\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            cookie_file = f.name

        try:
            state = netscape_cookies_to_storage_state(cookie_file)
            assert len(state["cookies"]) == 1
            cookie = state["cookies"][0]
            assert cookie["name"] == "session"
            assert cookie["value"] == "test-token"
            assert cookie["domain"] == "example.com"
            assert cookie["path"] == "/"
            assert cookie["secure"] is True
            assert cookie["httpOnly"] is False
            assert cookie["sameSite"] == "Lax"
            assert cookie["expires"] == 1781864091.0
        finally:
            Path(cookie_file).unlink()

    def test_multiple_cookies(self):
        content = (
            ".example.com\tTRUE\t/\tFALSE\t0\tsession\tabc\n"
            ".example.com\tTRUE\t/\tTRUE\t1781864091\ttoken\txyz\n"
            "sub.example.com\tFALSE\t/app\tFALSE\t2000000000\tuser\t123\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            cookie_file = f.name

        try:
            state = netscape_cookies_to_storage_state(cookie_file)
            assert len(state["cookies"]) == 3
            names = {c["name"] for c in state["cookies"]}
            assert names == {"session", "token", "user"}
        finally:
            Path(cookie_file).unlink()

    def test_duplicates_removed(self):
        content = (
            ".example.com\tTRUE\t/\tFALSE\t0\tsession\tfirst\n"
            ".example.com\tTRUE\t/\tFALSE\t0\tsession\tsecond\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            cookie_file = f.name

        try:
            state = netscape_cookies_to_storage_state(cookie_file)
            assert len(state["cookies"]) == 1
            assert state["cookies"][0]["value"] == "first"
        finally:
            Path(cookie_file).unlink()

    def test_domain_override(self):
        content = ".old.com\tTRUE\t/\tFALSE\t0\tsession\tabc\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            cookie_file = f.name

        try:
            state = netscape_cookies_to_storage_state(
                cookie_file, domain_override="new.com"
            )
            assert state["cookies"][0]["domain"] == "new.com"
        finally:
            Path(cookie_file).unlink()

    def test_quoted_values_stripped(self):
        content = '.example.com\tTRUE\t/\tFALSE\t0\ttoken\t"quoted-value"\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            cookie_file = f.name

        try:
            state = netscape_cookies_to_storage_state(cookie_file)
            assert state["cookies"][0]["value"] == "quoted-value"
        finally:
            Path(cookie_file).unlink()

    def test_real_deepseek_cookies(self):
        profile_dir = Path(__file__).parent.parent.parent.parent / "profiles"
        cookie_file = profile_dir / "deepseek_cookies.txt"
        if not cookie_file.exists():
            pytest.skip("deepseek_cookies.txt not found")
        state = netscape_cookies_to_storage_state(cookie_file)
        names = {c["name"] for c in state["cookies"]}
        assert "ds_session_id" in names
        assert all(c["domain"] in ("chat.deepseek.com", "deepseek.com") for c in state["cookies"])

    def test_real_zai_cookies(self):
        cookie_file = self._profile_dir() / "zai_cookies.txt"
        if not cookie_file.exists():
            pytest.skip("zai_cookies.txt not found")
        state = netscape_cookies_to_storage_state(cookie_file)
        names = {c["name"] for c in state["cookies"]}
        assert "token" in names
        assert "oauth_id_token" in names

    def test_real_xiaomimimo_cookies(self):
        cookie_file = self._profile_dir() / "xiaomimimo_cookies.txt"
        if not cookie_file.exists():
            pytest.skip("xiaomimimo_cookies.txt not found")
        state = netscape_cookies_to_storage_state(cookie_file)
        names = {c["name"] for c in state["cookies"]}
        assert "serviceToken" in names
        assert "userId" in names

    def test_real_minimax_cookies(self):
        cookie_file = self._profile_dir() / "minimax_cookies.txt"
        if not cookie_file.exists():
            pytest.skip("minimax_cookies.txt not found")
        state = netscape_cookies_to_storage_state(cookie_file)
        names = {c["name"] for c in state["cookies"]}
        assert "_token" in names
        assert "minimax_group_id_v2" in names

    def test_real_kimi_cookies(self):
        cookie_file = self._profile_dir() / "kimi_cookies.txt"
        if not cookie_file.exists():
            pytest.skip("kimi_cookies.txt not found")
        state = netscape_cookies_to_storage_state(cookie_file)
        names = {c["name"] for c in state["cookies"]}
        assert "kimi-auth" in names


class TestNewUIAdapterMockModes:
    """Verify all new adapters produce mock responses when mock_mode=True."""

    def _make_adapter(self, adapter_cls):
        return adapter_cls(mock_mode=True)

    def test_deepseek_ui_adapter_creation(self):
        adapter = DeepSeekUIAdapter(mock_mode=True)
        assert adapter.provider_name == "deepseek_ui"
        assert not adapter.supports_streaming

    def test_deepseek_ui_mock_send(self):
        adapter = self._make_adapter(DeepSeekUIAdapter)
        result = adapter._mock_send("Hello")
        assert isinstance(result, ProviderResponse)
        assert result.success
        assert "Hello" in result.content
        assert result.model == "deepseek-v4"

    def test_deepseek_ui_context_limit(self):
        adapter = DeepSeekUIAdapter(mock_mode=True)
        assert adapter.get_context_limit() == 1048576

    async def test_deepseek_ui_health_check(self):
        adapter = DeepSeekUIAdapter(mock_mode=True)
        assert await adapter.health_check()

    async def test_deepseek_ui_send_mock(self):
        adapter = DeepSeekUIAdapter(mock_mode=True)
        result = await adapter.send("Test prompt")
        assert result.success

    # ── Z.ai ──

    def test_zai_ui_adapter_creation(self):
        adapter = ZAIUIAdapter(mock_mode=True)
        assert adapter.provider_name == "z_ai_ui"

    def test_zai_ui_mock_send(self):
        adapter = ZAIUIAdapter(mock_mode=True)
        result = adapter._mock_send("Hello")
        assert isinstance(result, ProviderResponse)
        assert "Hello" in result.content

    async def test_zai_ui_health_check(self):
        adapter = ZAIUIAdapter(mock_mode=True)
        assert await adapter.health_check()

    async def test_zai_ui_send_mock(self):
        adapter = ZAIUIAdapter(mock_mode=True)
        result = await adapter.send("Test prompt")
        assert result.success

    # ── XiaomiMiMo ──

    def test_xiaomimimo_ui_adapter_creation(self):
        adapter = XiaomiMiMoUIAdapter(mock_mode=True)
        assert adapter.provider_name == "xiaomimimo_ui"

    async def test_xiaomimimo_ui_send_mock(self):
        adapter = XiaomiMiMoUIAdapter(mock_mode=True)
        result = await adapter.send("Test prompt")
        assert result.success

    async def test_xiaomimimo_ui_health_check(self):
        adapter = XiaomiMiMoUIAdapter(mock_mode=True)
        assert await adapter.health_check()

    # ── MiniMax ──

    def test_minimax_ui_adapter_creation(self):
        adapter = MiniMaxUIAdapter(mock_mode=True)
        assert adapter.provider_name == "minimax_ui"

    async def test_minimax_ui_send_mock(self):
        adapter = MiniMaxUIAdapter(mock_mode=True)
        result = await adapter.send("Test prompt")
        assert result.success

    async def test_minimax_ui_health_check(self):
        adapter = MiniMaxUIAdapter(mock_mode=True)
        assert await adapter.health_check()

    # ── Kimi ──

    def test_kimi_ui_adapter_creation(self):
        adapter = KimiUIAdapter(mock_mode=True)
        assert adapter.provider_name == "kimi_ui"

    async def test_kimi_ui_send_mock(self):
        adapter = KimiUIAdapter(mock_mode=True)
        result = await adapter.send("Test prompt")
        assert result.success

    async def test_kimi_ui_health_check(self):
        adapter = KimiUIAdapter(mock_mode=True)
        assert await adapter.health_check()
