"""ChatGPT UI (browser) adapter — Playwright-based for providers without public API."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse


class ChatGPTUIAdapter(ProviderAdapter):
    """ChatGPT accessed via Playwright browser automation.

    Set ``mock_mode=False`` to launch a real browser and interact with
    the ChatGPT web UI. Requires ``playwright`` to be installed.
    """

    provider_name = "chatgpt"
    supports_streaming = False
    supports_tools = False

    def __init__(
        self,
        headless: bool = True,
        mock_mode: bool = True,
        stealth: bool = True,
        timeout_ms: int = 60_000,
    ) -> None:
        super().__init__()
        self.headless = headless
        self._mock_mode = mock_mode
        self._stealth = stealth
        self._timeout_ms = timeout_ms
        self._browser = None
        self._context = None
        self._page = None

    async def send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        if self._mock_mode:
            return self._mock_send(prompt, context)
        return await self._real_send(prompt, context)

    async def health_check(self) -> bool:
        if self._mock_mode:
            return True
        try:
            page = await self._get_page()
            return page is not None
        except Exception:
            return False

    def get_context_limit(self) -> int:
        return 32768

    async def is_rate_limited(self) -> bool:
        return False

    async def refresh_session(self) -> bool:
        if not self._mock_mode and self._page is not None:
            try:
                await self._page.reload()
            except Exception:
                pass
        return True

    async def close(self) -> None:
        if self._page is not None:
            try:
                await self._page.close()
            except Exception:
                pass
            self._page = None
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

    # ── mock path ────────────────────────────────────────────────────

    def _mock_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"ChatGPT UI (browser) response to: {prompt[:50]}",
            model="gpt-4o",
            latency_ms=2500.0,
        )

    # ── real browser path ────────────────────────────────────────────

    async def _real_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        t0 = time.monotonic()
        try:
            page = await self._get_page()
            await self._type_prompt(page, prompt)
            content = await self._read_response(page)
            return ProviderResponse(
                content=content,
                model="gpt-4o",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ProviderResponse(
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    async def _get_page(self):
        if self._page is not None:
            return self._page

        from playwright.async_api import async_playwright

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=self.headless)
        context_opts = {}
        if self._stealth:
            context_opts["viewport"] = {"width": 1280, "height": 720}
        self._context = await browser.new_context(**context_opts)
        self._browser = browser
        self._page = await self._context.new_page()
        await self._page.goto("https://chat.openai.com/", timeout=self._timeout_ms)
        return self._page

    async def _type_prompt(self, page, prompt: str) -> None:
        await page.wait_for_selector(
            '[data-testid="chat-input"] textarea, textarea[placeholder*="Message"]',
            timeout=self._timeout_ms,
        )
        textarea = page.locator(
            '[data-testid="chat-input"] textarea, textarea[placeholder*="Message"]'
        ).first
        await textarea.fill(prompt)
        await textarea.press("Enter")

    async def _read_response(self, page) -> str:
        await page.wait_for_selector(
            '[data-message-author-role="assistant"], .prose',
            timeout=self._timeout_ms,
        )
        await asyncio.sleep(2)
        messages = page.locator(
            '[data-message-author-role="assistant"], .prose'
        )
        count = await messages.count()
        if count == 0:
            return ""
        return await messages.nth(count - 1).inner_text()
