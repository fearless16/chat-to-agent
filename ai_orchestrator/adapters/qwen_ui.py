"""Qwen UI (browser) adapter — Playwright-based for chat.qwen.ai."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse


class QwenUIAdapter(ProviderAdapter):
    """Qwen accessed via Playwright browser automation at chat.qwen.ai.

    Set ``mock_mode=False`` to launch a real browser.  Requires
    ``playwright`` to be installed.

    Parameters
    ----------
    headless:
        Run browser without UI.  Set ``False`` the first time you use
        ``persistent_profile`` so you can log in.
    mock_mode:
        When ``True`` (default) returns hardcoded responses — no browser.
    stealth:
        Apply anti-detection measures (viewport, user-agent, args).
    timeout_ms:
        Max time (ms) for page loads, selectors, and response polling.
    storage_state:
        Playwright ``auth.json`` file path or a dict with ``cookies``
        and ``origins`` keys.  **Note:** Qwen may validate sessions
        against browser fingerprint, so exported cookies may be
        rejected.  Prefer ``persistent_profile``.
    persistent_profile:
        Path to a Playwright persistent user-data directory.  The
        first time the adapter runs with ``headless=False`` the browser
        window opens so you can log in.  Subsequent runs reuse the
        saved session automatically.
    channel:
        Browser channel — ``"chromium"`` (default), ``"chrome"``, or
        ``"msedge"``.
    """

    provider_name = "qwen_ui"
    supports_streaming = False
    supports_tools = False

    def __init__(
        self,
        headless: bool = True,
        mock_mode: bool = True,
        stealth: bool = True,
        timeout_ms: int = 90_000,
        storage_state: Optional[str | Path | dict] = None,
        persistent_profile: Optional[str | Path] = None,
        channel: str = "chromium",
    ) -> None:
        super().__init__()
        self.headless = headless
        self._mock_mode = mock_mode
        self._stealth = stealth
        self._timeout_ms = timeout_ms
        self._storage_state = storage_state
        self._persistent_profile = persistent_profile
        self._channel = channel
        self._playwright = None
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
        return 131072

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
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    # ── mock path ────────────────────────────────────────────────────

    def _mock_send(
        self, prompt: str, context: Optional[list[dict]] = None
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"Qwen UI response to: {prompt[:50]}",
            model="qwen-max",
            latency_ms=2200.0,
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
                model="qwen-max",
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

        self._playwright = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
        ]

        if self._persistent_profile is not None:
            profile_path = Path(self._persistent_profile)
            profile_path.mkdir(parents=True, exist_ok=True)

            context_opts = {"channel": self._channel}
            if self._stealth:
                context_opts["viewport"] = {"width": 1280, "height": 720}
                context_opts["user_agent"] = (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                )

            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=self.headless,
                args=launch_args,
                **context_opts,
            )
            self._browser = None
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=launch_args,
                channel=self._channel,
            )

            context_opts = {}
            if self._stealth:
                context_opts["viewport"] = {"width": 1280, "height": 720}
                context_opts["user_agent"] = (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                )

            if self._storage_state is not None:
                context_opts["storage_state"] = self._storage_state

            self._context = await browser.new_context(**context_opts)
            self._browser = browser
            self._page = await self._context.new_page()

        await self._page.goto(
            "https://chat.qwen.ai",
            timeout=self._timeout_ms,
            wait_until="domcontentloaded",
        )

        await self._wait_for_app_ready(self._page)

        return self._page

    async def _wait_for_app_ready(self, page) -> None:
        for _ in range(30):
            title = await page.title()
            if "Qwen" in title or "通义千问" in title:
                return
            if title and title not in ("", "Just a moment..."):
                return
            await asyncio.sleep(1.5)

    async def _type_prompt(self, page, prompt: str) -> None:
        input_el = page.locator("textarea").first
        try:
            await input_el.wait_for(state="visible", timeout=10_000)
            await input_el.click()
            await input_el.fill(prompt)
        except Exception:
            pass

        await asyncio.sleep(0.5)

        send_selectors = [
            'button[type="submit"]',
            '[aria-label*="send" i]',
            '[aria-label*="Send" i]',
            '[data-testid="send-button"]',
        ]
        for sel in send_selectors:
            btn = page.locator(sel).first
            try:
                await btn.wait_for(state="visible", timeout=3_000)
                await btn.click()
                return
            except Exception:
                continue

        await page.keyboard.press("Enter")

    async def _read_response(self, page) -> str:
        msg_sel = '[class*="message"]'
        try:
            prev_count = await page.locator(msg_sel).count()
        except Exception:
            prev_count = 0

        for _ in range(45):
            await asyncio.sleep(2)
            try:
                curr_count = await page.locator(msg_sel).count()
            except Exception:
                curr_count = 0
            if curr_count > prev_count:
                await asyncio.sleep(2)
                try:
                    last = page.locator(msg_sel).last
                    text = await last.inner_text()
                    if text.strip():
                        return text.strip()
                except Exception:
                    pass
        return ""
