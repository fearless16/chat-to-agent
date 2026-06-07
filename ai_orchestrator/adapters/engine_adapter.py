"""EngineUIAdapter — browser adapter driven by BrowserIntelligenceEngine.

Replaces hardcoded selectors and `asyncio.sleep()` polling loops
with CDP-backed probabilistic state detection. The engine drives
the interaction flow: it tells us *when* to type, *when* to click
send, *when* a response is being generated, and *when* it's complete.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.adapters.cookie_validator import (
    check_cloudflare_challenge,
    check_post_navigation_auth,
    validate_storage_state,
)
from ai_orchestrator.adapters.errors import (
    AuthenticationError,
    CloudflareBlockError,
    ResponseExtractionError,
)
from ai_orchestrator.adapters.popup_manager import handle_popups
from ai_orchestrator.adapters.recovery_engine import (
    attempt_recovery,
    recovery_engine,
    CooldownError,
    MAX_RETRIES,
)
from ai_orchestrator.adapters.auto_cookie_update import save_cookies_from_context
from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

ENGINE_TICK_INTERVAL: float = 1.0
MAX_READY_TICKS: int = 60
MAX_RESPONSE_TICKS: int = 90

INPUT_SELECTORS = [
    'textarea',
    '[contenteditable="true"]',
    '[role="textbox"]',
    '[class*="prompt-textarea"]',
    '[class*="chat-input"]',
    '[class*="input-area"]',
    '[class*="composer"] textarea',
    '#prompt-textarea',
]

SEND_SELECTORS = [
    'button[type="submit"]',
    '[aria-label*="send" i]',
    '[data-testid="send-button"]',
    'button:has(svg)',
    '[class*="send-button"]',
    'button[aria-label*="Send"]',
    '#send-button',
]

# Status / role tokens that pollute response text when the SSE parser
# grabs them from metadata fields.  Qwen in particular emits
# {"type": "assistant", "status": "typing"} chunks whose values end up
# concatenated into the response: "assistanttypingassistantfinished".
_STATUS_LABEL_RE = re.compile(
    r"^(assistant|user|system|typing|thinking|finished|generating|pending|"
    r"streaming|complete|stopped|queued)$",
    re.IGNORECASE,
)

# Regex for runs of concatenated status labels that sneak through.
_STATUS_NOISE_RE = re.compile(
    r"(?:assistant|typing|thinking|finished|generating|streaming|complete|pending)"
    r"(?:assistant|typing|thinking|finished|generating|streaming|complete|pending)+",
    re.IGNORECASE,
)

RESPONSE_EXTRACT_SCRIPT = """() => {
    // Action-button labels that we never want to confuse with response text.
    const ACTION_RE = /^(copy|copied|regenerate|retry|edit|share|like|dislike|thumbs\\s*up|thumbs\\s*down|good\\s*response|bad\\s*response|speak|read\\s*aloud|listen)$/i;

    // Walk a node and concatenate text, but skip any element that is or
    // contains only action-button labels (copy / regenerate / like / etc.).
    function cleanText(node) {
        if (!node) return '';
        if (node.nodeType === Node.TEXT_NODE) {
            return node.nodeValue || '';
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return '';
        const tag = (node.tagName || '').toLowerCase();
        if (tag === 'script' || tag === 'style' || tag === 'svg' || tag === 'noscript') return '';
        if (tag === 'button' || node.getAttribute('role') === 'button') return '';
        const aria = (node.getAttribute('aria-label') || '').trim();
        const title = (node.getAttribute('title') || '').trim();
        if (aria && ACTION_RE.test(aria)) return '';
        if (title && ACTION_RE.test(title)) return '';
        if (tag === 'img' || tag === 'video' || tag === 'audio' || tag === 'iframe') return '';
        let out = '';
        for (const child of node.childNodes) {
            out += cleanText(child);
        }
        // Collapse runs of whitespace introduced by stripped children.
        return out.replace(/[ \\t]+\\n/g, '\\n').replace(/\\n{3,}/g, '\\n\\n');
    }

    // Pick the LAST assistant message container.  Try well-known shapes
    // first so we don't fall into a chat-history sidebar or copy-button
    // toolbar that happens to match a looser selector.
    const candidates = [
        () => document.querySelectorAll('[data-message-author-role="assistant"]'),
        () => document.querySelectorAll('[data-role="assistant"]'),
        () => document.querySelectorAll('article[data-author="assistant"]'),
        () => document.querySelectorAll('.assistant-message, .message-assistant, [class*="assistant-message"]'),
        () => document.querySelectorAll('.prose, .markdown-body, [class*="markdown-body"], [class*="MessageContent"]'),
        () => document.querySelectorAll('article'),
        () => document.querySelectorAll('[class*="message"]'),
    ];
    for (const factory of candidates) {
        const nodes = factory();
        if (nodes && nodes.length) {
            const visibleNodes = Array.from(nodes).filter(n => {
                return n.offsetWidth > 0 || n.offsetHeight > 0 || n.getClientRects().length > 0;
            });
            if (visibleNodes.length) {
                const last = visibleNodes[visibleNodes.length - 1];
                const t = cleanText(last).trim();
                if (t.length > 1) return t;
            }
        }
    }
    return '';
}"""

FETCH_INTERCEPT_SCRIPT = """(() => {
    if (window.__engine_adapter_hook__) return;
    window.__engine_adapter_hook__ = true;
    window.__engine_response_bodies__ = [];

    function _storeBody(url, text, transport) {
        if (text && text.length > 10) {
            window.__engine_response_bodies__.push({
                url: (url || '').substring(0, 200),
                text: text.substring(0, 8000),
                time: Date.now(),
                transport: transport || 'fetch',
            });
            if (window.__engine_response_bodies__.length > 20) {
                window.__engine_response_bodies__.shift();
            }
        }
    }

    function _readStreamBody(response, url) {
        if (!response.body || !response.body.getReader) return null;
        var reader = response.body.getReader();
        var decoder = new TextDecoder('utf-8');
        var chunks = [];
        var done = false;
        function pump() {
            return reader.read().then(function(result) {
                if (result.value) {
                    chunks.push(decoder.decode(result.value, {stream: true}));
                }
                if (result.done) {
                    done = true;
                    chunks.push(decoder.decode());
                    _storeBody(url, chunks.join(''), 'fetch_stream');
                    return;
                }
                return pump();
            }).catch(function() {});
        }
        pump();
        return null;
    }

    var origFetch = window.fetch;
    window.fetch = function(...args) {
        return origFetch.apply(this, args).then(function(resp) {
            var url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) || '';
            var ct = (resp.headers && resp.headers.get) ? (resp.headers.get('content-type') || '') : '';
            if (ct.indexOf('text/event-stream') !== -1) {
                var clone = resp.clone();
                _readStreamBody(clone, url);
                return resp;
            }
            var clone = resp.clone();
            clone.text().then(function(text) {
                _storeBody(url, text, 'fetch');
            }).catch(function() {});
            return resp;
        });
    };

    var origXHROpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url) {
        this.__engine_xhr_url = url;
        return origXHROpen.apply(this, arguments);
    };

    var origXHRSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function(body) {
        this.addEventListener('load', function() {
            if (this.responseText && this.responseText.length > 10) {
                window.__engine_response_bodies__.push({
                    url: (this.__engine_xhr_url || '').substring(0, 200),
                    text: this.responseText.substring(0, 8000),
                    time: Date.now(),
                });
                if (window.__engine_response_bodies__.length > 20) {
                    window.__engine_response_bodies__.shift();
                }
            }
        });
        return origXHRSend.apply(this, arguments);
    };

    if (typeof EventSource !== 'undefined') {
        var OrigEventSource = EventSource;
        window.EventSource = function(url, config) {
            var es = new OrigEventSource(url, config);
            var chunks = [];
            es.addEventListener('message', function(e) {
                if (e.data) chunks.push(e.data);
            });
            es.addEventListener('error', function() {
                if (chunks.length > 0) {
                    var text = chunks.join('');
                    if (text.length > 10 && text !== '[DONE]') {
                        window.__engine_response_bodies__.push({
                            url: (typeof url === 'string' ? url : '').substring(0, 200),
                            text: text.substring(0, 8000),
                            time: Date.now(),
                            transport: 'sse',
                        });
                    }
                }
            });
            var origClose = es.close.bind(es);
            es.close = function() {
                if (chunks.length > 0) {
                    var text = chunks.join('');
                    if (text.length > 10 && text !== '[DONE]') {
                        window.__engine_response_bodies__.push({
                            url: (typeof url === 'string' ? url : '').substring(0, 200),
                            text: text.substring(0, 8000),
                            time: Date.now(),
                            transport: 'sse',
                        });
                    }
                }
                return origClose();
            };
            return es;
        };
        window.EventSource.prototype = OrigEventSource.prototype;
        window.EventSource.CONNECTING = OrigEventSource.CONNECTING;
        window.EventSource.OPEN = OrigEventSource.OPEN;
        window.EventSource.CLOSED = OrigEventSource.CLOSED;
    }
})()"""

GET_INTERCEPTED_BODIES_SCRIPT = """() => {
    return (window.__engine_response_bodies__ || []).slice();
}"""

CLEAR_INTERCEPTED_BODIES_SCRIPT = """() => {
    window.__engine_response_bodies__ = [];
}"""


@dataclass
class SiteConfig:
    url: str
    name: str
    title_keyword: str


class EngineUIAdapter(ProviderAdapter):
    """Browser adapter driven by the Browser Intelligence Engine.

    Subclasses need only define ``_site`` and (optionally)
    ``mock_content_prefix`` / ``mock_model`` — everything else is
    handled by the engine's CDP-backed state machine.

    The flow:
      Navigate → Attach CDP → Tick until READY → Type prompt →
      Tick until GENERATING → Extract response → Return result.

    When the provider's SSE stream carries reasoning/thinking content
    (e.g. DeepSeek-R1 ``reasoning_content`` chunks, Z.ai ``phase:thinking``
    chunks, ChatGPT o-model reasoning), the adapter separates it into
    ``ProviderResponse.reasoning_content``.
    """

    supports_streaming = False
    supports_tools = False
    _site: SiteConfig | None = None
    mock_content_prefix: str = ""
    mock_model: str = ""
    mock_context_limit: int = 131_072
    _shared_playwright = None
    _shared_browser = None

    def __init__(
        self,
        headless: bool = True,
        mock_mode: bool = True,
        stealth: bool = True,
        timeout_ms: int = 120_000,
        storage_state: str | Path | dict | None = None,
        persistent_profile: str | Path | None = None,
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
        self._engine: BrowserIntelligenceEngine | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self, prompt: str, context: list[dict] | None = None
    ) -> ProviderResponse:
        if self._mock_mode:
            return self._mock_send(prompt)

        # Cooldown check (Layer 7: Provider Cooldown)
        try:
            recovery_engine.check_cooldown(self.provider_name)
        except CooldownError as exc:
            return ProviderResponse(
                success=False,
                error=f"CooldownError: {exc}",
                model=self.provider_name.replace("_", "-"),
            )

        # Retry loop with escalating recovery (Layer 7)
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            result = await self._real_send(prompt)
            if result.success:
                recovery_engine.record_success(self.provider_name)
                return result

            # Parse the error to decide recovery
            error_str = result.error or ""
            if "AuthenticationError" in error_str:
                last_error = AuthenticationError(self.provider_name, error_str)
            elif "CloudflareBlockError" in error_str:
                last_error = CloudflareBlockError(self.provider_name, error_str)
            else:
                last_error = RuntimeError(error_str)

            if attempt >= MAX_RETRIES:
                break

            # Attempt recovery
            action = await attempt_recovery(self, last_error, attempt)
            if action is None:
                log.warning(
                    "[%s] No recovery possible (attempt %d/%d)",
                    self.provider_name, attempt, MAX_RETRIES,
                )
                break

            log.info(
                "[%s] Recovery action: %s (attempt %d/%d)",
                self.provider_name, action.name, attempt, MAX_RETRIES,
            )

        # All retries exhausted — activate cooldown
        recovery_engine.record_failure(
            self.provider_name,
            str(last_error) if last_error else "unknown",
        )
        return result

    async def health_check(self) -> bool:
        try:
            await self._get_page()
            return self._page is not None
        except Exception:
            return False

    def get_context_limit(self) -> int:
        return self.mock_context_limit

    async def is_rate_limited(self) -> bool:
        return False

    async def refresh_session(self) -> bool:
        if not self._mock_mode and self._page is not None:
            try:
                await self._page.reload()
                await asyncio.sleep(3)
                return True
            except Exception:
                pass
        return False

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._page is not None:
                await self._page.close()
                self._page = None
        with contextlib.suppress(Exception):
            if self._context is not None:
                await self._context.close()
                self._context = None
        with contextlib.suppress(Exception):
            if self._browser is not None:
                await self._browser.close()
                self._browser = None
        with contextlib.suppress(Exception):
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None

    # ------------------------------------------------------------------
    # Mock path
    # ------------------------------------------------------------------

    def _mock_send(self, prompt: str) -> ProviderResponse:
        import time as _time
        _time.sleep(0.05)
        prefix = self.mock_content_prefix or self.provider_name
        content = f"{prefix} mock response to: {prompt}"
        return ProviderResponse(
            content=content,
            model=self.mock_model,
            latency_ms=50.0,
        )

    # ------------------------------------------------------------------
    # Fan-out (shared browser)
    # ------------------------------------------------------------------

    @classmethod
    async def fan_out(
        cls,
        adapters: list[ProviderAdapter],
        prompt: str,
        context: list[dict] | None = None,
        timeout: float | None = None,
        return_when: str = "ALL_COMPLETED",
    ) -> list[ProviderResponse]:
        """Send *prompt* to multiple adapters sharing one browser instance."""
        if not adapters:
            return []

        # Reuse a single Playwright + browser across all adapters.
        from playwright.async_api import async_playwright

        if cls._shared_playwright is None:
            cls._shared_playwright = await async_playwright().start()

        if cls._shared_browser is None:
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ]
            channel = getattr(adapters[0], "_channel", "chromium")
            cls._shared_browser = await cls._shared_playwright.chromium.launch(
                headless=getattr(adapters[0], "headless", True),
                args=launch_args,
                channel=channel,
            )

        async def _run(adapter: ProviderAdapter) -> ProviderResponse:
            try:
                return await adapter.send(prompt, context)
            except Exception as exc:
                return ProviderResponse(
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    model=adapter.provider_name,
                )
            finally:
                with contextlib.suppress(Exception):
                    await adapter.close()

        try:
            task_map: dict[asyncio.Task, int] = {}
            tasks: list[asyncio.Task] = []
            for i, a in enumerate(adapters):
                t = asyncio.create_task(_run(a), name=f"send:{a.provider_name}")
                tasks.append(t)
                task_map[t] = i

            results: list[ProviderResponse | None] = [None] * len(tasks)

            if return_when == "FIRST_COMPLETED":
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout)
                for d in done:
                    results[task_map[d]] = d.result()
                for p in pending:
                    p.cancel()
                for p in pending:
                    with contextlib.suppress(BaseException):
                        results[task_map[p]] = p.result()
            elif return_when == "FIRST_EXCEPTION":
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION, timeout=timeout)
                for d in done:
                    idx = task_map[d]
                    exc = d.exception()
                    if exc is not None:
                        results[idx] = ProviderResponse(
                            success=False, error=str(exc), model=adapters[idx].provider_name,
                        )
                    else:
                        results[idx] = d.result()
                for p in pending:
                    p.cancel()
                for p in pending:
                    with contextlib.suppress(BaseException):
                        results[task_map[p]] = p.result()
            else:
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                for i, outcome in enumerate(outcomes):
                    results[i] = outcome if isinstance(outcome, ProviderResponse) else ProviderResponse(
                        success=False, error=str(outcome), model=adapters[i].provider_name
                    )

            return [r for r in results if r is not None]

        finally:
            # Clean up shared browser resources
            if cls._shared_browser is not None:
                with contextlib.suppress(Exception):
                    await cls._shared_browser.close()
                cls._shared_browser = None
            if cls._shared_playwright is not None:
                with contextlib.suppress(Exception):
                    await cls._shared_playwright.stop()
                cls._shared_playwright = None

    # ── real browser path (engine-driven) ──────────────────────────

    async def _real_send(self, prompt: str) -> ProviderResponse:
        if self._site is None:
            return ProviderResponse(
                success=False, error="No SiteConfig — cannot navigate"
            )

        # Pre-flight: validate cookies exist and are parseable (AGENTS.md #7).
        if self._storage_state is not None:
            cv = validate_storage_state(self._storage_state)
            if not cv.is_valid:
                log.warning(
                    "[%s] Cookie pre-flight FAILED: %s",
                    self.provider_name,
                    "; ".join(cv.errors),
                )

        t0 = time.monotonic()
        try:
            page = await self._get_page()

            # Dead page guard: detect crashed/frozen pages
            if page.is_closed():
                log.warning("[%s] Page is closed/crashed — cannot proceed", self.provider_name)
                return ProviderResponse(
                    success=False,
                    error="Page is closed or crashed",
                    model=self.provider_name.replace("_", "-"),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            # Auto cookie save: extract fresh cookies after successful navigation
            if self._context is not None:
                await save_cookies_from_context(self._context, self.provider_name)

            engine = BrowserIntelligenceEngine()
            await engine.attach(page)
            self._engine = engine

            await self._wait_until_ready(engine, page)

            # Clear only the response capture buffers — NOT the full engine
            # state.  ``engine.reset()`` was destroying the readiness state
            # that _wait_until_ready just established, causing subsequent
            # input-element lookups to race against a still-loading page.
            engine._capture.reset()
            with contextlib.suppress(Exception):
                engine._composer._network._sse_observer._response_text = ""
                engine._composer._network._ws_observer._response_text = ""
                engine._composer._network._fetch_observer._response_text = ""

            await self._execute_type_prompt(page, prompt)

            content, reasoning = await self._wait_for_response_pair(engine, page)

            # Response validation contract (AGENTS.md #8):
            # Success is NOT empty string, NOT {"ResultObject":...}, NOT
            # pure JSON array.  It must be actual assistant response text.
            content = self._sanitize_response_text(content)
            reasoning = self._sanitize_response_text(reasoning) if reasoning else reasoning
            if not self._is_valid_response(content):
                log.warning(
                    "[%s] Response validation FAILED: content=%r",
                    self.provider_name,
                    content[:200] if content else "(empty)",
                )
                return ProviderResponse(
                    success=False,
                    error="ResponseExtractionError: captured text is empty or invalid",
                    model=self.provider_name.replace("_", "-"),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            return ProviderResponse(
                content=content,
                reasoning_content=reasoning if reasoning else None,
                model=self.provider_name.replace("_", "-"),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except (AuthenticationError, CloudflareBlockError) as exc:
            # Structured provider errors — surface them directly.
            return ProviderResponse(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ProviderResponse(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    # ── readiness / typing ──────────────────────────────────────────

    async def _wait_until_ready(self, engine: BrowserIntelligenceEngine, page) -> None:
        """Wait until the page is ready for prompt input.

        Strategy:
        1. First try the HMM-based engine (works best on Chromium with CDP).
        2. If the engine has no CDP session (Firefox), fall back to direct
           DOM polling for the chat input element.
        3. Log false positives: if the engine says READY but no input is
           visible, keep polling the DOM.
        """
        is_firefox = self._channel == "firefox"
        MIN_READY_CONSECUTIVE = 3
        consecutive_ready = 0
        login_dismissed = False

        for tick in range(MAX_READY_TICKS):
            # Always tick the engine for CDP-free sensors (DOM, a11y)
            await engine.tick(page)

            # Auto-dismiss popups that block the chat input (Phase 4).
            # Throttle to every 5th tick to avoid excessive page.evaluate calls.
            if tick % 5 == 0:
                await handle_popups(page)

            # On Firefox (no CDP), bypass the HMM and use direct DOM checks.
            if is_firefox:
                has_input = await page.evaluate("""() => {
                    const sels = ['textarea', '[contenteditable="true"]', '[role="textbox"]', '#prompt-textarea', '[class*="prompt-textarea"]'];
                    for (const sel of sels) {
                        try {
                            const el = document.querySelector(sel);
                            if (el && el.offsetParent !== null) return true;
                        } catch(_) {}
                    }
                    return false;
                }""")
                # Also check we're not on a login page
                is_auth = await page.evaluate("""() => {
                    const u = window.location.pathname.toLowerCase();
                    const a = ['/sign_in', '/signin', '/login', '/auth', '/log-in', '/sign-up'];
                    if (a.some(p => u.includes(p))) return true;
                    // Check for password input
                    return !!document.querySelector('input[type="password"]');
                }""")
                if is_auth and not login_dismissed:
                    login_dismissed = True
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(1)
                    for close_sel in (
                        '[aria-label="Close"]', '[aria-label="close"]',
                        '[class*="close"]', '[class*="dismiss"]',
                        'button:has-text("×")', 'button:has-text("✕")',
                        '.modal-close', '.dialog-close',
                    ):
                        try:
                            if await page.locator(close_sel).first.is_visible(timeout=1000):
                                await page.locator(close_sel).first.click()
                                await asyncio.sleep(0.5)
                        except Exception:
                            continue
                    await asyncio.sleep(2)
                    continue

                if has_input:
                    print(f"[TICK {tick}] DOM input visible, page ready")
                    return
                await asyncio.sleep(ENGINE_TICK_INTERVAL)
                continue

            # Chromium path: use HMM-based engine with CDP.
            if engine.most_likely_state == "AUTH_REQUIRED" and not login_dismissed:
                login_dismissed = True
                try:
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(1)
                    for close_sel in (
                        '[aria-label="Close"]', '[aria-label="close"]',
                        '[class*="close"]', '[class*="dismiss"]',
                        'button:has-text("×")', 'button:has-text("✕")',
                        '.modal-close', '.dialog-close',
                    ):
                        try:
                            if await page.locator(close_sel).first.is_visible(timeout=1000):
                                await page.locator(close_sel).first.click()
                                await asyncio.sleep(0.5)
                        except Exception:
                            continue
                except Exception:
                    pass
                await asyncio.sleep(2)
                continue

            if engine.is_ready_for_prompt:
                consecutive_ready += 1
                if consecutive_ready >= MIN_READY_CONSECUTIVE:
                    return
            else:
                consecutive_ready = 0
            if engine.is_error:
                raise RuntimeError(
                    f"Engine error state: {engine.most_likely_state or 'unknown'}"
                )
            await asyncio.sleep(ENGINE_TICK_INTERVAL)

        try:
            url_str = page.url
            title_str = await page.title()
        except Exception:
            url_str = "<unknown>"
            title_str = "<unknown>"
        raise TimeoutError(
            f"Timed out waiting for page readiness after {MAX_READY_TICKS}s "
            f"(url={url_str}, title={title_str!r})"
        )

    async def _execute_type_prompt(self, page, prompt: str) -> None:
        input_el = None

        # First pass: quick check of all selectors
        for sel in INPUT_SELECTORS:
            try:
                loc = page.locator(sel).first
                if await loc.is_visible():
                    input_el = loc
                    break
            except Exception:
                continue

        # Second pass: if not found, wait up to 30s for any selector to appear
        if input_el is None:
            for sel in INPUT_SELECTORS:
                try:
                    loc = page.locator(sel).first
                    await loc.wait_for(state="visible", timeout=30_000)
                    input_el = loc
                    break
                except Exception:
                    continue

        if input_el is None:
            url = page.url
            title = ""
            with contextlib.suppress(Exception):
                title = await page.title()
            raise RuntimeError(
                f"No visible input element found on {url} (title={title!r}). "
                f"The page may require re-authentication."
            )

        await input_el.click()
        await input_el.fill(prompt)
        # Clear the response-body buffer so the next /chat response
        # is the only thing we capture from here on.
        await page.evaluate(CLEAR_INTERCEPTED_BODIES_SCRIPT)
        await asyncio.sleep(0.3)
        await page.keyboard.press("Enter")

    # ── response extraction ─────────────────────────────────────────

    async def _wait_for_response_pair(
        self, engine: BrowserIntelligenceEngine, page
    ) -> tuple[str, str | None]:
        """Wait for the response to complete and return (content, reasoning).

        Returns a tuple of:
          - **content**: the final answer text
          - **reasoning**: the thinking/reasoning chain-of-thought, or None
        """
        extracted = ""
        extracted_reasoning = ""

        for _ in range(MAX_RESPONSE_TICKS):
            store = await engine.tick(page)
            latest = store.latest

            # Preferred source: the engine's network observers buffer
            # the actual SSE / WebSocket / fetch-chunked response text.
            text = engine.get_response_text()
            if text and len(text) > len(extracted):
                extracted = text

            if latest and latest.stream_active and latest.tokens_per_second > 0:
                if not extracted:
                    text = await self._extract_response_text(page)
                    if text and len(text) > len(extracted):
                        extracted = text

            done, _ = engine.is_response_complete()
            if done and len(extracted) > 20:
                break

            if latest and (latest.generation_completed or latest.stream_closed):
                await asyncio.sleep(1)
                text = engine.get_response_text()
                if text and len(text) > len(extracted):
                    extracted = text
                if not extracted:
                    text = await self._extract_response_text(page)
                    if text and len(text) > len(extracted):
                        extracted = text
                break

            await asyncio.sleep(ENGINE_TICK_INTERVAL)

        if not extracted:
            text = engine.get_response_text()
            if text:
                extracted = text
            else:
                extracted = await self._extract_response_text(page)

        # After we have the final content, try to extract reasoning
        # from the intercepted network bodies.
        extracted_reasoning = await self._extract_reasoning_text(page)

        # If reasoning was present in the SSE stream, it might be
        # duplicated in the extracted content. Try to strip it.
        if extracted_reasoning and extracted_reasoning in extracted:
            extracted = extracted.replace(extracted_reasoning, "").strip()

        return extracted, (extracted_reasoning if extracted_reasoning else None)

    # ── chat URL detection ──────────────────────────────────────────

    CHAT_URL_PATTERNS = (
        "chat/completions",
        "/chat/message",
        "/v1/messages",
        "/api/v1/chat",
        "/conversation/message",
        "sendmessage",
        "send_message",
        "/backend-api/conversation",
        "/backend-api/f/conversation",
        "/api/chat/send",
        "/api/chat/completions",
        # Kimi
        "/api/chat/segment",
        "/kimiplus/chat",
        # MiniMax
        "/v1/text/chatcompletion",
        "/agent/chat",
        # MiMo / Xiaomi
        "/api/v1/generate",
        "/api/chat/stream",
        # Qwen
        "/api/v1/qwen",
        "/api/chat/qwen",
    )

    @classmethod
    def _looks_like_chat_response(cls, entry: dict) -> bool:
        url = (entry.get("url") or "").lower()
        transport = (entry.get("transport") or "fetch").lower()
        if "/chats" in url or "/conversations" in url:
            return False
        if any(ext in url for ext in (".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".woff", ".woff2", ".ttf", ".eot", ".wasm")):
            return False
        if "chat/completions" in url or "/v1/messages" in url:
            return True
        if transport in ("sse", "fetch_stream"):
            return True
        if any(pat in url for pat in cls.CHAT_URL_PATTERNS):
            return True
        return False

    # ── Response sanitization & validation ───────────────────────────

    @staticmethod
    def _sanitize_response_text(text: str) -> str:
        """Strip known pollution patterns from extracted response text.

        Qwen's SSE stream includes metadata chunks like
        ``{"type": "assistant", "status": "typing"}`` whose values get
        concatenated into the response: ``assistanttypingassistantfinished``.
        This method removes those runs.
        """
        if not text:
            return text
        # Strip runs of concatenated status labels.
        text = _STATUS_NOISE_RE.sub("", text)
        # Trim any trailing/leading whitespace introduced.
        text = text.strip()
        return text

    @staticmethod
    def _is_valid_response(text: str) -> bool:
        """Check if extracted text is actual assistant response content.

        Per AGENTS.md rule #8:
          Success is NOT ``{}``, NOT ``{"ResultObject": true}``,
          NOT an empty string.  Only actual assistant response text.
        """
        if not text or len(text.strip()) < 2:
            return False
        stripped = text.strip()
        # Reject pure JSON that looks like a tracking ping.
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    # Tracking pings: {"ResultObject": true} etc.
                    if "ResultObject" in obj:
                        return False
                    # Empty API responses
                    if not any(k in obj for k in (
                        "choices", "message", "content", "text",
                        "delta", "response", "output",
                    )):
                        return False
            except Exception:
                pass
        # Reject pure JSON arrays (conversation lists, etc.)
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                arr = json.loads(stripped)
                if isinstance(arr, list):
                    return False
            except Exception:
                pass
        # Reject text that is only status labels.
        if _STATUS_LABEL_RE.match(stripped):
            return False
        return True

    # ── Delta / SSE parsing ─────────────────────────────────────────

    DELTA_KEYS = (
        "delta_content",
        "content",
        "text",
        "delta_text",
        "message",
        "reasoning_content",
    )

    REASONING_KEYS = (
        "reasoning_content",
    )

    REASONING_PHASE_MARKERS = (
        "thinking",
        "reasoning",
    )

    DELTA_PARENT_KEYS = (
        ("delta", "content"),
        ("delta", "text"),
        ("delta", "delta_content"),
        ("delta", "reasoning_content"),
        ("message", "content"),
        ("message", "reasoning_content"),
        ("choices", 0, "delta", "content"),
        ("choices", 0, "delta", "reasoning_content"),
        ("choices", 0, "message", "content"),
    )

    @classmethod
    def _extract_deltas_from_obj(cls, obj: dict | list) -> list[str]:
        """DFS walk pulling every string under a delta-like key.

        Skips known status/role labels (``assistant``, ``typing``,
        ``finished`` …) that pollute the response when a provider
        emits metadata chunks alongside content chunks.
        """
        found: list[str] = []
        if isinstance(obj, str):
            return found
        seen: set[int] = set()
        stack: list = [obj]
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if isinstance(cur, dict):
                for k, v in cur.items():
                    kl = k.lower()
                    if isinstance(v, str) and any(dk in kl for dk in cls.DELTA_KEYS):
                        if v and v != "[DONE]" and not _STATUS_LABEL_RE.match(v):
                            found.append(v)
                    elif isinstance(v, list) and any(dk in kl for dk in cls.DELTA_KEYS + ("parts",)):
                        for item in v:
                            if isinstance(item, str) and item and item != "[DONE]" and not _STATUS_LABEL_RE.match(item):
                                found.append(item)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                for v in cur:
                    if isinstance(v, (dict, list)):
                        stack.append(v)
        return found

    @classmethod
    def _extract_reasoning_deltas_from_obj(cls, obj: dict | list) -> list[str]:
        """DFS walk pulling only reasoning_content strings."""
        found: list[str] = []
        if isinstance(obj, str):
            return found
        seen: set[int] = set()
        stack: list = [obj]
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if isinstance(cur, dict):
                for k, v in cur.items():
                    kl = k.lower()
                    # Direct reasoning_content key
                    if any(rk in kl for rk in cls.REASONING_KEYS):
                        if isinstance(v, str) and v and v != "[DONE]" and not _STATUS_LABEL_RE.match(v):
                            found.append(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, str) and item and item != "[DONE]" and not _STATUS_LABEL_RE.match(item):
                                    found.append(item)
                    # Phase-marked chunks (Z.ai style: {"phase": "thinking", "delta_content": "..."})
                    if kl == "phase" and isinstance(v, str) and any(m in v.lower() for m in cls.REASONING_PHASE_MARKERS):
                        # Walk siblings for delta content
                        for sk, sv in cur.items():
                            if sk.lower() != kl and isinstance(sv, str) and sv:
                                found.append(sv)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                for v in cur:
                    if isinstance(v, (dict, list)):
                        stack.append(v)
        return found

    @classmethod
    def _sse_delta_to_text(cls, raw: str) -> str:
        out: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            # First try the well-known parent paths
            for path in cls.DELTA_PARENT_KEYS:
                cur = obj
                ok = True
                for k in path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok:
                    if isinstance(cur, str) and cur and not _STATUS_LABEL_RE.match(cur):
                        out.append(cur)
                        break
                    elif isinstance(cur, list) and all(isinstance(x, str) for x in cur):
                        joined = "".join(x for x in cur if not _STATUS_LABEL_RE.match(x))
                        if joined:
                            out.append(joined)
                            break
            else:
                # Fallback: deep walk and pull every delta-like string
                for s in cls._extract_deltas_from_obj(obj):
                    out.append(s)
        return "".join(out)

    @classmethod
    def _sse_reasoning_to_text(cls, raw: str) -> str:
        """Extract only reasoning/thinking content from an SSE stream."""
        out: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            # Try parent paths that lead to reasoning
            for path in cls.DELTA_PARENT_KEYS:
                # Only consider paths ending in reasoning_content
                if path[-1] not in cls.REASONING_KEYS:
                    continue
                cur = obj
                ok = True
                for k in path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok:
                    if isinstance(cur, str) and cur and not _STATUS_LABEL_RE.match(cur):
                        out.append(cur)
                        break
            else:
                # Fallback: deep walk for reasoning-only deltas
                for s in cls._extract_reasoning_deltas_from_obj(obj):
                    out.append(s)
        return "".join(out)

    @staticmethod
    def _is_invalid_json_response(text: str) -> bool:
        text = text.strip()
        if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
            return False
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return True
            if isinstance(obj, dict):
                for k in ("items", "conversations", "chats", "history", "models", "categories"):
                    if k in obj:
                        return True
                if "data" in obj and isinstance(obj["data"], list):
                    if len(obj["data"]) > 0 and isinstance(obj["data"][0], dict):
                        first = obj["data"][0]
                        if any(k in first for k in ("title", "uuid", "id", "created_at", "updated_at")):
                            return True
                has_chat_keys = any(k in obj for k in ("choices", "message", "content", "text", "delta", "response", "output"))
                if not has_chat_keys:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    async def _extract_response_text(page) -> str:
        # Strategy 1: Intercepted fetch/XHR response bodies — prefer chat URLs
        try:
            bodies = await page.evaluate(GET_INTERCEPTED_BODIES_SCRIPT)
            if bodies:
                chat_bodies = [b for b in bodies if EngineUIAdapter._looks_like_chat_response(b)]
                if chat_bodies:
                    for entry in reversed(chat_bodies):
                        text = (entry.get("text") or "").strip()
                        if not text or len(text) < 20:
                            continue
                        if EngineUIAdapter._is_invalid_json_response(text):
                            continue
                        delta = EngineUIAdapter._sse_delta_to_text(text)
                        if delta and len(delta) > 1:
                            return delta
                        if text.startswith("data:"):
                            continue
                        if text.startswith("[") and text.endswith("]"):
                            continue
                        return text
        except Exception:
            pass

        # Strategy 2: DOM-based extraction
        try:
            text = (await page.evaluate(RESPONSE_EXTRACT_SCRIPT)).strip()
            if text:
                return text
        except Exception:
            pass

        return ""

    @classmethod
    async def _extract_reasoning_text(cls, page) -> str:
        """Extract reasoning/thinking content from intercepted network bodies."""
        try:
            bodies = await page.evaluate(GET_INTERCEPTED_BODIES_SCRIPT)
            if not bodies:
                return ""

            chat_bodies = [b for b in bodies if cls._looks_like_chat_response(b)]
            if not chat_bodies:
                return ""

            for entry in reversed(chat_bodies):
                text = (entry.get("text") or "").strip()
                if not text or len(text) < 10:
                    continue
                if cls._is_invalid_json_response(text):
                    continue

                reasoning = cls._sse_reasoning_to_text(text)
                if reasoning and len(reasoning) > 1:
                    return reasoning

            return ""
        except Exception:
            return ""

    # ── browser lifecycle ──────────────────────────────────────────

    async def _get_page(self):
        if self._page is not None:
            return self._page

        from playwright.async_api import async_playwright

        # Firefox path — simple launch, no Chromium-specific flags
        if self._channel == "firefox":
            self._playwright = await async_playwright().start()
            if self._persistent_profile is not None:
                profile_path = Path(self._persistent_profile)
                profile_path.mkdir(parents=True, exist_ok=True)
                self._context = await self._playwright.firefox.launch_persistent_context(
                    user_data_dir=str(profile_path),
                    headless=self.headless,
                    viewport={"width": 1280, "height": 720} if self._stealth else None,
                )
                self._browser = None
                self._page = (
                    self._context.pages[0]
                    if self._context.pages
                    else await self._context.new_page()
                )
                if self._storage_state is not None:
                    await self._context.add_cookies(self._storage_state.get("cookies", []))
            else:
                self._browser = await self._playwright.firefox.launch(
                    headless=self.headless,
                )
                ctx_opts = {}
                if self._stealth:
                    ctx_opts["viewport"] = {"width": 1280, "height": 720}
                if self._storage_state is not None:
                    ctx_opts["storage_state"] = self._storage_state
                self._context = await self._browser.new_context(**ctx_opts)
                self._page = await self._context.new_page()
            await self._page.add_init_script(FETCH_INTERCEPT_SCRIPT)
            await self._page.goto(
                self._site.url, timeout=self._timeout_ms, wait_until="domcontentloaded",
            )
            await asyncio.sleep(3)

            # Post-navigation checks (same as Chromium path).
            provider = self.provider_name or (self._site.name if self._site else "unknown")

            is_cf, cf_reason = await check_cloudflare_challenge(self._page)
            if is_cf:
                raise CloudflareBlockError(
                    provider,
                    f"Cloudflare challenge detected — {cf_reason}. "
                    "Do NOT retry. Detect → Pause → Notify.",
                    page_title=await self._page.title(),
                )

            is_auth, auth_reason = await check_post_navigation_auth(self._page)
            if not is_auth:
                raise AuthenticationError(
                    provider,
                    f"Not authenticated — {auth_reason}. "
                    "Cookies may be expired. Re-export cookies.",
                    url=self._page.url,
                )

            return self._page

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
        ]

        if EngineUIAdapter._shared_browser is not None:
            from playwright.async_api import BrowserContext
            if isinstance(EngineUIAdapter._shared_browser, BrowserContext):
                self._context = EngineUIAdapter._shared_browser
                self._page = await self._context.new_page()
            else:
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

                self._context = await EngineUIAdapter._shared_browser.new_context(**context_opts)
                self._page = await self._context.new_page()
        else:
            self._playwright = await async_playwright().start()

            # Use Google Chrome's actual profile if available and not overridden by persistent profile
            chrome_user_data_dir = Path("/Users/prajwalbairagi/Library/Application Support/Google/Chrome")
            if chrome_user_data_dir.exists() and self._persistent_profile is None and self._storage_state is None and False:
                launch_args.append("--profile-directory=Profile 2")
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(chrome_user_data_dir),
                    headless=self.headless,
                    channel="chrome",
                    args=launch_args,
                    viewport={"width": 1280, "height": 720},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                )
                self._browser = None
                self._page = (
                    self._context.pages[0]
                    if self._context.pages
                    else await self._context.new_page()
                )
            else:
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
                    self._page = (
                        self._context.pages[0]
                        if self._context.pages
                        else await self._context.new_page()
                    )
                    if self._storage_state is not None:
                        await self._context.add_cookies(self._storage_state.get("cookies", []))
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

        await self._page.add_init_script(FETCH_INTERCEPT_SCRIPT)

        await self._page.goto(
            self._site.url,
            timeout=self._timeout_ms,
            wait_until="domcontentloaded",
        )

        # Give SPAs time to hydrate and render the chat input
        await asyncio.sleep(5)

        # Post-navigation checks: auth redirect and Cloudflare challenge.
        # Fail fast instead of waiting 60s for a readiness timeout.
        provider = self.provider_name or (self._site.name if self._site else "unknown")

        is_cf, cf_reason = await check_cloudflare_challenge(self._page)
        if is_cf:
            log.warning("[%s] Cloudflare challenge detected: %s", provider, cf_reason)
            raise CloudflareBlockError(
                provider,
                f"Cloudflare challenge detected — {cf_reason}. "
                "Do NOT retry. Detect → Pause → Notify.",
                page_title=await self._page.title(),
            )

        is_auth, auth_reason = await check_post_navigation_auth(self._page)
        if not is_auth:
            log.warning("[%s] Auth check failed: %s", provider, auth_reason)
            raise AuthenticationError(
                provider,
                f"Not authenticated — {auth_reason}. "
                "Cookies may be expired. Re-export cookies.",
                url=self._page.url,
            )

        return self._page
