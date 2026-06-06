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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_orchestrator.adapters.base import ProviderAdapter, ProviderResponse
from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine

if TYPE_CHECKING:
    pass

ENGINE_TICK_INTERVAL: float = 1.0
MAX_READY_TICKS: int = 30
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
            const last = nodes[nodes.length - 1];
            const t = cleanText(last).trim();
            if (t.length > 1) return t;
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
    """

    supports_streaming = False
    supports_tools = False
    _site: SiteConfig | None = None
    mock_content_prefix: str = ""
    mock_model: str = ""
    mock_context_limit: int = 131_072

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

    # ── ProviderAdapter interface ──────────────────────────────────

    async def send(
        self, prompt: str, context: list[dict] | None = None  # noqa: ARG002
    ) -> ProviderResponse:
        if self._mock_mode:
            return self._mock_send(prompt)
        return await self._real_send(prompt)

    async def health_check(self) -> bool:
        if self._mock_mode:
            return True
        try:
            page = await self._get_page()
            return page is not None
        except Exception:
            return False

    def get_context_limit(self) -> int:
        return self.mock_context_limit

    async def is_rate_limited(self) -> bool:
        return False

    async def refresh_session(self) -> bool:
        if not self._mock_mode and self._page is not None:
            with contextlib.suppress(Exception):
                await self._page.reload()
        return True

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.detach()
            self._engine = None
        if self._page is not None:
            with contextlib.suppress(Exception):
                await self._page.close()
            self._page = None
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.close()
            self._context = None
        if self._browser is not None:
            with contextlib.suppress(Exception):
                await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            with contextlib.suppress(Exception):
                await self._playwright.stop()
            self._playwright = None

    # ── mock path ──────────────────────────────────────────────────

    def _mock_send(self, prompt: str) -> ProviderResponse:
        prefix = self.mock_content_prefix or f"{self.provider_name}"
        model = self.mock_model or self.provider_name.replace("_", "-")
        return ProviderResponse(
            content=f"{prefix} response to: {prompt[:50]}",
            model=model,
            latency_ms=1200.0,
        )

    # ── parallel fan-out ───────────────────────────────────────────

    @classmethod
    async def fan_out(
        cls,
        adapters: list[ProviderAdapter],
        prompt: str,
        context: list[dict] | None = None,
        return_when: str = "ALL_COMPLETED",
        timeout: float | None = None,
    ) -> list[ProviderResponse]:
        """Send *prompt* to every adapter in *adapters* in parallel.

        Each adapter is a self-contained object — typically constructed
        with its own ``storage_state`` / ``persistent_profile`` — and
        runs in its own Playwright context.  The call returns once the
        chosen ``return_when`` condition is met (``"ALL_COMPLETED"``,
        ``"FIRST_COMPLETED"``, or ``"FIRST_EXCEPTION"``).

        Always cleans up every adapter in a ``finally`` block so a
        single failure does not leak browser processes.
        """
        valid_modes = {"ALL_COMPLETED", "FIRST_COMPLETED", "FIRST_EXCEPTION"}
        if return_when not in valid_modes:
            raise ValueError(f"Invalid return_when={return_when!r}; expected one of {valid_modes}")

        if not adapters:
            return []

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

    # ── real browser path (engine-driven) ──────────────────────────

    async def _real_send(self, prompt: str) -> ProviderResponse:
        if self._site is None:
            return ProviderResponse(
                success=False, error="No SiteConfig — cannot navigate"
            )
        t0 = time.monotonic()
        try:
            page = await self._get_page()
            engine = BrowserIntelligenceEngine()
            await engine.attach(page)
            self._engine = engine

            await self._wait_until_ready(engine, page)

            engine.reset()
            engine._composer._network._sse_observer._response_text = ""
            engine._composer._network._ws_observer._response_text = ""
            engine._composer._network._fetch_observer._response_text = ""

            await self._execute_type_prompt(page, prompt)

            content = await self._wait_for_response(engine, page)

            return ProviderResponse(
                content=content,
                model=self.provider_name.replace("_", "-"),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return ProviderResponse(
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    async def _wait_until_ready(self, engine: BrowserIntelligenceEngine, page) -> None:
        for _ in range(MAX_READY_TICKS):
            await engine.tick(page)
            if engine.is_ready_for_prompt:
                return
            if engine.is_error:
                raise RuntimeError(
                    f"Engine error state: {engine.most_likely_state or 'unknown'}"
                )
            await asyncio.sleep(ENGINE_TICK_INTERVAL)
        raise TimeoutError("Timed out waiting for page readiness")

    async def _execute_type_prompt(self, page, prompt: str) -> None:
        input_el = None
        for sel in INPUT_SELECTORS:
            try:
                loc = page.locator(sel).first
                if await loc.is_visible():
                    input_el = loc
                    break
            except Exception:
                continue

        if input_el is None:
            raise RuntimeError("No visible input element found")

        await input_el.click()
        await input_el.fill(prompt)
        # Clear the response-body buffer so the next /chat response
        # is the only thing we capture from here on.
        await page.evaluate(CLEAR_INTERCEPTED_BODIES_SCRIPT)
        await asyncio.sleep(0.3)
        await page.keyboard.press("Enter")

    async def _wait_for_response(
        self, engine: BrowserIntelligenceEngine, page
    ) -> str:
        extracted = ""

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

        return extracted

    CHAT_URL_PATTERNS = (
        "chat/completions",
        "/chat/message",
        "/v1/messages",
        "/v1/chats",
        "/api/v1/chat",
        "/api/chat",
        "/conversation/message",
        "sendmessage",
        "send_message",
    )

    @classmethod
    def _looks_like_chat_response(cls, entry: dict) -> bool:
        url = (entry.get("url") or "").lower()
        transport = (entry.get("transport") or "fetch").lower()
        if "chat/completions" in url or "/v1/messages" in url:
            return True
        if transport == "sse":
            return True
        if any(pat in url for pat in cls.CHAT_URL_PATTERNS):
            return True
        return False

    DELTA_KEYS = (
        "delta_content",
        "content",
        "text",
        "delta_text",
        "message",
        "reasoning_content",
    )
    DELTA_PARENT_KEYS = (
        ("delta", "content"),
        ("message", "content"),
        ("data", "content"),
        ("data", "delta_content"),
        ("data", "text"),
        ("choices", 0, "delta", "content"),
        ("choices", 0, "message", "content"),
        ("choices", 0, "text"),
        ("response", "choices", 0, "delta", "content"),
        ("response", "choices", 0, "message", "content"),
    )

    @classmethod
    def _extract_deltas_from_obj(cls, obj) -> list[str]:
        """Recursively walk a JSON value collecting any string that lives under a
        delta-like key (delta_content/content/text/delta)."""
        found: list[str] = []
        stack: list = [obj]
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if isinstance(cur, dict):
                for k, v in cur.items():
                    kl = k.lower()
                    if isinstance(v, str) and any(dk in kl for dk in cls.DELTA_KEYS):
                        if v and v != "[DONE]":
                            found.append(v)
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
                if ok and isinstance(cur, str) and cur:
                    out.append(cur)
                    break
            else:
                # Fallback: deep walk and pull every delta-like string
                for s in cls._extract_deltas_from_obj(obj):
                    out.append(s)
        return "".join(out)

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
                        delta = EngineUIAdapter._sse_delta_to_text(text)
                        if delta and len(delta) > 1:
                            return delta
                        if text.startswith("data:"):
                            continue
                        if text.startswith("[") and text.endswith("]"):
                            continue
                        return text
                else:
                    for entry in reversed(bodies):
                        text = (entry.get("text") or "").strip()
                        if not text or len(text) < 20:
                            continue
                        delta = EngineUIAdapter._sse_delta_to_text(text)
                        if delta and len(delta) > 1:
                            return delta
                        if text.startswith("data:"):
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

    # ── browser lifecycle ──────────────────────────────────────────

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
            self._page = (
                self._context.pages[0]
                if self._context.pages
                else await self._context.new_page()
            )
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

        await asyncio.sleep(2)

        return self._page
