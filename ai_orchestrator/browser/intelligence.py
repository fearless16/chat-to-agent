"""UI Intelligence Layer — the eyes of the browser worker.

Pipeline (ordered by cost, lowest first):

1. Selector Cache         — hit: return immediately
2. Accessibility Tree     — low-cost, semantic, stable
3. DOM Snippets           — targeted extraction (candidate buttons/inputs)
4. DeepSeek DOM Analysis  — expensive, only when a11y + DOM fail
5. Vision                 — last resort, screenshot + coord recovery
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ai_orchestrator.browser.accessibility import AccessibilityRuntime, A11yNode
from ai_orchestrator.browser.selector_cache import SelectorCache
from ai_orchestrator.browser.schema import UIRole, UISelector, UISchema, UISchemaEngine

log = logging.getLogger(__name__)


@dataclass
class UIQueryResult:
    """Result of a UI element discovery query."""
    selector: UISelector
    latency_ms: float = 0.0
    pipeline_stage: str = "cache"  # cache | a11y | dom | deepseek | vision
    confidence: float = 1.0


class UIIntelligence:
    """Pipeline orchestrator for browser UI element discovery.

    Usage::

        ii = UIIntelligence(selector_cache=cache)
        result = await ii.locate_input(page, provider="qwen_ui")
        page.locator(result.selector.value).click()
    """

    def __init__(
        self,
        selector_cache: Optional[SelectorCache] = None,
        schema_engine: Optional[UISchemaEngine] = None,
    ) -> None:
        self._cache = selector_cache or SelectorCache()
        self._a11y = AccessibilityRuntime()
        self._schema = schema_engine or UISchemaEngine()
        self._provider_known_selectors: dict[str, dict[str, list[str]]] = {
            "qwen_ui": {
                UIRole.INPUT: [
                    "textarea",
                    '[contenteditable="true"]',
                    '[class*="input"]',
                    '[class*="prompt"]',
                ],
                UIRole.SEND: [
                    'button[type="submit"]',
                    '[aria-label*="send" i]',
                    '[aria-label*="Send" i]',
                    '[data-testid="send-button"]',
                    '[class*="send"]',
                ],
                UIRole.ASSISTANT_MESSAGE: [
                    '[class*="message"]',
                    '[class*="assistant"]',
                    '[class*="response"]',
                ],
            },
            "chatgpt_ui": {
                UIRole.INPUT: [
                    "#prompt-textarea",
                    '[contenteditable="true"]',
                    '[class*="prompt"]',
                ],
                UIRole.SEND: [
                    '[data-testid="send-button"]',
                    'button[aria-label*="Send" i]',
                ],
                UIRole.ASSISTANT_MESSAGE: [
                    '[class*="assistant"]',
                    '[class*="message"]',
                    "article[data-testid*='conversation']",
                ],
            },
        }

    # ── public API ──────────────────────────────────────────────────

    async def locate_input(
        self, page, provider: str, hint: str = "",
    ) -> UIQueryResult:
        return await self._discover(page, provider, UIRole.INPUT, hint or "message")

    async def locate_send(
        self, page, provider: str, hint: str = "",
    ) -> UIQueryResult:
        return await self._discover(page, provider, UIRole.SEND, hint or "send")

    async def locate_message_container(
        self, page, provider: str,
    ) -> UIQueryResult:
        return await self._discover(page, provider, UIRole.ASSISTANT_MESSAGE, "message")

    async def build_schema(
        self, page, provider: str,
    ) -> UISchema:
        """Build a full UISchema for the current page."""
        title = await page.title()
        url = page.url

        input_result = await self.locate_input(page, provider)
        send_result = await self.locate_send(page, provider)

        schema = UISchema(
            input_box=input_result.selector,
            send_button=send_result.selector,
            title=title,
            url=url,
        )
        self._schema.register(provider, schema)
        return schema

    # ── discovery pipeline ──────────────────────────────────────────

    async def _discover(
        self, page, provider: str, role: UIRole, hint: str,
    ) -> UIQueryResult:
        """Run the full discovery pipeline, cheapest first."""

        # Stage 1 — Selector Cache ------------------------------------
        cached = self._cache.get(provider, role)
        if cached:
            sel = UISelector(role=role, value=cached["value"], source="cache", confidence=cached["confidence"])
            log.debug("Selector cache HIT for %s/%s: %s", provider, role, sel.value)
            return UIQueryResult(selector=sel, pipeline_stage="cache", confidence=cached["confidence"])

        # Stage 2 — Accessibility Tree ---------------------------------
        a11y_result = await self._try_a11y(page, role, hint)
        if a11y_result:
            self._cache.set(provider, role, a11y_result.selector.value, source="a11y", confidence=a11y_result.confidence)
            return a11y_result

        # Stage 3 — DOM known selectors --------------------------------
        dom_result = await self._try_dom(page, provider, role)
        if dom_result:
            self._cache.set(provider, role, dom_result.selector.value, source="dom", confidence=dom_result.confidence)
            return dom_result

        # Stage 4 — DeepSeek DOM analysis (Tier 2) ---------------------
        ds_result = await self._try_deepseek(page, role, hint)
        if ds_result:
            self._cache.set(provider, role, ds_result.selector.value, source="deepseek", confidence=ds_result.confidence)
            return ds_result

        # Stage 5 — Vision fallback ------------------------------------
        log.warning("All stages failed for %s/%s — falling back to Vision", provider, role)
        vision_result = await self._try_vision(page, role)
        if vision_result:
            self._cache.set(provider, role, vision_result.selector.value, source="vision", confidence=vision_result.confidence)
            return vision_result

        log.error("UI Intelligence could not locate %s/%s — no fallback remaining", provider, role)
        return UIQueryResult(
            selector=UISelector(role=role, value="", source="none", confidence=0.0),
            pipeline_stage="none",
            confidence=0.0,
        )

    # ── stage implementations ───────────────────────────────────────

    async def _try_a11y(self, page, role: UIRole, hint: str) -> Optional[UIQueryResult]:
        try:
            if role == UIRole.INPUT:
                node = await self._a11y.find_input(page, hint)
            elif role == UIRole.SEND:
                node = await self._a11y.find_send_button(page, hint)
            else:
                return None

            if node is None:
                return None

            selector = self._a11y_node_to_css(node)
            if not selector:
                return None

            return UIQueryResult(
                selector=UISelector(role=role, value=selector, source="a11y", confidence=0.85),
                pipeline_stage="a11y",
                confidence=0.85,
            )
        except Exception:
            log.debug("A11y stage failed for %s", role, exc_info=True)
            return None

    async def _try_dom(self, page, provider: str, role: UIRole) -> Optional[UIQueryResult]:
        known = self._provider_known_selectors.get(provider, {}).get(role, [])
        for sel in known:
            try:
                el = page.locator(sel).first
                if await el.count() > 0 and await el.is_visible():
                    log.debug("DOM hit for %s/%s: %s", provider, role, sel)
                    return UIQueryResult(
                        selector=UISelector(role=role, value=sel, source="dom", confidence=0.9),
                        pipeline_stage="dom",
                        confidence=0.9,
                    )
            except Exception:
                continue
        return None

    async def _try_deepseek(self, page, role: UIRole, hint: str) -> Optional[UIQueryResult]:
        """Stage 4 — ask DeepSeek to analyse DOM snippets.

        This is a stub that would send a focused DOM snippet to DeepSeek.
        The actual DeepSeek call would go through the DeepSeek API adapter.
        """
        return None

    async def _try_vision(self, page, role: UIRole) -> Optional[UIQueryResult]:
        """Stage 5 — last-resort vision fallback.

        This is a stub that would screenshot, send to DeepSeek Vision,
        and recover coordinates.
        """
        return None

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _a11y_node_to_css(node: A11yNode) -> Optional[str]:
        if node.name:
            escaped = node.name.replace('"', '\\"')
            return f'[aria-label="{escaped}"]'
        return None
