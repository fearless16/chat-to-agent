"""DomSensor — extracts structural DOM features without text content.

One Playwright RPC per tick. All selector checks and node counts
are evaluated browser-side in a single page.evaluate() call.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_orchestrator.browser_intelligence.sensors.base import BaseSensor

_DOM_EVAL_SCRIPT = """
() => {
    function any(selectors) {
        for (const sel of selectors) {
            try {
                const el = document.querySelector(sel);
                if (el && el.offsetParent !== null) return true;
            } catch (_) {}
        }
        return false;
    }
    return {
        input_visible: any(['textarea','[contenteditable="true"]','[role="textbox"]','[class*="prompt-textarea"]']),
        send_visible: any(['button[type="submit"]','[aria-label*="send" i]','[data-testid="send-button"]','button:has(svg)']),
        stop_visible: any(['[aria-label*="stop" i]','[data-testid="stop-button"]','[class*="stop-generating"]']),
        regenerate_visible: any(['[aria-label*="regenerate" i]','[data-testid="regenerate"]']),
        error_visible: any(['[class*="error"]','[class*="alert"]','[role="alert"]']),
        auth_visible: any(['[class*="login"]','[class*="signin"]','input[type="password"]']),
        dom_nodes: document.querySelectorAll('*').length,
        interactive: document.querySelectorAll('button,input,textarea,[contenteditable]').length,
    };
}
"""


@dataclass
class DOMFeatures:
    input_visible: bool = False
    send_visible: bool = False
    stop_button_visible: bool = False
    regenerate_visible: bool = False
    error_banner_visible: bool = False
    auth_form_visible: bool = False
    dom_node_count: int = 0
    interactive_count: int = 0


class DOMSensor(BaseSensor):
    """Extracts structural DOM features — presence/absence of elements only.

    Never reads text content. Never makes decisions.
    Exactly one page.evaluate() call per tick.
    """

    async def sense(self, page) -> DOMFeatures:
        features = DOMFeatures()
        try:
            data = await page.evaluate(_DOM_EVAL_SCRIPT)
            features.input_visible = bool(data.get("input_visible", False))
            features.send_visible = bool(data.get("send_visible", False))
            features.stop_button_visible = bool(data.get("stop_visible", False))
            features.regenerate_visible = bool(data.get("regenerate_visible", False))
            features.error_banner_visible = bool(data.get("error_visible", False))
            features.auth_form_visible = bool(data.get("auth_visible", False))
            features.dom_node_count = int(data.get("dom_nodes", 0))
            features.interactive_count = int(data.get("interactive", 0))
        except Exception:
            pass
        return features
