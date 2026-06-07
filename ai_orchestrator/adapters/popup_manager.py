"""Popup Manager — auto-dismiss overlays that block the chat input.

Per ARCHITECTURE.md Phase 4:
  Auto dismiss: Cookie banners, Newsletter popups, Upgrade modals,
  Generic modal overlays.

Per AGENTS.md rule #5:
  Explicit popup states: COOKIE_BANNER, NEWSLETTER_MODAL, UPGRADE_MODAL,
  RATE_LIMIT_MODAL, CAPTCHA_MODAL, UNKNOWN_MODAL.
  Unknown modal → screenshot, log, pause. Don't randomly click.

This module is called each tick during `_wait_until_ready()` to detect
and dismiss popups that are blocking the chat input element.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto

log = logging.getLogger(__name__)


class PopupType(Enum):
    """Detected popup classification."""
    COOKIE_BANNER = auto()
    NEWSLETTER_MODAL = auto()
    UPGRADE_MODAL = auto()
    RATE_LIMIT_MODAL = auto()
    CAPTCHA_MODAL = auto()
    UNKNOWN_MODAL = auto()
    NONE = auto()


# CSS selectors for common cookie-consent banners.
_COOKIE_BANNER_SELECTORS: tuple[str, ...] = (
    '[class*="cookie-consent"]',
    '[class*="cookie-banner"]',
    '[class*="cookie-notice"]',
    '[class*="cookie-popup"]',
    '[id*="cookie-consent"]',
    '[id*="cookie-banner"]',
    '[id*="cookiebanner"]',
    '[class*="consent-banner"]',
    '[class*="cc-banner"]',
    '[class*="gdpr"]',
    '[aria-label*="cookie" i]',
    '[data-testid*="cookie" i]',
)

# CSS selectors for generic dismiss / close buttons.
_CLOSE_BUTTON_SELECTORS: tuple[str, ...] = (
    # Explicit labels
    '[aria-label="Close" i]',
    '[aria-label="close" i]',
    '[aria-label="Dismiss" i]',
    '[aria-label="dismiss" i]',
    # Class-based
    '[class*="close-button"]',
    '[class*="closeButton"]',
    '[class*="close-btn"]',
    '[class*="dismiss"]',
    '[class*="modal-close"]',
    '[class*="dialog-close"]',
    # Data-testid
    '[data-testid="close-button"]',
    '[data-testid="modal-close"]',
    # Symbols
    'button:has-text("×")',
    'button:has-text("✕")',
    'button:has-text("✖")',
    'button:has-text("Close")',
)

# Text patterns for cookie-accept buttons.
_COOKIE_ACCEPT_TEXT: tuple[str, ...] = (
    "Accept",
    "Accept All",
    "Accept Cookies",
    "Allow",
    "Allow All",
    "I Agree",
    "Agree",
    "OK",
    "Got it",
    "I understand",
    "Continue",
    "Consent",
)

# Text patterns for upgrade/newsletter dismissal.
_DISMISS_TEXT: tuple[str, ...] = (
    "No thanks",
    "No, thanks",
    "Not now",
    "Later",
    "Maybe later",
    "Skip",
    "Dismiss",
    "Close",
    "Cancel",
)


# ── Detection JS ─────────────────────────────────────────────────

_DETECT_POPUP_SCRIPT = """() => {
    // Returns { type: string, selector: string } or null.
    const html = document.documentElement.innerHTML.toLowerCase();
    const body = document.body;
    if (!body) return null;

    // Check for modal overlays that block interaction.
    // Strategy 1: Semantic selectors (role, aria-modal, class names).
    const overlays = body.querySelectorAll(
        '[class*="modal"][class*="overlay"], [class*="modal-backdrop"], ' +
        '[class*="dialog-overlay"], [class*="popup-overlay"], ' +
        '[role="dialog"], [aria-modal="true"], ' +
        // Strategy 2: Full-screen fixed overlays (MiniMax-style).
        '[class*="blanket"], [class*="backdrop"], ' +
        '[class*="dismiss-boundary"]'
    );

    let blockingOverlay = null;
    for (const o of overlays) {
        const style = window.getComputedStyle(o);
        if (style.display !== 'none' && style.visibility !== 'hidden' &&
            parseFloat(style.opacity) > 0.1) {
            blockingOverlay = o;
            break;
        }
    }

    // Strategy 3: Detect any fixed element covering the viewport with high z-index.
    if (!blockingOverlay) {
        const allFixed = body.querySelectorAll('*');
        for (const el of allFixed) {
            const st = window.getComputedStyle(el);
            if (st.position === 'fixed' && parseInt(st.zIndex) >= 1000) {
                const rect = el.getBoundingClientRect();
                if (rect.width >= window.innerWidth * 0.5 &&
                    rect.height >= window.innerHeight * 0.5) {
                    blockingOverlay = el;
                    break;
                }
            }
        }
    }

    if (!blockingOverlay) return null;

    const text = (blockingOverlay.textContent || '').toLowerCase();

    // Classify the popup.  Order matters — check most specific first.
    // CAPTCHA before RATE_LIMIT (captcha is higher priority).
    if (text.includes('captcha') || text.includes('are you human') ||
        text.includes('verify you are human') || text.includes('robot')) {
        return { type: 'CAPTCHA_MODAL' };
    }
    if (text.includes('cookie') || text.includes('consent') ||
        text.includes('gdpr') || text.includes('privacy policy')) {
        return { type: 'COOKIE_BANNER' };
    }
    if (text.includes('rate limit') || text.includes('too many request') ||
        text.includes('slow down') || text.includes('try again later') ||
        text.includes('usage limit')) {
        return { type: 'RATE_LIMIT_MODAL' };
    }
    if (text.includes('newsletter') || text.includes('subscribe') ||
        text.includes('sign up for') || text.includes('mailing list')) {
        return { type: 'NEWSLETTER_MODAL' };
    }
    if (text.includes('upgrade') || text.includes('premium') ||
        text.includes('pro plan') || text.includes('pricing')) {
        return { type: 'UPGRADE_MODAL' };
    }

    return { type: 'UNKNOWN_MODAL' };
}"""


async def detect_popup(page) -> tuple[PopupType, str]:
    """Detect if a popup/overlay is blocking the chat input.

    Returns (popup_type, description).
    """
    try:
        result = await page.evaluate(_DETECT_POPUP_SCRIPT)
        if result is None:
            return PopupType.NONE, ""
        popup_type_str = result.get("type", "UNKNOWN_MODAL")
        try:
            popup_type = PopupType[popup_type_str]
        except KeyError:
            popup_type = PopupType.UNKNOWN_MODAL
        return popup_type, popup_type_str
    except Exception as exc:
        log.debug("Popup detection failed: %s", exc)
        return PopupType.NONE, ""


async def dismiss_popup(page, popup_type: PopupType) -> bool:
    """Attempt to dismiss a detected popup.

    Returns True if we believe the popup was dismissed.

    Rules:
      - COOKIE_BANNER: click "Accept" / "Allow" / close button.
      - NEWSLETTER_MODAL / UPGRADE_MODAL: click "No thanks" / close.
      - RATE_LIMIT_MODAL: log and pause — do NOT dismiss.
      - CAPTCHA_MODAL: log and pause — do NOT dismiss or solve.
      - UNKNOWN_MODAL: screenshot + log + try close. Don't randomly click.
    """
    if popup_type in (PopupType.RATE_LIMIT_MODAL, PopupType.CAPTCHA_MODAL):
        log.warning(
            "[PopupManager] %s detected — pausing. Do NOT auto-dismiss.",
            popup_type.name,
        )
        return False

    # Strategy 1: For cookie banners, try accept buttons first.
    if popup_type == PopupType.COOKIE_BANNER:
        for text in _COOKIE_ACCEPT_TEXT:
            try:
                btn = page.locator(f'button:has-text("{text}")').first
                if await btn.is_visible(timeout=500):
                    await btn.click()
                    log.info("[PopupManager] Cookie banner dismissed via '%s'", text)
                    return True
            except Exception:
                continue

    # Strategy 2: For newsletter/upgrade, try dismiss text first.
    if popup_type in (PopupType.NEWSLETTER_MODAL, PopupType.UPGRADE_MODAL):
        for text in _DISMISS_TEXT:
            try:
                btn = page.locator(f'button:has-text("{text}")').first
                if await btn.is_visible(timeout=500):
                    await btn.click()
                    log.info("[PopupManager] %s dismissed via '%s'", popup_type.name, text)
                    return True
            except Exception:
                continue

    # Strategy 3: Try close buttons (all popup types).
    for sel in _CLOSE_BUTTON_SELECTORS:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=500):
                await btn.click()
                log.info("[PopupManager] %s dismissed via close button '%s'", popup_type.name, sel)
                return True
        except Exception:
            continue

    # Strategy 4: Press Escape.
    try:
        await page.keyboard.press("Escape")
        log.info("[PopupManager] %s — tried Escape key", popup_type.name)
        return True  # Optimistic; caller should re-check.
    except Exception:
        pass

    if popup_type == PopupType.UNKNOWN_MODAL:
        log.warning(
            "[PopupManager] UNKNOWN_MODAL could not be dismissed. "
            "Take screenshot and investigate.",
        )

    return False


async def handle_popups(page) -> bool:
    """Detect and auto-dismiss popups. Returns True if a popup was handled.

    Call this every tick during readiness polling.
    """
    popup_type, desc = await detect_popup(page)
    if popup_type == PopupType.NONE:
        return False

    log.info("[PopupManager] Detected %s", popup_type.name)
    dismissed = await dismiss_popup(page, popup_type)
    if dismissed:
        # Small delay to let the UI settle after dismissal.
        await asyncio.sleep(0.5)
    return dismissed
