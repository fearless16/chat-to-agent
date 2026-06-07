"""Auto Cookie Update — save browser cookies back to disk after success.

After a successful navigation + authentication check, extract the
current cookies from the Playwright browser context and write them
back to the Netscape cookie file on disk.  This keeps cookie files
fresh automatically — no manual re-export needed.

Also provides a utility to refresh cookies from a persistent browser
profile when the current ones expire.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _cookies_to_netscape(cookies: list[dict], domain_hint: str = "") -> str:
    """Convert Playwright cookie dicts to Netscape cookie file format."""
    lines = [
        "# Netscape HTTP Cookie File",
        "# Auto-exported by chat-to-agent",
        f"# Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for c in cookies:
        domain = c.get("domain", domain_hint)
        # Netscape format: domain  flag  path  secure  expiry  name  value
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expiry = str(int(c.get("expires", 0)))
        name = c.get("name", "")
        value = c.get("value", "")
        if not name:
            continue
        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
    lines.append("")
    return "\n".join(lines)


_SAMESITE_NORMALIZE = {
    "unspecified": "None",
    "no_restriction": "None",
    "lax": "Lax",
    "strict": "Strict",
    "none": "None",
}


def _normalize_samesite(ss: str) -> str:
    return _SAMESITE_NORMALIZE.get(ss.lower(), ss)


def _cookies_to_storage_state(cookies: list[dict]) -> dict:
    """Convert Playwright cookie list to a storage_state dict.
    Normalizes sameSite values to Playwright-accepted values.
    """
    for c in cookies:
        ss = c.get("sameSite", "")
        if ss:
            c["sameSite"] = _normalize_samesite(ss)
    return {"cookies": cookies, "origins": []}


async def save_cookies_from_context(
    context,
    provider: str,
    *,
    cookie_dir: str | Path = "profiles",
) -> bool:
    """Extract cookies from a live browser context and save to disk.

    Saves in both formats:
      - Netscape: ``profiles/<provider>_cookies.txt``
      - Playwright JSON: ``profiles/<provider>_auth.json``

    Returns True if cookies were saved successfully.
    """
    try:
        cookies = await context.cookies()
        if not cookies:
            log.warning("[%s] No cookies to save from browser context", provider)
            return False

        cookie_dir = Path(cookie_dir)
        cookie_dir.mkdir(parents=True, exist_ok=True)

        # Save Netscape format (must match _load_auth_for naming: {provider}_cookies.txt)
        netscape_path = cookie_dir / f"{provider}_cookies.txt"
        netscape_text = _cookies_to_netscape(cookies)
        netscape_path.write_text(netscape_text)
        log.info(
            "[%s] Saved %d cookies to %s",
            provider, len(cookies), netscape_path,
        )

        # Save Playwright storage_state JSON
        auth_path = cookie_dir / f"{provider}_auth.json"
        storage = _cookies_to_storage_state(cookies)
        auth_path.write_text(json.dumps(storage, indent=2))

        return True
    except Exception as exc:
        log.warning("[%s] Failed to save cookies: %s", provider, exc)
        return False


async def refresh_cookies_from_profile(
    playwright,
    provider: str,
    profile_dir: str | Path,
    target_url: str,
    *,
    cookie_dir: str | Path = "profiles",
    headless: bool = False,
) -> bool:
    """Launch a persistent browser profile, navigate to the provider URL,
    and extract fresh cookies.

    This is the nuclear option when saved cookies expire — it uses the
    user's actual browser profile (with saved logins, IndexedDB, etc.)
    to get fresh cookies without manual intervention.

    Returns True if fresh cookies were extracted and saved.
    """
    context = None
    try:
        log.info("[%s] Refreshing cookies from profile: %s", provider, profile_dir)

        context = await playwright.chromium.launch_persistent_context(
            str(profile_dir),
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )

        page = context.pages[0] if context.pages else await context.new_page()

        await page.goto(target_url, timeout=30_000, wait_until="domcontentloaded")
        # Wait for SPA hydration
        import asyncio
        await asyncio.sleep(5)

        # Check if we actually authenticated
        from ai_orchestrator.adapters.cookie_validator import (
            check_cloudflare_challenge,
            check_post_navigation_auth,
        )

        is_cf, _ = await check_cloudflare_challenge(page)
        if is_cf:
            log.warning("[%s] Cloudflare challenge during cookie refresh", provider)
            return False

        is_auth, reason = await check_post_navigation_auth(page)
        if not is_auth:
            log.warning("[%s] Not authenticated during cookie refresh: %s", provider, reason)
            return False

        # Extract and save cookies
        saved = await save_cookies_from_context(context, provider, cookie_dir=cookie_dir)
        if saved:
            log.info("[%s] Cookie refresh successful", provider)
        return saved

    except Exception as exc:
        log.error("[%s] Cookie refresh failed: %s", provider, exc)
        return False
    finally:
        if context is not None:
            import contextlib
            with contextlib.suppress(Exception):
                await context.close()
