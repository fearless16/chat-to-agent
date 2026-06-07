"""Cookie validation — pre-flight and post-navigation checks.

Per AGENTS.md rule #7:
  Before launching provider:
    cookie file exists → cookie parseable → cookie count > 0
  After navigation:
    authenticated?

Cookie load success ≠ auth success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# URL path segments that indicate a login/auth page.
# Use full path segments (with trailing slash or end-of-path check)
# to avoid false-positives on e.g. /auth/callback → /authorize.
AUTH_PATH_SEGMENTS: tuple[str, ...] = (
    "/sign_in",
    "/signin",
    "/login",
    "/auth/",        # Only match /auth/ as a directory, not /authorize
    "/log-in",
    "/sign-up",
    "/signup",
    "/register",
    "/oauth/",
    "/sso/",
)

# Page title substrings that indicate Cloudflare challenge.
CLOUDFLARE_TITLE_MARKERS: tuple[str, ...] = (
    "just a moment",
    "checking your browser",
    "verify you are human",
    "attention required",
    "one more step",
)

# Page content markers for Cloudflare / captcha.
CLOUDFLARE_CONTENT_MARKERS: tuple[str, ...] = (
    "turnstile",
    "cf-browser-verification",
    "challenge-platform",
    "cf_chl_opt",
    "ray id",
)


@dataclass
class CookieValidationResult:
    """Result of pre-flight cookie validation."""

    file_exists: bool = False
    parseable: bool = False
    cookie_count: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def is_valid(self) -> bool:
        return self.file_exists and self.parseable and self.cookie_count > 0


def validate_cookie_file(cookie_path: str | Path | None) -> CookieValidationResult:
    """Validate a Netscape cookie file before launching browser.

    Returns a CookieValidationResult with details on what passed/failed.
    """
    result = CookieValidationResult()

    if cookie_path is None:
        result.errors.append("No cookie path provided")
        return result

    path = Path(cookie_path)
    if not path.exists():
        result.errors.append(f"Cookie file not found: {path}")
        return result
    result.file_exists = True

    try:
        text = path.read_text()
    except Exception as exc:
        result.errors.append(f"Cannot read cookie file: {exc}")
        return result

    # Count actual cookie lines (skip comments and blank lines)
    cookie_count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            cookie_count += 1

    result.parseable = True
    result.cookie_count = cookie_count

    if cookie_count == 0:
        result.errors.append("Cookie file contains zero cookies")

    return result


def validate_storage_state(storage_state: dict | None) -> CookieValidationResult:
    """Validate a Playwright storage_state dict (already parsed)."""
    result = CookieValidationResult()

    if storage_state is None:
        result.errors.append("No storage state provided")
        return result

    result.file_exists = True  # It was loaded from somewhere
    result.parseable = True

    cookies = storage_state.get("cookies", [])
    result.cookie_count = len(cookies) if isinstance(cookies, list) else 0

    if result.cookie_count == 0:
        result.errors.append("Storage state contains zero cookies")

    return result


async def check_post_navigation_auth(page) -> tuple[bool, str]:
    """Check if the page landed on a login/auth page after navigation.

    Returns (is_authenticated, reason).
    True means we're NOT on a login page (good).
    False means we ARE on a login page (bad).

    Note: Cloudflare detection is handled separately by
    ``check_cloudflare_challenge()`` which should be called first.
    """
    try:
        url = page.url.lower()
        # Check URL for auth path segments
        for segment in AUTH_PATH_SEGMENTS:
            if segment in url:
                return False, f"URL contains auth path: {segment} (url={page.url})"

        # Check DOM for password input (strong login indicator)
        has_password = await page.evaluate("""() => {
            return !!document.querySelector('input[type="password"]');
        }""")
        if has_password:
            return False, "Page has a password input field — likely a login page"

        return True, "ok"
    except Exception as exc:
        log.warning("Post-navigation auth check failed: %s", exc)
        # If we can't check, assume ok to avoid blocking
        return True, f"check failed: {exc}"


async def check_cloudflare_challenge(page) -> tuple[bool, str]:
    """Check if the page is showing a Cloudflare challenge.

    Returns (is_blocked, reason).
    True means Cloudflare is blocking us.
    """
    try:
        title = (await page.title() or "").lower()
        for marker in CLOUDFLARE_TITLE_MARKERS:
            if marker in title:
                return True, f"Cloudflare title: '{title}'"

        # Check page content for Cloudflare markers
        has_cf = await page.evaluate("""() => {
            const html = document.documentElement.innerHTML.toLowerCase();
            const markers = ['turnstile', 'cf-browser-verification',
                            'challenge-platform', 'cf_chl_opt'];
            return markers.some(m => html.includes(m));
        }""")
        if has_cf:
            return True, "Cloudflare challenge elements detected in DOM"

        return False, "ok"
    except Exception as exc:
        log.warning("Cloudflare check failed: %s", exc)
        return False, f"check failed: {exc}"
