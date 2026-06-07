"""Provider-specific error hierarchy.

Clear, actionable exceptions so callers know *why* a provider failed —
not just that it timed out.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base for all provider-level failures."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class AuthenticationError(ProviderError):
    """Cookies expired, missing, or the provider redirected to a login page."""

    def __init__(self, provider: str, message: str, *, url: str = "") -> None:
        self.url = url
        super().__init__(provider, message)


class CloudflareBlockError(ProviderError):
    """Cloudflare / CAPTCHA challenge detected.

    Per ARCHITECTURE.md: detect → pause → notify.
    Do NOT retry, do NOT refresh, do NOT attempt to solve.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        screenshot_path: str = "",
        page_title: str = "",
    ) -> None:
        self.screenshot_path = screenshot_path
        self.page_title = page_title
        super().__init__(provider, message)


class ResponseExtractionError(ProviderError):
    """Generation appeared to complete but no valid response text was captured.

    Success is NOT ``{}``, NOT ``{"ResultObject": true}``,
    NOT an empty string.  It must be actual assistant response text.
    """

    def __init__(self, provider: str, message: str, *, raw: str = "") -> None:
        self.raw = raw
        super().__init__(provider, message)


class CookieValidationError(ProviderError):
    """Pre-flight cookie validation failed.

    Cookie file missing, unparseable, or contains zero cookies.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message)
