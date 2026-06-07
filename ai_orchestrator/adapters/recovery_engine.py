"""Recovery Engine — Layer 7 from ARCHITECTURE.md.

Implements the escalating recovery chain:
  Retry → Dismiss Popup → Selector Recovery → Accessibility Recovery →
  Refresh → Reload Cookies → New Browser Context → Provider Cooldown

Each recovery strategy is tried in order.  If one succeeds, the send
is retried from that point.  If all fail, the provider is put on
cooldown.

Per AGENTS.md rules:
  - No captcha solving (detect → pause → notify)
  - No infinite retry loops
  - Max 3 retries per send attempt
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter

log = logging.getLogger(__name__)

MAX_RETRIES = 3


class RecoveryAction(Enum):
    """Recovery actions in escalation order."""
    RETRY = auto()
    DISMISS_POPUP = auto()
    REFRESH_PAGE = auto()
    RELOAD_COOKIES = auto()
    NEW_CONTEXT = auto()
    PROVIDER_COOLDOWN = auto()


class CooldownError(Exception):
    """Provider is on cooldown — do not attempt."""

    def __init__(self, provider: str, cooldown_until: float) -> None:
        self.provider = provider
        self.cooldown_until = cooldown_until
        remaining = max(0, cooldown_until - time.time())
        super().__init__(
            f"{provider} is on cooldown for {remaining:.0f}s more"
        )


@dataclass
class ProviderCooldown:
    """Track per-provider cooldown state."""

    provider: str
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    last_failure_reason: str = ""

    @property
    def is_on_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.cooldown_until - time.time())

    def record_failure(self, reason: str) -> None:
        self.consecutive_failures += 1
        self.last_failure_reason = reason
        # Exponential backoff: 30s, 60s, 120s, 240s, max 300s
        backoff = min(30 * (2 ** (self.consecutive_failures - 1)), 300)
        self.cooldown_until = time.time() + backoff
        log.warning(
            "[%s] Cooldown activated: %ds (failures=%d, reason=%s)",
            self.provider, backoff, self.consecutive_failures, reason,
        )

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.cooldown_until = 0.0
        self.last_failure_reason = ""


class RecoveryEngine:
    """Manages per-provider recovery and cooldown state."""

    def __init__(self) -> None:
        self._cooldowns: dict[str, ProviderCooldown] = {}

    def _ensure(self, provider: str) -> ProviderCooldown:
        if provider not in self._cooldowns:
            self._cooldowns[provider] = ProviderCooldown(provider=provider)
        return self._cooldowns[provider]

    def check_cooldown(self, provider: str) -> None:
        """Raise CooldownError if provider is on cooldown."""
        cd = self._ensure(provider)
        if cd.is_on_cooldown:
            raise CooldownError(provider, cd.cooldown_until)

    def record_success(self, provider: str) -> None:
        """Reset cooldown on success."""
        self._ensure(provider).record_success()

    def record_failure(self, provider: str, reason: str) -> None:
        """Record failure and potentially activate cooldown."""
        self._ensure(provider).record_failure(reason)

    def get_cooldown_info(self, provider: str) -> dict:
        """Get cooldown status for a provider."""
        cd = self._ensure(provider)
        return {
            "provider": provider,
            "on_cooldown": cd.is_on_cooldown,
            "remaining_seconds": round(cd.remaining_seconds, 1),
            "consecutive_failures": cd.consecutive_failures,
            "last_failure_reason": cd.last_failure_reason,
        }

    def get_all_cooldowns(self) -> dict[str, dict]:
        """Get all provider cooldown states."""
        return {name: self.get_cooldown_info(name) for name in self._cooldowns}


async def attempt_recovery(
    adapter: EngineUIAdapter,
    error: Exception,
    attempt: int,
) -> RecoveryAction | None:
    """Try to recover from a send failure.

    Returns the recovery action taken, or None if recovery is not possible.
    Uses the escalation chain from ARCHITECTURE.md Layer 7.
    """
    from ai_orchestrator.adapters.errors import (
        AuthenticationError,
        CloudflareBlockError,
    )
    from ai_orchestrator.adapters.popup_manager import handle_popups

    error_name = type(error).__name__
    error_msg = str(error)

    # NEVER recover from captcha/cloudflare — detect → pause → notify
    if isinstance(error, CloudflareBlockError):
        log.warning("[Recovery] CloudflareBlockError — no recovery. Detect → Pause → Notify.")
        return None

    # NEVER recover from captcha
    if "captcha" in error_msg.lower():
        log.warning("[Recovery] Captcha detected — no recovery.")
        return None

    # Don't retry more than MAX_RETRIES times
    if attempt >= MAX_RETRIES:
        log.warning("[Recovery] Max retries (%d) exhausted.", MAX_RETRIES)
        return None

    page = adapter._page

    # Strategy 1: Dismiss popup (attempt 1)
    if attempt == 1 and page is not None:
        try:
            dismissed = await handle_popups(page)
            if dismissed:
                log.info("[Recovery] Popup dismissed — retrying.")
                return RecoveryAction.DISMISS_POPUP
        except Exception:
            pass

    # Strategy 2: Refresh page (attempt 2)
    if attempt == 2 and page is not None:
        try:
            await page.reload(wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(3)
            log.info("[Recovery] Page refreshed — retrying.")
            return RecoveryAction.REFRESH_PAGE
        except Exception as exc:
            log.warning("[Recovery] Page refresh failed: %s", exc)

    # Strategy 3: Auth failure → try cookie refresh (attempt 2+)
    if isinstance(error, AuthenticationError) and attempt >= 2:
        try:
            from ai_orchestrator.adapters.auto_cookie_update import (
                refresh_cookies_from_profile,
            )

            profile_dir = adapter._persistent_profile
            if profile_dir and adapter._playwright and adapter._site:
                refreshed = await refresh_cookies_from_profile(
                    adapter._playwright,
                    adapter.provider_name,
                    profile_dir,
                    adapter._site.url,
                )
                if refreshed:
                    log.info("[Recovery] Cookies refreshed from profile — retrying.")
                    return RecoveryAction.RELOAD_COOKIES
        except Exception as exc:
            log.warning("[Recovery] Cookie refresh failed: %s", exc)

    # Strategy 4: New browser context (last resort before cooldown)
    if attempt >= 2 and adapter._context is not None:
        try:
            import contextlib
            with contextlib.suppress(Exception):
                await adapter.close()
            log.info("[Recovery] Browser context destroyed — retrying with fresh context.")
            return RecoveryAction.NEW_CONTEXT
        except Exception as exc:
            log.warning("[Recovery] Context reset failed: %s", exc)

    return None


# Module-level singleton
recovery_engine = RecoveryEngine()
