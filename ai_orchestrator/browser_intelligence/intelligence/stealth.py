"""Browser Intelligence — stealth hardening.

Reduces the fingerprint surface that Cloudflare / AWS WAF /
DataDome / PerimeterX use to detect headless Chrome.

Each method here is opt-in and idempotent: calling `apply_stealth`
twice is safe. The module is pure-logic: it returns a dict of init
scripts, launch args, and context options for the caller to wire up.
No hardcoded provider name, no I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Init script — installed via `page.add_init_script()` once.
# Exposes a single global `__bis_stealth_v1` for tests to verify.
# ──────────────────────────────────────────────────────────────────────

_STEALTH_INIT_SCRIPT: str = r"""
(() => {
  if (window.__bis_stealth_v1) return;
  Object.defineProperty(window, '__bis_stealth_v1', {
    value: { installed_at: Date.now() },
    writable: false,
    enumerable: false,
    configurable: false,
  });

  // ── WebGL vendor / renderer spoof ────────────────────────────
  const spoofWebGL = (proto, vendorKey, rendererKey, vendor, renderer) => {
    const getter = proto && proto.__lookupGetter__ && proto.__lookupGetter__(vendorKey);
    const getter2 = proto && proto.__lookupGetter__ && proto.__lookupGetter__(rendererKey);
    try {
      if (getter) Object.defineProperty(proto, vendorKey, { get: () => vendor, configurable: true });
      if (getter2) Object.defineProperty(proto, rendererKey, { get: () => renderer, configurable: true });
    } catch (e) { /* ignore */ }
  };
  try {
    const ctx2d = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
      const c = ctx2d.call(this, type, ...rest);
      return c;
    };
  } catch (e) { /* ignore */ }

  // ── Canvas fingerprint noise ─────────────────────────────────
  try {
    const toDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function (...a) {
      const c = this.getContext('2d');
      if (c) {
        const noise = (window.__bis_stealth_seed || 0) % 256;
        c.globalAlpha = 0.0001;
        c.fillStyle = `rgb(${noise},${noise},${noise})`;
        c.fillRect(0, 0, 1, 1);
        c.globalAlpha = 1.0;
      }
      return toDataURL.apply(this, a);
    };
  } catch (e) { /* ignore */ }

  // ── Audio fingerprint noise ─────────────────────────────────
  try {
    const getChannelData = AudioBuffer.prototype.getChannelData;
    AudioBuffer.prototype.getChannelData = function (ch) {
      const data = getChannelData.call(this, ch);
      const noise = (window.__bis_stealth_seed || 0) / 1e6;
      for (let i = 0; i < data.length; i += 17) {
        data[i] = data[i] + noise;
      }
      return data;
    };
  } catch (e) { /* ignore */ }

  // ── Hardware concurrency spoof ───────────────────────────────
  try {
    Object.defineProperty(navigator, 'hardwareConcurrency', {
      get: () => window.__bis_hardware_concurrency || navigator.hardwareConcurrency,
      configurable: true,
    });
  } catch (e) { /* ignore */ }

  // ── Plugin / mimeTypes flattening ───────────────────────────
  try {
    Object.defineProperty(navigator, 'plugins', {
      get: () => [
        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', length: 1 },
        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', length: 1 },
        { name: 'Native Client', filename: 'internal-nacl-plugin', length: 2 },
      ],
      configurable: true,
    });
  } catch (e) { /* ignore */ }

  // ── webdriver flag stripping ────────────────────────────────
  try {
    Object.defineProperty(navigator, 'webdriver', { get: () => false, configurable: true });
  } catch (e) { /* ignore */ }

  // ── permissions API signature fix ───────────────────────────
  try {
    const origQuery = navigator.permissions && navigator.permissions.query;
    if (origQuery) {
      navigator.permissions.query = (p) => {
        if (p && p.name === 'notifications') {
          return Promise.resolve({ state: Notification.permission });
        }
        return origQuery.call(navigator.permissions, p);
      };
    }
  } catch (e) { /* ignore */ }
})();
"""


# ──────────────────────────────────────────────────────────────────────
# Configurable stealth profile
# ──────────────────────────────────────────────────────────────────────

# Pools of plausible vendor / renderer strings. Real-world GPUs only —
# we don't try to claim a 4090 in a headless container.
_GPU_PROFILES: tuple[tuple[str, str], ...] = (
    ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0)"),
    ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce GTX 1660 SUPER Direct3D11 vs_5_0 ps_5_0)"),
    ("Google Inc. (Intel)", "ANGLE (Intel, Intel(R) UHD Graphics 630 (CFL GT2), OpenGL 4.5)"),
    ("Google Inc. (AMD)", "ANGLE (AMD, AMD Radeon RX 580 (POLARIS10), OpenGL 4.5)"),
    ("Google Inc. (Intel)", "ANGLE (Intel, Intel(R) Iris Xe Graphics (0x46A6), OpenGL 4.5)"),
    ("Apple Inc.", "Apple M1"),
    ("Apple Inc.", "Apple M2"),
    ("Google Inc. (Qualcomm)", "ANGLE (Qualcomm, Adreno (TM) 640, OpenGL ES 3.2)"),
)

_LANGUAGES: tuple[tuple[str, ...], ...] = (
    ("en-US", "en"),
    ("en-GB", "en"),
    ("en-CA", "en"),
    ("de-DE", "de", "en"),
    ("fr-FR", "fr", "en"),
    ("ja-JP", "ja", "en"),
    ("zh-CN", "zh", "en"),
)

_TIMEZONES: tuple[str, ...] = (
    "America/Los_Angeles",
    "America/New_York",
    "America/Chicago",
    "Europe/London",
    "Europe/Berlin",
    "Europe/Paris",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Australia/Sydney",
)


@dataclass
class StealthProfile:
    """One concrete stealth configuration.

    The profile is deterministic given a seed so tests can pin
    expectations. The same seed → same vendor, renderer, locale,
    timezone, and so on.
    """

    seed: int
    webgl_vendor: str
    webgl_renderer: str
    hardware_concurrency: int
    device_memory: int
    languages: tuple[str, ...]
    timezone: str
    canvas_noise_seed: int
    audio_noise_seed: int
    user_agent: str

    def fingerprint(self) -> str:
        """Stable identifier for the profile — handy for telemetry."""
        h = hashlib.sha256()
        h.update(str(self.seed).encode())
        h.update(self.webgl_vendor.encode())
        h.update(self.webgl_renderer.encode())
        h.update(str(self.hardware_concurrency).encode())
        h.update(",".join(self.languages).encode())
        h.update(self.timezone.encode())
        return h.hexdigest()[:16]


def make_stealth_profile(
    seed: int | None = None,
    *,
    user_agent: str | None = None,
) -> StealthProfile:
    """Construct a stealth profile. Seed=None → random per-process."""
    if seed is None:
        seed = int(time.time()) ^ random.randint(0, 2**16)
    rng = random.Random(seed)
    vendor, renderer = rng.choice(_GPU_PROFILES)
    languages = rng.choice(_LANGUAGES)
    tz = rng.choice(_TIMEZONES)
    hardware_concurrency = rng.choice([4, 8, 12, 16])
    device_memory = rng.choice([4, 8, 16, 32])
    canvas_seed = rng.randint(1, 255)
    audio_seed = rng.randint(1, 1000)
    if user_agent is None:
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        )
    return StealthProfile(
        seed=seed,
        webgl_vendor=vendor,
        webgl_renderer=renderer,
        hardware_concurrency=hardware_concurrency,
        device_memory=device_memory,
        languages=languages,
        timezone=tz,
        canvas_noise_seed=canvas_seed,
        audio_noise_seed=audio_seed,
        user_agent=user_agent,
    )


# ──────────────────────────────────────────────────────────────────────
# Launch / context options
# ──────────────────────────────────────────────────────────────────────

_BASE_LAUNCH_ARGS: tuple[str, ...] = (
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process,AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--no-zygote",
    "--disable-setuid-sandbox",
    "--disable-accelerated-2d-canvas",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-infobars",
    "--disable-breakpad",
    "--disable-component-update",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--disable-default-apps",
    "--disable-extensions",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-sync",
    "--metrics-recording-only",
    "--mute-audio",
    "--use-mock-keychain",
    "--enable-automation=false",
    "--password-store=basic",
    "--lang=en-US,en",
)


def stealth_launch_args() -> list[str]:
    return list(_BASE_LAUNCH_ARGS)


def stealth_context_options(
    profile: StealthProfile,
    *,
    proxy: dict | None = None,
) -> dict[str, Any]:
    """Return a kwargs dict suitable for `browser.new_context(**opts)`.
    Includes locale, timezone, user agent, viewport, and
    hardware-concurrency-shaped fields."""
    opts: dict[str, Any] = {
        "user_agent": profile.user_agent,
        "locale": profile.languages[0] if profile.languages else "en-US",
        "timezone_id": profile.timezone,
        "viewport": {"width": 1280, "height": 800},
        "device_scale_factor": 1,
        "is_mobile": False,
        "has_touch": False,
        "color_scheme": "light",
        "accept_downloads": False,
        "java_script_enabled": True,
        "extra_http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": ",".join(profile.languages) + f";q=0.9,en;q=0.8",
            "sec-ch-ua": '"Chromium";v="130", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        },
    }
    if proxy:
        opts["proxy"] = proxy
    return opts


def stealth_init_script(profile: StealthProfile) -> str:
    """Return the JS init script customized for this profile."""
    return (
        "(window.__bis_stealth_seed = " + str(profile.canvas_noise_seed) + ");"
        + "(window.__bis_hardware_concurrency = " + str(profile.hardware_concurrency) + ");"
        + "(window.__bis_stealth_profile_fp = '" + profile.fingerprint() + "');"
        + _STEALTH_INIT_SCRIPT
    )


@dataclass
class StealthApplication:
    profile: StealthProfile
    launch_args: list[str] = field(default_factory=stealth_launch_args)
    context_options: dict[str, Any] = field(default_factory=dict)
    init_script: str = ""

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.profile.fingerprint(),
            "launch_args_count": len(self.launch_args),
            "context_keys": sorted(self.context_options.keys()),
            "init_script_len": len(self.init_script),
        }


def apply_stealth(
    seed: int | None = None,
    *,
    user_agent: str | None = None,
    proxy: dict | None = None,
) -> StealthApplication:
    """One-shot helper: build a profile + the launch options.

    The caller wires these into the actual Playwright calls; this
    module never imports playwright itself (so it stays unit-testable
    without a browser).
    """
    profile = make_stealth_profile(seed=seed, user_agent=user_agent)
    return StealthApplication(
        profile=profile,
        launch_args=stealth_launch_args(),
        context_options=stealth_context_options(profile, proxy=proxy),
        init_script=stealth_init_script(profile),
    )


__all__ = [
    "StealthProfile",
    "StealthApplication",
    "make_stealth_profile",
    "stealth_launch_args",
    "stealth_context_options",
    "stealth_init_script",
    "apply_stealth",
]
