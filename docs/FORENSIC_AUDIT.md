# CHAT-TO-AGENT FORENSIC CODE AUDIT

> Staff Engineer Technical Handoff | 2026-06-07
> Repository: `github.com/fearless16/chat-to-agent`

---

# AUTH FLOW — Full Trace

## From `POST /chat` to authenticated page detection

```
POST /chat  {provider:"chatgpt_ui", prompt:"Hello", mock_mode:false}
│
├─ orchestrator/main.py:216  _build_adapter("chatgpt_ui", mock_mode=False)
│   │
│   ├─ main.py:230  cls = _PROVIDER_CLASS_MAP["chatgpt_ui"]  → ChatGPTUIAdapter
│   │
│   ├─ main.py:242-248  Auth resolution:
│   │   ├─ profile_dir = _PERSISTENT_PROFILE_MAP["chatgpt_ui"]
│   │   │   └─ "chatgpt_browser_profile"  — CHECK IF EXISTS ON DISK
│   │   ├─ IF exists: persistent_profile = "chatgpt_browser_profile"
│   │   ├─ ELSE: _load_auth_for("chatgpt_ui")
│   │   │   ├─ Tries: profiles/chatgpt_ui_cookies.txt
│   │   │   ├─ Tries: profiles/chatgpt_cookies.txt
│   │   │   ├─ Tries: chatgpt_ui_auth.json
│   │   │   └─ Tries: chatgpt_auth.json  → netscape_cookies_to_storage_state()
│   │   │       └─ cookie_to_storage_state.py:75  sameSite hardcoded to "Lax" FOR ALL COOKIES
│   │   └─ channel = "firefox" (Cloudflare bypass attempt)
│   │
│   └─ main.py:251-258  ChatGPTUIAdapter(
│         headless=False, stealth=True, timeout_ms=120_000,
│         storage_state=storage_state,   ← Playwright storage_state dict
│         persistent_profile=persistent_profile,  ← OR string path
│         channel="firefox"
│       )
│
├─ adapter.send("Hello", context=None)
│   └─ engine_adapter.py:344  EngineUIAdapter.send()
│       ├─ engine_adapter.py:352  recovery_engine.check_cooldown("chatgpt_ui")
│       ├─ engine_adapter.py:363  adapter._real_send("Hello")
│       │
│       └─ engine_adapter.py:562  _real_send()
│           ├─ engine_adapter.py:569-576  PRE-FLIGHT: validate_storage_state(storage_state)
│           │   └─ cookie_validator.py:115  validate_storage_state()
│           │       └─ Returns CookieValidationResult(is_valid=bool)
│           │       └─ NOTE: Failure is LOGGED but NOT BLOCKING. Carries on anyway.
│           │
│           ├─ engine_adapter.py:580  page = await self._get_page()
│           │   └─ engine_adapter.py:1257  _get_page()
│           │       │
│           │       ├─ FIREFOX PATH (chatgpt_ui: channel="firefox")
│           │       │   ├─ engine_adapter.py:1265  async_playwright().start()
│           │       │   ├─ engine_adapter.py:1266-1279  IF persistent_profile:
│           │       │   │   └─ launch_persistent_context(user_data_dir=profile_path, headless=False)
│           │       │   │       └─ NOTE: Firefox persistent context DOES NOT support `storage_state`
│           │       │   │           The cookies are in the browser profile directory.
│           │       │   │
│           │       │   ├─ engine_adapter.py:1280-1290  ELSE (no persistent profile):
│           │       │   │   ├─ firefox.launch(headless=False)  ← NO LAUNCH ARGS!
│           │       │   │   │   └─ BUG: --disable-blink-features=AutomationControlled not applied
│           │       │   │   ├─ new_context(viewport={...})
│           │       │   │   │   └─ NOTE: Firefox new_context() DOES NOT accept `user_agent` param here
│           │       │   │   ├─ ctx_opts["storage_state"] = self._storage_state  ← cookies injected
│           │       │   │   └─ new_page()
│           │       │   ├─ engine_adapter.py:1291  add_init_script(FETCH_INTERCEPT_SCRIPT)
│           │       │   ├─ engine_adapter.py:1292  page.goto(url, timeout=120s, wait_until="domcontentloaded")
│           │       │   ├─ engine_adapter.py:1295  asyncio.sleep(3)
│           │       │   ├─ engine_adapter.py:1300  check_cloudflare_challenge(page)
│           │       │   │   └─ cookie_validator.py:166-191 → checks title + innerHTML for CF markers
│           │       │   └─ engine_adapter.py:1309  check_post_navigation_auth(page)
│           │       │       └─ cookie_validator.py:135-163 → checks URL + password inputs
│           │       │
│           │       └─ CHROMIUM PATH (all other providers)
│           │           ├─ engine_adapter.py:1320-1323  launch_args = ["--disable-blink-features=AutomationControlled", ...]
│           │           ├─ engine_adapter.py:1325-1344  SHARED BROWSER check
│           │           ├─ engine_adapter.py:1349-1365  CHROME PROFILE BLOCK:
│           │           │   └─ if chrome_user_data_dir.exists() and persistent_profile is None and storage_state is None and False:
│           │           │       └─ BUG: `and False` — THIS CODE IS DEAD. The entire real-Chrome-profile
│           │           │           path can NEVER execute because of the literal `False` guard.
│           │           ├─ engine_adapter.py:1367-1391  PERSISTENT PROFILE PATH
│           │           │   ├─ IF stealth: user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)..."
│           │           │   └─ launch_persistent_context(user_data_dir=profile_path, ...)
│           │           │       └─ NOTE: persistent context CANNOT take storage_state (cookies in profile dir)
│           │           └─ engine_adapter.py:1392-1413  STANDARD LAUNCH PATH
│           │               ├─ chromium.launch(headless=..., args=launch_args, channel=self._channel)
│           │               ├─ context_opts (viewport, user_agent, storage_state)
│           │               ├─ browser.new_context(**context_opts)
│           │               └─ context.new_page()
│           │
│           ├─ engine_adapter.py:593  save_cookies_from_context(context, provider)
│           │   └─ auto_cookie_update.py:70  Writes profiles/*_cookies.txt + profiles/*_auth.json
│           │
│           ├─ engine_adapter.py:596-597  BrowserIntelligenceEngine().attach(page)
│           │   └─ engine.py:369  composer._network.attach(page)
│           │       └─ network_sensor.py:107  page.context.new_cdp_session(page) ← ONLY works on Chromium
│           │       └─ ON FIREFOX: throws exception, logged as "CDP Network attach failed"
│           │           → engine.py:374  self._fusion.record_sensor_failure("network")
│           │           → All CDP-dependent features DISABLED on Firefox
│           │
│           └─ engine_adapter.py:600  _wait_until_ready(engine, page)
│               └─ engine_adapter.py:656  (see AUTH DETECTION below)
│
└─ Response extracted or error returned
```

## AUTH FAILURE POINTS

| # | File:Line | Description | How It Fails |
|---|-----------|-------------|-------------|
| 1 | `main.py:163-197` | `_load_auth_for()` — file resolution | Tries 4 paths. None may exist. Returns `None`. No fallback error. |
| 2 | `main.py:242-248` | Auth priority: profile dir > cookies | If `chatgpt_browser_profile/` exists on disk, uses it INSTEAD of storage_state. If profile is stale, cookies are ignored. |
| 3 | `cookie_to_storage_state.py:75` | `sameSite` forced to `"Lax"` | **All cookies** get `sameSite: "Lax"` regardless of original value. Breaks cross-site cookies that need `None`. |
| 4 | `engine_adapter.py:569-576` | Pre-flight validation non-blocking | `validate_storage_state()` returns `is_valid=False`, but execution continues. Warning-only. |
| 5 | `engine_adapter.py:1281-1283` | Firefox launch — no args | `firefox.launch(headless=False)` — no `--disable-blink-features`, no stealth args. Exposes WebDriver flags. |
| 6 | `engine_adapter.py:1291` | `add_init_script()` on Firefox | `FETCH_INTERCEPT_SCRIPT` patches `window.fetch`, `XMLHttpRequest`, `EventSource` — this is visible fingerprinting to anti-bot systems. |
| 7 | `engine_adapter.py:1285-1288` | Firefox context — missing user_agent | `new_context()` has viewport but NO user_agent override. Uses Playwright's default Firefox UA. |
| 8 | `cookie_validator.py:162-163` | Auth check swallows errors | If `check_post_navigation_auth()` itself fails, returns `(True, "check failed")` — treats error as authenticated. |
| 9 | `cookie_validator.py:1301-1316` | `CloudflareBlockError` prevents retry | Even if cookies ARE valid, CF challenge detection fires first and blocks ALL retries. Per AGENTS.md: "Do NOT retry." |
| 10 | `engine_adapter.py:1350` | Chrome profile guard `and False` | Literally hardcoded `False` — the real Chrome profile path can never execute. Dead code protecting nothing. |

---

# PLAYWRIGHT FLOW — Full Trace

## Browser launch to response extraction

```
engine_adapter.py:1257    _get_page()
│
├── PATH A: Firefox (chatgpt_ui only)
│   ├── line 1265:   async_playwright().start()
│   ├── line 1269:   firefox.launch_persistent_context(**opts)
│   │                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
│   │                OPTS: user_data_dir, headless, viewport
│   │                MISSING: channel (none for Firefox), args
│   │                MISSING: storage_state (not supported with persistent context)
│   │                BUG: Cookies from storage_state are IGNORED
│   │                     Only cookies from the profile directory are loaded
│   │
│   ├── line 1281:   firefox.launch(headless=...)  
│   │                ^^^^^^^^^^^^^^^^^^ NO ARGS
│   │                MISSING: --disable-blink-features=AutomationControlled
│   │                MISSING: All stealth launch args from stealth.py
│   │
│   ├── line 1288:   new_context(viewport={}, storage_state=storage_state)
│   │                ^^^^^^^^^^^^
│   │                MISSING: user_agent, locale, timezone_id
│   │                MISSING: extra_http_headers
│   │                MISSING: All stealth context options from stealth.py
│   │
│   ├── line 1291:   add_init_script(FETCH_INTERCEPT_SCRIPT)
│   │                ^^^^^^^^^^^^^^
│   │                RISK: Intercepts fetch/XHR/EventSource — fingerprinting surface
│   │
│   ├── line 1292:   page.goto(url, timeout=120_000, wait_until="domcontentloaded")
│   └── line 1295:   asyncio.sleep(3)
│
├── PATH B: Chromium (all other providers)
│   ├── line 1346:   async_playwright().start()
│   │
│   ├── line 1349-1365:   DEAD Chrome profile path (guarded by `and False`)
│   │                     ^^^^^^^^^^^^^^^^^^^^^^
│   │
│   ├── line 1367-1391:   Persistent profile path (when _persistent_profile is set)
│   │   ├── context_opts = {channel, viewport, user_agent}  (if _stealth)
│   │   ├── launch_persistent_context(user_data_dir=str(profile_path), ...)
│   │   │   BUG: NO storage_state — cookies come from profile directory
│   │   │   BUG: NO stealth launch args (only the 2 minimal args)
│   │   │   BUG: NO stealth context options (locale, timezone, http_headers)
│   │   └── BUG: launch_args only has 2 items, stealth.py has 28
│   │
│   ├── line 1392-1413:   Standard launch path (most common)
│   │   ├── line 1320:    launch_args = ["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"]
│   │   │                  ^^^^^^^^^^ ONLY 2 ARGS
│   │   │                  MISSING: 26 other stealth args from stealth.py
│   │   ├── line 1393:    chromium.launch(headless=..., args=launch_args, channel=self._channel)
│   │   ├── line 1399-1406: context_opts = {viewport, user_agent}  (if _stealth)
│   │   │                  MISSING: locale, timezone_id, extra_http_headers
│   │   ├── line 1408:    context_opts["storage_state"] = self._storage_state
│   │   ├── line 1411:    browser.new_context(**context_opts)
│   │   └── line 1413:    context.new_page()
│   │
│   ├── line 1415:   add_init_script(FETCH_INTERCEPT_SCRIPT)
│   ├── line 1417:   page.goto(url, timeout=120_000, wait_until="load")
│   └── line 1424:   asyncio.sleep(5)
│
└── Post-navigation (both paths)
    ├── line 1300/1430:   check_cloudflare_challenge(page)
    └── line 1309/1440:   check_post_navigation_auth(page)
```

## SUSPICIOUS PLAYWRIGHT PATTERNS

| # | File:Line | Issue | Impact |
|---|-----------|-------|--------|
| 1 | `engine_adapter.py:1320-1323` | Only **2** launch args. stealth.py defines **28**. None of the stealth args are used. | `navigator.webdriver=true` exposed, `--enable-automation` not disabled |
| 2 | `engine_adapter.py:1281` | Firefox: `launch()` with ZERO args. Not even `--disable-blink-features`. | Firefox reports `navigator.webdriver: true` to every site |
| 3 | `engine_adapter.py:1285-1288` | Firefox: `new_context()` has no `user_agent`. | Uses Playwright's default Firefox UA (different from real Firefox) |
| 4 | `engine_adapter.py:1334-1337` | Shared browser UA: `Chrome/131.0.0.0` on `macOS 10.15.7`. Fixed version. | Version fingerprint inconsistency with actual installed Chrome |
| 5 | `engine_adapter.py:1373-1377` | Persistent profile UA: `Chrome/131.0.0.0`. Same fixed version. | Same version fingerprint issue |
| 6 | `engine_adapter.py:1402-1405` | Standard launch UA: `Chrome/131.0.0.0`. Same fixed version. | Same version fingerprint issue |
| 7 | `engine_adapter.py:1285-1288` | `context_opts` in Firefox: NO `locale`, NO `timezone_id` | Browser reports UTC+0 regardless of actual timezone |
| 8 | `engine_adapter.py:1399-1406` | Chromium context: NO `locale`, NO `timezone_id`, NO `extra_http_headers` | Missing `Accept-Language`, `sec-ch-ua` headers |
| 9 | `engine_adapter.py:1350` | `and False` guard — entire Chrome-user-data path is dead | The one path that would use real browser fingerprints is disabled |
| 10 | `engine_adapter.py:1417-1420` | `wait_until="load"` on Chromium | Waits for ALL resources — slow, but also means headless detection scripts in 3rd-party JS can run |
| 11 | `engine_adapter.py:1292` | `wait_until="domcontentloaded"` on Firefox | Mismatch with Chromium's `"load"` — Firefox navigates faster but misses deferred scripts |

---

# CLOUDFLARE ANALYSIS

## Why Cloudflare detects these browsers

### 1. `navigator.webdriver` NOT removed

| Evidence | File | 
|----------|------|
| `engine_adapter.py:1320` | Only `--disable-blink-features=AutomationControlled` in launch args |
| `stealth.py:109` | `Object.defineProperty(navigator, 'webdriver', { get: () => false })` — EXISTS in stealth_init_script |
| **NON-USE** | `stealth_init_script()` is **NEVER called in production**. Only in tests. |

**Root cause**: The `stealth_init_script()` that sets `navigator.webdriver = false` is defined in `stealth.py:308-315` but **never injected** into any page. `add_init_script()` is only called with `FETCH_INTERCEPT_SCRIPT`, not with `stealth_init_script()`.

`navigator.webdriver === true` — this is the #1 Cloudflare detection signal and it's wide open.

### 2. CDP leak

| Evidence | File |
|----------|------|
| `engine.py:398` | `self._cdp_session = await page.context.new_cdp_session(page)` |
| `engine.py:399` | `await self._cdp_session.send("Network.enable")` |

**Root cause**: Opening a CDP session on the page creates `window.__playwright` and other debugging properties that Cloudflare's JS can detect. There is no mitigation. `--disable-blink-features=AutomationControlled` only blocks the `AutomationControlled` blink feature, not the entire CDP surface.

### 3. WebGL fingerprint inconsistency

| Evidence | File | 
|----------|------|
| `stealth.py:134-143` | `_GPU_PROFILES` — 8 GPU profiles including `Apple M1`, `NVIDIA RTX 3060` |
| `stealth.py:50-56` | WebGL spoof function in `_STEALTH_INIT_SCRIPT` — hooks `getParameter` |
| **NON-USE** | `stealth_init_script()` never injected |

WebGL vendor/renderer are NOT being spoofed. The browser reports whatever Playwright's bundled Chromium reports, which is typically `"Google SwiftShader"` or `"Google Inc."` for headless — a dead giveaway.

### 4. Canvas fingerprint NOT noised

| Evidence | File |
|----------|------|
| `stealth.py:58-72` | `toDataURL` override adds 1px noise |
| **NON-USE** | Script never injected |

Canvas fingerprinting produces identical hashes across all sessions — trivially detectable by Cloudflare's Turnstile JS.

### 5. User-agent inconsistency

| File:Line | UA String | Used Where |
|-----------|-----------|------------|
| `engine_adapter.py:1334-1337` | `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ... Chrome/131.0.0.0 Safari/537.36` | Shared browser context |
| `engine_adapter.py:1374-1377` | Same | Persistent profile context |
| `engine_adapter.py:1402-1405` | Same | Standard launch context |
| `engine_adapter.py:1281-1284` | **NONE** (Playwright default Firefox UA) | Firefox launch |

**Inconsistency**: 
- All Chromium paths use `macOS 10.15.7` + `Chrome/131` — a 2019 macOS version with a 2024 Chrome version
- `sec-ch-ua-platform` not set (stealth.py sets it to `"Windows"`, but stealth is unused)
- Firefox path reports a different UA entirely (Playwright default)
- The UA does NOT match the actual installed Chrome version, creating a fingerprint mismatch with `navigator.userAgentData.getHighEntropyValues()`

### 6. Timezone NOT set

| Evidence | File |
|----------|------|
| `stealth.py:287` | `"timezone_id": profile.timezone` — in `stealth_context_options()` |
| `main.py:254` | `stealth=True` passed to adapter — but this is a boolean, not a stealth application |

None of the browser contexts set `timezone_id`. The browser defaults to whatever the container/host reports. If the host is UTC and the UA claims macOS 10.15.7 (likely America/Los_Angeles), there's a timezone mismatch detectable via `Intl.DateTimeFormat().resolvedOptions().timeZone`.

### 7. `Accept-Language` NOT set

| Evidence | File |
|----------|------|
| `stealth.py:297` | `"Accept-Language": ",".join(profile.languages) + ";q=0.9,en;q=0.8"` — in `stealth_context_options()` |
| `engine_adapter.py:1320-1420` | Nowhere in the actual context options |

Browser sends default headers. Cloudflare checks `Accept-Language` against expected patterns for the claimed locale.

### 8. Session continuity broken

| Evidence | File |
|----------|------|
| `main.py:242-248` | Prefers `persistent_profile` over `storage_state` |
| `engine_adapter.py:1367-1391` | Persistent profile: `launch_persistent_context()` creates NEW browser every call |
| `main.py:202-203` | Only `chatgpt_ui` and `qwen_ui` have persistent profiles mapped |

**Root cause**: 
- Each `send()` call creates a FRESH `BrowserIntelligenceEngine` (line 596: `engine = BrowserIntelligenceEngine()`)
- Each call creates a FRESH browser/page pair
- Cloudflare expects session continuity — the same browser fingerprint across requests
- Browser restart between calls resets all fingerprint entropy

### 9. Plugin / mimeTypes fingerprint

| Evidence | File |
|----------|------|
| `stealth.py:96-105` | Flattens `navigator.plugins` to 3 standard Chrome plugins |
| **NON-USE** | Script never injected |

Playwright's Chromium reports zero plugins. Real Chrome always reports at least Chrome PDF Plugin. `navigator.plugins.length === 0` is a strong headless detection signal.

---

# STEALTH REPORT

## `stealth.py` — defined but NEVER USED in production

### Every call site for stealth functions:

| Function | Call Sites in Production | Call Sites in Tests |
|----------|------------------------|---------------------|
| `apply_stealth()` | **0** | `tests/unit/test_browser_intelligence/test_new_modules.py:683` |
| `make_stealth_profile()` | **0** | `tests/unit/test_browser_intelligence/test_new_modules.py:650` |
| `stealth_launch_args()` | **0** | `tests/unit/test_browser_intelligence/test_new_modules.py:662` |
| `stealth_init_script()` | **0** | `tests/unit/test_browser_intelligence/test_new_modules.py:667` |
| `stealth_context_options()` | **0** | `tests/unit/test_browser_intelligence/test_new_modules.py:674` |
| `StealthApplication` | **0** | `tests/unit/test_browser_intelligence/test_new_modules.py:683` |

### Where stealth IS used:

| File:Line | Usage | What It Does |
|-----------|-------|-------------|
| `main.py:254` | `stealth=True` | Passed as boolean to adapter — only controls `viewport` width + `user_agent` string |
| `engine_adapter.py:320` | `stealth: bool = True` | Constructor param |
| `engine_adapter.py:329` | `self._stealth = stealth` | Stored as instance var |
| `engine_adapter.py:1272` | `if self._stealth:` | Controls viewport size only |
| `engine_adapter.py:1285` | `if self._stealth:` | Controls viewport + user_agent |
| `engine_adapter.py:1332` | `if self._stealth:` | Controls viewport + user_agent |
| `engine_adapter.py:1372` | `if self._stealth:` | Controls viewport + user_agent |
| `engine_adapter.py:1400` | `if self._stealth:` | Controls viewport + user_agent |

### Conclusion: STEALTH IS NOT USED.

The entire `stealth.py` module (363 lines, 8 GPU profiles, WebGL spoofing, canvas noise, audio noise, plugin flattening, timezone, locale, 28 launch args, HTTP headers) is **100% dead code in production**. The `stealth` boolean only controls 2 things: a viewport size and a hardcoded user-agent string.

---

# DEAD CODE REPORT

## 1. `browser_intelligence/intelligence/stealth.py` — ENTIRE MODULE
- **Files**: `stealth.py` (363 lines)
- **Why unused**: `apply_stealth()` never called. Boolean `stealth=True` does NOT invoke stealth module.
- **Evidence**: 0 production call sites for `apply_stealth`, `stealth_init_script`, `stealth_launch_args`, `stealth_context_options`

## 2. `browser_intelligence/intelligence/provider_brain.py`
- **File**: `provider_brain.py` — `ProviderBrain`, `ProviderBrainState`, `ProviderName`
- **Why unused**: Never imported by `engine.py` or any production file. Only test imports.
- **Evidence**: `engine.py` uses `ProviderReliabilityStore` from `learning/` instead.

## 3. `browser_intelligence/intelligence/drift_detector.py`
- **File**: `drift_detector.py` — `DriftDetector`
- **Why unused**: Never imported in production.
- **Evidence**: 0 production imports. 3 test files reference it.

## 4. `browser_intelligence/intelligence/shadow_ban_detector.py`
- **File**: `shadow_ban_detector.py` — `ShadowBanDetector`
- **Why unused**: Never imported in production.
- **Evidence**: 0 production imports. Only referenced in engine.py's `_compute_available_actions()` as a state name, never instantiated.

## 5. `memory/` — ENTIRE PACKAGE
- **Files**: `hot_warm_cold.py`, `summarizer.py`, `token_budget.py`
- **Why unused**: No production code imports anything from `ai_orchestrator.memory`.
- **Evidence**: 0 non-test imports anywhere in the codebase.

## 6. `adapters/cookie_validator.py:validate_cookie_file()`
- **File:Line**: `cookie_validator.py:73`
- **Why unused**: Defined but never called. Only `validate_storage_state()` is used.
- **Evidence**: `validate_cookie_file` has 0 cross-references. AGENTS.md rule #7 specifically requires cookie file validation.

## 7. `adapters/base.py:61` — `ResponseValidator` lazy import
- **File:Line**: `base.py:59-62`
- **Why unused**: Guarded by `validate_responses=False` which is always `False`.
- **Evidence**: `validate_responses=True` appears 0 times across the entire codebase.

## 8. `engine_adapter.py:1350` — Chrome user data path
- **File:Line**: `engine_adapter.py:1349-1365`
- **Why unused**: Guarded by literal `and False` — can never execute.
- **Evidence**: `if chrome_user_data_dir.exists() and self._persistent_profile is None and self._storage_state is None and False:`

## 9. `engine_adapter.py:447` — Redundant `import time as _time`
- **File:Line**: `engine_adapter.py:447`
- **Why unused**: `time` already imported at line 16. The `import time as _time` is a no-op duplicate.

## 10. `_STATUS_LABEL_RE` — triplicate definition
- **Files**:
  - `engine_adapter.py:76`
  - `engine.py:65`
  - `response_capture.py:280`
- **Why unused (as shared):** Same regex defined 3 times in 3 files. None imported from another.

---

# TOP 20 MOST LIKELY ROOT CAUSES FOR CURRENT FAILURES

---

## #1 — `navigator.webdriver` NOT masked (85%)

**Probability**: 85%

**Evidence**: 
- `engine_adapter.py:1415` — Only `add_init_script(FETCH_INTERCEPT_SCRIPT)` is injected
- `stealth.py:109` — The code to set `navigator.webdriver = false` exists but is never injected
- `engine_adapter.py:1320-1323` — Launch args only include `--disable-blink-features=AutomationControlled` — this is NOT sufficient
- `engine_adapter.py:1281` — Firefox gets ZERO launch args

**Files**: `engine_adapter.py:1291`, `engine_adapter.py:1415`, `stealth.py:308-315`

**Code snippets**:
```python
# engine_adapter.py:1320-1323 — Chromium launch args (2 items)
launch_args = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
]

# stealth.py:240-269 — 28 launch args that exist but are NEVER applied:
_BASE_LAUNCH_ARGS = (
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process,AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    # ... 24 more args that disable automation features ...
)

# engine_adapter.py:1291 — Only script injected, NOT stealth_init_script
await self._page.add_init_script(FETCH_INTERCEPT_SCRIPT)

# What should ALSO be injected (stealth.py:308-315):
await self._page.add_init_script(stealth_init_script(profile))
```

---

## #2 — `stealth.py` never wired into adapter (80%)

**Probability**: 80%

**Evidence**:
- `main.py:254` passes `stealth=True` but this is the adapter boolean, NOT `apply_stealth()`
- `engine_adapter.py` has its own `_stealth` boolean that only controls viewport + UA
- `stealth.py:334-352` `apply_stealth()` provides `StealthApplication` with profile, launch_args, context_options, init_script — **none** wired into any adapter

**Files**: `engine_adapter.py:316-338`, `stealth.py:334-352`, `main.py:244-258`

**Code snippets**:
```python
# main.py:251-258 — stealth is just a boolean:
return cls(
    mock_mode=mock_mode,
    headless=False if not mock_mode else True,
    stealth=True,          # ← This boolean only enables viewport + UA
    timeout_ms=120_000,
    storage_state=storage_state,
    persistent_profile=persistent_profile,
    channel=channel,
)

# What SHOULD happen:
from ai_orchestrator.browser_intelligence.intelligence import apply_stealth
stealth_app = apply_stealth()
return cls(
    mock_mode=mock_mode,
    headless=False,
    stealth_app=stealth_app,  # ← Pass the actual StealthApplication
    ...
)
```

---

## #3 — Firefox has ZERO anti-fingerprinting (75%)

**Probability**: 75% (chatgpt_ui specifically)

**Evidence**:
- `engine_adapter.py:1281-1283` — `firefox.launch(headless=False)` with NO args
- `engine_adapter.py:1285-1288` — `new_context({})` with minimal opts
- `engine_adapter.py:1291` — `add_init_script(FETCH_INTERCEPT_SCRIPT)` on Firefox
- `engine_adapter.py:666` — `is_firefox` check: bypasses HMM engine entirely, uses DOM polling
- CDP does NOT work on Firefox, so `engine.attach()` fails silently

**Files**: `engine_adapter.py:1264-1318`, `engine_adapter.py:369-376`

**Code snippets**:
```python
# engine_adapter.py:1281 — Firefox launch: NO args
self._browser = await self._playwright.firefox.launch(
    headless=self.headless,
)
# Missing: args=[...] for basic anti-detection

# engine_adapter.py:1285-1288 — Context: bare minimum
ctx_opts = {}
if self._stealth:
    ctx_opts["viewport"] = {"width": 1280, "height": 720}
# Missing: user_agent, locale, timezone_id, extra_http_headers

# engine.py:369-376 — CDP fails on Firefox silently
try:
    await self._composer._network.attach(page)
    self._fusion.record_sensor_success("network")
except Exception as exc:
    self._fusion.record_sensor_failure("network")
    log.warning("CDP Network attach failed: %s — network intelligence disabled", exc)
# ALL CDP-dependent features are disabled: stream capture, SSE observing, WS, etc.
```

---

## #4 — `sameSite` forced to "Lax" for ALL cookies (70%)

**Probability**: 70% (for auth failures specifically)

**Evidence**:
- `cookie_to_storage_state.py:75` — `"sameSite": _normalize_samesite("Lax")` — hardcoded to "Lax"
- `auto_cookie_update.py:55-56` — `_normalize_samesite()` normalizes "unspecified" → "None" but the bug is in `cookie_to_storage_state.py` which bypasses normalization entirely

**Files**: `cookie_to_storage_state.py:67-76`, `auto_cookie_update.py:46-56`

**Code snippets**:
```python
# cookie_to_storage_state.py:67-76 — BUG: sameSite always "Lax"
cookies.append({
    "name": name,
    "value": value,
    "domain": domain,
    "path": cookie_path,
    "expires": expires,
    "httpOnly": False,
    "secure": secure,
    "sameSite": _normalize_samesite("Lax"),  # ← ALWAYS "Lax" regardless of actual cookie
})
# A cookie that needs sameSite=None for cross-origin auth will be broken.
```

---

## #5 — Cloudflare check FAILS on Firefox because CDP disabled (65%)

**Probability**: 65%

**Evidence**:
- `engine_adapter.py:1300-1301` — `check_cloudflare_challenge(page)` runs AFTER navigation
- But the engine is NOT yet attached at this point (attach happens at line 597)
- On Firefox, the `_wait_until_ready` bypass path at line 680 ignores engine state
- Firefox-specific Dom detection at lines 682-699 ONLY checks for input visibility, NOT CF

**Files**: `engine_adapter.py:1300-1316`, `engine_adapter.py:656-723`

---

## #6 — Persistent profile and storage_state are mutually exclusive (60%)

**Probability**: 60%

**Evidence**:
- `main.py:242-248` — If `persistent_profile` directory exists, uses it. `storage_state` is IGNORED.
- `engine_adapter.py:1266-1279` — Firefox persistent profile path does NOT inject storage_state cookies
- `engine_adapter.py:1367-1391` — Chromium persistent profile path does NOT inject storage_state cookies
- Result: If a stale browser profile exists on disk, fresh cookie updates are NEVER loaded

**Files**: `main.py:242-248`, `engine_adapter.py:1266-1291`, `engine_adapter.py:1367-1391`

**Code snippets**:
```python
# main.py:242-248 — Priority: profile > cookies. No merge.
if not mock_mode:
    profile_dir = _PERSISTENT_PROFILE_MAP.get(provider)
    if profile_dir and Path(profile_dir).exists():
        persistent_profile = profile_dir      # ← Uses stale profile
    else:
        storage_state = _load_auth_for(provider) # ← Fresh cookies IGNORED
```

---

## #7 — Each `send()` call creates fresh browser = CF resets fingerprint (55%)

**Probability**: 55%

**Evidence**:
- `engine_adapter.py:596` — `engine = BrowserIntelligenceEngine()` — fresh engine every call
- `engine_adapter.py:1257` — `_get_page()` creates new page/browser each time
- `main.py:551-565` — `adapter.send()` then `adapter.close()` — no browser reuse unless fan_out
- Cloudflare tracks fingerprint continuity across requests

**Files**: `engine_adapter.py:562-652`

---

## #8 — Pre-flight cookie validation is WARNING-only, non-blocking (50%)

**Probability**: 50%

**Evidence**:
- `engine_adapter.py:569-576` — `validate_storage_state()` returns `is_valid=False`, but `_real_send` continues
- AGENTS.md rule #7 explicitly says: "Cookie load success ≠ auth success" and requires blocking
- Execution continues to launch browser even with ZERO cookies

**Files**: `engine_adapter.py:569-576`, `cookie_validator.py:115-132`

**Code snippets**:
```python
# engine_adapter.py:569-576 — Warning only, never blocks
if self._storage_state is not None:
    cv = validate_storage_state(self._storage_state)
    if not cv.is_valid:
        log.warning(
            "[%s] Cookie pre-flight FAILED: %s",
            self.provider_name,
            "; ".join(cv.errors),
        )
# ← Should return ProviderResponse(success=False) here, but doesn't
# Execution continues to launch browser without valid cookies
```

---

## #9 — `ConnectionRefused` / timeout for 5 providers (45%)

**Probability**: 45%

**Evidence**:
- Diagnostic runs show `page.goto: net::ERR_CONNECTION_REFUSED` for qwen, kimi, zai, minimax, xiaomimimo
- `engine_adapter.py:1417-1420` — `wait_until="load"` on Chromium, `timeout=120_000`
- No proxy configured, no DNS resolution check, no network reachability test
- Many Chinese providers (qwen, kimi, zai, minimax, xiaomimimo) may require VPN from outside China

**Files**: `engine_adapter.py:1292-1293`, `engine_adapter.py:1417-1421`

---

## #10 — Auth check swallows errors and treats as authenticated (40%)

**Probability**: 40%

**Evidence**:
- `cookie_validator.py:160-163` — If `check_post_navigation_auth()` throws, returns `(True, "check failed")`
- This means ANY error in the auth check is treated as "authenticated"
- A crashed page, a 500 error, or a timeout all become "authenticated"

**Files**: `cookie_validator.py:135-163`

**Code snippets**:
```python
# cookie_validator.py:160-163 — Error = authenticated
except Exception as exc:
    log.warning("Post-navigation auth check failed: %s", exc)
    # If we can't check, assume ok to avoid blocking
    return True, f"check failed: {exc}"  # ← BUG: should return False
```

---

## #11 — `_STATUS_LABEL_RE` defined 3 times, inconsistent updates (35%)

**Probability**: 35%

**Evidence**:
- `engine_adapter.py:76-80` — 10 status labels
- `engine.py:65-69` — 10 status labels (same)
- `response_capture.py:280` — different regex entirely (multiple lines)
- If one definition gets updated and others don't, response filtering breaks

**Files**: `engine_adapter.py:76-80`, `engine.py:65-69`, `response_capture.py:279-280`

---

## #12 — `fan_out()` shares browser but exposes CDP to all adapters (30%)

**Probability**: 30%

**Evidence**:
- `engine_adapter.py:461-558` — `fan_out()` uses ONE shared browser
- Line 486: `cls._shared_browser = await cls._shared_playwright.chromium.launch()` — only Chromium
- Multiple pages share one browser context — CDP sessions accumulate
- Race condition: adapters close each other's pages (line 553: `cls._shared_browser.close()`)

**Files**: `engine_adapter.py:461-558`

---

## #13 — Recovery engine escalation can never reach NEW_CONTEXT (25%)

**Probability**: 25%

**Evidence**:
- `recovery_engine.py:163` — `if attempt >= MAX_RETRIES: return None` before checking strategies
- `recovery_engine.py:209-217` — Strategy 4 (NEW_CONTEXT) checked only if `attempt >= 2`
- But `attempt >= MAX_RETRIES` (3) is checked at line 163 BEFORE we reach strategy 4
- Strategy 4 is dead: attempt 2 reaches it, but Strategy 3 (RELOAD_COOKIES) at line 188 also requires `attempt >= 2`

**Files**: `recovery_engine.py:161-219`

**Code snippets**:
```python
# recovery_engine.py:161-163
if attempt >= MAX_RETRIES:  # MAX_RETRIES = 3
    log.warning("[Recovery] Max retries (%d) exhausted.", MAX_RETRIES)
    return None
# Strategy 4 at line 209:
if attempt >= 2 and adapter._context is not None:  # Checked at attempt=2
    # ...NEW_CONTEXT
# This works for attempt=2, but Strategy 3 also triggers at attempt=2.
# The problem is at attempt=3 — MAX_RETRIES check kills it before Strategy 4.
```

---

## #14 — `_is_valid_response` rejects valid JSON chat responses (20%)

**Probability**: 20%

**Evidence**:
- `engine_adapter.py:942-979` — `_is_valid_response()` rejects JSON objects without `choices/message/content/text/delta/response/output` keys
- Some providers (e.g., Kimi MiniMax) return responses formatted as `{"type":"answer","data":{"content":"..."}}` — rejected
- Rejects `{"id":"...", "object": "chat.completion", ...}` if `choices` is structured differently

**Files**: `engine_adapter.py:942-979`

---

## #15 — `FETCH_INTERCEPT_SCRIPT` is a fingerprinting liability (15%)

**Probability**: 15%

**Evidence**:
- `engine_adapter.py:188-271` — Monkey-patches `window.fetch`, `XMLHttpRequest`, `EventSource`
- `engine_adapter.py:145` — Guard `if (window.__engine_adapter_hook__) return;` — single global
- Any anti-bot scanner can detect override of native `fetch` function via `fetch.toString()`
- `XMLHttpRequest.prototype.open` and `.send` are also overridden — visible via `toString()`

**Files**: `engine_adapter.py:145-272`

---

## #16 — Two independent recovery systems with overlapping responsibilities (15%)

**Probability**: 15%

**Evidence**:
- `adapters/recovery_engine.py` — `RecoveryEngine`, `attempt_recovery()` — Layer 7
- `browser_intelligence/recovery/__init__.py` — `RecoveryCascade`, `build_default_cascade()` — Phase 4
- Both handle recovery, but independently. `RecoveryCascade` is wired into `engine.py:250` but `RecoveryEngine` in `engine_adapter.py:380` calls `attempt_recovery()` which uses its own logic.

**Files**: `recovery_engine.py:90-219`, `engine.py:250`

---

## #17 — `Page.is_closed()` check is pointless (10%)

**Probability**: 10%

**Evidence**:
- `engine_adapter.py:583` — `if page.is_closed(): return error`
- But `_get_page()` at line 1258 returns immediately `if self._page is not None: return self._page`
- If the page was closed (e.g., by a previous error), `self._page` is already set to `None` in `close()` method
- So `page.is_closed()` can never be `True` here because the page was just created

**Files**: `engine_adapter.py:582-590`, `engine_adapter.py:424-428`

---

## #18 — `_wait_until_ready` loop: 60 second hard limit with no Cloudflare re-check (10%)

**Probability**: 10%

**Evidence**:
- `engine_adapter.py:671` — `for tick in range(MAX_READY_TICKS):` — 60 iterations max
- Cloudflare check is done at lines 1300/1430 BEFORE the loop
- The page might redirect to Cloudflare AFTER the initial check (e.g., delayed JS challenge)
- No re-check for Cloudflare during the 60-tick readiness polling

**Files**: `engine_adapter.py:656-769`

---

## #19 — `_PERSISTENT_PROFILE_MAP` only has 2 of 8 providers (5%)

**Probability**: 5%

**Evidence**:
- `main.py:201-204` — Only `chatgpt_ui` and `qwen_ui` mapped
- `main.py:243-245` — All other providers ALWAYS fall back to `storage_state` (cookies) with no persistent session
- Persistent profiles provide IndexedDB, localStorage, Service Workers — all needed for Cloudflare session integrity

**Files**: `main.py:201-204`

---

## #20 — `get_context_limit()` always returns mock value (5%)

**Probability**: 5%

**Evidence**:
- `engine_adapter.py:409` — `return self.mock_context_limit` — always returns the class attribute
- `chatgpt_ui.py:12` — `mock_context_limit = 131_072`
- No logic to actually query the provider's real context limit from an API or from the page

**Files**: `engine_adapter.py:408-409`

---

# IF YOU WERE THE ENGINEER

## The 10 code changes I would make tomorrow morning, ordered by impact:

### 1. Wire `stealth.py` into `engine_adapter._get_page()`

**File**: `ai_orchestrator/orchestrator/main.py` (lines 244-258)
**File**: `ai_orchestrator/adapters/engine_adapter.py` (lines 320-338, 1257-1450)

Inject the stealth init script into every page, use stealth launch args for all browser launches, use stealth context options for all browser contexts. This is the #1 root cause of Cloudflare detection.

```python
# In _get_page(), after creating page, before goto:
from ai_orchestrator.browser_intelligence.intelligence.stealth import (
    apply_stealth, stealth_init_script,
)
stealth_app = apply_stealth(seed=random.randint(0, 2**16))
launch_args = stealth_app.launch_args  # 28 args instead of 2
ctx_opts.update(stealth_app.context_options)  # locale, timezone, headers
await self._page.add_init_script(stealth_app.init_script)  # navigator.webdriver=false
```

### 2. Fix `cookie_to_storage_state.py` — don't force `sameSite: "Lax"` for all cookies

**File**: `ai_orchestrator/adapters/cookie_to_storage_state.py` (line 75)

```python
# Change from:
"sameSite": _normalize_samesite("Lax"),
# To:
"sameSite": _normalize_samesite(parts[7] if len(parts) > 7 else "Lax"),
```

Parse the Netscape cookie's sameSite field if present (column 8+ in some variants).

### 3. Add `navigator.webdriver = false` injection for Firefox

**File**: `ai_orchestrator/adapters/engine_adapter.py` (line 1281-1291)

Currently Firefox gets zero anti-detection. Add a minimal init script that strips `navigator.webdriver` even on Firefox.

```python
# After line 1291:
await self._page.add_init_script("""() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
}""")
```

### 4. Fix `check_post_navigation_auth` — return False on error, not True

**File**: `ai_orchestrator/adapters/cookie_validator.py` (lines 160-163)

```python
# Change from:
return True, f"check failed: {exc}"
# To:
return False, f"auth check threw exception: {exc}"
```

### 5. Make pre-flight cookie validation BLOCK execution

**File**: `ai_orchestrator/adapters/engine_adapter.py` (lines 569-576)

```python
# After line 576, add:
if not cv.is_valid:
    return ProviderResponse(
        success=False,
        error=f"Cookie pre-flight validation FAILED: {'; '.join(cv.errors)}",
        model=self.provider_name.replace("_", "-"),
    )
```

### 6. Merge persistent profile cookies WITH storage_state cookies

**File**: `ai_orchestrator/orchestrator/main.py` (lines 242-248)

Currently mutual exclusive. If both exist, load both.

```python
# Change to:
if not mock_mode:
    profile_dir = _PERSISTENT_PROFILE_MAP.get(provider)
    has_profile = profile_dir and Path(profile_dir).exists()
    storage_state = _load_auth_for(provider)
    if has_profile:
        persistent_profile = profile_dir
    # Always provide storage_state even with persistent profile
    # Playwright merges cookies from both sources
```

### 7. Enable real Chrome user-data path (remove `and False`)

**File**: `ai_orchestrator/adapters/engine_adapter.py` (line 1350)

```python
# Change from:
if chrome_user_data_dir.exists() and self._persistent_profile is None and self._storage_state is None and False:
# To:
if chrome_user_data_dir.exists() and self._persistent_profile is None:
```

This was clearly a debugging guard left in — it literally hardcodes the path as dead.

### 8. Add Cloudflare re-check during _wait_until_ready polling

**File**: `ai_orchestrator/adapters/engine_adapter.py` (line 671-768)

```python
# Add inside the tick loop, every 10th tick:
if tick > 0 and tick % 10 == 0:
    is_cf, cf_reason = await check_cloudflare_challenge(page)
    if is_cf:
        raise CloudflareBlockError(provider, f"Delayed Cloudflare challenge: {cf_reason}")
```

### 9. Centralize `_STATUS_LABEL_RE` — single source of truth

**Files**: `engine_adapter.py:76`, `engine.py:65`, `response_capture.py:280`

Delete definitions from `response_capture.py:280` and `engine.py:65`. Import from `engine_adapter.py` or create a shared constants module.

### 10. Remove the redundant `import time as _time` in `_mock_send`

**File**: `ai_orchestrator/adapters/engine_adapter.py` (line 447)

```python
# Delete line 447: import time as _time
# Change line 448: _time.sleep(0.05) → time.sleep(0.05)
```
