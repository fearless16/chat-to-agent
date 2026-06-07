# Chat-to-Agent Architecture Report

> Generated: 2026-06-07 | Codebase analysis with dependency tracing and call-flow mapping

---

## 1. Project Overview

**chat-to-agent** is an AI orchestration platform that uses browser automation (Playwright) to interact with web-based AI providers (ChatGPT, DeepSeek, Qwen, Kimi, Z.AI, MiniMax, XiaomiMiMo). It wraps these web UIs behind a FastAPI HTTP API with OpenAI-compatible endpoints, adding task orchestration, resource scheduling, and automated test-fix cycles.

**Core Insight**: Instead of calling provider APIs (which cost money and have rate limits), it drives browser sessions on free-tier web chat UIs. This requires solving auth, popup dismissal, response extraction, and Cloudflare bypass — all without API tokens.

---

## 2. Architecture Layers (Top-Down)

```
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 7: Entry Point                                             │
│ orchestrator/main.py — FastAPI app (uvicorn)                     │
│ /health  /tasks  /chat  /v1/chat/completions  /provider-health  │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 6: Orchestrator Services                                   │
│ WorkflowEngine  ControlPlane  LeaseManager  ProviderRouter       │
│ ResourceScheduler  ProviderHealthTracker  DeadLetterQueue        │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 5: Provider Adapters                                       │
│ EngineUIAdapter (core) → chatgpt_ui  deepseek_ui  qwen_ui        │
│   kimi_ui  zai_ui  minimax_ui  xiaomimimo_ui  local_llm         │
│ RecoveryEngine  PopupManager  CookieValidator  AutoCookieUpdate  │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 4: Browser Intelligence OS                                 │
│ BrowserIntelligenceEngine — HMM-based state estimation           │
│ Sensors: Network > Accessibility > DOM > Vision                  │
│ ResponseCapture  TrafficClassifier  FeatureStore  EventBus       │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 3: Runtime & Testing                                       │
│ RuntimeLoop (build→test→fix cycle)  TestRunner  Sandbox          │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 2: Workspace & Storage                                     │
│ FileWorkspace  GitWorkspace  PostgresORM  RedisClient            │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 1: Security & Validation                                   │
│ PromptGuard  CredentialVault  ResponseValidator                  │
├──────────────────────────────────────────────────────────────────┤
│ LAYER 0: Data Models                                             │
│ Account  Task  Lease  CapabilityVector  ProviderCapabilities     │
│ ProviderResponse  AgentResult  FeatureVector  BeliefState        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Call Flow: "Send a prompt to ChatGPT"

This is the **most important path** in the codebase. Here is exactly who calls what, in order:

```
POST /chat  {provider:"chatgpt_ui", prompt:"Hello"}
│
├─ orchestrator/main.py:_chat_endpoint()
│   │
│   ├─ _build_adapter("chatgpt_ui", mock_mode=False)
│   │   └─ ChatGPTUIAdapter.__init__()
│   │       └─ EngineUIAdapter.__init__()
│   │           ├─ Sets _site = SiteConfig(url="https://chatgpt.com")
│   │           ├─ Reads chatgpt_auth.json (or profiles/*_cookies.txt)
│   │           │   └─ cookie_to_storage_state.py → Playwright storage state
│   │           └─ cookie_validator.py → pre-flight validation
│   │
│   ├─ adapter.send(prompt, context)
│   │   └─ EngineUIAdapter.send()
│   │       ├─ recovery_engine.check_cooldown(provider)  → raise if cooling
│   │       ├─ _get_page()                                → launch browser
│   │       │   ├─ playwright.chromium.launch(headless, stealth, channel)
│   │       │   ├─ context.add_cookies(storage_state)
│   │       │   └─ page.goto(_site.url)
│   │       │
│   │       ├─ _real_send(prompt)
│   │       │   ├─ _engine = BrowserIntelligenceEngine()
│   │       │   ├─ _engine.attach(page)           → wire CDP session
│   │       │   ├─ _wait_until_ready(engine, page) → tick until READY
│   │       │   │   ├─ engine.tick(page)           → feature pipeline
│   │       │   │   ├─ popup_manager.detect_popup(page)
│   │       │   │   ├─ popup_manager.dismiss_popup(page)
│   │       │   │   └─ async_detect_auth/login      → AuthenticationError?
│   │       │   ├─ _execute_type_prompt(page, prompt) → fill input, press Enter
│   │       │   ├─ _wait_for_response_pair(engine, page)
│   │       │   │   └─ engine.tick(page) until COMPLETE or timeout(60s)
│   │       │   ├─ _extract_response_text(page)
│   │       │   │   ├─ engine.get_response_text()     → ResponseCapture
│   │       │   │   └─ fallback → DOM extraction
│   │       │   ├─ _sanitize_response_text(text)
│   │       │   └─ _is_valid_response(text) → reject {}, arrays, noise
│   │       │
│   │       └─ auto_cookie_update.py → save refreshed cookies to disk
│   │
│   └─ health_tracker.record_success(provider, latency_ms)
│
└─ Response: {content:"...", model:"gpt-4o", success:true}
```

---

## 4. Key Component Details

### 4.1 Entry Point (`orchestrator/main.py`)

The **FastAPI app** is the integration point. It:

| Endpoint | Method | What It Does |
|----------|--------|--------------|
| `/health` | GET | System health + RAM watermark + active leases |
| `/chat` | POST | **Main LLM call**: takes `provider` + `prompt`, returns `ProviderResponse`. Supports `fan_out` mode (parallel across providers) |
| `/tasks` | POST | Submit a task → lease acquisition → workflow planning |
| `/tasks/{id}` | GET/POST | Get task status / Execute next step / Halt / Resume |
| `/accounts` | GET/POST/DELETE | Manage provider accounts |
| `/leases` | GET | View active leases |
| `/provider-health` | GET | Per-provider success/auth/captcha dashboard |
| `/tasks/{id}/workspace` | POST | Create git-initialized workspace for task |
| `/tasks/{id}/run-loop` | POST | Start build→test→fix cycle |
| `/v1/chat/completions` | POST | OpenAI-compatible API (via `openai_compat.py`) |
| `/v1/models` | GET | List available models |

**Adapter Construction (`_build_adapter`)**:
```python
_PROVIDER_CLASS_MAP = {
    "chatgpt_ui": ChatGPTUIAdapter,
    "deepseek_ui": DeepSeekUIAdapter,
    "qwen_ui": QwenUIAdapter,
    "kimi_ui": KimiUIAdapter,
    "z_ai_ui": ZAIUIAdapter,
    "minimax_ui": MiniMaxUIAdapter,
    "xiaomimimo_ui": XiaomiMiMoUIAdapter,
    "local_llm": LocalLLMAdapter,
}
_PROVIDER_CHANNEL_MAP = {
    "chatgpt_ui": "chrome",   # tries Firefox for CF bypass, but mapped to chrome
    ...
}
```

### 4.2 EngineUIAdapter (`adapters/engine_adapter.py`)

**The central adapter** — all 7 web-based providers inherit from this single class. Each subclass only needs:

```python
class ChatGPTUIAdapter(EngineUIAdapter):
    def _site(self):
        return SiteConfig(
            url="https://chatgpt.com",
            input_selector="div#prompt-textarea",  # or auto-detect
            send_selector="button[data-testid='send-button']",
        )
```

**Key Methods** (execution order in `send()`):

| Method | Purpose |
|--------|---------|
| `_get_page()` | Launch Chromium/Firefox via Playwright, load cookies, navigate to provider URL |
| `_wait_until_ready()` | Tick the BI engine 1Hz until `belief_state == READY`; dismiss popups; check Cloudflare/auth |
| `_execute_type_prompt()` | Find input element, click, type prompt, press Enter |
| `_wait_for_response_pair()` | Tick engine until `COMPLETE` or timeout (60s); collect content + reasoning |
| `_extract_response_text()` | Preferred: `engine.get_response_text()` (CDP-intercepted). Fallback: DOM query |
| `_sanitize_response_text()` | Strip noise like "assistanttyping..." status messages |
| `_is_valid_response()` | Reject `{}`, `{"ResultObject": true}`, empty content, raw JSON arrays |
| `fan_out()` | **classmethod** — shared browser: open one context, run multiple adapters sequentially |

**Error Handling** in `send()`:
```
1. recovery_engine.check_cooldown()     → CooldownError if provider recently failed
2. _get_page() fails                    → RecoveryEngine escalates: page→context→browser restart
3. _wait_until_ready() detects CF      → CloudflareBlockError, screenshot saved
4. _wait_until_ready() detects auth    → AuthenticationError, cookie refresh triggered
5. Response invalid                    → ResponseExtractionError, retry with fresh page
6. Timeout (60s)                       → Retry up to max_retries (2)
```

### 4.3 Browser Intelligence OS (`browser_intelligence/`)

**Problem It Solves**: Hard-coded CSS selectors break when providers change their UI. The BI OS uses probabilistic state estimation instead.

**Pipeline**: `Sense → Compose → Estimate → Decide`

```
tick(page) — runs every 1 second
│
├─ Sense (6 sensors in priority order)
│   ├─ NetworkSensor: CDP Network domain → protocol detect, SSE/WS/fetch observe
│   ├─ AccessibilitySensor: a11y tree for selector recovery
│   ├─ DOMSensor: structural DOM (no text)
│   ├─ MutationSensor: DOM mutation rate/acceleration
│   ├─ PerformanceSensor: JS heap, page stability
│   └─ VisualSensor: screenshot (last resort, expensive)
│
├─ Compose (FeatureComposer)
│   └─ Assembles 30-dim FeatureVector:
│       • 18 binary features: input_visible, send_enabled, stream_active, new_text_present
│       • 12 continuous features: mutation_rate, tokens_per_second, bytes_received, etc.
│
├─ Estimate (HMM Engine)
│   ├─ EmissionModel: P(observation | hidden_state)  → learned from historic features
│   ├─ TransitionMatrix: P(state_t | state_{t-1})    → fixed Markov transitions
│   ├─ KalmanFilter: Response rate smoothing
│   └─ BeliefState: current probabilities over HiddenState enum
│
├─ Decide (Completion + Confidence + Evidence Fusion)
│   ├─ CompletionEngine: is_response_complete() → stream activity + mutation rate
│   ├─ ConfidenceEngine: how sure are we about current state?
│   ├─ EvidenceFusion: combine Network(0.4) + Accessibility(0.25) + DOM(0.2) + Vision(0.15)
│   └─ UtilityEngine: what action should we take next? (WAIT/RETRY/EXTRACT/RECOVER)
│
└─ Store (FeatureStore: ring buffer, 300 samples at 1Hz = 5 min window)
```

**HiddenState Enum** (the states the engine tracks):
```
BOOTING → AUTH_REQUIRED → READY → PROMPT_SENT → THINKING → GENERATING → COMPLETE
                                                      ↓
                                              RATE_LIMITED
                                              ERROR
                                              SHADOW_BANNED
```

**Priority Rule** (from AGENTS.md):
```
Generation State:  Network > Accessibility > DOM > Vision
Auth State:        Accessibility > DOM > Vision > Network
Response Text:     Network > DOM
Popup Detection:   DOM > Accessibility > Vision
```

### 4.4 ResponseCapture (`browser_intelligence/intelligence/response_capture.py`)

Captures response bodies from the browser's network layer (CDP) — bypasses DOM altogether.

**Flow**:
```
CDP Network.requestWillBeSent
├─ TrafficClassifier.classify(url, method, content_type, headers)
│   └─ Returns: CHAT_RESPONSE | CONVERSATION_LIST | ANALYTICS | TELEMETRY | AUTH | STATIC
│
├─ If CHAT_RESPONSE:
│   ├─ begin_response(request_id, url, method, status, content_type)
│   └─ On CDP Network.dataReceived:
│       └─ append_chunk(request_id, chunk) → accumulate body
│
├─ On CDP Network.loadingFinished:
│   └─ close_response(request_id) → CapturedResponse (full body text)
│
└─ get_response_text() → concatenate all CHAT_RESPONSE bodies
   get_response_text_sse() → SSE-specific bodies only
```

**Traffic Categories**:
| Category | What It Matches | Action |
|----------|----------------|--------|
| CHAT_RESPONSE | Completion API responses, SSE streams | CAPTURE |
| CONVERSATION_LIST | Chat history / sidebar loads | IGNORE |
| ANALYTICS | Google Analytics, telemetry endpoints | IGNORE |
| TELEMETRY | Error reporting, perf monitoring | IGNORE |
| AUTH | Login/token endpoints | IGNORE (but logged) |
| STATIC | CSS/JS/images | IGNORE |

### 4.5 RecoveryEngine (`adapters/recovery_engine.py`)

**Escalating recovery chain** when a provider fails:

```
Level 1: page.reload()                        — quick refresh
Level 2: page.close() + new page              — fresh tab
Level 3: context.close() + new context        — new session
Level 4: browser.close() + new browser        — full restart
Level 5: cooldown (provider, 60s)             — back off
```

Also maintains per-provider cooldown with exponential backoff.

### 4.6 PopupManager (`adapters/popup_manager.py`)

Detects and dismisses UI popups that block interaction:

| Popup Type | Detection | Dismissal |
|------------|-----------|-----------|
| COOKIE_BANNER | Text "cookie" / "accept" in dialog | Click "Accept" |
| NEWSLETTER_MODAL | "subscribe" / "newsletter" | Click ✕ or Escape |
| UPGRADE_MODAL | "upgrade" / "pro" / "premium" | Click ✕ |
| RATE_LIMIT_MODAL | "rate limit" / "too many" | Pause + notify |
| CAPTCHA_MODAL | "captcha" / "verify you" | **Never auto-solve** — pause + notify |
| UNKNOWN_MODAL | Can't classify | Screenshot + log + pause |

### 4.7 Recovery Engine + Event Bus Integration

```
EngineUIAdapter.send() failure
│
├─ PopupManager.detect_popup(page)
│   └─ If CAPTCHA → CaptchaDetected event → notify → pause (no auto-solve)
│   └─ If RATE_LIMIT → RateLimitDetected event → cooldown provider
│
├─ RecoveryEngine handling:
│   ├─ page reload → if still fails → new page → new context → new browser
│   └─ On exhaustion → CooldownError → provider enters cooldown
│
├─ GoogleRecoveryEngine (capability: stale_page, popup_disturbance, auth_expired, cf_blocked)
│   └─ self-healing: stale → quick refresh, popup → dismiss, auth → cookie refresh
│
├─ RecoveryCascade (browser_intelligence/recovery/)
│   └─ Adaptive based on provider history
│   └─ Actions: REFRESH, REAUTH, STASH_AND_RETRY, TERMINATE
│
└─ EventBus.publish(EventType.RECOVERY_STARTED, {provider, action})
```

### 4.8 Orchestrator Services

**WorkflowEngine** — FSM for multi-step tasks:
```
IDLE → PLANNING → EXECUTING → TESTING → REVIEWING → FIXING → VERIFICATION
  │        │           │           │          │          │          │
  └────────┴───────────┴───────────┴──────────┴──────────┴──→ DONE
                                                          └──→ FAILED
                                                          └──→ HALTED
                                                          └──→ DLQ (dead letter queue)
```

**ControlPlane** — 3-tier routing:
- **Tier 0**: Deterministic keyword matching (fast, no LLM)
- **Tier 1**: Cheap LLM for classification (stub — not implemented)
- **Tier 2**: DeepSeek for planning/review (stub — not implemented)

**LeaseManager** — Account pooling with exclusive leases:
```
Account lifecycle:  IDLE → WARMUP → ACTIVE → COOLDOWN → JAIL
Lease lifecycle:    REQUESTED → ACTIVE → RENEWING → EXPIRED → RELEASED

Reactive events:
  account_jailed() → force_expire all leases → notify workflow → REPLAN
```

**ProviderRouter** — Scores accounts on:
- Capability match (reasoning, coding, translation, multimodality, speed, reliability, cost)
- Penalties (health_score, consecutive_failures, rate_limits, latency)
- Weights tuned per task type

**ResourceScheduler** — Watermark-based admission (for low-RAM constraint):
```
NORMAL (>4GB free)  → accept all
WARNING (2-4GB)     → accept HIGH priority only, throttle MEDIUM
CLEANUP (1-2GB)     → reclaim expired leases, refuse LOW
EMERGENCY (<1GB)    → stop accepting, force-expire idle
CRITICAL (<512MB)   → halt all new work
```

### 4.9 Other Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **CredentialVault** | `security/vault.py` | Fernet-encrypted key-value store for provider credentials |
| **PromptGuard** | `security/prompt_guard.py` | Regex-based injection detection (9 patterns: ignore_previous, DAN mode, etc.) |
| **Sandbox** | `security/sandbox.py` | Subprocess execution with timeout (30s), output cap (1 MiB), minimal env |
| **FileWorkspace** | `workspace/manager.py` | Safe path-confined file ops, atomic writes, snapshots, search |
| **RuntimeLoop** | `runtime/loop.py` | Build→test→fix cycle: run tests → analyze failures → fix → commit → repeat (max 5 iterations) |
| **TestRunner** | `testrunner/runner.py` | Auto-detect pytest/npm/go, run in sandbox, parse results |
| **CookieValidator** | `adapters/cookie_validator.py` | Pre-flight: file exists, parseable, count>0. Post-nav: authenticated? |
| **CookieToStorageState** | `adapters/cookie_to_storage_state.py` | Converts Netscape cookie format → Playwright storage_state dict, normalizes sameSite |
| **AutoCookieUpdate** | `adapters/auto_cookie_update.py` | Saves refreshed cookies from browser context back to disk after successful run |
| **ResponseValidator** | `validation/validator.py` | L1 deterministic (schema, JSON, Python syntax) + L2 DeepSeek review (stub) |
| **Observability** | `observability/` | structlog + Prometheus (counters, gauges, histograms) + OpenTelemetry tracing |

---

## 5. Dependency Graph (Who Calls Whom)

```
                    orchestrator/main.py
                    │
    ┌───────────────┼───────────────────────────────┐
    │               │                               │
workflow_engine  lease_manager              provider_router
    │               │                               │
control_plane       │                               │
    │               │                               │
    └───────────────┴───────────────────────────────┘
                    │
              adapters/ (all 8 adapters)
                    │
         ┌─────────┴──────────┐
         │                    │
engine_adapter          local_llm.py
         │              (HTTP-based)
         │
┌────────┴─────────────────────────────┐
│                                      │
BrowserIntelligenceEngine         Support Modules
│   ├─ sensors (6)                ├─ recovery_engine.py
│   │   └─ network (5 observers)  ├─ popup_manager.py
│   ├─ estimation (HMM, Kalman)   ├─ cookie_validator.py
│   ├─ decision (completion,      ├─ cookie_to_storage_state.py
│   │          confidence,        └─ auto_cookie_update.py
│   │          evidence_fusion,
│   │          utility)
│   ├─ features (composer, store)
│   ├─ intelligence (capture,
│   │               classifier,
│   │               stealth,
│   │               provider_brain,
│   │               drift_detector,
│   │               shadow_ban_detector)
│   ├─ events (event bus)
│   ├─ learning (reliability store)
│   ├─ recovery (cascade)
│   └─ scheduling (adaptive)
│
runtime/             workspace/            security/
├─ loop.py           ├─ manager.py         ├─ vault.py
│  → git.py          │  → exceptions.py    ├─ prompt_guard.py
│  → sandbox.py      ├─ git.py             └─ sandbox.py
│  → testrunner.py   ├─ artifacts.py
│                    └─ ast_patch.py
storage/              models/               observability/
├─ redis_client.py    ├─ account.py         ├─ metrics.py
├─ postgres_models.py ├─ task.py            ├─ logging.py
└─ journal.py         ├─ lease.py           └─ telemetry.py
                      └─ capabilities.py
```

---

## 6. Data Models

### 6.1 Core Domain Models

```
Account
├── id: str, provider: str, provider_kind: ProviderKind
├── state: AccountState (IDLE → WARMUP → ACTIVE → COOLDOWN → JAIL)
├── health_score: float, consecutive_failures: int
├── rate_limit_rpm/tpm: int, current_rate_usage: int
├── context_limit: int, avg_latency_ms: float
├── last_used: datetime, cooldown_until: datetime
└── record_success/failure/rate_limit(), mark_idle/active/jail()

Task
├── id: str, status: TaskStatus (IDLE → PLANNING → EXECUTING → ... → DONE/FAILED/DLQ)
├── type: TaskType, priority: TaskPriority
├── user_id: str, prompt: str
├── assigned_account_id: str, assigned_agent: str
├── retry_count: int, max_retries: int, error_message: str
└── transition_to(), mark_failed()

Lease
├── id: str, account_id: str, task_id: str, agent_id: str
├── state: LeaseState (REQUESTED → ACTIVE → RENEWING → EXPIRED → RELEASED)
├── acquired_at: datetime, expires_at: datetime, heartbeat_at: datetime
├── ttl_seconds: int, renewal_count: int, max_renewals: int
└── activate(), heartbeat(), renew(), release(), expire()
```

### 6.2 Capability Models

```
CapabilityVector
├── reasoning, coding, translation: float [0,1]
├── multimodality, speed, reliability: float [0,1]
└── cost_efficiency, long_context: float [0,1]

TaskRequirements
├── context_length: int
├── requires_reasoning/coding/translation/multimodality: bool
└── priority: dict[str, float]

ProviderCapabilities
├── provider_name: str, transport: str
├── capabilities: CapabilityVector
├── context_limit: int, max_concurrent: int
├── supports_streaming: bool, supports_tools: bool
└── metadata: dict

PROVIDER_PROFILES — static capability profiles for each provider
```

### 6.3 Browser Intelligence Models

```
FeatureVector (30 dimensions)
├── 18 binary: input_visible, send_enabled, stream_active, new_text_present, ...
├── 12 continuous: mutation_rate, tokens_per_second, bytes_received, ...
└── to_list() → list[float]

FeatureStore (ring buffer)
├── capacity: 300 (5 minutes at 1Hz)
├── push(fv), latest(), window(n)
├── ema(field), mean(field), std(field)
├── derivative(field), second_derivative(field)

BeliefState
├── probabilities: dict[HiddenState, float]
├── most_likely: HiddenState, confidence: float, entropy: float
└── HiddenState: BOOTING→AUTH_REQUIRED→READY→PROMPT_SENT→THINKING→GENERATING→COMPLETE→ERROR...

ProviderResponse
├── content: str, reasoning_content: str | None, model: str
├── usage: dict | None, latency_ms: float
├── success: bool, error: str | None
└── is_valid: property (non-empty content, no error)
```

---

## 7. Provider Support Matrix (from AGENTS.md)

| Provider | Auth | Input | Send | Generate | Extract |
|----------|------|-------|------|----------|---------|
| chatgpt_ui | Success | Success | Success | Success | Success |
| z_ai_ui | Success | Success | Success | Success | Success |
| qwen_ui | Success | Success | Success | Success | Success |
| deepseek_ui | Success | Fail | Fail | Fail | Fail |
| kimi_ui | Success | Fail | Fail | Fail | Fail |
| minimax_ui | Success | Fail | Fail | Fail | Fail |
| xiaomimimo_ui | Success | Fail | Fail | Fail | Fail |

**Status from recent diagnosis (2026-06-07)**:

| Provider | Auth File | sameSite Fix | Browser Launch | Page Load | CF Block | Response |
|----------|-----------|-------------|----------------|-----------|----------|----------|
| chatgpt_ui | OK | N/A | OK | CF blocked | YES | FAIL |
| deepseek_ui | Fixed | YES | OK | CF blocked | YES | FAIL |
| qwen_ui | OK | YES | OK | Timeout | NO | FAIL |
| kimi_ui | Fixed | YES | OK | Timeout | NO | FAIL |
| z_ai_ui | Fixed | YES | OK | Timeout | NO | FAIL |
| minimax_ui | Fixed | YES | OK | Timeout | NO | FAIL |
| xiaomimimo_ui | Fixed | YES | OK | Timeout | NO | FAIL |

---

## 8. Known Issues & Gaps

### 8.1 Critical

1. **Cloudflare Blocking (chatgpt_ui, deepseek_ui)**: Both `chatgpt.com` and `chat.deepseek.com` are detecting Playwright/automated browsers and showing Cloudflare challenges. The `stealth.py` module exists but needs persistent browser profiles with real user fingerprints.

2. **Connection/Timeout (qwen, kimi, zai, minimax, xiaomimimo)**: After fixing sameSite cookie normalization, pages fail with `ERR_CONNECTION_REFUSED` or timeout. Possible causes: DNS resolution, proxy/VPN requirement, rate limiting at connection level.

3. **Tier 1 and Tier 2 ControlPlane** are stub implementations. Only Tier 0 (keyword-based) works. This means task classification and replanning are limited to hardcoded rules.

4. **HMM calibration not evident**: The Hidden Markov Model in BI Engine has `TransitionMatrix`, `EmissionModel`, and calibration parameters, but there's no evidence of training data or learned values being loaded.

### 8.2 Medium

5. **Cookie refresh coupling**: `auto_cookie_update.py` saves cookies after each successful call, but if Cloudflare blocks, cookies can't refresh, leading to stale cookie stalemate.

6. **No parallel browser limit enforcement in code**: AGENTS.md specifies max 2 browsers, 3 tabs per browser, 4 concurrent providers, but the `ResourceScheduler` only checks RAM — not browser/session count.

7. **Popup classification is regex-based**, not ML-based. Unknown popups (UNKNOWN_MODAL) trigger a screenshot + pause, which can deadlock automation.

8. **ResponseValidator L2 (DeepSeek review)** is a stub — semantic validation of LLM responses is not implemented.

### 8.3 Minor

9. **Agents are mostly stubs**: The 7 agent classes (CoderAgent, ExecutorAgent, FixerAgent, PlannerAgent, ResearcherAgent, ReviewerAgent, TesterAgent) extend `BaseAgent` but contain minimal logic.

10. **RecoveryCascade** in `browser_intelligence/recovery/` is separate from `RecoveryEngine` in `adapters/` — two recovery systems with overlapping responsibility.

11. **Observation weights**: The feature pipeline uses magical thinking rather than empirical calibration. The sensor fusion weights (network 0.4, accessibility 0.25, DOM 0.2, vision 0.15) are hardcoded.

12. **Capability profiles (`PROVIDER_PROFILES`)** for all 7 providers are hardcoded estimates — not based on real benchmarking.

---

## 9. Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Build system (hatchling), deps, ruff/mypy/pytest config |
| `pytest.ini` | Pytest markers: unit, integration, slow, browser, security |
| `chatgpt_auth.json` | Playwright storage_state for ChatGPT (Netscape cookie format) |
| `profiles/*_auth.json` | Playwright storage_state for each provider |
| `profiles/*_cookies.txt` | Netscape-format cookie backups |
| `profiles/*_ui_cookies.txt` | Alternative cookie files per provider |
| `AGENTS.md` | Project rules: phase ordering, constraints, provider matrix |
| `ARCHITECTURE.md` | Original architecture design document |
| `docker/Dockerfile` | Container build for orchestrator |
| `docker/docker-compose.yml` | Full stack: orchestrator + Redis + Postgres + monitoring |
| `monitoring/prometheus.yml` | Scrape config for orchestrator metrics |
| `monitoring/otel-collector.yml` | OpenTelemetry collector config |

---

## 10. Test Coverage

```
ai_orchestrator/tests/
├── unit/ (38 test files)
│   ├── Models: test_account_model.py, test_task_model.py, test_lease_model.py
│   ├── Adapters: test_adapters.py, test_new_adapters.py, test_provider_integration.py
│   ├── Orchestrator: test_control_plane.py, test_workflow_engine.py, test_provider_router.py
│   ├── Scheduler: test_resource_scheduler.py, test_lease_manager.py
│   ├── Runtime: test_runtime_loop.py, test_main.py
│   ├── Infrastructure: test_security.py, test_storage.py, test_utils.py
│   ├── Workspace: test_workspace.py, test_artifacts.py, test_git.py
│   ├── Integration: test_browser_intelligence/ (12 test files)
│   └── Others: test_memory.py, test_observability.py, test_testrunner.py
└── integration/ (1 file)
```

Pytest markers: `unit`, `integration`, `slow`, `browser`, `security`

---

## 11. Scripts (Manual Testing & Diagnostics)

| Script | Purpose |
|--------|---------|
| `scripts/test_all_providers_real.py` | End-to-end provider test (direct + HTTP streaming) |
| `scripts/live_test.py` | Non-headless browser test with real-time monitoring |
| `scripts/quick_test.py` | Quick single-provider test (MiniMax focused) |
| `scripts/diagnose_all_providers.py` | Comprehensive diagnostic: screenshots, a11y, engine monitoring for all 7 providers |
| `scripts/diagnose_zai.py` | Z.ai specific diagnostic |
| `scripts/diagnose_zai_extraction.py` | Z.ai response extraction diagnostic |
| `scripts/test_all_ui_adapters.py` | Test all UI adapters in parallel |
| `scripts/test_other_providers.py` | Test non-primary providers |
| `scripts/test_minimax_single.py` | MiniMax single-provider deep test |
| `scripts/run_parallel_fanout.py` | Parallel fan-out across all providers |
| `scripts/check_browser.py` | Browser connectivity + Playwright health check |
| `scripts/parse_usage_cookies.py` | Cookie file parsing utility |
