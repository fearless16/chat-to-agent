# Browser Intelligence OS — Current State & Shortcomings

> Snapshot of `ai_orchestrator/browser_intelligence/` as it stands today, what
> actually works, and where the system falls short.

---

## 1. Where it lives

```
ai_orchestrator/browser_intelligence/
├── engine.py                  # top-level orchestrator (the "kernel")
├── sensors/                   # 6 raw observation collectors
│   ├── dom_sensor.py
│   ├── accessibility_sensor.py
│   ├── network_sensor.py      # CDP-driven transport intelligence
│   ├── network/               # per-protocol observers
│   │   ├── sse_observer.py
│   │   ├── ws_observer.py
│   │   ├── fetch_observer.py
│   │   ├── protocol_detector.py
│   │   └── stream_parser.py
│   ├── mutation_sensor.py
│   ├── visual_sensor.py
│   ├── performance_sensor.py
│   └── base.py
├── features/
│   ├── feature_vector.py      # 30-dim observation
│   └── feature_composer.py    # rolls all sensors into a vector per tick
├── estimation/                # stochastic state inference
│   ├── belief_state.py        # P(s | o_1..t) over 10 hidden states
│   ├── hmm_engine.py          # forward filter
│   ├── emission_model.py      # learnable P(o | s) with calibration
│   ├── transition_matrix.py   # P(s_t | s_{t-1})
│   └── kalman_filter.py       # response-length smoother
├── decision/                  # action selection
│   ├── completion.py          # "is the response done?" detector
│   ├── confidence.py          # sensor-fused confidence
│   ├── entropy.py             # state-uncertainty metric
│   ├── evidence_fusion.py     # multi-sensor confidence mixing
│   └── utility.py             # argmax_a Σ_s b(s)·R(a,s)
├── events/                    # (empty placeholder)
├── scheduling/                # (empty placeholder)
├── recovery/                  # (empty placeholder)
├── learning/                  # (empty placeholder)
└── intelligence/              # (empty placeholder)
```

**Code size:** ~3,920 LOC of Python in 27 files. The five leaf packages
(`events/`, `intelligence/`, `learning/`, `recovery/`, `scheduling/`) are
empty `__init__.py` placeholders — the architecture document names them but
no code is wired into them.

---

## 2. What actually works

### 2.1 Tick loop (engine.py)

`BrowserIntelligenceEngine` runs a clean 1 Hz pipeline:

```
attach(page)              # wire CDP Network domain
tick(page) → FeatureStore # sense → compose → estimate → decide
is_ready_for_prompt       # P(READY) ≥ adaptive_threshold (0.30 → 0.75)
is_generating             # P(GENERATING) > 0.4 OR P(THINKING) > 0.4
is_error / is_rate_limited
is_response_complete()    # delegated to CompletionEngine
get_response_text()       # assembled text from network observers
detach() / reset()
```

The HMM adaptive threshold rises as the emission model calibrates on
provider-specific signal patterns — this is the core claim of the design and
the **unit tests confirm the math is right** (252/252 pass in
`tests/unit/test_browser_intelligence/`).

### 2.2 Network intelligence (sensors/network_sensor.py + observers)

CDP `Network.enable` is the source of truth. Three observers decode the
traffic:

| Observer | Triggers on | Tracks | New: buffers payload? |
|----------|-------------|--------|------------------------|
| `SSEObserver` | `Network.eventSourceMessageReceived` | event count, bytes, `[DONE]` | ✅ `data` field concatenated |
| `WSObserver` | `Network.webSocketFrameReceived` | frame count, bytes, `[DONE]` | ✅ `payloadData` concatenated |
| `FetchObserver` | `Network.responseReceived` + `dataReceived` (chunked / `text/plain` / `application/x-ndjson`) | chunk count, bytes | ✅ `Network.getResponseBody` on `loadingFinished` |

The `StreamParser` rolls the three streams into a unified `StreamState`
(token rate, idle time, lifecycle flags) using an EMA on chunk timestamps.

The `ProtocolDetector` classifies the response transport as SSE / WebSocket /
fetch-stream / xhr-poll by sniffing `Content-Type` and URL patterns — no
hardcoded provider names.

### 2.3 Estimation (estimation/)

- **HMMEngine** forward-filters over 10 hidden states (`BOOTING`,
  `AUTH_REQUIRED`, `READY`, `PROMPT_SENT`, `GENERATING`, `THINKING`,
  `COMPLETE`, `ERROR`, `RATE_LIMITED`, `SHADOW_BANNED`).
- **EmissionModel** learns `P(o | s)` online and exposes a
  `calibration_score(state)` used to raise the readiness threshold.
- **TransitionMatrix** encodes the legal state graph.
- **BeliefState** holds the full posterior distribution — never a boolean.
- **ResponseKalmanFilter** smooths the per-tick `response_length_delta` to
  detect the "growth went to zero" moment that signals completion.

### 2.4 Decision (decision/)

- **UtilityEngine** has a hand-tuned reward table and picks
  `argmax_a Σ_s b(s)·R(a,s)`.
- **CompletionEngine** says "yes, the response is finished" using a
  velocity/acceleration zero-crossing AND a stream-idle timeout, so no
  `sleep()` is on the critical path.
- **ConfidenceEngine / EntropyEngine** expose `P(state)` quality metrics for
  observability and adaptive thresholding.
- **EvidenceFusion** mixes per-sensor confidence into a single number that
  the adaptive threshold consults.

### 2.5 FeatureComposer (features/)

Every tick, six sensors fire (max one Playwright RPC per sensor) and roll
into a 30-dimensional `FeatureVector` (input visible, send enabled, stop
visible, regenerate visible, error banner, auth form, text-input count,
button count, thinking/error/rate-limit/streaming markers, transport,
generation started/completed, tokens/sec, idle time, chunks, bytes,
request rate, mutation rate, mutation acceleration, JS heap, page
stability, response length, response length delta, visual stability, title,
URL).

The composer also runs a built-in `_default_extract_response_length(page)`
which is a tiny DOM probe — **a legacy escape hatch, not a primary path.**

### 2.6 Tests

- **1,228 unit tests pass** project-wide.
- **252 of those** are brain-specific (`tests/unit/test_browser_intelligence/`):
  emission model calibration, feature vector composition, network observers
  with synthetic CDP events, runtime improvements (stream-stalled detection,
  evidence fusion, adaptive threshold).
- The OBSERVABILITY coverage shows the engine's own modules at 76% line
  coverage; many leaf decision modules hit 100%.

---

## 3. How the engine is wired into the FastAPI app

The engine is **the** readiness / streaming detector for every browser
adapter. The 7 UI adapters (`chatgpt_ui`, `deepseek_ui`, `kimi_ui`,
`minimax_ui`, `qwen_ui`, `xiaomimimo_ui`, `zai_ui`) all subclass
`EngineUIAdapter` and use it like this:

```python
async def _real_send(self, prompt):
    page = await self._get_page()
    engine = BrowserIntelligenceEngine()
    await engine.attach(page)            # CDP Network.enable

    await self._wait_until_ready(engine, page)   # poll until is_ready_for_prompt
    engine.reset()                       # clear observer buffers
    await self._execute_type_prompt(page, prompt) # click + fill + Enter
    content = await self._wait_for_response(engine, page)  # poll completion
```

The FastAPI `/chat` endpoint picks an adapter (or fans out across
providers), wires auth from `profiles/{name}_cookies.txt` or
`{name}_auth.json`, and calls `EngineUIAdapter.fan_out()` for parallel
multi-provider calls.

---

## 4. Shortcomings (the honest list)

### 4.1 The engine detects streaming but does not return the chat text reliably

`engine.get_response_text()` exists, but the path is shaky in practice.

- **SSEObserver** only fires for the browser's `EventSource` API. **Every
  modern chat UI uses `fetch()` with `ReadableStream`**, so the SSE observer
  gets zero chat traffic. It does fire for the post-load pings
  (e.g. login-status, "is the user signed in") and those pollute the buffer.
- **WSObserver** is correct (WebSocket frames carry the payload), but
  ChatGPT / DeepSeek / Qwen / Kimi / z.ai / MiniMax / Xiaomi MiMo **do not
  use WebSockets** — they all use fetch streams.
- **FetchObserver** needs `Network.loadingFinished` to call
  `Network.getResponseBody`, which **only works after the stream has fully
  closed**. For a 10-second model response that means the body is
  unavailable until the very end, and any prior non-chat fetch response in
  the same buffer slot wins the "longest text" comparison and gets
  returned as the "answer".
- Live test result on z.ai: `engine.get_response_text()` returned
  `{"ResultObject":true,"RequestId":"...","Code":"200"}` — a 65-byte
  tracking ping, not the chat response. **The engine knows the stream
  happened but cannot give you the words.**

**Workaround in production today:** the engine drives the *lifecycle*
(when to type, when to click, when the response is complete). The actual
text is captured by a **JS-side `window.fetch` interceptor** installed
via `page.add_init_script()`, which clones each `fetch()` response and
stores its body in `window.__engine_response_bodies__`. The
`EngineUIAdapter._extract_response_text` static method reads that buffer
and runs my SSE-aware parser. This works — it is what produced the
successful `"1. **Analyze the Request:** … 2+2=4 …"` reply — but it
sidesteps the engine rather than using it.

### 4.2 No live body capture for fetch streams

The only CDP API that would solve this is `Network.streamResource` /
`Network.streamResourceContent`, which streams response body chunks in
real time. It is **not wired up**. Without it, fetch-based SSE has a
fundamental gap: the body is opaque to the engine until the stream ends.

### 4.3 Chat-URL filter is too loose

`EngineUIAdapter._looks_like_chat_response` uses substring matches like
`/v1/messages` and `/chat/message` which also fire for non-chat
endpoints (analytics, conversation list, friend/thread messages). On
DeepSeek in particular, the conversation-list API returns a body that
matches the filter and is mistaken for the chat response. Combined with
the buffer-priority logic in `_extract_response_text`, the wrong body
wins.

### 4.4 DOM extractor is needed as a backstop, but copies button labels

`RESPONSE_EXTRACT_SCRIPT` reads `innerText` from the assistant message
bubble. The bubble contains the action toolbar (Copy / Regenerate /
Thumbs up / Thumbs down / Share), so the extracted text often includes
those labels. A regex filter on `aria-label` / `title` / `<button>`
elements was just added, but the JS-side filter is brittle: providers
ship new button labels and the regex has to be updated.

### 4.5 No body-text filter on the network observer buffer

The fetch/SSE/WS observers buffer whatever the network produces.
`engine.reset()` clears the buffer, but **anything fired between
`engine.reset()` and the chat response arriving is also captured.** A
small request like a post-prompt telemetry ping (e.g. `ResultObject` /
`RequestId`) wins on length, gets coerced into "no delta keys found,
return raw text", and is reported as the answer.

### 4.6 Many advertised submodules are empty placeholders

`events/`, `intelligence/`, `learning/`, `recovery/`, `scheduling/` are
all single-line `__init__.py` files. The architecture diagram and
markdown claim:

- A **learning** module that updates emission probabilities from
  per-session reward.
- A **recovery** module that re-runs the flow when the engine lands in
  `ERROR` or `SHADOW_BANNED`.
- A **scheduling** module that times out stuck pages.
- An **events** module for emitting a structured event log.

None of these exist as code. The engine's `record_reward()` is a `pass`
stub.

### 4.7 Engine emits "ERROR" too aggressively on some providers

`MiniMax` and `Kimi` probe runs reported `is_error == True` from
`P(ERROR) > 0.5`. The page was actually ready — the HMM emission model
just hasn't been calibrated for those providers' signal patterns, so a
harmless class name like `[class*="error"]` for a hidden toast inflates
the error probability. There is no per-provider emission pre-training
or warm-up.

### 4.8 Cookie freshness is not validated by the engine

`/chat` happily loads any `*_cookies.txt` file from `profiles/` and hands
it to Playwright. If the cookies are expired, the engine sees a
login form, transitions to `AUTH_REQUIRED`, raises, and the request
fails with `"Engine error state: AUTH_REQUIRED"`. There is no
preflight that says "these cookies are stale, refresh them" — the
caller is left to figure it out. DeepSeek and Xiaomi MiMo are stuck
exactly here.

### 4.9 The HMM is not retrained across sessions

`EmissionModel` is learnable in principle, but every `BrowserIntelligenceEngine()`
instance starts with a fresh random calibration. The "adaptive threshold
rises as the emission model learns" claim only holds **within a single
session**. There is no cross-session persistence — open the engine twice
on the same provider and you start at 0.45 again.

### 4.10 No headless-cloud bypass

`chromium.launch(headless=True)` is what every UI adapter calls. Some
providers (ChatGPT, DeepSeek) raise a Cloudflare challenge or AWS WAF
captcha on headless Chrome. The `_stealth` flag exists
(`--disable-blink-features=AutomationControlled`, custom UA, viewport)
but the bypass is shallow. There is no `x-evil-puppeteer`, no real
fingerprint randomization, no `playwright-stealth` plugin.

### 4.11 The shadow_banned state has no recovery path

`HiddenState.SHADOW_BANNED` is in the state graph and the utility
matrix rewards `recover` and `quarantine` actions — but the
`EngineUIAdapter` has no implementation of either. The adapter just
raises and the request fails.

### 4.12 Engine is single-page-per-instance

A new `BrowserIntelligenceEngine()` is constructed for every send. The
CDP session is attached, the tick loop runs, then the engine is
discarded. This means calibration is per-call, not per-page. Fan-out
across 7 providers in parallel spawns 7 isolated engines with zero
shared knowledge. The architecture doc implies a per-page long-lived
runtime, but the API adapter doesn't use it that way.

### 4.13 Test coverage of the integration path is 0%

`tests/unit/test_adapters.py` has 48 unit tests, all of them mocking out
the browser. The end-to-end path (real Chromium, real provider, real SSE
stream) has **zero automated coverage**. A successful `/chat` call was
confirmed manually on z_ai, but no CI gate exists.

### 4.14 No completion acknowledgement for the response

Once `_wait_for_response` returns, the engine is detached and the
playwright page is left open. There is no "is the model still
generating?" re-check before the adapter hands the text to the caller.
On a flaky network the adapter can return a truncated response with no
way for the caller to know.

---

## 5. Quick verdict

**The engine's value today is its lifecycle detection**, not its content
capture. It tells you *when* to type, *when* the response is happening,
and *when* it's done — that part is solid (252 unit tests, sound math,
real HMM). The actual response text still comes from a JS-side fetch
interceptor that runs alongside the engine.

To close the gap, the engine needs:
1. **Network.streamResource** wired into `NetworkSensor` so fetch-stream
   bodies are captured chunk-by-chunk, not just at `loadingFinished`.
2. **A chat-URL filter on the fetch/SSE/WS observer buffers** so
   non-chat responses (analytics, login, conversation list) cannot
   pollute `_response_text`.
3. **Per-page engine reuse** so emission calibration accumulates
   across sends, plus **persistence** so a re-opened browser doesn't
   start cold.
4. **The four empty submodules filled in** (`events`, `learning`,
   `recovery`, `scheduling`) — they are not just decorative; the
   current system has no event log, no cross-session learning, no
   recovery, and no scheduling.
5. **An integration test harness** that exercises a real provider in
   CI, even if it's gated behind a cookie-availability env var.
