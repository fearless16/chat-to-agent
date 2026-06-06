# Browser Intelligence OS

Autonomous browser runtime for AI chat providers with probabilistic state estimation.

---

## Architecture

```
Sensor → Feature → Estimation → Decision → Action
  ↑        ↑           ↑            ↑         ↑
  DOM     Vector      HMM        Utility    Type
  A11y    (30-dim)    Emission   Completion  Click
  Net     Store       Kalman     Confidence  Extract
  Mut     RingBuf     Belief     Entropy     Recover
  Perf                State
  Visual
```

The pipeline runs at 1 Hz. Each tick: six sensors produce raw observations (max 1 Playwright
RPC per sensor), the FeatureComposer assembles a 30-dimensional feature vector, the HMM
engine forward-filters to update the belief state with online emission learning, and the
decision layer selects the utility-maximizing action. No LLM is on the critical path.
Readiness detection uses an adaptive threshold (0.30 → 0.75) that rises as the emission
model calibrates.

---

## Key Principles

- **No LLM in critical path.** Inference runs purely on signal processing — frequency,
  velocity, acceleration, entropy. LLMs invoked only for high-level reasoning when state
  confidence crosses a threshold.
- **No `sleep()` or fixed timers.** Idle detection uses wall-clock deltas from the last
  chunk timestamp; completion uses stream-idle thresholds.
- **Network-first priority.** Stream signals (tokens/sec, chunk counts, transport state)
  are the source of truth. DOM and accessibility are rendering artifacts. Hierarchy:
  `stream > accessibility > DOM > vision`.
- **Probabilistic belief states, not booleans.** Every state judgment carries a
  probability distribution. Binary flags replaced with `P(GENERATING) = 0.73,
  P(COMPLETE) = 0.21`.
- **Multi-hypothesis tracking.** Belief state holds a full distribution over 10 hidden
  states. Forward algorithm maintains all hypotheses; Viterbi decodes the most likely
  sequence.
- **Accessibility tree is canonical representation.** When network signals are
  insufficient, the ARIA tree provides structured UI semantics (role, state, name) that
  DOM selectors cannot match.

---

## Hidden States

The provider page is modeled as a hidden-state system with 10 states:

| State | Description |
|---|---|
| `BOOTING` | Page loading, DOM not yet stable |
| `AUTH_REQUIRED` | Login form detected, session expired |
| `READY` | Input visible, send enabled, waiting for prompt |
| `PROMPT_SENT` | Prompt just typed, transitioning toward generation |
| `THINKING` | Reasoning phase (e.g., DeepSeek-R1 chain-of-thought) |
| `GENERATING` | Tokens flowing, stream active, response growing |
| `COMPLETE` | Generation finished, stop button gone, regenerate visible |
| `RATE_LIMITED` | Error banner + rate-limit marker (429 or platform cap) |
| `ERROR` | Error banner, no streaming, page may be broken |
| `SHADOW_BANNED` | Page renders normally but requests silently dropped |

---

## Sensor Architecture

Six sensors feed the FeatureComposer. Sensors are stateless — all time-series state
lives in the FeatureStore ring buffer.

### DOM Sensor
Single `page.evaluate()` call runs all selector checks and node counts browser-side.
6 selector groups evaluated in one JS pass: input, send, stop, regenerate, error, auth.
Boolean presence flags only — never reads text content.

### Accessibility Sensor
Reads the Playwright accessibility snapshot (ARIA tree). Counts text inputs and buttons,
detects thinking markers ("Thinking…", "Reasoning…"), error markers, rate-limit markers,
and streaming indicators via ARIA role/state patterns.

### Network Sensor (KEY UPGRADE)
Intercepts all network traffic via CDP `Network` domain. Owns four sub-components:

- **`ProtocolDetector`** — Auto-detects transport protocol (SSE/WS/Fetch/XHR) from
  content-type, status code, URL patterns, and transfer-encoding. No provider names are
  hardcoded.
- **`SSEObserver`** — Intercepts `Network.eventSourceMessageReceived` to track
  EventSource streams. Counts data chunks, bytes, and `[DONE]` markers.
- **`WSObserver`** — Tracks WebSocket connections (`webSocketCreated`,
  `webSocketFrameReceived`) and analyzes frame patterns for stream lifecycle.
- **`FetchObserver`** — Monitors chunked transfer-encoding responses via
  `Network.dataReceived`, detecting streaming even for `application/json`.

Network signals have priority over DOM for generation/completion detection.

### Mutation Sensor
Tracks DOM mutation rate and acceleration. High mutation rate (>5/tick) + positive
acceleration → GENERATING. Mutation spike with stream inactive → PROMPT_SENT. Rate
approaching zero with stream closed → COMPLETE.

### Performance Sensor
Collects JS heap size (`Performance.getMetrics`), page stability (resource timing
variance), and frame rate metrics via CDP `Performance` domain.

### Visual Sensor
Screenshot-based stability check. Compares consecutive viewport hashes or pixel diff
ratios to detect layout shifts during generation.

---

## Feature Vector (v2)

30-dimensional observation vector produced every tick. FEATURE_DIM = 30.

**18 binary features (indices 0-17):**
`input_visible`, `send_enabled`, `stop_button_visible`, `regenerate_visible`,
`error_banner_visible`, `auth_form_visible`, `text_input_count`, `button_count`,
`has_thinking_marker`, `has_error_marker`, `has_rate_limit_marker`,
`has_streaming_marker`, `stream_active`, `transport_detected`, `generation_started`,
`generation_completed`, `stream_closed`, `generation_stop_detected`.

**12 continuous features (indices 18-29):**
`mutation_rate`, `mutation_acceleration`, `js_heap_used_mb`, `page_stability`,
`response_length`, `response_length_delta`, `visual_stability`, `tokens_per_second`
(EMA-smoothed), `stream_idle_time` (seconds since last chunk), `total_chunks`,
`bytes_received`, `network_request_rate`.

FeatureStore: 300-tick ring buffer (5 minutes at 1 Hz) with EMA, derivative,
second-derivative, mean, std, and `aged_mean()` (exponential decay with configurable
half-life).

---

## Estimation Engine

### HMM Engine
Hidden Markov Model with the forward algorithm for online belief updates:

```
b_t(s_j) = P(O_t | S_t=s_j) * sum_i b_{t-1}(s_i) * A[i][j]
b_t = normalize(b_t)
```

### Transition Matrix
Empirically-derived transition probabilities between all 10 hidden states. Stored as
log-probabilities with configurable epsilon smoothing. `validate_stochastic()` checks
Σ_j A[i][j] = 1.0 per row. `enforce_stochastic()` re-normalizes. `is_ergodic()` verifies
every state reachable from every other state. Key transitions: READY →
PROMPT_SENT (0.15), PROMPT_SENT → GENERATING (0.40), GENERATING → COMPLETE (0.20),
GENERATING → GENERATING (0.70, self-loop).

### Emission Model (FEATURE_DIM=30)
Mixture of Bernoulli (binary) + Gaussian (continuous) log-likelihoods. Each hidden
state has a characteristic feature signature — e.g., GENERATING expects
`stop_button_visible=true, has_streaming_marker=true, mutation_rate ~8.0,
tokens_per_second ~15.0`. Online learning via soft assignment: each state gets a
fraction of the update proportional to belief mass, with learning rate scaled by
belief (lr = 0.01 + belief * 0.07). `calibration_score()` reports model readiness.
Parameters clamped: binary p ∈ [0.01, 0.99]; sigma ≥ 0.1.

### Kalman Filter (ResponseKalmanFilter)
State vector [length, velocity, acceleration]^T with constant-acceleration dynamics.
Smooths the noisy `response_length` signal. Used by CompletionEngine to detect
velocity → 0 and acceleration → 0 conditions.

### Belief State
Probability distribution over 10 hidden states summing to 1.0. Properties:
`most_likely`, `confidence` (max p), `entropy` (Shannon bits). Initialized uniform
(0.1 each). Readiness uses adaptive threshold rising from 0.30 (cold start)
to 0.75 (fully calibrated). Confidence gates: action when `is_confident(threshold)`.

---

## Decision Layer

### Completion Detection
`CompletionEngine` — no `sleep()`. Observes stream idle time, Kalman velocity, and
transport state:

- Velocity: `|v| < 2.0` for ≥ 3 consecutive ticks (Kalman-smoothed)
- Acceleration: `|a| < 1.0`
- Stream: `!has_streaming_marker && !stop_button_visible` OR `generation_stop_detected`
- Content guard: `response_length > 20`
- StreamParser idle: `idle > 5s + transport_disconnected` OR `idle > 10s` (hard timeout)

### State Transitions via Stream Signals
- Last chunk received + idle > threshold + transport disconnected → COMPLETE
- Tokens flowing + stream_active → GENERATING
- Mutation spike + stream inactive → PROMPT_SENT
- Stream active but idle > 5s + zero tokens → stream stalled (triggers recover action)

### Utility-Based Action Selection
`UtilityEngine` computes `E[U(a)] = sum_s P(s) * R(a, s)` for candidate actions gated by
the most-likely state. READY → `type_prompt`, PROMPT_SENT → `click_send`, GENERATING →
`wait`, COMPLETE → `extract_response`, ERROR → `recover/refresh`, RATE_LIMITED →
`wait/quarantine`, AUTH_REQUIRED → `relogin`, SHADOW_BANNED → `quarantine`, BOOTING →
`wait/refresh`, THINKING → `wait`.

### Evidence Fusion
Multi-sensor confidence tracking per sensor (6 trackers). `EvidenceFusion` weights
readings by real-time confidence scores with exponential aging. Consecutive sensor
failures (≥5) drop confidence to 0.1. Stale data (>60s without success) penalized.

### Recovery Cascade
When ERROR or SHADOW_BANNED detected: `selector_cache → a11y → graph → network →
session → worker → provider → replan`.

### Confidence & Entropy Gates
`ConfidenceEngine` blends observation quality, historical success, and sensor reliability
(observation 0.25, historical 0.25, selector 0.20, accessibility 0.15, network 0.15).
`EntropyEngine` suppresses expensive actions when entropy > 2.0 bits. Triggers recovery
when both GENERATING and READY probabilities fall below 0.3.

---

## Network-First Intelligence

The NetworkSensor replaces brittle DOM-watching with CDP-level network interception:

```
CDP Network.enable
  ├── Network.requestWillBeSent       → ProtocolDetector
  ├── Network.responseReceived        → ProtocolDetector + Observers
  ├── Network.dataReceived            → FetchObserver (chunked transfer)
  ├── Network.loadingFinished/Failed  → Transport lifecycle signals
  ├── Network.webSocketCreated/Closed → WSObserver
  ├── Network.webSocketFrameReceived  → WSObserver (frame analysis)
  └── Network.eventSourceMessageReceived → SSEObserver
```

**`ProtocolDetector`** auto-detects transport without hardcoding providers:
- SSE: `text/event-stream` → confidence 0.85
- WebSocket: status 101 → 0.90; explicit `webSocketCreated` → 0.95
- Fetch stream: `application/json` + `transfer-encoding: chunked` → 0.70
- XHR poll: `x-requested-with: xmlhttprequest` → 0.30

**`StreamParser`** produces protocol-agnostic metrics:
- `tokens_per_second`: EMA-smoothed chunk rate over 5s sliding window
- `stream_idle_time`: wall-clock time since last `dataReceived`
- `total_chunks` / `bytes_received`: cumulative across all transports
- Lifecycle: first chunk → `generation_started`; idle > threshold + disconnected →
  `stream_closed`

Stream signals override DOM signals. If StreamParser says `stream_active=true` and DOM
says `stop_button_visible=false`, the network signal wins.

---

## Quick Start

```python
from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine

engine = BrowserIntelligenceEngine()
await engine.attach(page)

# Tick loop for ready detection
while True:
    await engine.tick(page)
    if engine.is_ready_for_prompt:
        break
    if engine.is_error:
        raise RuntimeError(f"State: {engine.most_likely_state}")

# Send prompt, then wait for completion
await page.click("#prompt-textarea")
await page.keyboard.type("Write a Python function...")

while True:
    await engine.tick(page)
    done, confidence = engine.is_response_complete()
    if done:
        break
    if engine.most_likely_state == HiddenState.ERROR:
        break
```

---

## Testing

```bash
.venv/bin/pytest ai_orchestrator/tests/unit/test_browser_intelligence/ -q
# 266 passed (0 fail, 0 skip)
```

Unit tests cover all components — FeatureVector/FeatureStore, all six sensors, BeliefState,
TransitionMatrix, EmissionModel, HMMEngine, KalmanFilter, ConfidenceEngine, EntropyEngine,
CompletionEngine, UtilityEngine, FeatureComposer, and BrowserIntelligenceEngine with a
full READY → GENERATING → COMPLETE pipeline simulation. Pure data, no Playwright needed.
100% deterministic.

---

## Design Invariants

1. **No `sleep()` anywhere in BIOS.** All timing from `time.monotonic()` deltas.
2. **FeatureStore is the sole time-series memory.** Sensors are stateless; state decays
   through the ring buffer.
3. **BeliefState always sums to 1.0.** Normalization enforced in `__post_init__`.
4. **Stream signals override DOM.** Network `stream_active` wins over DOM indicators.
5. **Entropy gates expensive actions.** `recover()` blocked if entropy > 2.0 and neither
   GENERATING nor READY has P > 0.3.
6. **Emission parameters clamped.** Binary p in [0.01, 0.99]; sigma ≥ 0.1.
7. **HMM forward algorithm uses log-space** for numerical stability.
8. **Transition matrix: Laplace smoothing** guarantees all transitions > 1e-6 in
   log-space.
9. **FeatureStore capacity = 300 ticks** (5 min at 1 Hz); oldest entries silently evicted.
10. **FEATURE_DIM = 30.** 18 Bernoulli (binary) + 12 Gaussian (continuous).
11. **Transition matrix validated.** `validate_stochastic()` checks Σ=1.0 per row;
    `enforce_stochastic()` re-normalizes. `is_ergodic()` verifies all-reachable.
12. **FeatureStore capacity ≥ 1.** Raises `ValueError` for capacity=0.
13. **Emission learning gated.** Soft-assignment updates skip states with belief < 1e-6.
    Learning rate scales with belief mass: lr = 0.01 + belief * 0.07.
14. **Adaptive readiness threshold.** Starts at 0.30 (uncalibrated), asymptotes to 0.75
    as emission model collects observations. Formula: `base + (max-base) * calibration`.
