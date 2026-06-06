# BROWSER INTELLIGENCE OPERATING SYSTEM — V2

> **A browser runtime that thinks in probability, not language.**
>
> No LLM in the critical path. Hidden-state inference, network-first signals, 30-dim
> feature vectors, stochastic state estimation running at 1 Hz on every tick.

---

## 1. ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           TICK (1 Hz)                                    │
│                                                                          │
│  SENSOR LAYER          FEATURE LAYER       ESTIMATION        DECISION    │
│  ┌──────────┐          ┌───────────┐       ┌─────────┐       ┌───────┐  │
│  │ DOM      │──┐       │ Feature   │       │ HMM     │       │Utility│  │
│  │ A11y     │  │       │ Vector    │──────▶│ Forward │──────▶│Engine │  │
│  │ Network  │──┤──────▶│ (30-dim)  │       │ Algo    │       │       │  │
│  │ Mutation │  │       │           │       │         │       ├───────┤  │
│  │ Visual   │  │       └───────────┘       │ Emission│       │Comple-│  │
│  │ Perf     │──┘               │           │ Model   │       │tion   │  │
│  └──────────┘                  │           ├─────────┤       │Detect │  │
│                                ▼           │ Kalman  │       └───────┘  │
│                         ┌───────────┐      │ Filter  │           │      │
│                         │ Feature   │      └─────────┘           │      │
│                         │ Store     │            │                │      │
│                         │ RingBuf   │            ▼                ▼      │
│                         │ (300 cap) │      ┌─────────┐     ┌──────────┐  │
│                         └───────────┘      │ Belief  │     │ Action   │  │
│                                            │ State   │────▶│ Selection│  │
│                                            │ P(s|O)  │     └──────────┘  │
│                                            └─────────┘                  │
└──────────────────────────────────────────────────────────────────────────┘
```

Six independent sensors produce raw observations. The FeatureComposer assembles a
30-dimensional feature vector. The HMM forward-filters to update the belief
distribution over 10 hidden states. The utility engine selects the
reward-maximizing action. Completion is detected by velocity/acceleration
zero-crossing AND stream-idle timeout — no `sleep()` anywhere.

---

## 2. KEY PRINCIPLES

| Principle | Implementation |
|-----------|---------------|
| **No LLM in critical path** | Brute-force signal processing: frequency, velocity, acceleration, entropy |
| **No `sleep()` / fixed timers** | Wall-clock deltas from `time.monotonic()`. Idle thresholds, not wait periods |
| **Network-first** | Stream signals (tokens/sec, chunks, transport state) are truth. DOM/a11y are artifacts |
| **Probabilistic, not boolean** | `P(GENERATING) = 0.73` — never `is_generating = True` |
| **Multi-hypothesis** | Belief state holds full distribution. Forward algorithm maintains all. Viterbi decodes |
| **Accessibility canonical** | `page.aria_snapshot()` — semantic, stable, structured. CSS selectors are fallback |

### Signal Priority

```
Network Stream (tokens/sec, chunk frequency, transport state)
    ↓
Accessibility Tree (roles, states, semantic markers)
    ↓
DOM (element presence/absence only, no text content)
    ↓
Visual (screenshot stability, layout shift)
```

---

## 3. HIDDEN STATES

The provider is modeled as a **partially observable stochastic system**. The true
state is never directly observed. Only evidence is.

```
BOOTING ──────▶ AUTH_REQUIRED ──▶ READY ◀────────────────────┐
    │                │               │                         │
    ▼                ▼               ▼                         │
  ERROR           ERROR        PROMPT_SENT                     │
    ▲                             │                           │
    │                    ┌────────┴────────┐                  │
    │                    ▼                 ▼                  │
    │               THINKING          GENERATING               │
    │                    │                 │                   │
    │                    ▼                 ▼                   │
    │                 ERROR            COMPLETE ───────────────┘
    │                    ▲                 │
    └────────────────────┘                 ▼
                        RATE_LIMITED ◀── ERROR
                            │
                            ▼
                      SHADOW_BANNED
```

| # | State | Evidence Signature |
|---|-------|-------------------|
| 0 | `BOOTING` | High DOM mutation, low a11y confidence, page stability < 0.5 |
| 1 | `AUTH_REQUIRED` | auth_form_visible=True, send_enabled=False, input_visible=False |
| 2 | `READY` | input_visible=True, send_enabled=True, stream_idle=high, mutation=0 |
| 3 | `PROMPT_SENT` | Mutation spike, network request burst, generation_started=True |
| 4 | `THINKING` | streaming_marker=True, thinking_marker=True, stream_idle=moderate |
| 5 | `GENERATING` | stream_active=True, tokens_per_second > 0, total_chunks growing |
| 6 | `COMPLETE` | stream_closed=True, idle > 5s, velocity≈0, acceleration≈0 |
| 7 | `RATE_LIMITED` | error_banner=True, rate_limit_marker=True, stream_closed=True |
| 8 | `ERROR` | error_banner=True, error_marker=True, stream_closed=True |
| 9 | `SHADOW_BANNED` | Appears COMPLETE but response=empty, quality=low, stream_closed=True |

---

## 4. SENSOR ARCHITECTURE

Every sensor is an independent observer. No sensor accesses another sensor's
state. No sensor makes decisions. Sensors only emit features.

### 4.1 DOM Sensor
**Output:** `DOMFeatures` — 8 fields
```
input_visible, send_visible, stop_button_visible, regenerate_visible,
error_banner_visible, auth_form_visible, dom_node_count, interactive_count
```
Uses broad CSS selector categories (not specific class names). Falls back
gracefully — all features return `False` on failure.

### 4.2 Accessibility Sensor
**Output:** `AccessibilityFeatures` — 9 fields
```
text_input_count, button_count, article_count, has_thinking_marker,
has_error_marker, has_rate_limit_marker, has_streaming_marker,
semantic_tree_depth, accessibility_confidence
```
Uses `page.aria_snapshot()` via `AccessibilityRuntime`. Extracts semantic markers
from text content. Confidence computed as fraction of named nodes.

### 4.3 Network Sensor (v2 — FULL REWRITE)
**Output:** `NetworkFeatures` — 15 fields

The network sensor is the **primary intelligence source**. It orchestrates four
sub-observers and a protocol-agnostic stream parser.

```
NetworkSensor
    ├── ProtocolDetector     — auto-detect SSE/WS/Fetch/XHR, confidence 0.0–1.0
    ├── SSEObserver          — intercept event-stream, count chunks, detect [DONE]
    ├── WSObserver           — intercept WS frames, count data frames, detect [DONE]
    ├── FetchObserver        — detect chunked transfer-encoding, count data chunks
    └── StreamParser         — protocol-agnostic: EMA token rate, idle time, lifecycle
```

**Key fields:**
```
stream_active: bool           — data received within last 2 seconds
tokens_per_second: float      — EMA-smoothed chunk rate (α=0.3)
stream_idle_time: float       — seconds since last chunk (wall-clock)
total_chunks: int             — cumulative chunks from all observers
bytes_received: int           — cumulative bytes from all observers
generation_started: bool      — first chunk after >5s idle
generation_completed: bool    — stream_closed AND generation_was_active
stream_closed: bool           — idle > 5s + transport disconnected OR > 10s hard
transport_protocol: str       — "sse"|"websocket"|"fetch_stream"|"xhr"|"unknown"
transport_detected: bool
transport_confidence: float   — 0.0–1.0
```

**Transport detection is protocol-agnostic — zero hardcoded provider names.**

### 4.4 Mutation Sensor
**Output:** `MutationFeatures` — 3 fields
```
mutation_count, mutation_rate, mutation_acceleration
```
Injects a `MutationObserver` into the page. Mutation rate is the most powerful
signal for detecting DOM activity (generation spikes to 20–50/sec).

### 4.5 Performance Sensor
**Output:** `PerformanceFeatures` — 5 fields
```
js_heap_used_mb, js_heap_limit_mb, dom_node_count,
layout_shift_count, page_load_stable
```
Uses `performance.memory` and `document.querySelectorAll('*').length`.
Page stability = DOM node count delta < 50 between ticks.

### 4.6 Visual Sensor
**Output:** numerical `visual_stability` — 0.0–1.0
```
visual_stability: float
```
Screenshot-based change detection. Used as a fallback when a11y fails.

---

## 5. FEATURE VECTOR — 30 DIMENSIONS

Every tick, the `FeatureComposer` assembles all sensor outputs into a single
30-dimensional feature vector.

### Dimension Layout

```
Binary features (0–17, Bernoulli-distributed):
 0: input_visible           6: text_input_count
 1: send_enabled            7: button_count
 2: stop_button_visible     8: has_thinking_marker
 3: regenerate_visible      9: has_error_marker
 4: error_banner_visible   10: has_rate_limit_marker
 5: auth_form_visible      11: has_streaming_marker
                           12: stream_active          ← NETWORK (new)
                           13: transport_detected     ← NETWORK (new)
                           14: generation_started     ← NETWORK (new)
                           15: generation_completed   ← NETWORK (new)
                           16: stream_closed          ← NETWORK (new)
                           17: generation_stop_detected

Continuous features (18–29, Gaussian-distributed):
18: mutation_rate           24: visual_stability
19: mutation_acceleration   25: tokens_per_second      ← NETWORK (new)
20: js_heap_used_mb         26: stream_idle_time       ← NETWORK (new)
21: page_stability          27: total_chunks           ← NETWORK (new)
22: response_length         28: bytes_received         ← NETWORK (new)
23: response_length_delta   29: network_request_rate   ← was binary, now continuous
```

Total: 18 binary + 12 continuous = 30 dimensions.

### FeatureStore — Ring Buffer
- Capacity: 300 ticks (5 minutes at 1 Hz)
- Provides: `window(n)`, `ema(field, n)`, `derivative(field)`, `mean(field)`, `std(field)`

---

## 6. ESTIMATION ENGINE

### 6.1 HMM Forward Algorithm

Core belief update equation:

```
b_t(s) = η · P(O_t | S_t = s) · Σ_{s'} P(S_t = s | S_{t-1} = s') · b_{t-1}(s')
```

Where:
- `b_t(s)` — belief that current state is `s` at time `t`
- `P(O_t | S_t = s)` — emission probability (how likely is this observation given state `s`?)
- `P(S_t = s | S_{t-1} = s')` — transition probability
- `η` — normalization constant

Implementation runs in O(N^2) where N=10 states. At 1 Hz, this is negligible.

### 6.2 Emission Model — P(O | S)

**FEATURE_DIM = 30** (18 binary + 12 continuous)

**Binary features (0–17):** Bernoulli likelihood
```
P(f_i | S=s) = p_i^f_i · (1 − p_i)^(1 − f_i)
```
where `p_i ∈ [0.01, 0.99]` is the learned Bernoulli parameter.

**Continuous features (18–29):** Gaussian likelihood
```
P(f_j | S=s) = (1 / √(2πσ²)) · exp(−(f_j − μ)^2 / 2σ²)
```
where `μ` and `σ=50.0` (wide tolerance initially, narrows with learning).

**Log-space computation** for numerical stability:
```
log P(O|S) = Σ_i [f_i·log(p_i) + (1−f_i)·log(1−p_i)] + Σ_j [−½·(f_j−μ_j)²/σ_j² − log(σ_j·√(2π))]
```

Response extraction fallback: `page.evaluate()` finds `[data-message-author-role="assistant"]`,
then `article`, then `[class*="message"]` — returns last block's `innerText` length.

### 6.3 Emission Defaults — Per-State Signatures

| State | Binary Key Signals | Continuous Key Signals |
|-------|-------------------|----------------------|
| BOOTING | all 0.3 | heap=80μ, stability=0.3μ, stream_idle=30μ |
| AUTH_REQUIRED | auth=0.7, input/send=0.3 | all near 0 |
| READY | input=0.7, send=0.7, stream_active=0.3 | response=100μ, stream_idle=10μ |
| PROMPT_SENT | transport=0.7, gen_started=0.7 | tps=1μ, stream_idle=2μ, req_rate=3μ |
| THINKING | thinking=0.7, transport=0.7, gen_started=0.7 | tps=0μ, stream_idle=2μ |
| GENERATING | stream_active=0.7, transport=0.7, gen_started=0.7 | **tps=15μ**, idle=0.2μ, chunks=50μ, bytes=5000μ |
| COMPLETE | gen_completed=0.7, stream_closed=0.7, stop=0.7 | **tps=0μ**, **idle=5μ**, chunks=200μ, bytes=20000μ |
| RATE_LIMITED | error=0.7, rate_limit=0.7, stream_closed=0.7 | stream_idle=20μ, req_rate=0.5μ |
| ERROR | error=0.7, stream_closed=0.7 | all near 0 |
| SHADOW_BANNED | transport=0.7, stream_closed=0.7 | stream_idle=30μ |

### 6.4 Transition Matrix

```
READY → READY (0.80), PROMPT_SENT (0.15), RATE_LIMITED (0.02), ERROR (0.03)
PROMPT_SENT → THINKING (0.40), GENERATING (0.40), ERROR (0.10)
GENERATING → GENERATING (0.70), COMPLETE (0.20), ERROR (0.08)
COMPLETE → READY (0.55), COMPLETE (0.40), ERROR (0.05)
```

Laplace smoothing applied. Log-probability storage for numerical stability.
Baum-Welch re-estimation from Viterbi-decoded sequences for offline learning.

### 6.5 Kalman Filter — Response Length Smoothing

**State vector:** `[length, velocity, acceleration]^T`

```
Predict:  x_{t|t−1} = F · x_{t−1|t−1}
Update:   x_{t|t} = x_{t|t−1} + K · (z_t − H · x_{t|t−1})
```

State transition (constant-acceleration model):
```
F = [[1, 1, 0.5],
     [0, 1, 1.0],
     [0, 0, 1.0]]
```

Used by the CompletionEngine to detect velocity→0 and acceleration→0.

### 6.6 Belief State

```python
BeliefState {
    probabilities: dict[HiddenState, float]  # Σ p(s) = 1.0
    most_likely: HiddenState                  # argmax
    confidence: float                        # max(p)
    entropy: float                           # −Σ p·log₂(p)
}
```

- High entropy (> 2.0) → system is confused → avoid expensive actions
- Low entropy (< 0.5) → system is confident → proceed
- Confidence > 0.85 → strong belief

---

## 7. DECISION LAYER

### 7.1 Completion Detection

**No sleep(). No fixed timers.**

Multiple converging signals:

```
Velocity → 0         AND
Acceleration → 0     AND
(stream_closed       OR
 stream_idle > 5s    OR
 generation_completed OR
 not_streaming)
```

The CompletionEngine uses:
1. Kalman-smoothed response_length velocity (must be < 2.0 px/tick)
2. Kalman-smoothed response_length acceleration (must be < 1.0 px/tick²)
3. Network stream_idle_time > 5 seconds (primary signal)
4. OR generation_completed flag from stream parser
5. AND has_content (response > 20 chars OR total_chunks > 5)

**Stable count:** 3 consecutive ticks of velocity < threshold → confidence >= 0.85 → DONE.

### 7.2 Utility-Based Action Selection

```
a* = argmax_a Σ_s b(s) · R(a, s)
```

Key rewards:
- `type_prompt` + READY → +10
- `extract_response` + COMPLETE → +10
- `extract_response` + GENERATING → −5 (penalty for premature extraction)
- `recover` + GENERATING → −30 (NEVER recover during generation)

### 7.3 Entropy Gating

- entropy > 2.0 → is_confused → suppress actions, gather more observations
- entropy > 3.0 → is_critical → trigger recovery cascade
- `should_recover` = entropy > 2.0 AND P(GENERATING) < 0.3 AND P(READY) < 0.3

### 7.4 Recovery Cascade — Increasing Cost Order

```
Selector Cache Recovery
    ↓
Accessibility Recovery
    ↓
DOM Graph Similarity Recovery
    ↓
Network Protocol Re-detection
    ↓
Session Refresh
    ↓
Worker Replacement
    ↓
Provider Replacement
    ↓
Workflow Replan
```

Never jump directly to expensive recovery. Only high-confidence, low-entropy
states trigger actions.

---

## 8. NETWORK-FIRST INTELLIGENCE — THE KEY UPGRADE

### 8.1 Why Network-First

Before V2, the engine detected generation by:
- DOM:text was growing → GENERATING
- DOM:stop_button_visible → GENERATING
- Response_length increasing → GENERATING

**Problem:** When ChatGPT re-rendered its DOM after prompt send, CSS selectors went stale,
response_length dropped to 0, and the HMM fell back to ERROR. The engine correctly
detected DOM mutation activity but had **no stream signal** to know what was happening.

**Solution:** The network sensor directly intercepts the SSE/WebSocket stream that
carries the actual tokens. The DOM is a rendering artifact — the network stream is
the source of truth.

### 8.2 Protocol Detection

```python
class ProtocolDetector:
    def feed_response(url, status, content_type, headers):
        # SSE detection
        if content_type in text/event-stream:
            confidence += 0.85

        # WebSocket detection
        if status == 101:
            confidence += 0.90

        # Fetch stream detection
        if transfer_encoding == 'chunked' and content_type == application/json:
            confidence += 0.70

        # URL pattern boost (conversation, chat, completion, stream, etc.)
        url_boost = min(0.05 * url_hits, 0.25)

        # Return (protocol, confidence)
```

No provider names anywhere. Purely protocol characteristics.

### 8.3 Stream Parser — The Core Intelligence

```python
StreamParser:
    push_event(data, timestamp)      # from any observer
    push_bytes(byte_count, timestamp)
    signal_transport_connected()
    signal_transport_disconnected()
    evaluate(now) → StreamState       # called every sense()

StreamState:
    tokens_per_second: float         # EMA(α=0.3) of chunk rate
    stream_idle_time: float          # now - last_chunk_timestamp
    total_chunks: int
    bytes_received: int
    stream_active: bool              # idle < 2s AND chunks > 0
    generation_started: bool         # first chunk after 5s idle
    stream_closed: bool              # idle > 5s + disconnected OR > 10s
```

**Token rate computation:**
```python
def _compute_instantaneous_rate(now):
    recent = chunks in last 5 seconds
    if len(recent) < 2:
        return total_chunks / (now - first_chunk)
    return len(recent) / (now - oldest_recent)
```

EMA is applied: `rate = α · current + (1−α) · previous` where α=0.3.

### 8.4 CDP Integration

```
page.context.new_cdp_session(page) → cdp

cdp.on("Network.requestWillBeSent",   callback)
cdp.on("Network.responseReceived",    callback)
cdp.on("Network.dataReceived",        callback)
cdp.on("Network.loadingFinished",     callback)
cdp.on("Network.loadingFailed",       callback)
cdp.on("Network.webSocketCreated",    callback)
cdp.on("Network.webSocketClosed",     callback)
cdp.on("Network.webSocketFrameReceived", callback)
cdp.on("Network.eventSourceMessageReceived", callback)  # Chrome-only
```

All callbacks are non-blocking — they queue events. The `sense()` method drains
the queue. Events are dispatched to the appropriate observer by type.

### 8.5 SSE Observer

```python
class SSEObserver:
    on_response_received(event):
        if content_type is text/event-stream:
            mark stream active, track request_id

    on_event_source_message_received(event):
        event_count += 1
        data_chunk_count += 1
        bytes_received += len(data)
        if "[DONE]" in data:
            done_seen = True
            stream_closed = True

    on_loading_finished(event):
        if last stream closed:
            stream_closed = True
```

### 8.6 WebSocket Observer

```python
class WSObserver:
    on_ws_created(event):
        connection_open = True
        track URL

    on_ws_frame_received(event):
        frame_count += 1
        if payload not empty:
            data_frame_count += 1
            bytes_received += len(payload)
            stream_active = True
            if "[DONE]" in payload:
                done_seen = True

    on_ws_closed(event):
        connection_open = False
        stream_closed = True
```

### 8.7 Fetch Observer

```python
class FetchObserver:
    on_response_received(event):
        if chunked transfer-encoding:
            mark stream active

    on_data_received(event):
        chunk_count += 1
        bytes_received += data_length
        stream_active = True

    on_loading_finished(event):
        stream_closed = True
```

### 8.8 Impact of Network-First

| Scenario | Before (V1) | After (V2) |
|----------|------------|-----------|
| ChatGPT prompt sent | rlen→0, belief→ERROR | SSE detected, stream_active=True, belief→GENERATING(0.9+) |
| Qwen streaming | rlen=0 (stale selectors) | FetchObserver detects chunks, tps>0 |
| Claude thinking | has_thinking_marker only | Transport detected + stream_idle tracking |
| Rate limited | error_banner check | 429 response → error_codes tracked |
| Completion | wait_for_stable_text() | stream_idle>5s + stream_closed + velocity≈0 |

---

## 9. LEARNING SYSTEM

### 9.1 Online Learning

```python
# Emission parameter update
p_new = p_old + lr * (observed - p_old)
μ_new = μ_old + lr * (observed - μ_old)
σ_new = (1−lr) * σ_old + lr * |observed − μ_old|
```

Learning rate = 0.05 per observation. Parameters converge slowly (hundreds of ticks).

### 9.2 Offline Learning (Baum-Welch)

1. Collect sequences of (observation, Viterbi-decoded state)
2. Re-estimate transition matrix from state pair counts
3. Re-estimate emission parameters from observation distributions per state
4. Update with Laplace smoothing

### 9.3 Bayesian Reliability Tracking

Track per-session, per-provider:
- `selector_reliability` — fraction of successful element interactions
- `accessibility_reliability` — fraction of ticks with a11y confidence > 0.5
- `network_reliability` — fraction of ticks with transport_detected=True
- `provider_reliability` — overall session success rate
- `session_stability` — consecutive ticks without recovery

Old observations decay: `reliability_t = α · obs + (1−α) · reliability_{t−1}`

---

## 10. EDGE CASES & FAILURE MODES

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Stale DOM selectors | response_length = 0, but network stream active | Network signals override, continue detection from stream |
| Protocol change | transport_confidence drops, stream_idle spikes | ProtocolDetector re-detects, observers re-attach |
| CDP unavailable (headless) | attach() fails gracefully, sense() returns zeros | Fall back to a11y + DOM sensors |
| Rogue mutation spikes (ads) | high mutation_rate + 0 stream activity | Network sensor confirms no generation, HMM stays READY |
| Page crash | all sensors fail, page_stability→0 | Entropy spikes, triggers session recovery |
| Network partition | loadingFailed events accumulate | Network errors tracked, streams marked closed |
| Shadow ban | appears COMPLETE but response empty | Historical baseline comparison triggers SHADOW_BANNED belief |
| OOM imminent | heap > 500MB, cpu spiking | Resource governor reduces concurrency before crash |

---

## 11. PERFORMANCE CHARACTERISTICS

| Metric | Target |
|--------|--------|
| Tick latency | < 500 ms (warning at 500, critical at 2000) |
| READY detection | 1–3 ticks (80%+ confidence) |
| GENERATING detection | 1 tick after first stream chunk |
| COMPLETE detection | 3–5 ticks after last chunk (stable window) |
| Memory | FeatureStore ring buffer: 300 × 30 floats ≈ 72 KB |
| CDP overhead | Event queue drain < 1 ms per tick |
| Sensor parallelism | All 6 sensors can run concurrently (not yet parallelized) |

---

## 12. IMPLEMENTATION MODULES

```
ai_orchestrator/browser_intelligence/
├── engine.py              # BrowserIntelligenceEngine — top-level orchestrator
├── sensors/
│   ├── base.py            # BaseSensor ABC
│   ├── dom_sensor.py      # DOMSensor — element presence/absence
│   ├── accessibility_sensor.py  # AccessibilitySensor — ARIA tree features
│   ├── mutation_sensor.py       # MutationSensor — DOM mutation rate
│   ├── performance_sensor.py    # PerformanceSensor — JS heap, page stability
│   ├── visual_sensor.py         # VisualSensor — screenshot stability
│   ├── network_sensor.py        # NetworkSensor — orchestrator for network observers
│   └── network/
│       ├── __init__.py
│       ├── protocol_detector.py # TransportProtocol enum + detection
│       ├── sse_observer.py      # SSE stream interception
│       ├── ws_observer.py       # WebSocket frame interception
│       ├── fetch_observer.py    # Fetch/XHR stream detection
│       └── stream_parser.py     # Protocol-agnostic stream metrics
├── features/
│   ├── feature_vector.py  # FeatureVector (30-dim) + FeatureStore (ring buffer)
│   └── feature_composer.py # FeatureComposer — assembles sensors → vector
├── estimation/
│   ├── belief_state.py    # HiddenState enum, BeliefState distribution
│   ├── hmm_engine.py      # Forward algorithm + Viterbi decoder
│   ├── emission_model.py  # Bernoulli+Gaussian, FEATURE_DIM=30
│   ├── transition_matrix.py # Log-probability transition matrix
│   └── kalman_filter.py   # 3-state Kalman (length, velocity, acceleration)
├── decision/
│   ├── completion.py      # CompletionEngine — zero-velocity + stream-idle
│   ├── utility.py         # UtilityEngine — expected utility maximization
│   ├── confidence.py      # ConfidenceEngine — weighted confidence scores
│   └── entropy.py         # EntropyEngine — uncertainty gating
├── recovery/              # (Phase 4: selector recovery, graph similarity)
├── learning/              # (Phase 5: Baum-Welch, Bayesian tracking)
├── scheduling/            # (Phase 6: adaptive scheduling, resource governor)
├── events/                # (Phase 7: event sourcing, journal)
└── intelligence/          # (Phase 8: provider drift detection)
```

---

## 13. TESTING

```
pytest ai_orchestrator/tests/unit/test_browser_intelligence/ -q
```

| Test File | Category | Count |
|-----------|----------|-------|
| `test_core.py` | FeatureVector, FeatureStore, BeliefState | ~70 |
| `test_network.py` | ProtocolDetector, StreamParser, SSE/WS/Fetch observers, NetworkSensor | 83 |
| `test_feature_vector_v2.py` | 30-dim layout verification | 9 |
| `test_emission_model_v2.py` | Emission probabilities, defaults, FEATURE_DIM=30 | 16 |
| **Total** | | **178+** |

- All tests are pure Python — no browser, no Playwright, no network
- Sensor tests simulate CDP events directly
- StreamParser tests verify EMA computation, idle time, lifecycle detection
- Emission tests verify all 10 states have 30-dim defaults

---

## 14. USAGE

```python
from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine
from ai_orchestrator.browser_intelligence.estimation.belief_state import HiddenState

engine = BrowserIntelligenceEngine()
await engine.attach(page)

# Wait for READY state
while True:
    await engine.tick(page)
    if engine.is_ready_for_prompt:
        break
    if engine.is_error:
        raise RuntimeError("Page error")

# Type prompt
await page.locator('textarea').first.fill("What is 2+2?")
await page.locator('[data-testid="send-button"]').click()

# Monitor generation
while True:
    await engine.tick(page)
    if engine.is_generating:
        print(f"Generating: {engine.confidence:.2%}, "
              f"tps={engine._store.latest.tokens_per_second:.1f}")
    done, conf = engine.is_response_complete()
    if done:
        print(f"Complete: {conf:.2%}")
        break
    if engine.is_error:
        print("Error during generation")
        break

# Current belief distribution
for state, prob in engine.state_probabilities().items():
    if prob > 0.01:
        print(f"  {state:20s} {prob:.4f}")
```

### Diagnostic Properties

```python
engine.most_likely_state    # HiddenState enum
engine.confidence           # float 0.0–1.0
engine.entropy              # float (0 = certain, >2 = confused)
engine.is_ready_for_prompt  # bool
engine.is_generating        # bool
engine.is_error             # bool
engine.is_rate_limited      # bool
engine.recommended_action   # str ("type_prompt", "wait", "extract_response", ...)
engine.state_probabilities() # dict[str, float]
engine.action_utilities()   # dict[str, float]
```

---

## 15. ROADMAP

| Phase | Description | Status |
|-------|------------|--------|
| 0 | Core data structures (FeatureVector, FeatureStore, BeliefState, HiddenState) | DONE |
| 1 | Sensors (DOM, A11y, Mutation, Performance, Visual) | DONE |
| 2 | Estimation (HMM, Emission Model, Transition Matrix, Kalman Filter) | DONE |
| 3 | Decision (Completion, Utility, Confidence, Entropy) | DONE |
| 3.5 | **Network-First Intelligence** (SSE/WS/Fetch/CDP observers, StreamParser) | DONE |
| 4 | Selector Recovery Engine (graph similarity, historical success) | PLANNED |
| 5 | Learning System (Baum-Welch, Bayesian tracking, decay) | PLANNED |
| 6 | Resource Governor + Adaptive Scheduler | PLANNED |
| 7 | Event Sourcing + Journal | PLANNED |
| 8 | Provider Drift Detection + Shadow Ban Detection | PLANNED |

---

*Version 2.0 — Network-First Browser Intelligence Operating System*
