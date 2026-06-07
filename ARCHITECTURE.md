
# ARCHITECTURE

## Layer 0 - Account/Auth

### Current Decision

```text
Cookies Only
```

Abhi Chrome profile migration/CDP attach pe time waste nahi karna.

Jo chal raha hai usko stable karo.

### States

```text
UNKNOWN
AUTHENTICATED
AUTH_REQUIRED
HUMAN_VERIFICATION
RATE_LIMITED
BANNED
```

### Checks

Before every prompt:

```text
Cookie Loaded?
↓
Chat Page Loaded?
↓
Input Visible?
↓
Send Button Visible?
```

If no:

```text
Auth Failure
```

---

# Layer 1 - Browser Worker

One Provider

```text
Qwen Worker

DeepSeek Worker

Kimi Worker

...
```

Each Worker:

```text
Browser
Context
Page
Cookie Jar
State Machine
```

---

# Layer 2 - Page Sensors

Priority:

```text
Network
 ↓
Accessibility
 ↓
DOM
 ↓
Vision
```

---

## Network Sensor

Watch:

```text
XHR

Fetch

SSE

WebSocket
```

Collect:

```text
Request URL

Response URL

Payload

Content-Type

Timestamp
```

---

## DOM Sensor

Collect:

```text
Input Visible

Send Visible

Stop Visible

Error Visible

Auth Visible
```

No text extraction.

No decisions.

---

## Accessibility Sensor

Collect:

```text
Role

Name

State

Hierarchy
```

Used for selector recovery.

---

## Vision Sensor

Only if everything else fails.

Never primary.

---

# Layer 3 - State Machine

States:

```text
BOOTING

AUTH_REQUIRED

HUMAN_VERIFICATION

READY

PROMPT_TYPED

PROMPT_SENT

GENERATING

COMPLETE

RATE_LIMITED

ERROR
```

---

Transitions:

```text
READY
 ↓
PROMPT_TYPED
 ↓
PROMPT_SENT
 ↓
GENERATING
 ↓
COMPLETE
```

---

# Layer 4 - Response Engine

Most important layer.

Current bug:

```text
Network sees response

But content lost
```

Fix:

```text
Chunk
 ↓
Buffer
 ↓
Parser
 ↓
Final Text
```

Store:

```text
Request ID

Provider

Timestamp

Payload
```

Never:

```python
push_event(data=None)
```

---

# Layer 5 - Response Classifier

Before buffering:

Classify:

```text
CHAT

ANALYTICS

TELEMETRY

AUTH

UNKNOWN
```

Only:

```text
CHAT
```

goes to buffer.

---

# Layer 6 - Completion Detection

Never:

```python
sleep(60)
```

Use:

```text
Network Activity

Mutation Rate

Response Growth
```

Completion:

```text
No Stream Activity
+
No Response Growth
+
No DOM Mutations
```

---

# Layer 7 - Recovery

Recovery Order:

```text
Retry

↓

Dismiss Popup

↓

Selector Recovery

↓

Accessibility Recovery

↓

Refresh

↓

Reload Cookies

↓

New Browser Context

↓

Provider Cooldown
```

---

# Popup Handling

Detect:

```text
Cookie Banner

Newsletter Popup

Upgrade Popup

Rate Limit Popup

Modal Overlay
```

Rule:

```text
If modal blocks input

Find close button

Dismiss immediately
```

---

# Cloudflare / Captcha

New State:

```text
HUMAN_VERIFICATION
```

Detect:

```text
Just a moment

Checking your browser

Verify you are human

Turnstile

Captcha
```

Actions:

```text
Pause Worker

Capture Screenshot

Raise Event
```

Do NOT loop.

Do NOT spam refresh.

---

# Metrics

Per Provider:

```text
Success Rate

Auth Rate

Response Rate

Average Latency

Captcha Frequency

Popup Frequency

Recovery Frequency
```

---

# Mathematical Model

Readiness:

P(READY)=\sum_i w_i x_i

Where:

```text
x_i = sensor signal

w_i = learned weight
```

---

Provider Score:

Score=P(success)\times Availability/Latency

Used for routing.

---

# Execution Plan

## Phase 1 (Current Blocker)

Fix:

```text
Response Capture
```

Goal:

```text
Qwen
↓
Prompt
↓
Actual Answer
```

No empty string.

---

## Phase 2

Fix:

```text
Response Classifier
```

Remove:

```text
Telemetry

Analytics

Tracking Pings
```

---

## Phase 3

Fix:

```text
Auth Detection
```

Separate:

```text
AUTH_REQUIRED

HUMAN_VERIFICATION
```

---

## Phase 4

Popup Manager

Auto dismiss:

```text
Cookie

Newsletter

Upgrade

Modal
```

---

## Phase 5

Cloudflare / Captcha Detection

No solving.

Only:

```text
Detect

Pause

Notify
```

---

## Phase 6

Provider Health Dashboard

Show:

```text
Ready

Auth

Generating

Complete

Error
```

---

# NON-NEGOTIABLE RULE

Before building:

```text
Bayesian Learning
Entropy Engine
Provider Brain
Scheduler
Shadow Ban Detector
```

Prove this works:

```text
Open Provider
↓
Load Cookies
↓
Input Found
↓
Prompt Sent
↓
Response Captured
↓
Correct Text Returned
```

For:

```text
Qwen
DeepSeek
Kimi
ZAI
MiniMax
MiMo
ChatGPT
```

Until that flow is stable, no new architecture work is allowed.

```
