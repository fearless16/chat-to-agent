# Missing Constraints (Must Add)

## 1. Single Source of Truth per Task

Abhi problem:

```text
Network says COMPLETE
DOM says GENERATING
Accessibility says READY
```

LLM confuse ho jayega.

Rule:

```text
Generation State:
Network > Accessibility > DOM > Vision

Auth State:
Accessibility > DOM > Vision > Network

Response Text:
Network > DOM

Popup Detection:
DOM > Accessibility > Vision
```

Never mix priorities.

---

## 2. Agent Ownership

Current plan me conflict ho sakta hai.

Add:

```text
Agent Auth
Can modify:
- auth/
- cookies/
- state detection

Cannot modify:
- response pipeline

Agent Response
Can modify:
- network/
- parser/
- buffer/

Cannot modify:
- auth/

Agent UI
Can modify:
- popup/
- selectors/
- accessibility/

Cannot modify:
- network/

Agent Test
Read-only except tests.
```

No overlapping ownership.

---

## 3. Hard Stop on Hallucinated Fixes

Add:

```text
Before any code change:

1. Reproduce bug
2. Capture evidence
3. Identify root cause
4. Implement fix
5. Reproduce again
6. Verify fixed

If root cause is not proven:
DO NOT CODE.
```

---

## 4. Provider Matrix

Maintain:

| Provider | Auth | Input | Send | Generate | Extract |
| -------- | ---- | ----- | ---- | -------- | ------- |
| chatgpt_ui | Success | Success | Success | Success | Success |
| z_ai_ui | Success | Success | Success | Success | Success |
| qwen_ui | Success | Success | Success | Success | Success |
| deepseek_ui | Success | Fail | Fail | Fail | Fail |
| kimi_ui | Success | Fail | Fail | Fail | Fail |
| minimax_ui | Success | Fail | Fail | Fail | Fail |
| xiaomimimo_ui | Success | Fail | Fail | Fail | Fail |

Every commit updates matrix.

No guessing.

---

## 5. Popup Handling States

Add explicit states:

```text
COOKIE_BANNER
NEWSLETTER_MODAL
UPGRADE_MODAL
RATE_LIMIT_MODAL
CAPTCHA_MODAL
UNKNOWN_MODAL
```

Unknown modal:

```text
Screenshot
Log
Pause
```

Don't randomly click.

---

## 6. Captcha Rule

Very important.

```text
Never solve captcha automatically.

Detect.
Pause.
Notify.

Do not:
- refresh loop
- retry loop
- click random boxes
```

Otherwise account ban risk.

---

## 7. Cookie Validation

Before launching provider:

Check:

```text
cookie file exists
cookie parseable
cookie count > 0
```

After navigation:

Check:

```text
authenticated?
```

Cookie load success ≠ auth success.

---

## 8. Response Extraction Contract

Success is NOT:

```json
{}
```

Success is NOT:

```json
{"ResultObject":true}
```

Success is:

```text
Actual assistant response text
```

Add explicit validator.

---

## 9. Parallel Execution Limits

Because low RAM concern.

Rule:

```text
Max Browsers = 2

Max Tabs Per Browser = 3

Max Concurrent Providers = 4

Keep:
Free RAM >= 2GB
```

If RAM below threshold:

```text
Pause new work.
```

---

## 10. Commit Gate

Before merge:

Must pass:

```text
Provider Validation
Response Extraction
Auth Detection
Popup Detection
```

If any fail:

```text
No merge.
```

---

# Final Order (Frozen)

```text
Phase 0
Evidence

Phase 1
Response Capture

Phase 2
Response Classifier

Phase 3
Auth Detection

Phase 4
Popup Manager

Phase 5
Captcha Detection

Phase 6
Provider Health

Phase 7
Integration Tests

ONLY THEN:

Learning
Recovery
Scheduler
Provider Brain
```

And the most important line to put at the top:

```text
The project is not allowed to add new architecture until at least 3 providers can reliably complete:

Open → Authenticate → Send Prompt → Extract Correct Response

in real mode.
```