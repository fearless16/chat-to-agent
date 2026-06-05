# MULTI-PROVIDER AGENTIC PLATFORM — V6 (PRODUCTION ARCHITECTURE)

> Philosophy:
>
> Deterministic First
> → AI Second
> → Vision Last

The system should never invoke an LLM when deterministic systems can solve the problem.

The system should never invoke Vision when DOM, Accessibility, or structured UI information is available.

---

# CORE PRINCIPLES

## Principle 1

Providers are replaceable.

```text
ChatGPT
Qwen
DeepSeek
Kimi
Future Providers
```

All providers are workers.

The system must never depend on any specific provider.

---

## Principle 2

Workflow is king.

```text
Workflow
>
Provider
>
Model
```

Providers execute work.

Workflow owns state.

---

## Principle 3

DeepSeek is consulted.

DeepSeek is not the operating system.

Use DeepSeek only when deterministic logic cannot solve the problem.

---

## Principle 4

Browser workers are first-class citizens.

Browser transports are not fallback systems.

Browser workers are equal to API workers.

---

# HIGH LEVEL ARCHITECTURE

```text
User
 ↓

Gateway
 ↓

Workflow Engine
 ↓

Scheduler
 ↓

Control Plane
 ↓

Workspace Runtime
 ↓

Provider Runtime
 ↓

Transport Runtime

 ├── Browser Workers
 ├── API Workers
 └── Local Workers

 ↓

Response Validation
 ↓

Memory
 ↓

Event Journal
```

---

# WORKFLOW ENGINE

The Workflow Engine is the most important component.

The system is workflow-driven.

Not provider-driven.

Not model-driven.

---

## State Machine

```text
CREATED
 ↓

PLANNING
 ↓

EXECUTING
 ↓

TESTING
 ↓

REVIEWING
 ↓

DONE
```

---

## Failure Loop

```text
EXECUTING
 ↓

FAILED
 ↓

REPLAN
 ↓

EXECUTING
```

---

## Workflow Responsibilities

- state ownership
- retries
- replanning
- checkpoints
- resumability
- task graph generation

---

# CONTROL PLANE

The Control Plane decides:

- task classification
- provider selection
- routing
- replanning
- validation escalation

---

## Tiered Intelligence

### Tier 0

No AI

```text
Selector Cache
Rule Engine
Heuristics
Known Patterns
```

---

### Tier 1

Cheap Intelligence

Used for:

```text
classification
basic validation
dom labeling
light summarization
```

Only invoked when Tier 0 fails.

---

### Tier 2

DeepSeek

Used for:

```text
planning
replanning
code review
architecture review
complex dom analysis
workflow repair
```

DeepSeek should never be used for routine operations.

---

# PROVIDER RUNTIME

Provider and Transport are separate.

---

## Provider Layer

```text
ChatGPT
Qwen
DeepSeek
Kimi
```

---

## Transport Layer

```text
API
Browser
Local
```

---

## Example

```text
Qwen
 ├── API
 └── Browser

ChatGPT
 ├── API
 └── Browser

DeepSeek
 └── API

Kimi
 └── Browser
```

---

# BROWSER WORKER POOL

Each browser worker owns:

```text
Session
Lease
Context
```

Examples:

```text
ChatGPT-1
ChatGPT-2

Qwen-1
Qwen-2

Kimi-1
```

---

# UI INTELLIGENCE LAYER

This is the most important browser component.

Never parse full HTML blindly.

---

## Pipeline

```text
Selector Cache
 ↓

Accessibility Tree
 ↓

DOM Snippets
 ↓

DeepSeek Analysis
 ↓

Vision
```

---

# Stage 1

Selector Cache

```json
{
  "input": "...",
  "send": "...",
  "assistant": "..."
}
```

Fast path.

No AI.

---

# Stage 2

Accessibility Snapshot

Use:

```python
page.accessibility.snapshot()
```

Advantages:

- compact
- semantic
- stable
- low token cost

---

# Stage 3

DOM Snippet Extraction

Extract only:

```text
candidate buttons
candidate inputs
candidate message containers
```

Never send full HTML.

---

# Stage 4

DeepSeek DOM Analysis

Input:

```json
{
  "nodes": [...]
}
```

Output:

```json
{
  "input": "...",
  "send": "...",
  "assistant": "...",
  "confidence": 0.92
}
```

---

# Stage 5

Vision Fallback

Last resort only.

Flow:

```text
Screenshot
 ↓

DeepSeek Vision
 ↓

Recovered Coordinates
```

Vision must never be the default path.

---

# UI SCHEMA ENGINE

All providers are normalized into one schema.

```json
{
  "messages": [],
  "input_box": {},
  "send_button": {},
  "streaming": false
}
```

The rest of the system consumes only this schema.

---

# ACCOUNT MANAGEMENT

Separate:

```text
Account Health
```

from

```text
Lease State
```

---

## Account Health States

```text
READY
DEGRADED
COOLDOWN
JAIL
```

---

## Lease States

```text
FREE
LEASED
EXPIRED
RELEASED
```

---

# REACTIVE LEASE MANAGER

Critical component.

---

## Event Flow

```text
Account
 ↓

JAIL
 ↓

Lease Manager
 ↓

Force Expire Lease
 ↓

Workflow Engine
 ↓

REPLAN
```

No stale leases allowed.

---

# SCHEDULER

Resource-based.

Never CPU-only.

---

## Inputs

```text
Available RAM
Browser Context Count
Provider Limits
Queue Depth
```

---

## Formula

```text
MaxRunners =
min(
 AvailableBrowserContexts,
 AvailableRAM / AvgContextRAM,
 ProviderConcurrency,
 ConfiguredLimit
)
```

---

# MEMORY SYSTEM

Three-tier model.

---

## HOT

RAM

Contains:

```text
recent turns
active state
live workflow context
```

---

## WARM

Compressed Memory

Contains:

```text
summaries
compressed conversations
```

---

## COLD

Persistent Storage

Contains:

```text
artifacts
logs
history
screenshots
```

---

# WORKSPACE RUNTIME

The Workspace Runtime owns code.

Not OpenCode.

Not Aider.

---

## Components

```text
Workspace
Git
AST Engine
Sandbox
```

---

## File Operations

```python
read_file()
write_file()
patch_file()
search_code()
list_tree()
```

---

# AST PATCH ENGINE

Preferred editing mechanism.

Avoid:

```text
regex
string replacement
```

Use:

```text
Tree-sitter
AST-based patching
```

Benefits:

- syntax safety
- precise edits
- lower corruption rate

---

## Validation

Every patch must pass:

```python
ast.parse()
```

or language-specific validation.

Before commit.

---

# SANDBOX RUNTIME

All code execution happens in isolation.

---

## Responsibilities

```text
run tests
execute code
lint
format
verify
```

---

## Constraints

```text
resource limits
timeouts
filesystem isolation
```

---

# OPENCODE INTEGRATION

OpenCode is an execution backend.

Not the core architecture.

---

## Runtime

```text
Workspace Runtime
 ├── OpenCode
 ├── Aider
 └── Native Runtime
```

---

# RESPONSE VALIDATION

Every provider response passes validation.

---

## Flow

```text
Provider
 ↓

Validator
 ↓

Accept
Retry
Fallback
```

---

## Validation Levels

### Level 1

Deterministic

```text
schema checks
json checks
syntax checks
```

### Level 2

DeepSeek Review

Only if required.

---

# EVENT JOURNAL

Append-only.

Never mutate.

---

## Example

```json
{
  "task": "123",
  "agent": "coder",
  "action": "patch_file",
  "status": "success",
  "timestamp": "..."
}
```

---

# ARTIFACT STORE

Per-task storage.

```text
artifacts/

task_id/

  plans/
  code/
  reports/
  screenshots/
  patches/
```

---

# SELF-HEALING SYSTEM

## UI Failure

```text
Selector Broken
 ↓

Selector Cache Miss
 ↓

Accessibility Tree
 ↓

DOM Analysis
 ↓

Recovered
```

---

## If Recovery Fails

```text
DOM Analysis Failed
 ↓

Vision Recovery
 ↓

Recovered
```

---

## If Vision Fails

```text
Provider Quarantine
 ↓

Alert
 ↓

Fallback Provider
```

---

# DEEPSEEK RESPONSIBILITIES

DeepSeek should only perform:

```python
plan()
replan()
review()
architecture_review()
complex_dom_analysis()
```

DeepSeek should not be used for:

```python
simple_classification()
basic_validation()
cached_dom_recovery()
```

---

# P0 IMPLEMENTATION ROADMAP

## P0.1

Accessibility Runtime

```python
page.accessibility.snapshot()
```

---

## P0.2

UI Schema Engine

Provider-independent UI representation.

---

## P0.3

Reactive Lease Manager

Account state event system.

---

## P0.4

Workspace Runtime

Git + AST + Sandbox.

---

## P0.5

DeepSeek Control Plane

Planning and workflow intelligence.

---

# FINAL SYSTEM MENTAL MODEL

```text
Workflow Engine
     = Soul

DeepSeek
     = Brain

Workspace Runtime
     = Hands

Browser Workers
     = Labour

UI Intelligence Layer
     = Eyes

Scheduler
     = Nervous System

Event Journal
     = Memory
```

If providers change:

System survives.

If selectors change:

System survives.

If accounts fail:

System survives.

If workflows survive:

The platform survives.
