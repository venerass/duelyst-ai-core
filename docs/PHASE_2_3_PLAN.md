# Duelyst.ai Core — Implementation Plan: Streaming, Examples & PyPI Polish

> Finishing the OSS core before building the product platform.

**Context:** Phase 1 of the OSS engine is complete (154 tests, ruff/mypy clean, working CLI). The [PRD](PRD_duelyst_ai.md) defines five product phases, where Phase 2 (API + Frontend) requires the core to expose streaming events that the FastAPI backend can relay via SSE. This plan covers the **remaining OSS core work** that bridges Phase 1 → PRD Phase 2.

**What this plan covers (duelyst-ai-core only):**
- Event callback infrastructure so the API can stream debate progress
- Rich live terminal display for the CLI
- Public streaming API for library consumers
- Working example scripts for developer adoption
- PyPI publish workflow hardening

**What this plan does NOT cover (separate repos):**
- FastAPI backend (`duelyst-ai-api`) — auth, persistence, SSE endpoints, billing
- Next.js frontend (`duelyst-ai-app`) — debate creator, real-time viewer, published pages
- Supabase schema, Railway deployment, Vercel setup

---

## Why This Work Matters

The PRD's Month 3 checkpoint requires 200+ GitHub stars and 1,000+ PyPI downloads. The streaming infrastructure + examples are what convert "interesting project" into "I can actually use this." And the PRD's Phase 2 (API backend) needs the callback protocol to power SSE streaming — without it, the API would have to reinvent event emission from scratch.

```
Phase 1 (done)          This Plan              PRD Phase 2
┌──────────────┐   ┌─────────────────┐   ┌──────────────────────┐
│ Engine + CLI │ → │ Streaming +     │ → │ FastAPI API consuming │
│ 154 tests    │   │ Examples + PyPI │   │ core callbacks → SSE │
└──────────────┘   └─────────────────┘   └──────────────────────┘
```

---

## Step 1: Event Callback Protocol

**Goal:** Define the interface the core uses to emit events. Both the CLI (Rich Live) and the API (SSE relay) will implement this.

**Files:**
- `orchestrator/callbacks.py` — new file

**Design:**

```python
class DebateEventCallback(Protocol):
    async def on_event(self, event: DebateEvent) -> None: ...
```

A simple async protocol. The orchestrator accepts an optional callback; consumers implement it. This keeps the library decoupled — Rich display, SSE relay, logging, webhooks are all just callback implementations.

Also include:
- `NullCallback` — no-op default so nodes don't need `if callback:` guards
- `CollectorCallback` — buffers events in a list for testing and batch processing

**Why this design:** The PRD requires SSE streaming in the API. The API will create a callback that writes events to an SSE channel. By defining this as a protocol in the core, the API doesn't need to subclass anything — it just needs an async `on_event` method.

**Tests:** `tests/test_orchestrator/test_callbacks.py`
- `NullCallback.on_event()` is a no-op
- `CollectorCallback` captures events in order
- Custom class satisfying the protocol works

---

## Step 2: Emit Events from Orchestrator Nodes

**Goal:** Every orchestrator node emits typed events during execution, making the full debate lifecycle observable.

**Files:**
- `orchestrator/engine.py` — modify `__init__()` and all node methods

**Changes:**

`DebateOrchestrator.__init__()` gains `callback: DebateEventCallback | None`. Each node emits:

| Node | Events |
|------|--------|
| `init_debate` | `DebateStarted(config=...)` |
| `run_debater_a` | `TurnStarted(agent="a")` → invoke → `TurnCompleted(agent="a", response=...)` |
| `run_debater_b` | `TurnStarted(agent="b")` → invoke → `TurnCompleted(agent="b", response=...)` |
| `check_convergence` | `ConvergenceUpdate(score_a, score_b, is_converged)`, `RoundStarted(next)` if continuing |
| `run_judge` | `SynthesisStarted()` → invoke → `SynthesisCompleted(synthesis=...)` |

**Safety:** All `await callback.on_event(...)` wrapped in try/except — a buggy callback never crashes a debate. Errors are logged.

**Tests:** Extend `tests/test_orchestrator/test_engine.py`:
- Mock callback, assert events emitted in correct order
- Assert callback exceptions caught and logged, not propagated

---

## Step 3: Rich Live Display

**Goal:** Replace the static CLI spinner with a real-time display that updates per-turn.

**Files:**
- `cli/display.py` — rewrite `run_debate()`, add `RichDisplayCallback`
- `cli/live_panel.py` — new file with Rich layout components

**Design:**

`RichDisplayCallback` implements `DebateEventCallback` and drives `rich.live.Live`:

```
┌─── Debate: Should startups use microservices? ──────────────────┐
│ Model A: claude-haiku  │  Model B: gpt-mini  │  Max: 5 rounds  │
├─────────────────────────────────────────────────────────────────┤
│ Round 1                                                         │
│ ┌── Agent A ──────────────────────────────────────────────────┐ │
│ │ Monoliths offer simplicity in deployment and debugging...   │ │
│ │ Convergence: 3/10                                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌── Agent B ──────────────────────────────────────────────────┐ │
│ │ Microservices enable independent scaling...                 │ │
│ │ Convergence: 2/10                                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ Convergence: A=3, B=2 — not converged                          │
│                                                                 │
│ Round 2                                                         │
│ ⠋ Agent A is thinking...                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Event → display mapping:**

| Event | Display update |
|-------|---------------|
| `DebateStarted` | Render config header |
| `RoundStarted` | Add "Round N" section |
| `TurnStarted` | Show spinner: "Agent X is thinking..." |
| `TurnCompleted` | Replace spinner with colored argument panel (cyan A, magenta B) |
| `ConvergenceUpdate` | Convergence scores line below the round |
| `SynthesisStarted` | Spinner: "Judge is synthesizing..." |
| `SynthesisCompleted` | Synthesis panel |
| `DebateError` | Red error panel |

**Graph execution:** Switch from `ainvoke()` to `astream()`. The callback receives granular events during node execution; `astream()` yields state diffs after each node for collecting the final state.

**Output mode behavior:**
- `--output rich` (default): Live display active
- `--output markdown` / `--output json`: No live display, just run and print formatted output

**Tests:** `tests/test_cli/test_display.py` — new file:
- `RichDisplayCallback` accumulates renderables on each event type
- Markdown/JSON modes skip live display

---

## Step 4: Public Streaming API

**Goal:** Expose streaming for programmatic consumers — critical for the API backend.

**Files:**
- `orchestrator/engine.py` — add `arun_with_events()` method
- `__init__.py` — export new symbols

**Design:**

```python
async def arun_with_events(self) -> AsyncGenerator[DebateEvent, None]:
    """Run the debate and yield events as they occur."""
    ...
```

This is the primary interface the API backend will consume:

```python
# In duelyst-ai-api (FastAPI SSE endpoint):
orchestrator = DebateOrchestrator(config, tools=tools)
async for event in orchestrator.arun_with_events():
    yield f"data: {event.model_dump_json()}\n\n"  # SSE format
```

Implementation uses `asyncio.Queue` internally: a callback enqueues events, the generator dequeues them. The graph runs in a background task.

**New public API exports:**
- `DebateEventCallback` — protocol for custom callbacks
- Event types: `DebateEvent` (union), `DebateStarted`, `TurnCompleted`, `ConvergenceUpdate`, `SynthesisCompleted`, `DebateCompleted`

**Tests:** `tests/test_orchestrator/test_streaming.py`:
- `arun_with_events()` yields events in order
- Generator terminates after debate completion
- Queue doesn't deadlock on errors

---

## Step 5: Example Scripts

**Goal:** Working examples that developers copy-paste. Required for OSS adoption (Month 3 metrics: stars + downloads).

### `examples/basic_debate.py`

Minimal debate: configure two cheap models, run, print Markdown output. ~40 lines.

```python
"""Basic debate — two models argue a topic.

Usage:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key
    python examples/basic_debate.py
"""
```

### `examples/streaming_debate.py`

Real-time events via `arun_with_events()`. Shows each turn as it completes. ~50 lines.

### `examples/custom_debate.py`

Specific models, custom instructions, web search enabled. Shows the full configuration surface. ~60 lines.

### `examples/json_output.py`

Machine-consumable output for pipelines. Run → format → stdout. ~30 lines.

### `examples/README.md`

Index with prerequisites, API key setup, and a table of what each example does.

---

## Step 6: PyPI Publish Workflow Hardening

**Goal:** Don't publish broken packages. The existing `publish.yml` has no quality gates.

**File:** `.github/workflows/publish.yml` — modify

**Changes:**

1. **Add `test` job** before `publish` — runs lint + mypy + pytest on Python 3.11–3.13
2. **`publish` depends on `test`** via `needs: [test]`
3. **Version consistency check** — verify git tag matches `pyproject.toml` version
4. **Build verification** — install built wheel in clean venv, import `duelyst_ai_core`
5. **GitHub Release** — auto-create release with generated notes

**New file:** `.github/release.yml` — configure auto-generated release note categories

---

## Step 7: Test Sweep & Documentation

**Goal:** All tests pass after streaming changes. README reflects new capabilities.

**Changes:**
- Update mocked orchestrator calls in `test_main.py` (`astream()` instead of `ainvoke()`)
- Verify `--output markdown` and `--output json` still work
- Ruff + mypy clean on all new files
- README additions: "Streaming Events" section, link to examples, updated Python API usage

---

## Summary

| Step | Files | New/Modified | Deliverable |
|------|-------|-------------|-------------|
| 1 | `orchestrator/callbacks.py` | New | Event callback protocol |
| 2 | `orchestrator/engine.py` | Modified | Event emission from all nodes |
| 3 | `cli/display.py`, `cli/live_panel.py` | Modified + New | Rich live terminal display |
| 4 | `orchestrator/engine.py`, `__init__.py` | Modified | `arun_with_events()` + public exports |
| 5 | `examples/*.py`, `examples/README.md` | New (5 files) | Working example scripts |
| 6 | `.github/workflows/publish.yml`, `.github/release.yml` | Modified + New | Hardened publish workflow |
| 7 | `README.md`, tests | Modified | Documentation + test sweep |

**New files:** `orchestrator/callbacks.py`, `cli/live_panel.py`, 4 example scripts, `examples/README.md`, `.github/release.yml`, `tests/test_cli/test_display.py`, `tests/test_orchestrator/test_streaming.py`, `tests/test_orchestrator/test_callbacks.py`

**Modified files:** `orchestrator/engine.py`, `cli/display.py`, `cli/main.py`, `__init__.py`, `.github/workflows/publish.yml`, `README.md`, `tests/test_cli/test_main.py`, `tests/test_orchestrator/test_engine.py`

**Estimated new tests:** ~50

---

## Execution Order

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 7
                                      ↑
Step 5 (parallel, no streaming dep    │
        except streaming_debate.py) ──┘
Step 6 (independent) ────────────────┘
```

Steps 1–4 are strictly sequential (each builds on the previous). Steps 5 and 6 can start in parallel after Step 1. Step 7 runs last as a sweep.

---

## What Comes After This

Once these steps are complete, the core is ready for the PRD's Phase 2:

1. **`duelyst-ai-api`** imports `duelyst-ai-core`, creates a `DebateOrchestrator` per request, and relays `arun_with_events()` as SSE to the frontend.
2. **`duelyst-ai-app`** connects to the SSE endpoint and renders turns in real-time.
3. The callback protocol means the API needs zero custom streaming code in the core — it just consumes the generator.

The core remains product-independent. The API adds auth, persistence, rate limiting, and SSE transport. The frontend adds UX.
