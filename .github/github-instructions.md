# Duelyst.ai Core — Repository Context

## Purpose

`duelyst-ai-core` is the open-source debate engine behind Duelyst.ai. It is a standalone Python package
published on PyPI and must remain product-independent.

What belongs here:

- debate agents and orchestration
- model registry and alias resolution
- optional tool integrations
- streaming events and callback interfaces
- CLI, output formatters, examples, tests, and docs

What does not belong here:

- authentication or user accounts
- billing or subscription logic
- database persistence
- API endpoints or frontend code
- secrets or real API keys

## Core Architecture

### Agents

Agents are thin wrappers around `langchain.agents.create_agent`.

- `src/duelyst_ai_core/agents/debater.py`: debater agent with optional tools and structured `AgentResponse`
- `src/duelyst_ai_core/agents/judge.py`: judge agent with structured `JudgeSynthesis`

Important rule: user-controlled topic/instructions/transcript content goes into user messages, never into
system prompts. Prompt templates in `agents/prompts.py` are static by design.

### Orchestrator

The outer control plane is a LangGraph `StateGraph` in `src/duelyst_ai_core/orchestrator/engine.py`.

Topology:

```text
START -> init_debate -> run_debater_a -> run_debater_b -> check_convergence
                      ^                                       |
                      |___________ continue _________________ |
                                                              |
                                            converged/max -> run_judge -> END
```

The orchestrator owns the real debate lifecycle:

- round progression
- transcript accumulation
- convergence tracking
- synthesis invocation
- event emission

Sub-agents do not own debate state. They receive messages and return structured output.

### State And Schemas

- `src/duelyst_ai_core/orchestrator/state.py`: `DebateConfig`, `ModelConfig`, `DebateStatus`, `OrchestratorState`
- `src/duelyst_ai_core/agents/schemas.py`: public Pydantic models such as `AgentResponse`, `DebateTurn`, `JudgeSynthesis`, and `DebateResult`

Important detail: `OrchestratorState.turns` uses `Annotated[..., operator.add]`, so nodes append turn deltas
instead of replacing the full list.

### Events, Callbacks, And Streaming

- `src/duelyst_ai_core/orchestrator/events.py`: typed runtime events
- `src/duelyst_ai_core/orchestrator/callbacks.py`: `DebateEventCallback`, `NullCallback`, `CollectorCallback`
- `DebateOrchestrator.arun_with_events()`: async generator API built on an `asyncio.Queue`

The core emits debate lifecycle events so the CLI, tests, or a future API server can observe progress without
coupling orchestration to a transport layer.

Key nuance: `DebateCompleted` and `DebateError` are terminal wrapper events produced by `arun_with_events()`,
not graph-node events.

### CLI And Live Display

- `src/duelyst_ai_core/cli/main.py`: Typer entry point
- `src/duelyst_ai_core/cli/display.py`: debate runner and output-mode switch
- `src/duelyst_ai_core/cli/live_panel.py`: `RichDisplayCallback` that turns events into a live Rich UI

`--output rich` uses the live callback-driven display. Markdown and JSON output skip live rendering and use a
simple spinner before formatting the final `DebateResult`.

### Model And Tool Layer

- `src/duelyst_ai_core/models/registry.py`: `resolve_alias()`, `create_model()`, `get_judge_model()`
- `src/duelyst_ai_core/tools/search.py`: optional Tavily search tool with graceful degradation

There is no custom adapter class hierarchy. The model registry returns LangChain chat models directly.

## Important Design Decisions

1. Agents use `create_agent()` rather than custom multi-node agent graphs.
2. The orchestrator is small and explicit; the complexity lives in state and event flow.
3. Structured output is the public contract, not raw model text.
4. Tools are optional and should never be required for a debate to run.
5. Callback failures are logged and swallowed so observers do not break debate execution.
6. The package exports streaming types and callbacks as part of the public API.

## Testing And Validation

Local commands should mirror CI:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest -v
```

Test strategy:

- unit tests mock model creation and agent `ainvoke()` calls
- callback and streaming tests validate event order and terminal behavior
- CLI tests validate live vs non-live execution paths
- integration tests live under `tests/integration/` and are separate from the default test run

## GitHub Actions And Publishing

Workflow files:

- `.github/workflows/ci.yml`: Ruff, format check, mypy, and pytest on push/PR
- `.github/workflows/publish.yml`: reruns the quality gate on version tags, verifies tag/version alignment, builds the package, smoke-tests the wheel in a clean virtualenv, publishes to PyPI, and creates a GitHub Release
- `.github/release.yml`: release-note categories

Release flow:

1. bump `project.version` in `pyproject.toml`
2. run local checks
3. push to `main`
4. push a tag like `v0.1.0`

## Documentation To Keep In Sync

- `README.md`: public-facing usage and release overview
- `docs/ARCHITECTURE.md`: detailed guided architecture walkthrough
- `docs/graph.png`: combined architecture diagram used by the README and architecture doc

When architecture changes, update the code and these docs together.
