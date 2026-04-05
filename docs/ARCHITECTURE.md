# Duelyst.ai Core Architecture

This document is the guided map for understanding how `duelyst-ai-core` works end to end.
It is written for developers who want to study the debate engine, extend it, or integrate it
into another project.

## What This Repository Owns

`duelyst-ai-core` is the product-independent debate engine. Its responsibilities are:

- define the debater and judge agents
- orchestrate turns, convergence, and synthesis
- expose structured events and streaming APIs
- provide a CLI and output formatters
- stay independent from auth, persistence, billing, and frontend concerns

That separation matters. The core should be importable from any other Python project without
learning anything about a specific web stack.

## Visual Map

![Duelyst.ai Core architecture diagram](graph.png)

The outer graph in the image is the real `StateGraph` implemented in
`src/duelyst_ai_core/orchestrator/engine.py`. The two sub-agent boxes are conceptual views of
the `create_agent()` graphs returned by LangChain. They emphasize the behavior this repository
cares about: prompt in, optional tools, structured output out.

## Best Reading Order

If you are new to the codebase, read the files in this order:

1. `src/duelyst_ai_core/orchestrator/state.py`
2. `src/duelyst_ai_core/agents/schemas.py`
3. `src/duelyst_ai_core/orchestrator/engine.py`
4. `src/duelyst_ai_core/orchestrator/events.py`
5. `src/duelyst_ai_core/orchestrator/callbacks.py`
6. `src/duelyst_ai_core/agents/debater.py`
7. `src/duelyst_ai_core/agents/judge.py`
8. `src/duelyst_ai_core/agents/prompts.py`
9. `src/duelyst_ai_core/cli/display.py`
10. `src/duelyst_ai_core/cli/live_panel.py`
11. `src/duelyst_ai_core/models/registry.py`
12. `src/duelyst_ai_core/tools/search.py`

That order moves from data model to runtime flow.

## Runtime Layers

The runtime has five layers.

### 1. Configuration and Model Resolution

The debate starts with `DebateConfig` and `ModelConfig` in `state.py`.

- `DebateConfig` describes the topic, models, max rounds, convergence thresholds, custom instructions, and enabled tools.
- `ModelConfig` describes one provider/model pair plus generation settings.
- `ToolType` is a small enum that lets the CLI and programmatic callers request optional capabilities.

Model creation is delegated to `models/registry.py`.

- `resolve_alias()` maps friendly CLI names like `claude-sonnet` or `gpt-mini` to provider/model pairs.
- `create_model()` returns a LangChain `BaseChatModel` directly.
- `get_judge_model()` picks a judge from a different provider than the debaters when the caller does not set one explicitly.

The important design choice here is that there is no custom adapter class hierarchy. The core uses
LangChain's native chat model interface directly.

### 2. Agent Subgraphs

The two debate participants and the judge are thin wrappers around `langchain.agents.create_agent()`.

#### DebaterAgent

`src/duelyst_ai_core/agents/debater.py` builds a LangGraph agent that:

- receives debate context as messages
- may call tools if tools were supplied
- must return a structured `AgentResponse`

`AgentResponse` includes:

- the main argument text
- key points
- evidence
- convergence score and reasoning

#### JudgeAgent

`src/duelyst_ai_core/agents/judge.py` builds another `create_agent()` graph that:

- receives the full transcript as a user message
- has no tools
- must return `JudgeSynthesis`

The judge is intentionally a separate model choice so synthesis is not just another turn from one
of the debaters.

### 3. Outer Orchestrator Graph

The real control plane lives in `src/duelyst_ai_core/orchestrator/engine.py`.

Topology:

```text
START -> init_debate -> run_debater_a -> run_debater_b -> check_convergence
                      ^                                       |
                      |___________ continue _________________ |
                                                              |
                                            converged/max -> run_judge -> END
```

This graph is intentionally small. The complexity sits in the data and event flow, not in a large
number of nodes.

#### Node Responsibilities

| Node | Main job | Key outputs |
| --- | --- | --- |
| `init_debate` | initialize first round and status | `current_round=1`, running status |
| `run_debater_a` | invoke debater A | append one turn |
| `run_debater_b` | invoke debater B | append one turn |
| `check_convergence` | inspect the latest round scores | append history, maybe update status, maybe increment round |
| `run_judge` | invoke judge synthesis | set `synthesis` |

#### Why The Orchestrator Owns The Truth

The sub-agents do not maintain debate state. They only receive messages and return structured output.
The orchestrator owns:

- round count
- accumulated turns
- convergence history
- final status
- synthesis

That makes the debate lifecycle predictable and keeps provider/tool logic out of the control plane.

## State Model

The most important type in the system is `OrchestratorState`.

```python
class OrchestratorState(TypedDict):
    config: DebateConfig
    turns: Annotated[list[dict[str, object]], operator.add]
    current_round: int
    current_agent: Literal["a", "b"]
    convergence_history: list[tuple[int, int]]
    status: DebateStatus
    synthesis: JudgeSynthesis | None
    error: str | None
```

Important details:

- `turns` uses `Annotated[..., operator.add]`, so nodes append to the list instead of replacing it.
- Nodes therefore return deltas like `{"turns": [turn.model_dump(mode="python")]}` rather than the full transcript.
- The state stores plain dictionaries for turns, not `DebateTurn` instances, because LangGraph state works best with serializable runtime values.
- `current_agent` exists in state for clarity and future flexibility, even though the current routing is encoded directly in graph edges.

### Pydantic Models vs Graph State

There are two data layers by design.

- Graph state is lightweight and mutation-friendly.
- Public outputs use frozen Pydantic models for validation and stable API boundaries.

That is why `DebateDisplay.run_debate()` and `DebateOrchestrator.arun_with_events()` rebuild
`DebateTurn` and `DebateResult` after the graph finishes.

## Structured Schemas

`src/duelyst_ai_core/agents/schemas.py` contains the public data model.

Core types:

- `Evidence`
- `AgentResponse`
- `DebateTurn`
- `JudgeSynthesis`
- `DebateMetadata`
- `DebateResult`

There are also `Reflection` and `ToolCallRecord` models. Those are useful extension points.
The current orchestrator persists the final agent response for each turn, but the schema already
has room for richer introspection if the debate loop later starts extracting reflection or tool-call
records from agent runs.

## Prompts And Security Boundaries

`src/duelyst_ai_core/agents/prompts.py` contains a design rule that matters more than the strings
themselves:

- system prompts are static
- user topic, user instructions, and transcript are placed in user messages only

This is an explicit prompt-injection boundary. User-controlled content is never interpolated into the
system prompt.

Two more important details:

- debaters receive a formatted transcript built from agent label, round, and argument text
- evidence metadata stays in stored state but is not currently inserted into the prompt transcript

That keeps prompt context smaller and easier to reason about, at the cost of not replaying every
structured field back into later turns.

## Convergence Logic

`src/duelyst_ai_core/orchestrator/convergence.py` is intentionally a pure function.

```python
check_convergence(history, threshold, required_rounds) -> bool
```

This is good architecture for two reasons:

- convergence rules are easy to test in isolation
- changing convergence behavior does not require touching agent code or graph plumbing

The orchestrator extracts the two latest convergence scores after both debaters speak, appends that
pair to `convergence_history`, and asks the pure function whether the debate should stop.

## Events, Callbacks, And Streaming

This is the part that makes the core useful both for the CLI and for future API servers.

### Event Types

`src/duelyst_ai_core/orchestrator/events.py` defines nine event models:

- `DebateStarted`
- `RoundStarted`
- `TurnStarted`
- `TurnCompleted`
- `ConvergenceUpdate`
- `SynthesisStarted`
- `SynthesisCompleted`
- `DebateCompleted`
- `DebateError`

All events are frozen Pydantic models with a literal `event` discriminator.

### Which Events Come From Where

| Event | Emitted by |
| --- | --- |
| `DebateStarted` | `init_debate()` |
| `RoundStarted` | `init_debate()` and `check_convergence()` when continuing |
| `TurnStarted` | `_run_debater()` before invoking a sub-agent |
| `TurnCompleted` | `_run_debater()` after structured response is available |
| `ConvergenceUpdate` | `check_convergence()` |
| `SynthesisStarted` | `run_judge()` before invoking the judge |
| `SynthesisCompleted` | `run_judge()` after synthesis returns |
| `DebateCompleted` | `arun_with_events()` after the graph finishes successfully |
| `DebateError` | `arun_with_events()` if the run raises |

The last two are worth calling out. They are not graph-node events. They are synthetic wrapper events
added by the streaming API so a consumer always receives a terminal event.

### Callback Protocol

`src/duelyst_ai_core/orchestrator/callbacks.py` defines `DebateEventCallback`.

```python
class DebateEventCallback(Protocol):
    async def on_event(self, event: DebateEvent) -> None: ...
```

There are two built-in implementations:

- `NullCallback` for the default no-op path
- `CollectorCallback` for tests and buffering

Why this matters:

- the CLI can render a live panel from events
- a FastAPI or SSE server can forward events to a browser
- tests can assert exact event order without parsing logs

### `_emit()` And Failure Isolation

`DebateOrchestrator._emit()` wraps callback invocation in `try/except` and logs callback failures
instead of propagating them. A broken observer should not break the debate itself.

That is a deliberate tradeoff: observability is best-effort, debate execution is primary.

### `arun_with_events()`

`DebateOrchestrator.arun_with_events()` is the main programmatic streaming API.

Architecture:

1. create an `asyncio.Queue`
2. install a queue-backed callback
3. run the graph in a background task
4. push every event into the queue
5. yield events from the async generator until a sentinel arrives

This queue bridge is simple and robust. It decouples event production from event consumption and makes
the API natural for callers:

```python
async for event in orchestrator.arun_with_events():
    ...
```

## Async Execution Model

The codebase is async because network-bound LLM calls dominate runtime.

Important async boundaries:

- orchestrator nodes are `async def`
- sub-agent invocations use `await debater.graph.ainvoke(...)`
- callbacks use `async def on_event(...)`
- `arun_with_events()` runs the graph in a background task and yields from a queue

The main practical consequence is that every extension point that touches runtime events should also be
async. Trying to force synchronous hooks here would either block LLM I/O or push complexity upward.

## CLI Architecture

The CLI is in `src/duelyst_ai_core/cli/`.

### `main.py`

This is the Typer entry point. It:

- loads `.env`
- resolves model aliases
- builds `DebateConfig`
- chooses output mode
- delegates execution to `DebateDisplay`

### `display.py`

`DebateDisplay` is the CLI runtime wrapper.

Two modes exist:

- `live=True` for rich terminal output
- `live=False` for markdown/json output where a simple spinner is enough

`run_debate()` does not just execute the graph. It also:

- constructs optional tools
- measures timing metadata
- rebuilds a public `DebateResult`

### `live_panel.py`

`RichDisplayCallback` translates debate events into Rich renderables.

The interesting part is `_run_live()` in `display.py`:

- create a `RichDisplayCallback`
- pass it into `DebateOrchestrator`
- wrap it with a tiny callback that also calls `Live.update(...)`

That keeps rendering state inside the callback and terminal refresh logic inside the display layer.

## Tools

`src/duelyst_ai_core/tools/search.py` exposes the optional Tavily search tool.

Design choices:

- web search is optional per debate
- missing API keys or optional dependencies degrade gracefully
- the search tool is a normal LangChain tool passed into `create_agent()`

The core never assumes that tools exist.

## Output Formatters

Formatters convert `DebateResult` into different surfaces.

- `RichTerminalFormatter`
- `MarkdownFormatter`
- `JsonFormatter`

The formatter layer is separate from the orchestrator on purpose. Debate execution returns structured
data first. Rendering is a later concern.

## Public API Design

`src/duelyst_ai_core/__init__.py` exports the main public surface and lazy-loads symbols.

Why lazy loading matters:

- importing the package stays lightweight
- provider SDKs do not need to load until a caller uses them
- side effects at import time stay smaller

The package now publicly exposes not just debate types but also streaming-related symbols like events
and callback interfaces.

## Runtime Typing Quirks

Two implementation details are easy to miss but important if you modify the code.

### 1. Runtime imports for LangGraph state typing

LangGraph resolves type hints at runtime. That is why some imports in `state.py` and `engine.py` are
real runtime imports, not hidden behind `TYPE_CHECKING`.

### 2. Forward-reference rebuilds for Pydantic

`events.py` and `schemas.py` explicitly rebuild forward refs once dependent types exist at runtime.

Without that, circular references such as `DebateResult -> DebateConfig` would be fragile.

## Testing Strategy

The test suite is designed to validate behavior without spending API credits.

### Local Test Layers

- unit tests mock `create_model()` and agent graph `ainvoke()` calls
- callback tests validate protocol behavior and event ordering
- streaming tests validate the queue bridge and terminal events
- CLI tests validate output mode decisions and display behavior
- integration tests live under `tests/integration/` and are kept separate from the default test run

### Local Commands

Use the same commands the CI uses:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest -v
```

If you are changing packaging or docs only, the important question is still whether the repository can
pass the same gate that GitHub Actions will run.

## GitHub Actions Architecture

Two workflows matter for the repository lifecycle.

### `ci.yml`

This workflow runs on pushes and pull requests to `main`.

Jobs:

- `lint`: Ruff check and format check
- `type-check`: mypy
- `test`: pytest on Python 3.11, 3.12, and 3.13

This workflow answers the day-to-day question: is the repository healthy enough to merge?

### `publish.yml`

This workflow runs only on tags matching `v*`.

It has two stages:

1. `test` quality gate
2. `publish` release job

The release job then:

- verifies the git tag matches `project.version` in `pyproject.toml`
- builds the package with `uv build`
- installs the built wheel in a clean virtualenv and imports `duelyst_ai_core`
- publishes to PyPI via trusted publishing
- creates a GitHub Release and attaches built artifacts

This means publishing is not a separate manual path. It is the same verification path plus packaging.

## How To Test Changes With GitHub Actions In Mind

If you want high confidence before pushing:

1. run the same local quality commands used by `ci.yml`
2. if you changed packaging, also run `uv build`
3. install the built wheel in a fresh virtualenv and import the package

Example:

```bash
uv build
python -m venv /tmp/verify-venv
/tmp/verify-venv/bin/pip install dist/*.whl
/tmp/verify-venv/bin/python -c "import duelyst_ai_core; print(duelyst_ai_core.__version__)"
```

That last step mirrors the publish workflow's smoke test.

## How Publishing Works

Publishing is tag-driven.

Typical release sequence:

1. bump `project.version` in `pyproject.toml`
2. run local checks
3. merge or push the release commit to `main`
4. create and push a tag like `v0.1.0`
5. let GitHub Actions run `publish.yml`

The workflow assumes PyPI trusted publishing is already configured for the repository. If that PyPI-side
configuration is missing, the publish step will fail even if the build is correct.

## Practical Extension Points

If you want to extend the system, these are the safest seams.

- add new event consumers by implementing `DebateEventCallback`
- add new tools by building LangChain tools and passing them into `DebateOrchestrator`
- adjust termination behavior by modifying `check_convergence()`
- add new formatter outputs without touching orchestration
- expand agent result extraction by evolving the structured schemas

These changes preserve the existing layering.

## Mental Model Summary

If you only remember one architecture sentence, it should be this:

The repository is a small outer LangGraph state machine that delegates reasoning to two thin
`create_agent()` subgraphs, stores the debate as structured state, and exposes that runtime through
events, callbacks, and formatters.