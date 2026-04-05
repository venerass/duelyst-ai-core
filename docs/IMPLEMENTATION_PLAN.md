# Duelyst.ai Core — Implementation Plan (v2)

> Phase 1: OSS Engine + CLI — from zero to a working `pip install duelyst-ai-core` that runs multi-model debates from the terminal.

---

## Completed Steps

### Step 1: Project Scaffolding & Tooling ✅

- `pyproject.toml` with Hatch build backend, all dependencies, ruff/mypy/pytest config
- `src/` layout with full directory structure and `__init__.py` files
- `.pre-commit-config.yaml`, `.env.example`, `LICENSE` (MIT)
- `.github/workflows/ci.yml` for lint/type-check/test on push/PR
- `src/duelyst_ai_core/py.typed` PEP 561 marker

### Step 2: Data Models & State Schema ✅

**Files:**
- `orchestrator/state.py` — `ModelConfig`, `DebateConfig`, `DebateStatus`, `ToolType`, `OrchestratorState`
- `agents/schemas.py` — `Evidence`, `Reflection`, `ToolCallRecord`, `AgentResponse`, `DebateTurn`, `JudgeSynthesis`, `DebateMetadata`, `DebateResult`
- `exceptions.py` — `DuelystError` → `ModelError`, `ConfigError`, `ToolError`, `ConvergenceError`
- `orchestrator/events.py` — Streaming event types (`DebateStarted`, `TurnCompleted`, `ConvergenceUpdate`, `DebateCompleted`, etc.)

**Tests:** `tests/test_models/test_schemas.py`, `tests/test_orchestrator/test_events.py`

### Step 3: Model Registry ✅

**Architecture decision:** Removed the abstract `BaseModelAdapter` pattern. LangChain's `BaseChatModel` is the interface. Provider-specific logic consolidated into private factory functions in `registry.py`.

**Files:**
- `models/registry.py` — `create_model()`, `resolve_alias()`, `get_judge_model()`, `MODEL_ALIASES`

**Deleted files:** `models/base.py`, `models/anthropic.py`, `models/openai.py`, `models/google.py` (adapter wrapping was unnecessary)

**Tests:** `tests/test_models/test_adapters.py`, `tests/test_models/test_registry.py`

### Step 4: Agent Implementation ✅

**Architecture decision:** Agents use `create_agent` from `langchain.agents` (prebuilt ReAct agent factory) instead of custom multi-node `StateGraph` graphs. Removed the `BaseAgent` ABC — unnecessary abstraction for two simple wrapper classes.

Key parameter mapping:
- `system_prompt` — static prompt string
- `response_format` — Pydantic model class for structured output
- `tools` — list of LangChain tools (empty for judge)
- Output: `result["structured_response"]` contains the Pydantic model instance

**Files:**
- `agents/debater.py` — `DebaterAgent` class wrapping `create_agent` with `response_format=AgentResponse`
- `agents/judge.py` — `JudgeAgent` class wrapping `create_agent` with `response_format=JudgeSynthesis`
- `agents/prompts.py` — `DEBATER_SYSTEM_PROMPT`, `JUDGE_SYSTEM_PROMPT`, `REFLECTION_PROMPT`, message builders

**Deleted files:** `agents/base.py` (BaseAgent ABC)

**Tests:** `tests/test_agents/test_debater.py`, `tests/test_agents/test_judge.py`, `tests/test_agents/test_prompts.py`

### Step 5: Orchestrator ✅

**Files:**
- `orchestrator/engine.py` — `DebateOrchestrator` class with LangGraph `StateGraph`, nodes for init/debater_a/debater_b/check_convergence/judge, conditional routing, `visualize()` and `visualize_ascii()` methods
- `orchestrator/convergence.py` — `check_convergence()` pure function
- `orchestrator/state.py` — `OrchestratorState` TypedDict with `operator.add` reducer on `turns`

**Subgraph pattern:** Wrapper function (Pattern B) — the orchestrator builds messages, invokes the agent's `create_agent` graph, reads `structured_response`, and maps back to `OrchestratorState`.

**Tests:** `tests/test_orchestrator/test_engine.py`, `tests/test_orchestrator/test_convergence.py`

**Test count:** 117 tests, all passing. Ruff, mypy clean.

---

## Remaining Steps

### Step 6: Output Formatters

**Goal:** Transform `DebateResult` into human-readable or machine-readable formats.

**Files to create:**
- `formatters/base.py` — abstract `BaseFormatter`
- `formatters/markdown.py` — `MarkdownFormatter` (headers, quotes, evidence lists)
- `formatters/json_fmt.py` — `JsonFormatter` (`result.model_dump_json(indent=2)`)
- `formatters/rich_terminal.py` — `RichTerminalFormatter` (Rich panels, tables, colored output)

The Rich formatter should support real-time CLI streaming — render each turn as it arrives.

**Tests:** `tests/test_formatters/`

### Step 7: Web Search Tool (Tavily)

**Goal:** LangChain-compatible tool that agents can invoke for real-time research.

**Files to create:**
- `tools/search.py` — Tavily integration, returns structured results
- `tools/__init__.py` — tool registry/factory

Design notes:
- Uses `tavily-python` SDK
- Returns structured results (title, URL, snippet, relevance score)
- Rate limiting and error handling built in
- Graceful degradation: if no API key, tool is unavailable (not an error)
- Tool is a standard LangChain `BaseTool` — works directly with `create_agent`'s `tools` parameter

**Tests:** `tests/test_tools/test_search.py`

### Step 8: CLI

**Goal:** `duelyst debate "topic"` runs a full debate with real-time terminal output.

**Files to create:**
- `cli/main.py` — Typer app with `debate` command
- `cli/display.py` — Rich live display for streaming debate progress

CLI responsibilities:
- Parse arguments → build `DebateConfig` (using `resolve_alias` for model names)
- Create `DebateOrchestrator` and invoke the graph
- Render events in real-time using Rich (spinners, progress, panels)
- On completion, output final result in requested format
- Handle Ctrl+C gracefully

```python
@app.command()
def debate(
    topic: str,
    model_a: str = "claude-sonnet",
    model_b: str = "gpt-4o",
    rounds: int = 5,
    tools: str | None = None,
    output: str = "rich",
    ...
): ...
```

**Tests:** `tests/test_cli/test_main.py`

### Step 9: Public API & Polish

**Files to update:**
- `__init__.py` — finalize public API exports (`run_debate`, `DebateConfig`, `DebateResult`, etc.)
- `README.md` — comprehensive with examples, installation, usage
- `examples/` — working example scripts
- `.github/workflows/publish.yml` — PyPI publish on tag

---

## Architecture Decisions Log

### Decision 1: No custom model adapter layer

**What:** Removed `BaseModelAdapter`, `AnthropicAdapter`, `OpenAIAdapter`, `GoogleAdapter` in favor of direct `BaseChatModel` usage via factory functions in `registry.py`.

**Why:** Each adapter was ~15 lines wrapping LangChain's native methods. The abstraction added indirection without value — consumer code already depends on LangChain.

### Decision 2: `create_agent` instead of custom agent graphs

**What:** Replaced custom multi-node `StateGraph` agents (reflect → research → formulate) with `create_agent` from `langchain.agents`.

**Why:** The debater IS a ReAct agent — it reasons, optionally uses tools, and produces structured output. `create_agent` does this in 5 lines. The custom graph artificially split what should be one reasoning step into multiple LLM calls.

### Decision 3: No BaseAgent ABC

**What:** Removed the `BaseAgent` abstract base class that `DebaterAgent` and `JudgeAgent` inherited from.

**Why:** With `create_agent`, each agent class is ~20 lines. An ABC for two simple classes adds abstraction without value. If more agent types are needed later, patterns can emerge organically.

### Decision 4: Orchestrator state is separate from agent state

**What:** Agents use `MessagesState` internally (via `create_agent`). The orchestrator has its own `OrchestratorState` TypedDict. The orchestrator maps between them.

**Why:** Different concerns. Agent state is messages in/out. Orchestrator state tracks rounds, convergence, turns, synthesis. Coupling them would complicate both.

### Decision 5: Runtime imports for TypedDict fields

**What:** Types referenced in `OrchestratorState` fields (like `JudgeSynthesis`) must be runtime imports, not behind `TYPE_CHECKING`.

**Why:** LangGraph calls `get_type_hints()` at `StateGraph` construction time. With `from __future__ import annotations`, string annotations are evaluated lazily — but `get_type_hints()` forces evaluation, requiring the types to exist in the module namespace.

---

## Dependencies

```toml
[project]
dependencies = [
    "langgraph>=1.1,<1.2",
    "langchain-core>=1.2,<1.3",
    "langchain>=1.2.15",
    "langchain-anthropic>=1.4,<2.0",
    "langchain-openai>=1.1,<2.0",
    "langchain-google-genai>=4.2,<5.0",
    "pydantic>=2.12,<3.0",
    "typer>=0.24,<1.0",
    "rich>=14.0,<15.0",
]
```

---

## Testing Strategy

| Layer | Type | Approach |
|-------|------|----------|
| Data models | Unit | Direct Pydantic validation, serialization |
| Model registry | Unit | Mock LangChain chat model constructors, test factory dispatch |
| Agents | Unit | Patch `create_agent`, verify init params and graph invocation |
| Orchestrator | Integration | Mock agent graphs, test full debate flow end-to-end |
| CLI | Integration | Typer CliRunner, mock orchestrator |
| Real APIs | Integration | `@pytest.mark.integration`, separate CI job, real API keys in secrets |

Current: 117 tests, all passing.
