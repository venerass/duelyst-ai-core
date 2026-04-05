# Duelyst.ai Core — Implementation Plan (v2)

> Phase 1: OSS Engine + CLI — from zero to a working `pip install duelyst-ai-core` that runs multi-model debates from the terminal.

**Status: Phase 1 Complete** — All steps implemented. 154 tests passing. Ruff, mypy clean.

---

## Completed Steps

### Step 1: Project Scaffolding & Tooling

- `pyproject.toml` with Hatch build backend, all dependencies, ruff/mypy/pytest config
- `src/` layout with full directory structure and `__init__.py` files
- `.pre-commit-config.yaml`, `.env.example`, `LICENSE` (MIT)
- `.github/workflows/ci.yml` for lint/type-check/test on push/PR
- `src/duelyst_ai_core/py.typed` PEP 561 marker

### Step 2: Data Models & State Schema

**Files:**
- `orchestrator/state.py` — `ModelConfig`, `DebateConfig`, `DebateStatus`, `ToolType`, `OrchestratorState`
- `agents/schemas.py` — `Evidence`, `Reflection`, `ToolCallRecord`, `AgentResponse`, `DebateTurn`, `JudgeSynthesis`, `DebateMetadata`, `DebateResult`
- `exceptions.py` — `DuelystError` → `ModelError`, `ConfigError`, `ToolError`, `ConvergenceError`
- `orchestrator/events.py` — Streaming event types (`DebateStarted`, `TurnCompleted`, `ConvergenceUpdate`, `DebateCompleted`, etc.)

**Tests:** `tests/test_models/test_schemas.py`, `tests/test_orchestrator/test_events.py`

### Step 3: Model Registry

**Architecture decision:** Removed the abstract `BaseModelAdapter` pattern. LangChain's `BaseChatModel` is the interface. Provider-specific logic consolidated into private factory functions in `registry.py`.

**Files:**
- `models/registry.py` — `create_model()`, `resolve_alias()`, `get_judge_model()`, `MODEL_ALIASES`

**Deleted files:** `models/base.py`, `models/anthropic.py`, `models/openai.py`, `models/google.py` (adapter wrapping was unnecessary)

**Tests:** `tests/test_models/test_adapters.py`, `tests/test_models/test_registry.py`

### Step 4: Agent Implementation

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

### Step 5: Orchestrator

**Files:**
- `orchestrator/engine.py` — `DebateOrchestrator` class with LangGraph `StateGraph`, nodes for init/debater_a/debater_b/check_convergence/judge, conditional routing, `visualize()` and `visualize_ascii()` methods
- `orchestrator/convergence.py` — `check_convergence()` pure function
- `orchestrator/state.py` — `OrchestratorState` TypedDict with `operator.add` reducer on `turns`

**Subgraph pattern:** Wrapper function (Pattern B) — the orchestrator builds messages, invokes the agent's `create_agent` graph, reads `structured_response`, and maps back to `OrchestratorState`.

**Tests:** `tests/test_orchestrator/test_engine.py`, `tests/test_orchestrator/test_convergence.py`

### Step 6: Output Formatters

**Files:**
- `formatters/base.py` — abstract `BaseFormatter` with `format(result: DebateResult) -> str`
- `formatters/markdown.py` — `MarkdownFormatter` (headers, quoted arguments, evidence lists, synthesis sections)
- `formatters/json_fmt.py` — `JsonFormatter` (`result.model_dump_json(indent=2)`)
- `formatters/rich_terminal.py` — `RichTerminalFormatter` (Rich panels with agent colors, tables, styled synthesis)
- `formatters/__init__.py` — exports all three formatters

**Tests:** `tests/test_formatters/test_markdown.py` (11 tests), `tests/test_formatters/test_json.py` (4 tests), `tests/test_formatters/test_rich.py` (5 tests)

### Step 7: Web Search Tool (Tavily)

**Files:**
- `tools/search.py` — `create_search_tool()` returns `TavilySearchResults` from `langchain-community`, `is_search_available()` for graceful degradation
- `tools/__init__.py` — exports both functions

Design: standard LangChain `BaseTool` that works with `create_agent`'s `tools` parameter. Raises `ConfigError` if no API key, `ToolError` if package missing.

**Tests:** `tests/test_tools/test_search.py` (5 tests)

### Step 8: CLI

**Files:**
- `cli/main.py` — Typer app with `debate` command, all options (model-a, model-b, judge, instructions, rounds, threshold, convergence-rounds, tools, output, verbose)
- `cli/display.py` — `DebateDisplay` class with `show_config()` and `run_debate()`, Rich Status spinner, builds `DebateResult` from graph output

**Tests:** `tests/test_cli/test_main.py` (12 tests)

### Step 9: Public API & Polish

**Files:**
- `__init__.py` — finalized public API exports (13 symbols)
- `README.md` — comprehensive with installation, CLI usage, Python API, model table, architecture, CLI options, development setup, project structure
- `tests/conftest.py` — shared fixtures (`sample_config`, `sample_debate_result`)

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

[project.optional-dependencies]
search = ["tavily-python>=0.7,<1.0"]
```

---

## Testing Strategy

| Layer | Type | Approach |
|-------|------|----------|
| Data models | Unit | Direct Pydantic validation, serialization |
| Model registry | Unit | Mock LangChain chat model constructors, test factory dispatch |
| Agents | Unit | Patch `create_agent`, verify init params and graph invocation |
| Orchestrator | Integration | Mock agent graphs, test full debate flow end-to-end |
| Formatters | Unit | Test output format, content presence, structure |
| Tools | Unit | Mock imports and env vars, test creation and availability |
| CLI | Integration | Typer CliRunner, mock orchestrator |
| Real APIs | Integration | `@pytest.mark.integration`, separate CI job, real API keys in secrets |

Current: 154 tests, all passing.

---

## Future Phases

### Phase 2: Streaming & Real-time Display
- Event-based streaming from orchestrator
- Rich live display that updates per-turn during debate

### Phase 3: Examples & Documentation
- Working example scripts (`examples/`)
- PyPI publish workflow (`.github/workflows/publish.yml`)

### Phase 4: Code Execution & Visualization
- E2B / Docker code execution tool
- Matplotlib/Plotly chart generation as debate evidence
