# Duelyst.ai Core — Implementation Plan (v1)

> Phase 1: OSS Engine + CLI — from zero to a working `pip install duelyst-ai-core` that runs multi-model debates from the terminal.

---

## 1. Project Scaffolding & Tooling

**Goal:** Production-grade Python package skeleton with modern tooling.

### 1.1 `pyproject.toml` (PEP 621 + Hatch build backend)

```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- Package metadata: name, version (0.1.0), description, license (MIT), Python >=3.11
- `src/` layout (`src/duelyst_ai_core/`)
- CLI entry point: `[project.scripts] duelyst = "duelyst_ai_core.cli.main:app"`
- Dependency groups:
  - **core**: `langgraph`, `langchain-core`, `pydantic>=2.0`, `typer`, `rich`
  - **models**: `langchain-anthropic`, `langchain-openai`, `langchain-google-genai`
  - **tools**: `tavily-python` (optional extra)
  - **dev**: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`

### 1.2 Developer tooling

| Tool | Purpose | Config location |
|------|---------|-----------------|
| **Ruff** | Linting + formatting (replaces flake8/black/isort) | `pyproject.toml [tool.ruff]` |
| **mypy** | Static type checking (strict mode) | `pyproject.toml [tool.mypy]` |
| **pre-commit** | Git hooks (ruff, mypy, trailing whitespace) | `.pre-commit-config.yaml` |
| **pytest** | Testing with asyncio support | `pyproject.toml [tool.pytest]` |

### 1.3 Files to create

```
pyproject.toml
.pre-commit-config.yaml
.env.example
LICENSE                          # MIT
src/duelyst_ai_core/__init__.py  # version, public API re-exports
src/duelyst_ai_core/py.typed     # PEP 561 marker
```

### 1.4 CI/CD — GitHub Actions

- `.github/workflows/ci.yml`: lint (ruff) → type-check (mypy) → test (pytest) on push/PR
- Matrix: Python 3.11, 3.12, 3.13
- `.github/workflows/publish.yml`: build + publish to PyPI on tag push

---

## 2. Data Models & State Schema

**Goal:** Define the Pydantic models that represent every data structure in the system. This is the foundation — everything else depends on these types.

### 2.1 `src/duelyst_ai_core/orchestrator/state.py` — Debate State

```python
class DebateConfig(BaseModel):
    """Immutable configuration for a debate session."""
    topic: str
    model_a: ModelConfig
    model_b: ModelConfig
    judge_model: ModelConfig | None     # auto-selected if None
    instructions_a: str | None
    instructions_b: str | None
    max_rounds: int = 5                 # default 5 rounds
    tools_enabled: list[ToolType] = []  # empty = no tools
    
class ModelConfig(BaseModel):
    """Configuration for a single model."""
    provider: Literal["anthropic", "openai", "google"]
    model_id: str                       # e.g. "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096

class ToolType(str, Enum):
    SEARCH = "search"
    CODE = "code"                       # Phase 4, but define enum now
```

### 2.2 `src/duelyst_ai_core/agents/schemas.py` — Agent I/O Models

```python
class Evidence(BaseModel):
    """A piece of evidence cited by an agent."""
    claim: str
    source: str | None                  # URL or "reasoning"
    source_type: Literal["web", "code", "reasoning"]

class AgentResponse(BaseModel):
    """Structured output from a debater agent turn."""
    argument: str                       # The main argument text
    key_points: list[str]               # Bullet-point summary
    evidence: list[Evidence]
    convergence_score: int = Field(ge=0, le=10)
    convergence_reasoning: str          # Why this score
    
class Reflection(BaseModel):
    """Internal reflection before formulating response."""
    opponent_strong_points: list[str]
    opponent_weak_points: list[str]
    strategy: str                       # How to approach this turn

class DebateTurn(BaseModel):
    """A single turn in the debate history."""
    agent: Literal["a", "b"]
    round_number: int
    response: AgentResponse
    reflection: Reflection | None       # None for first turn
    tool_calls: list[ToolCallRecord] = []
    timestamp: datetime
```

### 2.3 `src/duelyst_ai_core/orchestrator/state.py` — LangGraph State

```python
class DebateState(TypedDict):
    """LangGraph state object — the single source of truth during a debate."""
    config: DebateConfig
    turns: Annotated[list[DebateTurn], operator.add]
    current_round: int
    current_agent: Literal["a", "b"]
    convergence_history: list[tuple[int, int]]  # (score_a, score_b) per round
    status: Literal["running", "converged", "max_rounds", "error"]
    error: str | None
```

### 2.4 `src/duelyst_ai_core/agents/schemas.py` — Judge Output

```python
class JudgeSynthesis(BaseModel):
    """Structured output from the judge agent."""
    summary_side_a: str
    summary_side_b: str
    key_evidence_a: list[Evidence]
    key_evidence_b: list[Evidence]
    points_of_agreement: list[str]
    points_of_disagreement: list[str]
    conclusion: str
    winner: Literal["a", "b", "draw"] | None  # Optional explicit winner

class DebateResult(BaseModel):
    """Complete result of a finished debate."""
    config: DebateConfig
    turns: list[DebateTurn]
    synthesis: JudgeSynthesis
    status: Literal["converged", "max_rounds"]
    total_rounds: int
    metadata: DebateMetadata

class DebateMetadata(BaseModel):
    """Metadata about the debate execution."""
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    total_tokens_used: int | None       # If available from providers
```

**Implementation notes:**
- All models use `model_config = ConfigDict(frozen=True)` where appropriate (configs are immutable, state is mutable)
- Pydantic v2 with `from __future__ import annotations` for modern type syntax
- Every model has a docstring explaining its role

**Tests:** `tests/test_models/test_schemas.py`
- Validation: convergence_score bounds (0-10), required fields, enum values
- Serialization: round-trip JSON serialize/deserialize
- Edge cases: empty evidence lists, None optional fields

---

## 3. Model Adapters

**Goal:** Uniform async interface for calling Claude, GPT, and Gemini. The orchestrator never knows which model it's talking to.

### 3.1 `src/duelyst_ai_core/models/base.py` — Abstract Interface

```python
class BaseModelAdapter(ABC):
    """Abstract base for all LLM model adapters."""
    
    @abstractmethod
    async def generate(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
        response_format: type[BaseModel] | None = None,
    ) -> AIMessage: ...

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
        response_model: type[T],
    ) -> T: ...
    
    @abstractmethod
    def supports_structured_output(self) -> bool: ...
```

**Key design decisions:**
- Uses LangChain's `ChatModel` under the hood (langchain-anthropic, langchain-openai, langchain-google-genai) — these handle auth, retries, rate limiting
- `generate_structured` uses LangChain's `.with_structured_output()` for providers that support it natively (Anthropic, OpenAI), falls back to JSON-mode + Pydantic parsing for others
- Adapters handle provider-specific quirks (e.g., Google's different message format) behind the uniform interface

### 3.2 Concrete Adapters

| File | Class | LangChain model |
|------|-------|-----------------|
| `models/anthropic.py` | `AnthropicAdapter` | `ChatAnthropic` |
| `models/openai.py` | `OpenAIAdapter` | `ChatOpenAI` |
| `models/google.py` | `GoogleAdapter` | `ChatGoogleGenerativeAI` |

Each adapter:
- Accepts `ModelConfig` in constructor
- Reads API key from environment (never as parameter)
- Validates the model_id is supported at construction time
- Wraps provider errors in a `ModelError` exception with context

### 3.3 `src/duelyst_ai_core/models/registry.py` — Factory

```python
MODEL_REGISTRY: dict[str, tuple[str, type[BaseModelAdapter]]] = {
    "claude-sonnet": ("anthropic", AnthropicAdapter),
    "claude-opus": ("anthropic", AnthropicAdapter),
    "claude-haiku": ("anthropic", AnthropicAdapter),
    "gpt-4o": ("openai", OpenAIAdapter),
    "gpt-4o-mini": ("openai", OpenAIAdapter),
    "gemini-pro": ("google", GoogleAdapter),
    "gemini-flash": ("google", GoogleAdapter),
}

def create_model(model_config: ModelConfig) -> BaseModelAdapter: ...
def get_judge_model(model_a: str, model_b: str) -> ModelConfig: ...
```

`get_judge_model` auto-selects a judge from a different provider than both debaters.

**Tests:** `tests/test_models/`
- Unit tests mock the LangChain chat models
- Test structured output parsing with known response fixtures
- Test error wrapping (API timeout, auth failure, rate limit)
- Test registry resolution and judge auto-selection logic

---

## 4. Agent Implementation

**Goal:** A single LangGraph subgraph that represents one debater, reusable for both sides.

### 4.1 `src/duelyst_ai_core/agents/prompts.py` — System Prompts

Static templates, no user input injected:

```python
DEBATER_SYSTEM_PROMPT = """You are a rigorous debate participant...
You MUST respond with structured JSON matching the required schema.
You MUST score convergence honestly — do not artificially agree or disagree.
..."""

JUDGE_SYSTEM_PROMPT = """You are an impartial debate judge...
Analyze the complete debate transcript and produce a balanced synthesis.
..."""
```

User-specific context (topic, side, instructions) goes into the user message, never the system prompt.

### 4.2 `src/duelyst_ai_core/agents/debater.py` — Debater Agent Graph

LangGraph subgraph with these nodes:

```
reflect → [research] → formulate → score_convergence
```

- **reflect**: Analyzes opponent's last turn, outputs `Reflection`. Skipped on first turn.
- **research** (conditional): If tools enabled AND agent decides research is needed, calls web search. Controlled by a conditional edge.
- **formulate**: Takes reflection + research results + full history, generates `AgentResponse` via structured output.
- **score_convergence**: Already part of `AgentResponse` — this node validates the score is consistent with the argument (sanity check).

```python
def create_debater_graph(
    model: BaseModelAdapter,
    agent_role: Literal["a", "b"],
    tools: list[BaseTool] | None = None,
) -> CompiledGraph: ...
```

### 4.3 `src/duelyst_ai_core/agents/judge.py` — Judge Agent

Simpler graph — single node:

```
analyze_and_synthesize → output
```

- Receives full debate transcript
- Uses structured output to produce `JudgeSynthesis`
- Uses a different model from both debaters (enforced by orchestrator)

**Tests:** `tests/test_agents/`
- Mock model adapter returning canned responses
- Test reflect node with various debate histories
- Test convergence scoring validation
- Test judge synthesis with a sample transcript
- Test that user input never appears in system prompt

---

## 5. Orchestrator

**Goal:** The outer LangGraph graph that manages the full debate lifecycle.

### 5.1 `src/duelyst_ai_core/orchestrator/engine.py`

LangGraph graph:

```
initialize → agent_a_turn → agent_b_turn → check_convergence → [loop or synthesize]
                  ↑                              |
                  └──────────────────────────────┘
```

Nodes:
- **initialize**: Validate config, create model adapters, set up state
- **agent_a_turn**: Invoke debater subgraph for agent A, append turn to state
- **agent_b_turn**: Invoke debater subgraph for agent B, append turn to state
- **check_convergence**: Evaluate convergence criteria
- **synthesize**: Invoke judge agent, produce final `DebateResult`

Conditional edges:
- After `check_convergence`: if converged or max_rounds → `synthesize`, else → `agent_a_turn`

```python
async def run_debate(config: DebateConfig) -> DebateResult: ...
```

This is the primary public API of the library.

### 5.2 `src/duelyst_ai_core/orchestrator/convergence.py`

```python
def check_convergence(
    convergence_history: list[tuple[int, int]],
    threshold: int = 7,
    consecutive_rounds: int = 2,
) -> bool: ...
```

- Both agents must score >= threshold for `consecutive_rounds` consecutive rounds
- Returns `True` when debate should end

### 5.3 Streaming support

The orchestrator should yield events as the debate progresses:

```python
async def run_debate_stream(
    config: DebateConfig,
) -> AsyncIterator[DebateEvent]: ...
```

Event types: `DebateStarted`, `TurnStarted`, `TurnCompleted`, `ConvergenceUpdate`, `SynthesisStarted`, `DebateCompleted`, `DebateError`

This is critical for the CLI (show progress in real-time) and for the future API (SSE streaming).

**Tests:** `tests/test_orchestrator/`
- Convergence detection: various score sequences, edge cases (first round, all zeros, immediate convergence)
- Full orchestrator flow with mocked agents (verify turn alternation, state accumulation, termination conditions)
- Error handling: model failure mid-debate, invalid config
- Streaming: verify event sequence and types

---

## 6. Output Formatters

**Goal:** Transform `DebateResult` into human-readable or machine-readable formats.

### 6.1 `src/duelyst_ai_core/formatters/base.py`

```python
class BaseFormatter(ABC):
    @abstractmethod
    def format(self, result: DebateResult) -> str: ...
```

### 6.2 Concrete Formatters

| File | Class | Output |
|------|-------|--------|
| `formatters/markdown.py` | `MarkdownFormatter` | Clean Markdown with headers, quotes, evidence lists |
| `formatters/json_fmt.py` | `JsonFormatter` | `result.model_dump_json(indent=2)` with metadata |
| `formatters/rich_terminal.py` | `RichTerminalFormatter` | Rich panels, tables, colored output for terminal |

The Rich formatter is special — it's used for real-time CLI output during streaming, not just final results. It renders each turn as it arrives.

**Tests:** `tests/test_formatters/`
- Markdown: verify structure, headers, evidence formatting
- JSON: round-trip parse, schema validation
- Rich: snapshot tests or verify Panel/Table construction

---

## 7. Web Search Tool (Tavily)

**Goal:** LangChain-compatible tool that agents can invoke for real-time research.

### 7.1 `src/duelyst_ai_core/tools/search.py`

```python
class WebSearchTool:
    """Web search tool using Tavily API."""
    
    def __init__(self, api_key: str | None = None):
        # Falls back to TAVILY_API_KEY env var
        ...
    
    def as_langchain_tool(self) -> BaseTool:
        """Return a LangChain-compatible tool for use in agents."""
        ...
```

- Uses `tavily-python` SDK
- Returns structured results (title, URL, snippet, relevance score)
- Rate limiting and error handling built in
- Graceful degradation: if no API key, tool is unavailable (not an error)

**Tests:** `tests/test_tools/`
- Mock Tavily API responses
- Test result parsing and structuring
- Test missing API key behavior

---

## 8. CLI

**Goal:** `duelyst debate "topic"` runs a full debate with real-time terminal output.

### 8.1 `src/duelyst_ai_core/cli/main.py`

Typer application:

```python
app = typer.Typer(name="duelyst", help="AI-powered debate engine")

@app.command()
def debate(
    topic: str = typer.Argument(..., help="The debate topic"),
    model_a: str = typer.Option("claude-sonnet", help="Model for side A"),
    model_b: str = typer.Option("gpt-4o", help="Model for side B"),
    judge: str | None = typer.Option(None, help="Judge model (auto-selected if omitted)"),
    instructions_a: str | None = typer.Option(None, help="Instructions for side A"),
    instructions_b: str | None = typer.Option(None, help="Instructions for side B"),
    rounds: int = typer.Option(5, help="Maximum debate rounds"),
    tools: str | None = typer.Option(None, help="Comma-separated tools: search,code"),
    output: str = typer.Option("rich", help="Output format: rich, markdown, json"),
    verbose: bool = typer.Option(False, help="Show debug information"),
): ...
```

CLI responsibilities:
- Parse arguments → build `DebateConfig`
- Call `run_debate_stream()` 
- Render events in real-time using Rich (spinners, progress, panels)
- On completion, output final result in requested format
- Handle Ctrl+C gracefully (show partial results)

### 8.2 `src/duelyst_ai_core/cli/display.py`

Rich-based live display:
- Show current round and active agent
- Render each turn as it completes (Rich Panel with agent color coding)
- Show convergence scores as a progress bar or gauge
- Final synthesis in a highlighted panel

**Tests:** `tests/test_cli/`
- Use Typer's `CliRunner` for testing
- Test argument parsing and config building
- Test output format selection
- Test error messages for missing API keys

---

## 9. Public API & Package Exports

### 9.1 `src/duelyst_ai_core/__init__.py`

Clean public API:

```python
from duelyst_ai_core.orchestrator.engine import run_debate, run_debate_stream
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig
from duelyst_ai_core.agents.schemas import DebateResult, DebateTurn, JudgeSynthesis
from duelyst_ai_core.models.registry import create_model

__version__ = "0.1.0"
__all__ = [
    "run_debate",
    "run_debate_stream", 
    "DebateConfig",
    "ModelConfig",
    "DebateResult",
    "DebateTurn",
    "JudgeSynthesis",
    "create_model",
]
```

This is the programmatic API — a developer can `from duelyst_ai_core import run_debate, DebateConfig` and run a debate without the CLI.

---

## 10. Error Handling & Logging

### 10.1 `src/duelyst_ai_core/exceptions.py`

```python
class DuelystError(Exception): """Base exception."""
class ModelError(DuelystError): """LLM provider error (auth, rate limit, timeout)."""
class ConfigError(DuelystError): """Invalid debate configuration."""
class ToolError(DuelystError): """Tool execution failure."""
class ConvergenceError(DuelystError): """Unexpected convergence state."""
```

### 10.2 Logging strategy

- Library code uses `logging.getLogger(__name__)` — no handlers attached (let the consumer configure)
- CLI attaches a Rich handler with configurable level (`--verbose` → DEBUG)
- Structured log messages with context: `logger.error("Model API call failed", extra={"provider": "anthropic", "model": model_id, "error": str(e)})`

---

## 11. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Prompt injection | System prompts are static constants. User input (topic, instructions) goes only in user messages. |
| API key exposure | Keys read from env vars only. Never logged, never in error messages, never serialized. `.env` in `.gitignore`. |
| Tool output injection | Tool results (web search) are treated as untrusted data. Sanitized before inclusion in prompts. |
| Dependency security | Pin major versions. `dependabot` or `renovate` for updates. |
| Code execution (Phase 4) | Sandboxed via E2B (cloud) or Docker with resource limits. Not in Phase 1. |

---

## 12. Implementation Order

Strict dependency order — each step builds on the previous.

### Step 1: Scaffolding
- [ ] `pyproject.toml` with all dependencies and tool configs
- [ ] `src/duelyst_ai_core/__init__.py` (empty, version only)
- [ ] `.pre-commit-config.yaml`
- [ ] `.env.example`
- [ ] `LICENSE` (MIT)
- [ ] `.github/workflows/ci.yml`
- [ ] Create full directory structure with `__init__.py` files

### Step 2: Data Models
- [ ] `orchestrator/state.py` — `DebateConfig`, `ModelConfig`, `DebateState`, `ToolType`
- [ ] `agents/schemas.py` — `Evidence`, `AgentResponse`, `Reflection`, `DebateTurn`, `JudgeSynthesis`, `DebateResult`, `DebateMetadata`
- [ ] `exceptions.py` — custom exception hierarchy
- [ ] Tests for all models (validation, serialization, edge cases)

### Step 3: Model Adapters
- [ ] `models/base.py` — abstract `BaseModelAdapter`
- [ ] `models/anthropic.py` — `AnthropicAdapter`
- [ ] `models/openai.py` — `OpenAIAdapter`
- [ ] `models/google.py` — `GoogleAdapter`
- [ ] `models/registry.py` — factory and judge auto-selection
- [ ] Tests with mocked LangChain models

### Step 4: Agent Implementation
- [ ] `agents/prompts.py` — system prompt templates
- [ ] `agents/debater.py` — LangGraph debater subgraph
- [ ] `agents/judge.py` — judge agent
- [ ] Tests with mocked model adapters

### Step 5: Orchestrator
- [ ] `orchestrator/convergence.py` — convergence detection
- [ ] `orchestrator/engine.py` — main debate graph + `run_debate` / `run_debate_stream`
- [ ] Streaming event types
- [ ] Tests for full debate flow (mocked agents)

### Step 6: Formatters
- [ ] `formatters/base.py` — abstract formatter
- [ ] `formatters/markdown.py`
- [ ] `formatters/json_fmt.py`
- [ ] `formatters/rich_terminal.py` — including live streaming display
- [ ] Tests with fixture debate results

### Step 7: Web Search Tool
- [ ] `tools/search.py` — Tavily integration
- [ ] `tools/__init__.py` — tool registry
- [ ] Tests with mocked API

### Step 8: CLI
- [ ] `cli/main.py` — Typer app with `debate` command
- [ ] `cli/display.py` — Rich live display
- [ ] Tests with CliRunner

### Step 9: Polish & Publish
- [ ] `README.md` — comprehensive with examples, installation, usage
- [ ] `examples/` — working example scripts
- [ ] `__init__.py` — finalize public API exports
- [ ] `.github/workflows/publish.yml` — PyPI publish on tag
- [ ] Final pass: ruff, mypy, all tests green

---

## 13. Testing Strategy

| Layer | Type | Approach |
|-------|------|----------|
| Data models | Unit | Direct Pydantic validation, serialization |
| Model adapters | Unit | Mock LangChain chat models, test structured output parsing |
| Agents | Unit | Mock model adapters, verify prompt construction and response parsing |
| Orchestrator | Integration | Mock agents, test full debate flow end-to-end |
| CLI | Integration | Typer CliRunner, mock orchestrator |
| Real APIs | Integration | `@pytest.mark.integration`, separate CI job, real API keys in secrets |

Target: >90% coverage on library code, excluding integration tests.

```
tests/
├── conftest.py                  # Shared fixtures (sample configs, mock responses)
├── test_models/
│   ├── test_schemas.py
│   ├── test_anthropic.py
│   ├── test_openai.py
│   ├── test_google.py
│   └── test_registry.py
├── test_agents/
│   ├── test_debater.py
│   └── test_judge.py
├── test_orchestrator/
│   ├── test_convergence.py
│   ├── test_engine.py
│   └── test_streaming.py
├── test_formatters/
│   ├── test_markdown.py
│   ├── test_json.py
│   └── test_rich.py
├── test_tools/
│   └── test_search.py
├── test_cli/
│   └── test_main.py
└── integration/                 # Real API tests, not in default test run
    ├── test_real_debate.py
    └── conftest.py
```

---

## 14. Dependencies (pinned ranges)

```toml
[project]
dependencies = [
    "langgraph>=0.4,<1.0",
    "langchain-core>=0.3,<1.0",
    "langchain-anthropic>=0.3,<1.0",
    "langchain-openai>=0.3,<1.0",
    "langchain-google-genai>=2.1,<3.0",
    "pydantic>=2.0,<3.0",
    "typer>=0.15,<1.0",
    "rich>=13.0,<14.0",
]

[project.optional-dependencies]
search = ["tavily-python>=0.5,<1.0"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "ruff>=0.9",
    "mypy>=1.14",
    "pre-commit>=4.0",
]
```

---

## 15. Milestones & Acceptance Criteria

### M1: Foundation (Steps 1-2)
- `pip install -e ".[dev]"` works
- All data models validate correctly
- `ruff check` and `mypy` pass
- Tests pass

### M2: Models + Agents (Steps 3-4)
- Can instantiate each model adapter (with mocked API)
- Debater agent produces valid `AgentResponse` from mocked model
- Judge agent produces valid `JudgeSynthesis`
- All tests pass

### M3: Working Debate (Steps 5-6)
- `run_debate()` executes a full debate with mocked models
- Convergence detection works correctly
- All three formatters produce valid output
- Streaming events fire in correct order

### M4: CLI + Tools (Steps 7-8)
- `duelyst debate "topic"` runs with real API keys
- Web search tool works when enabled
- Rich terminal shows live debate progress
- All output formats work

### M5: Ship It (Step 9)
- README is complete and accurate
- Examples run successfully
- CI pipeline green
- `pip install duelyst-ai-core` from PyPI works
