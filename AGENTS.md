# Duelyst.ai Core — Agent Onboarding

## 1. Project Overview

`duelyst-ai-core` is the open-source debate engine for the Duelyst.ai platform, published on PyPI (`pip install duelyst-ai-core`). It orchestrates structured multi-turn AI debates using LangGraph: two adversarial agents argue a topic, then a judge synthesizes the result. It is a product-independent library — it has no concept of users, billing, or persistence.

**Consumers:** `duelyst-ai-api` (via PyPI), standalone library users (via CLI or Python import), and open-source contributors.

**Current phase:** post-MVP, actively maintained. Phases 1–3 complete (engine, CLI, streaming events). Phase 4 (code execution, data visualization) not yet implemented.

---

## 2. Architecture

### Pattern

LangGraph `StateGraph` orchestrating three agents (debater A, debater B, judge). Agents are thin wrappers around LangChain's `create_agent` factory. Streaming is delivered via an async generator (`arun_with_events()`).

### Data Flow

```
DebateConfig
    │
    ▼
DebateOrchestrator.arun_with_events()     ← primary streaming interface
    │
    ├── asyncio.Queue bridges callback → async generator
    │
    ▼
LangGraph StateGraph
    │
    ├── init_debate        → emits DebateStarted, RoundStarted
    ├── run_debater_a      → DebaterAgent → emits TurnStarted, TurnCompleted
    ├── run_debater_b      → DebaterAgent → emits TurnStarted, TurnCompleted
    ├── check_convergence  → emits ConvergenceUpdate, RoundStarted (if continuing)
    └── run_judge          → JudgeAgent  → emits SynthesisStarted, SynthesisCompleted
    │
    ▼
DebateCompleted | DebateError event
```

**Termination conditions:** both agents score ≥ `convergence_threshold` for `convergence_rounds` consecutive rounds, OR `max_rounds` reached.

### Key Directories

| Path | Owns |
|------|------|
| `src/duelyst_ai_core/orchestrator/` | `DebateOrchestrator`, state, events, convergence, callbacks |
| `src/duelyst_ai_core/agents/` | `DebaterAgent`, `JudgeAgent`, prompts, data schemas |
| `src/duelyst_ai_core/models/` | `registry.py` — model creation, alias resolution, judge auto-selection |
| `src/duelyst_ai_core/tools/` | Web search tool (Tavily); search availability checks |
| `src/duelyst_ai_core/formatters/` | `MarkdownFormatter`, `JsonFormatter`, `RichTerminalFormatter` |
| `src/duelyst_ai_core/cli/` | Typer CLI, Rich live display, `RichDisplayCallback` |
| `tests/` | Unit tests (mocked LLMs), integration tests (real APIs, excluded by default) |

### Critical Files

- **`orchestrator/engine.py`** — `DebateOrchestrator`. The graph is built here. `arun_with_events()` is the interface `duelyst-ai-api` calls. Every event type is emitted from this file. Change carefully.
- **`orchestrator/state.py`** — `DebateConfig`, `ModelConfig`, `OrchestratorState`, `ToolType`, `DebateStatus`. Every other module depends on these types.
- **`orchestrator/events.py`** — All 9 event types (`DebateStarted` through `DebateError`) plus the `DebateEvent` union. Changing event fields breaks `duelyst-ai-api`'s runner.
- **`agents/schemas.py`** — `AgentResponse`, `DebateTurn`, `JudgeSynthesis`, `DebateResult`, `Evidence`. The response shapes that propagate into SSE events consumed by the frontend.
- **`models/registry.py`** — Model alias table, `create_model()`, `get_judge_model()`. Adding a new provider or model only requires changes here.
- **`__init__.py`** — Public API surface with lazy loading. All exported symbols are declared in `__all__` and `_LAZY_EXPORTS`. Adding a public export requires updating both.

---

## 3. Repository Connections

| Repo | Relationship | How |
|------|-------------|-----|
| `duelyst-ai-api` | **Consumer** (downstream) | PyPI package `duelyst-ai-core[search]`; imported only through `integrations/duelyst_core.py` |
| `duelyst-ai-app` | No direct dependency | Consumed through `duelyst-ai-api` only |

**Dependency direction:** `duelyst-ai-core` → `duelyst-ai-api` → `duelyst-ai-app`

**Changes here that would break `duelyst-ai-api`:**
- Removing or renaming `arun_with_events()` on `DebateOrchestrator`
- Changing the fields of any event type (`TurnCompleted`, `SynthesisCompleted`, `DebateCompleted`, `DebateError`)
- Changing `DebateConfig`, `ModelConfig`, or `resolve_alias()` signatures
- Moving or renaming `ToolType` in `orchestrator/state.py`
- Removing any symbol from `__all__`

**This repo must remain product-independent.** Never import or reference authentication, user accounts, database schemas, or billing from this package.

---

## 4. Tech Stack

- **Python 3.11–3.14** — actively tested across this range
- **LangGraph `>=1.1,<1.2`** — `StateGraph` orchestrator
- **LangChain `>=1.2.15`** — `create_agent` factory, `BaseChatModel` interface
- **langchain-anthropic `>=1.4,<2.0`**, **langchain-openai `>=1.1,<2.0`**, **langchain-google-genai `>=4.2,<5.0`** — provider SDKs
- **Pydantic `>=2.12,<3.0`** — all data models
- **Typer `>=0.24`** — CLI framework
- **Rich `>=14.0`** — terminal output and live display
- **python-dotenv `>=1.1`** — `.env` loading in CLI
- **langchain-tavily `>=0.2,<0.3`** — web search (optional extra: `[search]`)
- **pytest `>=9.0`** + **pytest-asyncio `>=1.3`** — all tests async by default (`asyncio_mode="auto"`)
- **Ruff `>=0.15`** + **mypy `>=1.20`** — lint, format, type checking

**External APIs consumed:** Anthropic, OpenAI, Google (Generative AI), Tavily (optional search).

---

## 5. Business Rules & Domain Logic

1. **Agents never receive user input in their system prompt.** Topic, instructions, and debate history go into the user message (via `build_debater_user_message()`). System prompts are static strings. This is a hard security boundary against prompt injection.

2. **The judge model is always from a different provider.** `get_judge_model(model_a, model_b)` refuses to pick a model from the same provider as either debater. The judge defaults are: Anthropic debates → `claude-haiku-4-5`, OpenAI debates → `gpt-5.4-mini`, Google debates → `gemini-2.5-flash`.

3. **Convergence requires both agents to agree.** `check_convergence()` returns `True` only if the last `convergence_rounds` entries in `convergence_history` all have `score_a >= threshold AND score_b >= threshold`. One agent cannot unilaterally end the debate.

4. **`ToolType.CODE` is defined but non-functional.** The enum value exists in `state.py` for future Phase 4 work. Do not wire it to any tool execution path yet.

5. **`DebateTurn.reflection` is always `None` currently.** `Reflection` and `ToolCallRecord` schemas are defined for future use; the debater agent does not produce them in the current implementation.

6. **`arun_with_events()` always ends with exactly one terminal event.** It emits either `DebateCompleted` (success) or `DebateError` (failure). Consumers must handle both; only the last event signals stream end.

7. **`DebateResult.status`** is `"converged"` or `"max_rounds"` — only those two terminal outcomes are possible for successful runs. `DebateError` covers crashes.

### Domain Model

```
DebateConfig
  ├── topic, model_a, model_b, judge_model (optional)
  ├── instructions_a, instructions_b (optional, ≤5000 chars each)
  ├── max_rounds (1–20, default 5)
  ├── convergence_threshold (1–10, default 7)
  ├── convergence_rounds (1–5, default 2)
  └── tools_enabled: list[ToolType]

DebateResult
  ├── config: DebateConfig
  ├── turns: list[DebateTurn]   # agent: "a" | "b"
  ├── synthesis: JudgeSynthesis
  ├── status: "converged" | "max_rounds"
  └── metadata: DebateMetadata

DebateTurn
  └── response: AgentResponse
        ├── argument, key_points, evidence: list[Evidence]
        └── convergence_score (0–10)

JudgeSynthesis
  ├── summary_side_a/b, key_evidence_a/b
  ├── points_of_agreement, points_of_disagreement
  ├── conclusion
  └── winner: "a" | "b" | "draw" | None
```

---

## 6. Code Standards & Conventions

- **Style:** Ruff with `select = ["E","W","F","I","N","UP","B","SIM","TCH","RUF","S","T20","PTH","ASYNC"]`, line length 100. Run `ruff format` before `ruff check`.
- **Type checking:** mypy strict mode. All functions must have type annotations. No bare `Any` in public signatures.
- **Docstrings:** Google style on all public classes and functions.
- **No `print()` in library code** — use `logging.getLogger(__name__)`. The CLI layer owns all user-facing output via Rich.
- **LangGraph runtime types** — all types used in `TypedDict` state fields (e.g., `OrchestratorState`) must be regular imports, not under `TYPE_CHECKING`. LangGraph calls `get_type_hints()` at graph construction time and will fail on forward references from `TYPE_CHECKING` blocks.
- **Async everywhere for I/O** — LLM calls and tool execution are async. Do not call sync LLM methods in an async context.
- **Conventional commits** — format: `type: subject` lowercase, no period. E.g. `feat: add gemini-2.5-ultra alias`.
- **Import order:** stdlib → third-party → local (Ruff `I` rules enforce this).

---

## 7. Non-Obvious Patterns

### Pattern 1: Lazy exports in `__init__.py`

All public symbols are loaded lazily via `__getattr__` + `_LAZY_EXPORTS` dict. This avoids import-time side effects (e.g., LangChain trying to validate API keys at `import duelyst_ai_core`). When adding a new public export:
1. Add to `_LAZY_EXPORTS = {"SymbolName": "module.path"}` in `__init__.py`
2. Add to `__all__`

**Anti-pattern:** Direct top-level import of provider SDKs in `__init__.py`.

### Pattern 2: `asyncio.Queue` bridges callback to `arun_with_events()`

`arun_with_events()` cannot directly `yield` from callback invocations because the LangGraph node execution is not in the generator's call stack. Instead, an internal `_QueueCallback` puts events into an `asyncio.Queue`, and the async generator `get()`s from that queue. The `None` sentinel signals end-of-stream. Always publish the terminal event (`DebateCompleted` or `DebateError`) before putting `None`.

### Pattern 3: Agent classes are thin `create_agent` wrappers

`DebaterAgent` and `JudgeAgent` are ~20-line classes. They differ only in `system_prompt`, `response_format`, and whether `tools` is passed. Do not add orchestration logic to these classes — it belongs in `engine.py` nodes. If you need a new agent type, follow the same pattern: one class, one system prompt, one response format.

### Pattern 4: Model aliases are the only user-facing model names

Never expose raw provider model IDs to users. All user-facing model names go through `resolve_alias()`. Adding a new model requires only a new entry in `MODEL_ALIASES` in `registry.py`. The alias is what the CLI `--model-a` flag accepts and what `duelyst-ai-api` validates against.

**Current aliases:**

| Alias | Provider | Model ID |
|-------|----------|----------|
| `claude-opus` | anthropic | `claude-opus-4-6` |
| `claude-sonnet` | anthropic | `claude-sonnet-4-6` |
| `claude-haiku` | anthropic | `claude-haiku-4-5` |
| `gpt-5` | openai | `gpt-5.4` |
| `gpt-mini` | openai | `gpt-5.4-mini` |
| `gpt-nano` | openai | `gpt-5.4-nano` |
| `gpt-4o` | openai | `gpt-4o` *(legacy compat)* |
| `gpt-4o-mini` | openai | `gpt-4o-mini` *(legacy compat)* |
| `gpt-4.1` | openai | `gpt-4.1` *(legacy compat)* |
| `gpt-4.1-mini` | openai | `gpt-4.1-mini` *(legacy compat)* |
| `gemini-pro` | google | `gemini-2.5-pro` |
| `gemini-flash` | google | `gemini-2.5-flash` |
| `gemini-flash-lite` | google | `gemini-2.5-flash-lite` |

---

## 8. Lessons Learned & Pitfalls

- **LangGraph runtime type resolution.** `OrchestratorState` is a `TypedDict`. Any type referenced in its fields (e.g., `DebateTurn`, `DebateConfig`) must be a real import — not a string annotation or `TYPE_CHECKING` import. Placing them behind `TYPE_CHECKING` produces a `NameError` when LangGraph builds the graph.

- **`TurnStarted` and `SynthesisStarted` are NOT in `__all__`** — they are internal streaming events used only by `duelyst-ai-api`'s runner. Do not add them to the public API without a coordinated update to the API package.

- **`NullCallback` is also not exported.** It's the default when no callback is passed. Library consumers who want to collect events should use `CollectorCallback`.

- **Python 3.14 LangChain `PydanticV1` warning.** `_runtime.py` suppresses a `DeprecationWarning` from LangChain's internal Pydantic v1 compatibility layer on Python 3.14+. It is called automatically at import time. Do not remove it.

- **The `[search]` extra is required for web search.** `create_search_tool()` raises `ToolError` (not a soft warning) if `langchain-tavily` is not installed. `is_search_available()` checks for both the installation and the `TAVILY_API_KEY` env var — use it before calling `create_search_tool()`.

- **`DebateTurn.reflection` schema exists but is always `None`.** The `Reflection` model and the `reflection` field on `DebateTurn` are defined for future Phase 4 use. Do not write code that depends on `reflection` being populated.

- **`E2B_API_KEY` in `.env.example` is Phase 4 only.** Cloud code execution via E2B is not implemented. The key is documented to avoid confusion when developers see it.

- **`ModelAlias` must not carry product display concerns.** Fields like `reasoning`, `context_window`, and `description` belong in the API's product catalog (`domain/models/catalog.py`), not here. `ModelAlias` is intentionally thin: `(provider, model_id, tier)` only. Adding display fields back will cause them to leak into the OSS library and couple it to product decisions.

- **Two aliases can map to the same `model_id` with different tiers.** This is intentional for reasoning variants (e.g., `gpt-5` standard vs `gpt-5-high` pro both map to `gpt-5.4`). The alias is the unit of tier gating and display; the `model_id` is what the provider API receives.

- **Legacy compatibility aliases must stay in `MODEL_ALIASES` indefinitely.** Existing debates in the database store `model_id` values resolved from old aliases. Removing a legacy alias (`gpt-4o`, `gemini-2.5-flash`, etc.) will break `build_debate_config()` in the API for those historical debates. Legacy aliases are not shown in the product catalog — they exist only for resolution.

- **`uv run pytest` may silently use the wrong venv if a foreign `VIRTUAL_ENV` is active.** Use `uv run -- python -m pytest` when another project's venv is activated, or deactivate first. The `VIRTUAL_ENV` mismatch warning is printed but execution still succeeds using the project's own venv.

- **Bumping the version in `pyproject.toml` is the only release step needed.** Push a `vX.Y.Z` tag and the GitHub Actions workflow handles build, quality gate, and PyPI publish automatically via Trusted Publishing — no token management required. The tag version must exactly match `pyproject.toml`.

- **Multiple agents may work on this repository in parallel.** When committing, only `git add` the specific files you changed — never use `git add .` or `git add -A`. Another agent may have uncommitted work in the same repo. Use `git add <file1> <file2>` explicitly. Same applies to `git stash` and `git reset` — scope them to your files only.

---

## 9. Development Setup & Testing

### Testing Standards

- **Run tests with `uv run -- python -m pytest -v`**, not bare `pytest`, to avoid venv confusion when multiple Python projects are open.
- **All unit tests mock LLM calls** — no API keys are needed for `uv run pytest`. The full suite (~191 tests) runs in ~3 seconds.
- **Integration tests (`tests/integration/`) require real API keys** and are excluded from CI by default (`-m integration` marker). Run them manually before major releases.
- **After adding a new model alias**, add a test in `test_registry.py` covering the alias, tier, and provider. After changing `_JUDGE_DEFAULTS`, update `test_with_judge` in `test_main.py` to assert the new judge model ID.

### Environment Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

cd duelyst-ai-core
uv sync --dev         # installs all deps including dev extras

cp .env.example .env
# Edit .env: fill the API keys for the providers you want to test against
```

### Required Environment Variables

| Variable | Required | Notes |
|----------|----------|-------|
| `ANTHROPIC_API_KEY` | For Claude models | |
| `OPENAI_API_KEY` | For GPT models | |
| `GOOGLE_API_KEY` | For Gemini models | |
| `TAVILY_API_KEY` | Optional | Required for `search` tool; omit to disable gracefully |
| `E2B_API_KEY` | Future Phase 4 | Cloud code execution; not implemented yet |

### Run Tests

```bash
# Full unit test suite (mocked LLM calls, no API keys needed)
uv run pytest -v

# Single module
uv run pytest tests/test_orchestrator/test_convergence.py -v

# By keyword
uv run pytest -k "streaming" -v

# With coverage report
uv run pytest --cov=duelyst_ai_core --cov-report=term-missing

# Integration tests (requires real API keys, NOT run in CI by default)
uv run pytest tests/integration/ -v -m integration
```

### Lint and Format

```bash
# Always format first (auto-fix), then check
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

### Run the CLI Locally

```bash
# Basic debate (uses .env for API keys)
uv run duelyst debate "Is remote work more productive than office work?"

# With specific models and tools
uv run duelyst debate "Nuclear energy vs renewables" \
  --model-a claude-haiku --model-b gpt-mini \
  --rounds 3 --tools search --output markdown

# JSON output
uv run duelyst debate "Topic" --output json > result.json
```

### Release & Publish to PyPI

Publishing is fully automated via GitHub Actions (`.github/workflows/publish.yml`). **No PyPI token or manual upload is needed.** The workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) with `id-token: write` permissions.

**To release a new version:**

```bash
# 1. Bump the version in pyproject.toml
#    Edit [project] version = "X.Y.Z"

# 2. Commit the version bump
git add pyproject.toml
git commit -m "chore: bump version to X.Y.Z"

# 3. Create and push the tag (must match pyproject.toml exactly)
git tag vX.Y.Z
git push origin main --tags
```

That's it. Pushing the tag triggers the workflow, which:
1. Runs the full quality gate (ruff, mypy, pytest) on Python 3.11, 3.12, and 3.13
2. Verifies the tag version matches `pyproject.toml`
3. Builds the sdist and wheel with `uv build`
4. Publishes to PyPI via `pypa/gh-action-pypi-publish`
5. Creates a GitHub Release with auto-generated notes and the build artifacts attached

Monitor progress in the **Actions** tab on GitHub. If the quality gate fails, fix and re-tag (`git tag -f vX.Y.Z && git push origin --tags --force`).

**After publish:** any `duelyst-ai-api` consumer can upgrade with:
```bash
uv add duelyst-ai-core[search]==X.Y.Z
```

---

## 10. Keeping This File Current

Update this file when:
- A new directory or critical file is added
- A model alias, provider, or tool integration is added or removed
- A business rule or domain constraint changes
- A new non-obvious pattern is established (used 3+ times)
- A significant bug, gotcha, or integration quirk is discovered
- The tech stack or Python version support range changes
- The relationship with `duelyst-ai-api` changes
- The release or CI/CD process changes

Do **not** update for routine bug fixes, dependency patch bumps, or changes that don't affect how an agent should write code.
