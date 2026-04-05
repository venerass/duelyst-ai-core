# Duelyst.ai Core — Project Context

## What is Duelyst.ai

Duelyst.ai is a platform where AI models debate topics in structured multi-turn exchanges. Two AI agents — each powered by a model chosen by the user (Claude, GPT, Gemini) — research, argue, challenge each other, and converge toward a synthesis.

The platform reduces two known LLM failure modes: **sycophancy** (a single model tends to agree with the user) and **hallucination** (fabricated facts go unchallenged). When two models debate adversarially, they challenge claims, request evidence, and surface disagreements.

## This Repository: duelyst-ai-core

This is the **open-source debate engine** — a standalone Python package published on PyPI (`pip install duelyst-ai-core`). It contains the LangGraph-based orchestration of multi-agent debates, model adapters, tool integrations, and a CLI.

This repo is the OSS core of an open-core product. It must remain **product-independent** — no references to the API, frontend, billing, or user accounts.

### What belongs here

- LangGraph agent and orchestrator definitions
- Model registry (Anthropic, OpenAI, Google)
- Tool integrations (web search)
- Debate orchestration logic (turns, convergence detection, synthesis)
- CLI interface
- Output formatters (Markdown, JSON, Rich terminal)
- Tests and documentation

### What does NOT belong here

- Authentication, user management, billing
- Database schemas or persistence logic
- Frontend code or API endpoints
- Any secret or API key (even in examples — use placeholders)

## Architecture

### Agent Design

Agents use `create_agent` from `langchain.agents` — a prebuilt ReAct agent factory that handles the tool-calling loop and structured output automatically. Each agent is a thin class wrapping `create_agent` with the appropriate system prompt and `response_format`.

- **DebaterAgent** (`agents/debater.py`): Receives debate context as messages, optionally uses tools (web search), produces structured `AgentResponse` with argument and convergence score.
- **JudgeAgent** (`agents/judge.py`): No tools. Receives full transcript, produces `JudgeSynthesis`.

### Orchestrator

The orchestrator (`orchestrator/engine.py`) is a LangGraph `StateGraph`:

```
START → init_debate → run_debater_a → run_debater_b → check_convergence
                      ↑                                       |
                      |_____ continue ________________________|
                                                              |
                                            converged/max → run_judge → END
```

### Model Layer

No custom adapter classes. `models/registry.py` provides:
- `create_model(config)` — returns a LangChain `BaseChatModel` directly
- `resolve_alias(name)` — maps short names like "claude-sonnet" to (provider, model_id)
- `get_judge_model(model_a, model_b)` — auto-selects a judge from a different provider

### Tools

Web search via Tavily (`tools/search.py`). Optional — agents work without it. Graceful degradation when API key is missing.

### Output Formatters

- `RichTerminalFormatter` — Rich panels with colors
- `MarkdownFormatter` — headers, quotes, evidence lists
- `JsonFormatter` — Pydantic model serialization

### CLI

Typer-based CLI (`cli/main.py`) with `debate` command, Rich live display (`cli/display.py`), supports all output formats.

## Tech Stack

- **Python 3.11+**
- **LangGraph 1.1+** — orchestrator graph
- **LangChain 1.2+** — agent factory (`create_agent`), model adapters, tool abstractions
- **Pydantic 2.12+** — data models and validation
- **Typer** — CLI framework
- **Rich** — terminal output formatting
- **pytest** — testing

## Coding Conventions

- **Type hints everywhere** — Pydantic models for data structures, type annotations for all function signatures.
- **Docstrings** — Google style. Every public class and function.
- **Error handling** — never swallow exceptions silently. Catch, log, and re-raise with context.
- **No print statements** — use `logging` module or Rich console.
- **Async where appropriate** — LLM calls and tool execution are async.
- **Tests** — pytest. Mock LLM APIs in unit tests. Integration tests in separate folder with `@pytest.mark.integration`.
- **LangGraph runtime types** — types in TypedDict state fields must be runtime imports (not behind `TYPE_CHECKING`).

## Key Design Decisions

1. **Agents use `create_agent`** — prebuilt ReAct agent factory. No custom multi-node agent graphs.
2. **No custom adapter layer** — LangChain's `BaseChatModel` is the interface.
3. **No BaseAgent ABC** — each agent is ~20 lines, abstraction unnecessary.
4. **System prompts are static** — user input goes in user message, never system prompt.
5. **Structured output** — `response_format` parameter, output in `state["structured_response"]`.
6. **Tools are optional** — debates work without tools.
7. **Product-independent** — no users, billing, or persistence.
8. **Orchestrator state is source of truth** — agents use `MessagesState` internally.

## Current Status

**Phase 1: Complete** — All deliverables implemented. 154 tests passing. Ruff, mypy clean.
