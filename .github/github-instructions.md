# Duelyst.ai Core — Project Context

## What is Duelyst.ai

Duelyst.ai is a platform where AI models debate topics in structured multi-turn exchanges. Two AI agents — each powered by a model chosen by the user (Claude, GPT, Gemini) — research, argue, challenge each other, and converge toward a synthesis. Debates are published as shareable content pages, creating a growing library of AI-generated analysis.

The platform reduces two known LLM failure modes: **sycophancy** (a single model tends to agree with the user) and **hallucination** (fabricated facts go unchallenged). When two models debate adversarially, they challenge claims, request evidence, and surface disagreements.

## This Repository: duelyst-ai-core

This is the **open-source debate engine** — a standalone Python package published on PyPI (`pip install duelyst-ai-core`). It contains the LangGraph-based orchestration of multi-agent debates, model adapters, tool integrations, and a CLI.

This repo is the OSS core of an open-core product. It must remain **product-independent** — no references to the API, frontend, billing, or user accounts. A developer installs it, provides their own API keys, and runs debates from the terminal or integrates into their own projects.

### What belongs here

- LangGraph agent and orchestrator definitions
- Model adapters (Anthropic, OpenAI, Google)
- Tool integrations (web search, code execution, data visualization)
- Debate orchestration logic (turns, convergence detection, synthesis)
- CLI interface
- Output formatters (Markdown, JSON, Rich terminal)
- Tests and documentation

### What does NOT belong here

- Authentication, user management, billing
- Database schemas or persistence logic
- Frontend code or API endpoints
- Supabase/Railway/Vercel configuration
- Any secret or API key (even in examples — use placeholders)

## Architecture

### Agent Design

Agents use `create_agent` from `langchain.agents` — a prebuilt ReAct agent factory that handles the tool-calling loop and structured output automatically. Each agent is a thin class wrapping `create_agent` with the appropriate system prompt and `response_format`.

- **DebaterAgent** (`agents/debater.py`): Receives debate context as messages, optionally uses tools (web search, code execution), and produces a structured `AgentResponse` with argument and convergence score.
- **JudgeAgent** (`agents/judge.py`): No tools. Receives the full debate transcript and produces a `JudgeSynthesis`.

### Orchestrator

The orchestrator (`orchestrator/engine.py`) is a LangGraph `StateGraph`:

```
START → init_debate → run_debater_a → run_debater_b → check_convergence
                      ↑                                       |
                      |_____ continue ________________________|
                                                              |
                                            converged/max → run_judge → END
```

- Alternates turns between Agent A and Agent B
- Tracks convergence scores from both agents
- Terminates when: both agents score >= threshold for N consecutive rounds, OR max rounds reached
- Triggers synthesis generation by a judge agent (third model, different from debaters)

### Model Layer

No custom adapter classes. `models/registry.py` provides:
- `create_model(config)` — returns a LangChain `BaseChatModel` directly
- `resolve_alias(name)` — maps short names like "claude-sonnet" to (provider, model_id)
- `get_judge_model(model_a, model_b)` — auto-selects a judge from a different provider

### Models Supported

| Provider | Models | SDK |
|----------|--------|-----|
| Anthropic | Claude Opus, Sonnet, Haiku | `langchain-anthropic` |
| OpenAI | GPT-4o, GPT-4o-mini, GPT-4.1 | `langchain-openai` |
| Google | Gemini Pro, Flash | `langchain-google-genai` |

### Tools

| Tool | Purpose | Provider |
|------|---------|----------|
| Web search | Find current data, statistics, evidence | Tavily (primary), configurable |
| Code execution | Run Python for calculations, charts, data analysis | E2B (cloud) or local Docker |
| Data visualization | Generate charts/graphs as debate evidence | Matplotlib/Plotly via code execution |

Tools are **optional per debate** — the orchestrator and agents work without them. When enabled, agents decide autonomously when to use tools based on the debate context.

## Tech Stack

- **Python 3.11+**
- **LangGraph 1.1+** — orchestrator graph
- **LangChain 1.2+** — agent factory (`create_agent`), model adapters, tool abstractions
- **Pydantic 2.12+** — data models and validation
- **Typer** — CLI framework
- **Rich** — terminal output formatting
- **pytest** — testing

## Coding Conventions

- **Type hints everywhere** — use Pydantic models for data structures, type annotations for all function signatures.
- **Docstrings** — Google style. Every public class and function gets a docstring.
- **Error handling** — never swallow exceptions silently. API errors from LLM providers should be caught, logged, and re-raised with context.
- **No print statements** — use `logging` module or Rich console for output. The CLI layer handles user-facing output; library code uses logging.
- **Async where appropriate** — LLM API calls and tool execution should be async. The CLI can use `asyncio.run()` as entry point.
- **Tests** — pytest. Mock LLM API calls in unit tests (don't burn API credits on CI). Integration tests that call real APIs are in a separate folder, marked with `@pytest.mark.integration`.

## Key Design Decisions

1. **Agents use `create_agent`** — the prebuilt agent factory from `langchain.agents` handles the ReAct loop. No custom multi-node agent graphs.
2. **No custom adapter layer** — LangChain's `BaseChatModel` is the interface. Provider-specific logic lives in private factory functions in `registry.py`.
3. **System prompts are static** — user input goes into the user message, never into the system prompt.
4. **Structured output** — agents return structured Pydantic models via `response_format` parameter.
5. **Tools are optional** — debates work without tools. Tools enhance quality but are not required.
6. **The core is product-independent** — no concept of users, billing, or persistence.

## Current Phase

**Phase 1: OSS Engine + CLI**

Completed: scaffolding, data models, model registry, agents (DebaterAgent, JudgeAgent), orchestrator, convergence detection, streaming events, 117 tests.

Next: CLI, output formatters, web search tool, README.

Code execution tool and data visualization are Phase 4 — do not implement now unless specifically asked.
