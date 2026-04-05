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

- **DebaterAgent** (`agents/debater.py`): Receives debate context as messages, optionally uses tools (web search, code execution), and produces a structured `AgentResponse` with argument and convergence score. The model decides autonomously when to use tools.
- **JudgeAgent** (`agents/judge.py`): No tools. Receives the full debate transcript and produces a `JudgeSynthesis`.

Both agents share the same pattern — they differ only in model, system prompt, and structured output schema.

### Orchestrator

The orchestrator (`orchestrator/engine.py`) is a LangGraph `StateGraph` that manages the full debate lifecycle:

```
START → init_debate → run_debater_a → run_debater_b → check_convergence
                      ↑                                       |
                      |_____ continue ________________________|
                                                              |
                                            converged/max → run_judge → END
```

- Alternates turns between Agent A and Agent B
- Passes full debate history to each agent on their turn
- Tracks convergence scores from both agents
- Terminates when: both agents score >= threshold for N consecutive rounds, OR max rounds reached
- Triggers synthesis generation by a judge agent (third model, different from debaters)

### Model Layer

No custom adapter classes. The `models/registry.py` module provides:
- `create_model(config)` — returns a LangChain `BaseChatModel` directly (ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI)
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

## Project Structure

```
duelyst-ai-core/
├── src/
│   └── duelyst_ai_core/
│       ├── __init__.py
│       ├── exceptions.py            # DuelystError hierarchy
│       ├── agents/
│       │   ├── debater.py           # DebaterAgent (create_agent wrapper)
│       │   ├── judge.py             # JudgeAgent (create_agent wrapper)
│       │   ├── prompts.py           # Static system prompts + message builders
│       │   └── schemas.py           # AgentResponse, JudgeSynthesis, DebateTurn, etc.
│       ├── models/
│       │   └── registry.py          # create_model(), resolve_alias(), get_judge_model()
│       ├── orchestrator/
│       │   ├── engine.py            # DebateOrchestrator (LangGraph StateGraph)
│       │   ├── state.py             # OrchestratorState, DebateConfig, ModelConfig
│       │   ├── convergence.py       # check_convergence() pure function
│       │   └── events.py            # Streaming event types (DebateEvent union)
│       ├── tools/                   # Tool integrations (Phase 2)
│       ├── formatters/              # Output formatters (Phase 2)
│       └── cli/
│           └── main.py              # CLI entry point (Phase 2)
├── tests/
│   ├── test_agents/                 # test_debater.py, test_judge.py, test_prompts.py
│   ├── test_orchestrator/           # test_engine.py, test_convergence.py, test_events.py
│   ├── test_models/                 # test_adapters.py, test_registry.py, test_schemas.py
│   └── integration/                 # Real API tests (not in default test run)
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── CLAUDE.md
└── .env.example
```

## Configuration

Users configure via environment variables. Never hardcode keys or defaults that assume a specific provider.

```bash
# .env.example
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here        # Optional: for web search tool
E2B_API_KEY=your-key-here            # Optional: for cloud code execution
```

## CLI Usage (target interface)

```bash
# Basic debate
duelyst debate "Should startups use microservices or monoliths?"

# Choose models
duelyst debate "Rust vs Go for backend in 2026" --model-a claude-sonnet --model-b gpt-4o

# Assign sides and instructions
duelyst debate "Will AI replace software engineers by 2030?" \
  --model-a claude-sonnet \
  --model-b gemini-pro \
  --instructions-a "Defend the position that AI will replace most software engineering jobs" \
  --instructions-b "Defend the position that AI will augment but not replace engineers" \
  --rounds 5

# Enable tools
duelyst debate "Bitcoin price prediction 2026" \
  --model-a claude-sonnet \
  --model-b gpt-4o \
  --tools search,code

# Output format
duelyst debate "..." --output markdown > debate.md
duelyst debate "..." --output json > debate.json
```

## Coding Conventions

- **Type hints everywhere** — use Pydantic models for data structures, type annotations for all function signatures.
- **Docstrings** — Google style. Every public class and function gets a docstring.
- **Error handling** — never swallow exceptions silently. API errors from LLM providers should be caught, logged, and re-raised with context.
- **No print statements** — use `logging` module or Rich console for output. The CLI layer handles user-facing output; library code uses logging.
- **Async where appropriate** — LLM API calls and tool execution should be async. The CLI can use `asyncio.run()` as entry point.
- **Tests** — pytest. Mock LLM API calls in unit tests (don't burn API credits on CI). Integration tests that call real APIs are in a separate folder, marked with `@pytest.mark.integration`.
- **LangGraph runtime types** — types used in TypedDict state fields must be runtime imports (not behind `TYPE_CHECKING`), because LangGraph calls `get_type_hints()` at graph construction time.

## Key Design Decisions

1. **Agents use `create_agent`** — the prebuilt agent factory from `langchain.agents` handles the ReAct loop (model → tools → model → structured output). No custom multi-node agent graphs.
2. **No custom adapter layer** — LangChain's `BaseChatModel` is the interface. Provider-specific logic lives in private factory functions in `registry.py`.
3. **System prompts are static** — user input (topic, instructions, sides) goes into the user message, never into the system prompt. This is a security boundary against prompt injection.
4. **Structured output** — agents return structured Pydantic models via `response_format` parameter. Output is in `state["structured_response"]`.
5. **Tools are optional** — a debate runs fine without web search or code execution. Tools enhance quality but are not required.
6. **The core is product-independent** — it has no concept of users, billing, or persistence. It takes a debate config, runs the debate, and returns the result.
7. **Orchestrator state is the source of truth** — agent subgraphs use `MessagesState` internally; the orchestrator maps between its own state and agent messages.

## Current Phase

**Phase 1: OSS Engine + CLI**

Completed:
1. Project scaffolding & tooling (pyproject.toml, ruff, mypy, pytest, CI)
2. Data models & Pydantic schemas (AgentResponse, JudgeSynthesis, DebateTurn, etc.)
3. Model registry (create_model, resolve_alias, get_judge_model)
4. Agent implementation (DebaterAgent, JudgeAgent using create_agent)
5. Orchestrator with turn management, convergence detection, and judge synthesis
6. Streaming event types
7. 117 tests passing

Next:
- CLI with basic debate command
- Output formatters (Markdown, JSON, Rich terminal)
- Web search tool integration (Tavily)
- README with examples
