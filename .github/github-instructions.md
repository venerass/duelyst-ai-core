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

Each agent is a **LangGraph graph**, identical in structure, parameterized by model and user instructions. Both agents in a debate share the same architecture — they differ only in configuration (which model, which side to defend, what instructions).

Agent turn flow:

1. **Receive state** — full debate history, topic, assigned side, instructions
2. **Reflect** — analyze opponent's last arguments, identify weak points and strong points
3. **Research** (optional) — web search for current data and evidence if needed
4. **Generate evidence** (optional) — execute code for calculations, charts, data analysis
5. **Formulate response** — argue based on reflection, research, and evidence
6. **Score convergence** — output a 0-10 score indicating agreement level with opponent

### Orchestrator

The orchestrator is the outer graph that manages the debate:

- Alternates turns between Agent A and Agent B
- Passes full debate history to each agent on their turn
- Tracks convergence scores from both agents
- Terminates when: both agents score ≥7 for 2 consecutive rounds, OR max rounds reached
- Triggers synthesis generation by a **judge agent** (third model, different from debaters)

### Judge / Synthesis

The judge receives the complete debate transcript and produces:

- Summary of each side's position
- Key evidence cited by each agent
- Points of agreement
- Points of persistent disagreement
- Balanced conclusion

The judge model is always different from the two debater models to avoid bias.

### Models Supported

| Provider | Models | SDK |
|----------|--------|-----|
| Anthropic | Claude Opus, Sonnet, Haiku | `anthropic` |
| OpenAI | GPT-4o, GPT-4o-mini | `openai` |
| Google | Gemini Pro, Flash | `google-generativeai` |

Model adapters must expose a uniform interface so the orchestrator is model-agnostic.

### Tools

| Tool | Purpose | Provider |
|------|---------|----------|
| Web search | Find current data, statistics, evidence | Tavily (primary), configurable |
| Code execution | Run Python for calculations, charts, data analysis | E2B (cloud) or local Docker |
| Data visualization | Generate charts/graphs as debate evidence | Matplotlib/Plotly via code execution |

Tools are **optional per debate** — the orchestrator and agents work without them. When enabled, agents decide autonomously when to use tools based on the debate context.

## Tech Stack

- **Python 3.11+**
- **LangGraph** — agent orchestration framework
- **LangChain** — model adapters and tool abstractions
- **Pydantic** — data models and validation
- **Typer** or **Click** — CLI framework
- **Rich** — terminal output formatting
- **pytest** — testing

## Project Structure (target)

```
duelyst-ai-core/
├── src/
│   └── duelyst_ai_core/
│       ├── __init__.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── debater.py          # LangGraph agent definition
│       │   ├── judge.py            # Synthesis/judge agent
│       │   └── prompts.py          # System prompts (static templates)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract model adapter interface
│       │   ├── anthropic.py        # Claude adapter
│       │   ├── openai.py           # GPT adapter
│       │   └── google.py           # Gemini adapter
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── search.py           # Web search tool (Tavily)
│       │   ├── code_exec.py        # Code execution (E2B / Docker)
│       │   └── visualization.py    # Chart generation
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   ├── engine.py           # Main debate orchestrator (LangGraph)
│       │   ├── state.py            # Debate state schema
│       │   └── convergence.py      # Convergence detection logic
│       ├── formatters/
│       │   ├── __init__.py
│       │   ├── markdown.py         # Markdown output
│       │   ├── json_fmt.py         # JSON output
│       │   └── rich_terminal.py    # Rich terminal output
│       └── cli/
│           ├── __init__.py
│           └── main.py             # CLI entry point
├── tests/
│   ├── test_agents/
│   ├── test_orchestrator/
│   ├── test_models/
│   └── test_tools/
├── examples/
│   ├── basic_debate.py
│   ├── debate_with_tools.py
│   └── custom_instructions.py
├── pyproject.toml
├── README.md
├── CLAUDE.md                       # This file
├── LICENSE
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

## Key Design Decisions

1. **Agents are stateless** — all state lives in the LangGraph state object, not in the agent. This makes agents composable and testable.
2. **Model adapters are pluggable** — adding a new model means implementing one adapter class. The orchestrator doesn't know which model it's talking to.
3. **Tools are optional** — a debate runs fine without web search or code execution. Tools enhance quality but are not required.
4. **System prompts are static** — user input (topic, instructions, sides) goes into the user message, never into the system prompt. This is a security boundary against prompt injection.
5. **Structured output** — agents return structured Pydantic models (argument text, evidence list, convergence score, tool calls made), not raw text. This makes parsing, formatting, and downstream processing reliable.
6. **The core is product-independent** — it has no concept of users, billing, or persistence. It takes a debate config, runs the debate, and returns the result. The product (duelyst-ai-api) wraps this with product logic.

## Current Phase

**Phase 1: OSS Engine + CLI (Weeks 1-3)**

Priority deliverables in order:

1. Debate state schema and Pydantic models
2. Model adapters for Claude, GPT, Gemini with uniform interface
3. Basic debater agent (LangGraph graph with reflect → respond → score convergence)
4. Orchestrator with turn management and convergence detection
5. Judge agent for synthesis generation
6. CLI with basic debate command
7. Markdown and JSON formatters
8. Web search tool integration (Tavily)
9. README with examples, usage, and installation instructions
10. PyPI publishing setup (pyproject.toml)

Code execution tool and data visualization are Phase 4 — do not implement now unless specifically asked.
