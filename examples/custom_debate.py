"""Custom debate — specific models, instructions, and web search.

Demonstrates the full configuration surface: custom model selection,
per-side instructions, web search tools, and convergence tuning.

Usage:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key
    export GOOGLE_API_KEY=your-key
    export TAVILY_API_KEY=your-key          # optional, for web search
    python examples/custom_debate.py
"""

import asyncio

from duelyst_ai_core import (
    DebateConfig,
    DebateOrchestrator,
    ModelConfig,
    RichTerminalFormatter,
)
from duelyst_ai_core.agents.schemas import DebateMetadata, DebateResult, DebateTurn
from duelyst_ai_core.orchestrator.state import ToolType
from duelyst_ai_core.tools.search import create_search_tool


async def main() -> None:
    # Pick specific models for each side and the judge
    config = DebateConfig(
        topic="Should governments regulate AI development?",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
        judge_model=ModelConfig(provider="google", model_id="gemini-2.5-flash"),
        # Give each side a specific stance
        instructions_a="Argue that strong government regulation is essential for AI safety.",
        instructions_b="Argue that self-regulation and market forces are sufficient.",
        # Tuning
        max_rounds=4,
        convergence_threshold=8,
        convergence_rounds=2,
        tools_enabled=[ToolType.SEARCH],
    )

    # Build tools — gracefully degrade if Tavily is unavailable
    tools = []
    try:
        tools.append(create_search_tool())
        print("Web search enabled.")
    except Exception:
        print("Web search unavailable — running without it.")

    orchestrator = DebateOrchestrator(config, tools=tools or None)

    from datetime import UTC, datetime

    started = datetime.now(UTC)

    result = await orchestrator.graph.ainvoke(
        {
            "config": config,
            "turns": [],
            "current_round": 0,
            "current_agent": "a",
            "convergence_history": [],
            "status": "running",
            "synthesis": None,
            "error": None,
        }
    )

    finished = datetime.now(UTC)
    duration = (finished - started).total_seconds()

    turns = [DebateTurn.model_validate(t) if isinstance(t, dict) else t for t in result["turns"]]
    debate_result = DebateResult(
        config=config,
        turns=turns,
        synthesis=result["synthesis"],
        status="converged" if result["status"] == "converged" else "max_rounds",
        total_rounds=result["current_round"],
        metadata=DebateMetadata(
            started_at=started,
            finished_at=finished,
            duration_seconds=duration,
        ),
    )

    print(RichTerminalFormatter().format(debate_result))


if __name__ == "__main__":
    asyncio.run(main())
