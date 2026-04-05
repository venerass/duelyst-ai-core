"""Basic debate — two models argue a topic.

Minimal example: configure two cheap models, run the debate, and print
the result as Markdown.

Usage:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key
    python examples/basic_debate.py
"""

import asyncio

from duelyst_ai_core import (
    DebateConfig,
    DebateOrchestrator,
    MarkdownFormatter,
    ModelConfig,
)


async def main() -> None:
    config = DebateConfig(
        topic="Should startups use microservices or monoliths?",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
        max_rounds=3,
    )

    orchestrator = DebateOrchestrator(config)

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

    # Build a DebateResult for the formatter
    from datetime import UTC, datetime

    from duelyst_ai_core.agents.schemas import DebateMetadata, DebateResult, DebateTurn

    turns = [DebateTurn.model_validate(t) if isinstance(t, dict) else t for t in result["turns"]]
    debate_result = DebateResult(
        config=config,
        turns=turns,
        synthesis=result["synthesis"],
        status="converged" if result["status"] == "converged" else "max_rounds",
        total_rounds=result["current_round"],
        metadata=DebateMetadata(
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            duration_seconds=0.0,
        ),
    )

    print(MarkdownFormatter().format(debate_result))


if __name__ == "__main__":
    asyncio.run(main())
