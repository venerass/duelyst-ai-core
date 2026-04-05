"""JSON output — machine-consumable debate results.

Runs a debate and outputs the full result as JSON to stdout.
Useful for pipelines, automation, or feeding into other tools.

Usage:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key
    python examples/json_output.py > debate.json
"""

import asyncio

from duelyst_ai_core import (
    DebateConfig,
    DebateOrchestrator,
    JsonFormatter,
    ModelConfig,
)
from duelyst_ai_core.agents.schemas import DebateMetadata, DebateResult, DebateTurn


async def main() -> None:
    config = DebateConfig(
        topic="Rust vs Go for backend services",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
        max_rounds=2,
    )

    orchestrator = DebateOrchestrator(config)

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
            duration_seconds=(finished - started).total_seconds(),
        ),
    )

    print(JsonFormatter().format(debate_result))


if __name__ == "__main__":
    asyncio.run(main())
