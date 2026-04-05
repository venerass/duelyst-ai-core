"""Streaming debate — receive real-time events as the debate progresses.

Uses ``arun_with_events()`` to consume events as an async generator.
This is the pattern the FastAPI backend uses for SSE streaming.

Usage:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key
    python examples/streaming_debate.py
"""

import asyncio

from duelyst_ai_core import (
    DebateConfig,
    DebateOrchestrator,
    ModelConfig,
)


async def main() -> None:
    config = DebateConfig(
        topic="Will AI replace software engineers by 2030?",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
        max_rounds=3,
    )

    orchestrator = DebateOrchestrator(config)

    # Stream events as they happen — each event is a Pydantic model
    async for event in orchestrator.arun_with_events():
        match event.event:
            case "debate_started":
                print(f"Debate started: {event.config.topic}")  # type: ignore[union-attr]
            case "round_started":
                print(f"\n--- Round {event.round_number} ---")  # type: ignore[union-attr]
            case "turn_started":
                print(f"  Agent {event.agent.upper()} is thinking...")  # type: ignore[union-attr]
            case "turn_completed":
                resp = event.response  # type: ignore[union-attr]
                print(f"  Agent {event.agent.upper()}: {resp.argument[:120]}...")  # type: ignore[union-attr]
                print(f"  Convergence: {resp.convergence_score}/10")
            case "convergence_update":
                print(f"  Scores: A={event.score_a}, B={event.score_b}")  # type: ignore[union-attr]
            case "synthesis_started":
                print("\nJudge is synthesizing...")
            case "synthesis_completed":
                print(f"Conclusion: {event.synthesis.conclusion}")  # type: ignore[union-attr]
            case "debate_completed":
                print(f"\nDebate finished! Status: {event.result.status}")  # type: ignore[union-attr]
            case "debate_error":
                print(f"\nError: {event.error_message}")  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
