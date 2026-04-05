"""Shared test fixtures for duelyst-ai-core."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from duelyst_ai_core.agents.schemas import (
    AgentResponse,
    DebateMetadata,
    DebateResult,
    DebateTurn,
    Evidence,
    JudgeSynthesis,
)
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig


@pytest.fixture
def sample_config() -> DebateConfig:
    """A minimal debate configuration for testing."""
    return DebateConfig(
        topic="Should startups use microservices or monoliths?",
        model_a=ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514"),
        model_b=ModelConfig(provider="openai", model_id="gpt-4o"),
        max_rounds=2,
        convergence_threshold=7,
        convergence_rounds=2,
    )


@pytest.fixture
def sample_debate_result(sample_config: DebateConfig) -> DebateResult:
    """A complete DebateResult fixture for formatter/CLI tests."""
    now = datetime.now(UTC)

    turns = [
        DebateTurn(
            agent="a",
            round_number=1,
            response=AgentResponse(
                argument=(
                    "Monoliths reduce operational complexity "
                    "for small teams with limited DevOps resources."
                ),
                key_points=["Simpler deployment", "Lower infrastructure cost"],
                evidence=[
                    Evidence(
                        claim="90% of startups have teams under 10 engineers",
                        source_type="reasoning",
                    ),
                ],
                convergence_score=3,
                convergence_reasoning="Fundamental disagreement on scalability trade-offs.",
            ),
            timestamp=now,
        ),
        DebateTurn(
            agent="b",
            round_number=1,
            response=AgentResponse(
                argument=(
                    "Microservices enable horizontal scaling and independent deployment cycles."
                ),
                key_points=["Horizontal scaling", "Independent deployment"],
                evidence=[
                    Evidence(
                        claim="Netflix migrated to microservices to handle 200M+ subscribers",
                        source="https://example.com/netflix",
                        source_type="web",
                    ),
                ],
                convergence_score=3,
                convergence_reasoning="Disagree on the premise that simplicity trumps scalability.",
            ),
            timestamp=now,
        ),
        DebateTurn(
            agent="a",
            round_number=2,
            response=AgentResponse(
                argument="Start with a monolith, extract services when scale demands it.",
                key_points=["Monolith-first approach", "Extract when needed"],
                convergence_score=7,
                convergence_reasoning="Agreeing on a pragmatic middle ground.",
            ),
            timestamp=now,
        ),
        DebateTurn(
            agent="b",
            round_number=2,
            response=AgentResponse(
                argument=(
                    "Agreed — premature microservices add complexity "
                    "without benefit at small scale."
                ),
                key_points=["Start simple", "Scale when needed"],
                convergence_score=8,
                convergence_reasoning="Near full agreement on pragmatic approach.",
            ),
            timestamp=now,
        ),
    ]

    synthesis = JudgeSynthesis(
        summary_side_a=(
            "Side A argued for monoliths, emphasizing simplicity "
            "and lower operational cost for small teams."
        ),
        summary_side_b=(
            "Side B argued for microservices, citing scalability "
            "and independent deployment as key advantages."
        ),
        key_evidence_a=[
            Evidence(
                claim="90% of startups have teams under 10",
                source_type="reasoning",
            )
        ],
        key_evidence_b=[
            Evidence(
                claim="Netflix scaled with microservices",
                source="https://example.com/netflix",
                source_type="web",
            )
        ],
        points_of_agreement=[
            "Start with a monolith for most startups",
            "Extract services when scale demands it",
        ],
        points_of_disagreement=[
            "When exactly to begin the migration to microservices",
        ],
        conclusion=(
            "Both sides converged on a monolith-first approach. "
            "The debate highlighted that architecture should follow "
            "team size and scale needs, not industry trends."
        ),
        winner="draw",
    )

    metadata = DebateMetadata(
        started_at=now,
        finished_at=now,
        duration_seconds=45.2,
        total_tokens_used=12500,
    )

    return DebateResult(
        config=sample_config,
        turns=turns,
        synthesis=synthesis,
        status="converged",
        total_rounds=2,
        metadata=metadata,
    )
