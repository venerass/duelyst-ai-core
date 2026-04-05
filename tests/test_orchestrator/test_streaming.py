"""Tests for the arun_with_events() streaming API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duelyst_ai_core.agents.schemas import AgentResponse, JudgeSynthesis
from duelyst_ai_core.orchestrator.engine import DebateOrchestrator
from duelyst_ai_core.orchestrator.events import (
    DebateCompleted,
    DebateError,
)
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig


def _create_orchestrator(config: DebateConfig) -> DebateOrchestrator:
    with (
        patch("duelyst_ai_core.orchestrator.engine.create_model") as mock_create,
        patch("duelyst_ai_core.orchestrator.engine.get_judge_model") as mock_judge,
    ):
        mock_create.return_value = MagicMock()
        mock_judge.return_value = ModelConfig(provider="google", model_id="gemini-2.5-flash")
        return DebateOrchestrator(config)


@pytest.fixture
def config() -> DebateConfig:
    return DebateConfig(
        topic="Test streaming",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
        max_rounds=1,
    )


@pytest.fixture
def mock_response_a() -> AgentResponse:
    return AgentResponse(
        argument="Argument A",
        key_points=["Point A"],
        convergence_score=5,
        convergence_reasoning="Some agreement.",
    )


@pytest.fixture
def mock_response_b() -> AgentResponse:
    return AgentResponse(
        argument="Argument B",
        key_points=["Point B"],
        convergence_score=5,
        convergence_reasoning="Some agreement.",
    )


@pytest.fixture
def mock_synthesis() -> JudgeSynthesis:
    return JudgeSynthesis(
        summary_side_a="A argued something.",
        summary_side_b="B argued something.",
        conclusion="Both have points.",
        winner="draw",
    )


class TestArunWithEvents:
    async def test_yields_events_in_order(
        self,
        config: DebateConfig,
        mock_response_a: AgentResponse,
        mock_response_b: AgentResponse,
        mock_synthesis: JudgeSynthesis,
    ) -> None:
        orchestrator = _create_orchestrator(config)

        orchestrator._debater_a.graph.ainvoke = AsyncMock(
            return_value={"structured_response": mock_response_a}
        )
        orchestrator._debater_b.graph.ainvoke = AsyncMock(
            return_value={"structured_response": mock_response_b}
        )
        orchestrator._judge.graph.ainvoke = AsyncMock(
            return_value={"structured_response": mock_synthesis}
        )

        events = []
        async for event in orchestrator.arun_with_events():
            events.append(event)

        event_types = [e.event for e in events]

        # Should see all intermediate events plus final DebateCompleted
        assert "debate_started" in event_types
        assert "turn_started" in event_types
        assert "turn_completed" in event_types
        assert "convergence_update" in event_types
        assert "synthesis_started" in event_types
        assert "synthesis_completed" in event_types
        assert event_types[-1] == "debate_completed"

    async def test_ends_with_debate_completed(
        self,
        config: DebateConfig,
        mock_response_a: AgentResponse,
        mock_response_b: AgentResponse,
        mock_synthesis: JudgeSynthesis,
    ) -> None:
        orchestrator = _create_orchestrator(config)

        orchestrator._debater_a.graph.ainvoke = AsyncMock(
            return_value={"structured_response": mock_response_a}
        )
        orchestrator._debater_b.graph.ainvoke = AsyncMock(
            return_value={"structured_response": mock_response_b}
        )
        orchestrator._judge.graph.ainvoke = AsyncMock(
            return_value={"structured_response": mock_synthesis}
        )

        last_event = None
        async for event in orchestrator.arun_with_events():
            last_event = event

        assert isinstance(last_event, DebateCompleted)
        assert last_event.result.total_rounds == 1

    async def test_error_yields_debate_error(self, config: DebateConfig) -> None:
        orchestrator = _create_orchestrator(config)

        orchestrator._debater_a.graph.ainvoke = AsyncMock(side_effect=RuntimeError("API failure"))

        events = []
        async for event in orchestrator.arun_with_events():
            events.append(event)

        # Should have some events before failure, then DebateError
        last_event = events[-1]
        assert isinstance(last_event, DebateError)
        assert "API failure" in last_event.error_message
