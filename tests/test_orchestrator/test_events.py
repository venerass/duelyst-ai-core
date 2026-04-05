"""Tests for streaming event types."""

import pytest
from pydantic import ValidationError

from duelyst_ai_core.agents.schemas import AgentResponse, Evidence
from duelyst_ai_core.orchestrator.events import (
    ConvergenceUpdate,
    DebateError,
    DebateStarted,
    RoundStarted,
    SynthesisStarted,
    TurnCompleted,
    TurnStarted,
)
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig


@pytest.fixture
def debate_config() -> DebateConfig:
    return DebateConfig(
        topic="Test topic",
        model_a=ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514"),
        model_b=ModelConfig(provider="openai", model_id="gpt-4o"),
    )


@pytest.fixture
def sample_response() -> AgentResponse:
    return AgentResponse(
        argument="Test argument",
        key_points=["Point 1"],
        evidence=[Evidence(claim="Test claim")],
        convergence_score=5,
        convergence_reasoning="Moderate agreement",
    )


class TestDebateStarted:
    def test_event_type(self, debate_config: DebateConfig) -> None:
        event = DebateStarted(config=debate_config)
        assert event.event == "debate_started"
        assert event.timestamp is not None

    def test_has_config(self, debate_config: DebateConfig) -> None:
        event = DebateStarted(config=debate_config)
        assert event.config.topic == "Test topic"


class TestRoundStarted:
    def test_valid(self) -> None:
        event = RoundStarted(round_number=1)
        assert event.event == "round_started"

    def test_invalid_round(self) -> None:
        with pytest.raises(ValidationError):
            RoundStarted(round_number=0)


class TestTurnStarted:
    def test_valid(self) -> None:
        event = TurnStarted(agent="a", round_number=1)
        assert event.event == "turn_started"

    def test_invalid_agent(self) -> None:
        with pytest.raises(ValidationError):
            TurnStarted(agent="c", round_number=1)


class TestTurnCompleted:
    def test_valid(self, sample_response: AgentResponse) -> None:
        event = TurnCompleted(agent="b", round_number=2, response=sample_response)
        assert event.event == "turn_completed"
        assert event.response.convergence_score == 5


class TestConvergenceUpdate:
    def test_valid(self) -> None:
        event = ConvergenceUpdate(
            round_number=3,
            score_a=7,
            score_b=8,
            is_converged=True,
        )
        assert event.event == "convergence_update"
        assert event.is_converged is True

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ConvergenceUpdate(round_number=1, score_a=11, score_b=5, is_converged=False)


class TestSynthesisStarted:
    def test_event_type(self) -> None:
        event = SynthesisStarted()
        assert event.event == "synthesis_started"


class TestDebateError:
    def test_valid(self) -> None:
        event = DebateError(
            error_type="ModelError",
            error_message="API rate limit exceeded",
            round_number=2,
            agent="a",
        )
        assert event.event == "debate_error"
        assert event.round_number == 2

    def test_minimal(self) -> None:
        event = DebateError(
            error_type="ConfigError",
            error_message="Invalid model",
        )
        assert event.round_number is None
        assert event.agent is None
