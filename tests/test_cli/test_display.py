"""Tests for RichDisplayCallback and DebateDisplay."""

from __future__ import annotations

import pytest

from duelyst_ai_core.agents.schemas import AgentResponse, Evidence, JudgeSynthesis
from duelyst_ai_core.cli.live_panel import RichDisplayCallback
from duelyst_ai_core.orchestrator.events import (
    ConvergenceUpdate,
    DebateError,
    DebateStarted,
    RoundStarted,
    SynthesisCompleted,
    SynthesisStarted,
    TurnCompleted,
    TurnStarted,
)
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig


@pytest.fixture
def debate_config() -> DebateConfig:
    return DebateConfig(
        topic="Test topic",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
    )


@pytest.fixture
def sample_response() -> AgentResponse:
    return AgentResponse(
        argument="Test argument text",
        evidence=[Evidence(claim="Test claim", source_type="web")],
        convergence_score=5,
        convergence_reasoning="Moderate agreement",
    )


@pytest.fixture
def sample_synthesis() -> JudgeSynthesis:
    return JudgeSynthesis(
        summary_side_a="Side A argued well.",
        summary_side_b="Side B argued well.",
        conclusion="Both sides have merit.",
        winner="draw",
    )


class TestRichDisplayCallback:
    async def test_starts_empty(self) -> None:
        cb = RichDisplayCallback()
        renderable = cb.build()
        assert renderable is not None

    async def test_debate_started_adds_section(self, debate_config: DebateConfig) -> None:
        cb = RichDisplayCallback()
        await cb.on_event(DebateStarted(config=debate_config))
        assert len(cb._sections) == 1

    async def test_round_started_adds_section(self) -> None:
        cb = RichDisplayCallback()
        await cb.on_event(RoundStarted(round_number=1))
        assert len(cb._sections) == 1

    async def test_turn_started_sets_spinner(self) -> None:
        cb = RichDisplayCallback()
        await cb.on_event(TurnStarted(agent="a", round_number=1))
        assert cb._spinner is not None

    async def test_turn_completed_clears_spinner(self, sample_response: AgentResponse) -> None:
        cb = RichDisplayCallback()
        await cb.on_event(TurnStarted(agent="a", round_number=1))
        assert cb._spinner is not None
        await cb.on_event(TurnCompleted(agent="a", round_number=1, response=sample_response))
        assert cb._spinner is None
        assert len(cb._sections) == 1

    async def test_convergence_update_adds_section(self) -> None:
        cb = RichDisplayCallback()
        await cb.on_event(
            ConvergenceUpdate(round_number=1, score_a=3, score_b=4, is_converged=False)
        )
        assert len(cb._sections) == 1

    async def test_synthesis_flow(self, sample_synthesis: JudgeSynthesis) -> None:
        cb = RichDisplayCallback()
        await cb.on_event(SynthesisStarted())
        assert cb._spinner is not None

        await cb.on_event(SynthesisCompleted(synthesis=sample_synthesis))
        assert cb._spinner is None
        assert len(cb._sections) == 1

    async def test_error_clears_spinner(self) -> None:
        cb = RichDisplayCallback()
        await cb.on_event(TurnStarted(agent="a", round_number=1))
        await cb.on_event(DebateError(error_type="TestError", error_message="something broke"))
        assert cb._spinner is None
        assert len(cb._sections) == 1

    async def test_full_event_sequence(
        self,
        debate_config: DebateConfig,
        sample_response: AgentResponse,
        sample_synthesis: JudgeSynthesis,
    ) -> None:
        """Run a complete event sequence and verify section count."""
        cb = RichDisplayCallback()

        events = [
            DebateStarted(config=debate_config),
            RoundStarted(round_number=1),
            TurnStarted(agent="a", round_number=1),
            TurnCompleted(agent="a", round_number=1, response=sample_response),
            TurnStarted(agent="b", round_number=1),
            TurnCompleted(agent="b", round_number=1, response=sample_response),
            ConvergenceUpdate(round_number=1, score_a=5, score_b=5, is_converged=False),
            SynthesisStarted(),
            SynthesisCompleted(synthesis=sample_synthesis),
        ]

        for event in events:
            await cb.on_event(event)

        # config + round + turn_a + turn_b + convergence + synthesis = 6 sections
        assert len(cb._sections) == 6
        assert cb._spinner is None

        renderable = cb.build()
        assert renderable is not None
