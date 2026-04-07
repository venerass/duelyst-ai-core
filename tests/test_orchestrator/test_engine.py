"""Tests for the DebateOrchestrator — mocked sub-agents, test graph routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duelyst_ai_core.agents.schemas import AgentResponse, JudgeSynthesis
from duelyst_ai_core.orchestrator.callbacks import CollectorCallback
from duelyst_ai_core.orchestrator.engine import DebateOrchestrator
from duelyst_ai_core.orchestrator.state import (
    DebateConfig,
    DebateStatus,
    ModelConfig,
    OrchestratorState,
)


@pytest.fixture
def debate_config() -> DebateConfig:
    return DebateConfig(
        topic="Should startups use microservices or monoliths?",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
        max_rounds=3,
        convergence_threshold=7,
        convergence_rounds=2,
    )


@pytest.fixture
def mock_response_a() -> AgentResponse:
    return AgentResponse(
        argument="Monoliths are simpler.",
        convergence_score=5,
        convergence_reasoning="Some agreement.",
    )


@pytest.fixture
def mock_response_b() -> AgentResponse:
    return AgentResponse(
        argument="Microservices scale better.",
        convergence_score=5,
        convergence_reasoning="Some agreement.",
    )


@pytest.fixture
def mock_synthesis() -> JudgeSynthesis:
    return JudgeSynthesis(
        summary_side_a="A argued for monoliths.",
        summary_side_b="B argued for microservices.",
        conclusion="Both have valid points.",
        winner="draw",
    )


def _make_debater_result(response: AgentResponse) -> dict[str, object]:
    """Simulate create_react_agent output."""
    return {"structured_response": response}


def _make_judge_result(synthesis: JudgeSynthesis) -> dict[str, object]:
    """Simulate judge create_react_agent output."""
    return {"structured_response": synthesis}


def _create_orchestrator(
    config: DebateConfig,
    callback: CollectorCallback | None = None,
) -> DebateOrchestrator:
    """Create an orchestrator with mocked models."""
    with (
        patch("duelyst_ai_core.orchestrator.engine.create_model") as mock_create,
        patch("duelyst_ai_core.orchestrator.engine.get_judge_model") as mock_judge,
    ):
        mock_create.return_value = MagicMock()
        mock_judge.return_value = ModelConfig(provider="google", model_id="gemini-2.5-flash")
        return DebateOrchestrator(config, callback=callback)


class TestOrchestratorInitDebate:
    async def test_sets_initial_state(self, debate_config: DebateConfig) -> None:
        state: OrchestratorState = {
            "config": debate_config,
            "turns": [],
            "current_round": 0,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }
        orchestrator = _create_orchestrator(debate_config)
        result = await orchestrator.init_debate(state)

        assert result["current_round"] == 1
        assert result["status"] == DebateStatus.RUNNING
        assert result["synthesis"] is None
        assert result["error"] is None


class TestOrchestratorCheckConvergence:
    async def test_not_converged(self, debate_config: DebateConfig) -> None:
        """When scores are below threshold, continues."""
        state: OrchestratorState = {
            "config": debate_config,
            "turns": [
                {
                    "agent": "a",
                    "round_number": 1,
                    "response": {"convergence_score": 3},
                },
                {
                    "agent": "b",
                    "round_number": 1,
                    "response": {"convergence_score": 4},
                },
            ],
            "current_round": 1,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }

        orchestrator = _create_orchestrator(debate_config)
        result = await orchestrator.check_convergence(state)

        assert result.get("current_round") == 2
        assert "status" not in result or result["status"] == DebateStatus.RUNNING

    async def test_max_rounds_reached(self, debate_config: DebateConfig) -> None:
        """When at max rounds, triggers judge."""
        state: OrchestratorState = {
            "config": debate_config,
            "turns": [
                {
                    "agent": "a",
                    "round_number": 3,
                    "response": {"convergence_score": 3},
                },
                {
                    "agent": "b",
                    "round_number": 3,
                    "response": {"convergence_score": 4},
                },
            ],
            "current_round": 3,
            "current_agent": "a",
            "convergence_history": [(3, 4), (4, 5)],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }

        orchestrator = _create_orchestrator(debate_config)
        result = await orchestrator.check_convergence(state)

        assert result["status"] == DebateStatus.MAX_ROUNDS


class TestOrchestratorRouting:
    def test_routes_to_judge_on_converged(self, debate_config: DebateConfig) -> None:
        state: OrchestratorState = {
            "config": debate_config,
            "turns": [],
            "current_round": 2,
            "current_agent": "a",
            "convergence_history": [(8, 8), (9, 9)],
            "status": DebateStatus.CONVERGED,
            "synthesis": None,
            "error": None,
        }
        assert DebateOrchestrator._route_after_convergence(state) == "judge"

    def test_routes_to_judge_on_max_rounds(self, debate_config: DebateConfig) -> None:
        state: OrchestratorState = {
            "config": debate_config,
            "turns": [],
            "current_round": 3,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.MAX_ROUNDS,
            "synthesis": None,
            "error": None,
        }
        assert DebateOrchestrator._route_after_convergence(state) == "judge"

    def test_routes_to_continue_when_running(self, debate_config: DebateConfig) -> None:
        state: OrchestratorState = {
            "config": debate_config,
            "turns": [],
            "current_round": 1,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }
        assert DebateOrchestrator._route_after_convergence(state) == "continue"


class TestOrchestratorFullGraph:
    async def test_single_round_max_rounds(
        self,
        mock_response_a: AgentResponse,
        mock_response_b: AgentResponse,
        mock_synthesis: JudgeSynthesis,
    ) -> None:
        """Test a complete debate with max_rounds=1."""
        config = DebateConfig(
            topic="Test topic",
            model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
            model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
            max_rounds=1,
        )

        orchestrator = _create_orchestrator(config)

        # Patch the sub-agent graphs
        orchestrator._debater_a.graph.ainvoke = AsyncMock(
            return_value=_make_debater_result(mock_response_a)
        )
        orchestrator._debater_b.graph.ainvoke = AsyncMock(
            return_value=_make_debater_result(mock_response_b)
        )
        orchestrator._judge.graph.ainvoke = AsyncMock(
            return_value=_make_judge_result(mock_synthesis)
        )

        initial_state = {
            "config": config,
            "turns": [],
            "current_round": 0,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }

        result = await orchestrator.graph.ainvoke(initial_state)

        # Should have 2 turns (one per debater)
        assert len(result["turns"]) == 2
        assert result["status"] == DebateStatus.MAX_ROUNDS
        assert result["synthesis"] is mock_synthesis

        # Sub-agents should have been called once each
        orchestrator._debater_a.graph.ainvoke.assert_awaited_once()
        orchestrator._debater_b.graph.ainvoke.assert_awaited_once()
        orchestrator._judge.graph.ainvoke.assert_awaited_once()


class TestOrchestratorEventEmission:
    async def test_full_graph_emits_events_in_order(
        self,
        mock_response_a: AgentResponse,
        mock_response_b: AgentResponse,
        mock_synthesis: JudgeSynthesis,
    ) -> None:
        """Run a single-round debate and verify event emission order."""
        config = DebateConfig(
            topic="Test topic",
            model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
            model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
            max_rounds=1,
        )

        collector = CollectorCallback()
        orchestrator = _create_orchestrator(config, callback=collector)

        orchestrator._debater_a.graph.ainvoke = AsyncMock(
            return_value=_make_debater_result(mock_response_a)
        )
        orchestrator._debater_b.graph.ainvoke = AsyncMock(
            return_value=_make_debater_result(mock_response_b)
        )
        orchestrator._judge.graph.ainvoke = AsyncMock(
            return_value=_make_judge_result(mock_synthesis)
        )

        initial_state = {
            "config": config,
            "turns": [],
            "current_round": 0,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }

        await orchestrator.graph.ainvoke(initial_state)

        event_types = [e.event for e in collector.events]
        assert event_types == [
            "debate_started",
            "round_started",  # round 1
            "turn_started",  # agent a
            "turn_completed",  # agent a
            "turn_started",  # agent b
            "turn_completed",  # agent b
            "convergence_update",
            "synthesis_started",
            "synthesis_completed",
        ]

    async def test_callback_error_does_not_crash_debate(
        self,
        mock_response_a: AgentResponse,
        mock_response_b: AgentResponse,
        mock_synthesis: JudgeSynthesis,
    ) -> None:
        """A buggy callback should not prevent the debate from completing."""

        class BuggyCallback:
            async def on_event(self, event: object) -> None:
                msg = "callback error"
                raise RuntimeError(msg)

        config = DebateConfig(
            topic="Test topic",
            model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
            model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
            max_rounds=1,
        )

        orchestrator = _create_orchestrator(config, callback=BuggyCallback())  # type: ignore[arg-type]

        orchestrator._debater_a.graph.ainvoke = AsyncMock(
            return_value=_make_debater_result(mock_response_a)
        )
        orchestrator._debater_b.graph.ainvoke = AsyncMock(
            return_value=_make_debater_result(mock_response_b)
        )
        orchestrator._judge.graph.ainvoke = AsyncMock(
            return_value=_make_judge_result(mock_synthesis)
        )

        initial_state = {
            "config": config,
            "turns": [],
            "current_round": 0,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }

        result = await orchestrator.graph.ainvoke(initial_state)

        # Debate should still complete despite buggy callback
        assert len(result["turns"]) == 2
        assert result["synthesis"] is mock_synthesis
