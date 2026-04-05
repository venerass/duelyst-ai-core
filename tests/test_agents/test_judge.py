"""Tests for the JudgeAgent — create_react_agent wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duelyst_ai_core.agents.judge import JudgeAgent
from duelyst_ai_core.agents.schemas import Evidence, JudgeSynthesis


@pytest.fixture
def mock_synthesis() -> JudgeSynthesis:
    return JudgeSynthesis(
        summary_side_a="Side A argued for monoliths based on simplicity.",
        summary_side_b="Side B argued for microservices based on scalability.",
        key_evidence_a=[Evidence(claim="Simplicity reduces bugs", source_type="reasoning")],
        key_evidence_b=[
            Evidence(
                claim="Netflix scaled with microservices",
                source="https://example.com",
                source_type="web",
            )
        ],
        points_of_agreement=["Both agree that architecture depends on team size"],
        points_of_disagreement=["Disagree on long-term maintenance cost"],
        conclusion="For most startups, monoliths are pragmatic initially.",
        winner="draw",
    )


class TestJudgeAgentInit:
    def test_creates_graph(self) -> None:
        """JudgeAgent wraps create_react_agent into self.graph."""
        with patch("duelyst_ai_core.agents.judge.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            model = MagicMock()
            agent = JudgeAgent(model=model)

            mock_create.assert_called_once()
            assert agent.graph is mock_create.return_value

    def test_no_tools(self) -> None:
        """Judge passes empty tools list."""
        with patch("duelyst_ai_core.agents.judge.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            model = MagicMock()
            JudgeAgent(model=model)

            call_kwargs = mock_create.call_args
            tools_arg = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
            assert tools_arg == []

    def test_uses_response_format(self) -> None:
        """create_react_agent is called with response_format=JudgeSynthesis."""
        with patch("duelyst_ai_core.agents.judge.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            model = MagicMock()
            JudgeAgent(model=model)

            call_kwargs = mock_create.call_args
            rf = call_kwargs.kwargs.get("response_format") or call_kwargs[1].get("response_format")
            assert rf is JudgeSynthesis


class TestJudgeGraphInvocation:
    async def test_invoke_returns_structured_response(self, mock_synthesis: JudgeSynthesis) -> None:
        """Graph invocation returns structured_response key."""
        with patch("duelyst_ai_core.agents.judge.create_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={"structured_response": mock_synthesis})
            mock_create.return_value = mock_graph
            agent = JudgeAgent(model=MagicMock())

        result = await agent.graph.ainvoke({"messages": [{"role": "user", "content": "test"}]})

        assert result["structured_response"] is mock_synthesis
        assert result["structured_response"].winner == "draw"

    async def test_winner_preserved(self) -> None:
        """Judge returns whatever winner the model declares."""
        synthesis = JudgeSynthesis(
            summary_side_a="A summary",
            summary_side_b="B summary",
            conclusion="A wins clearly",
            winner="a",
        )
        with patch("duelyst_ai_core.agents.judge.create_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={"structured_response": synthesis})
            mock_create.return_value = mock_graph
            agent = JudgeAgent(model=MagicMock())

        result = await agent.graph.ainvoke({"messages": [{"role": "user", "content": "test"}]})

        assert result["structured_response"].winner == "a"
