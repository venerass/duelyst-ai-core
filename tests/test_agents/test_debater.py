"""Tests for the DebaterAgent — create_react_agent wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duelyst_ai_core.agents.debater import DebaterAgent
from duelyst_ai_core.agents.schemas import AgentResponse


@pytest.fixture
def mock_response() -> AgentResponse:
    return AgentResponse(
        argument="Monoliths reduce operational complexity for small teams.",
        key_points=["Simpler deployment", "Lower cost"],
        convergence_score=3,
        convergence_reasoning="Opponent has valid scalability points but ignores startup context.",
    )


class TestDebaterAgentInit:
    def test_creates_graph(self) -> None:
        """DebaterAgent wraps create_react_agent into self.graph."""
        with patch("duelyst_ai_core.agents.debater.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            model = MagicMock()
            agent = DebaterAgent(model=model)

            mock_create.assert_called_once()
            assert agent.graph is mock_create.return_value

    def test_passes_tools(self) -> None:
        """Tools are forwarded to create_react_agent."""
        with patch("duelyst_ai_core.agents.debater.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            model = MagicMock()
            tools = [MagicMock(), MagicMock()]
            DebaterAgent(model=model, tools=tools)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs.get("tools") == tools or call_kwargs[1].get("tools") == tools

    def test_no_tools_defaults_to_empty(self) -> None:
        """When no tools provided, passes empty list."""
        with patch("duelyst_ai_core.agents.debater.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            model = MagicMock()
            DebaterAgent(model=model)

            call_kwargs = mock_create.call_args
            tools_arg = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
            assert tools_arg == []

    def test_uses_response_format(self) -> None:
        """create_react_agent is called with response_format=AgentResponse."""
        with patch("duelyst_ai_core.agents.debater.create_agent") as mock_create:
            mock_create.return_value = MagicMock()
            model = MagicMock()
            DebaterAgent(model=model)

            call_kwargs = mock_create.call_args
            rf = call_kwargs.kwargs.get("response_format") or call_kwargs[1].get("response_format")
            assert rf is AgentResponse


class TestDebaterGraphInvocation:
    async def test_invoke_returns_structured_response(self, mock_response: AgentResponse) -> None:
        """Graph invocation returns structured_response key."""
        with patch("duelyst_ai_core.agents.debater.create_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={"structured_response": mock_response})
            mock_create.return_value = mock_graph
            agent = DebaterAgent(model=MagicMock())

        result = await agent.graph.ainvoke({"messages": [{"role": "user", "content": "test"}]})

        assert result["structured_response"] is mock_response
        assert result["structured_response"].convergence_score == 3

    async def test_convergence_score_preserved(self) -> None:
        """The convergence score from the model is preserved."""
        response = AgentResponse(
            argument="We agree on this.",
            key_points=["Agreement"],
            convergence_score=9,
            convergence_reasoning="Near full agreement.",
        )
        with patch("duelyst_ai_core.agents.debater.create_agent") as mock_create:
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={"structured_response": response})
            mock_create.return_value = mock_graph
            agent = DebaterAgent(model=MagicMock())

        result = await agent.graph.ainvoke({"messages": [{"role": "user", "content": "test"}]})

        assert result["structured_response"].convergence_score == 9
