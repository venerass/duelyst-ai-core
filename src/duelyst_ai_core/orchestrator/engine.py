"""Debate orchestrator — outer LangGraph graph managing the full debate lifecycle.

Graph topology:
    START → init_debate → run_debater_a → run_debater_b → check_convergence
                          ↑                                       |
                          |_____ continue ________________________|
                                                                  |
                                                converged/max → run_judge → END
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from duelyst_ai_core.agents.debater import DebaterAgent
from duelyst_ai_core.agents.judge import JudgeAgent
from duelyst_ai_core.agents.prompts import (
    build_debater_user_message,
    build_judge_user_message,
    format_debate_history,
)
from duelyst_ai_core.agents.schemas import AgentResponse, DebateTurn, JudgeSynthesis
from duelyst_ai_core.models.registry import create_model, get_judge_model
from duelyst_ai_core.orchestrator.convergence import check_convergence

# Runtime import: LangGraph resolves node function annotations at runtime
from duelyst_ai_core.orchestrator.state import (
    DebateConfig,
    DebateStatus,
    OrchestratorState,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def _get_turn_argument(turn: dict[str, object]) -> str:
    """Safely extract the argument text from a serialized debate turn."""
    response = turn.get("response")
    if isinstance(response, dict):
        return str(response.get("argument", ""))
    return ""


class DebateOrchestrator:
    """Outer LangGraph graph that manages the full debate lifecycle.

    Creates debater and judge sub-agents and orchestrates the debate flow:
    alternating turns, convergence checking, and final synthesis.

    Args:
        config: The debate configuration.
        tools: Optional tools available to debater agents.
    """

    def __init__(
        self,
        config: DebateConfig,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self._config = config

        # Create models and sub-agents
        model_a = create_model(config.model_a)
        model_b = create_model(config.model_b)

        judge_config = config.judge_model or get_judge_model(config.model_a, config.model_b)
        judge_model = create_model(judge_config)

        self._debater_a = DebaterAgent(model=model_a, tools=tools)
        self._debater_b = DebaterAgent(model=model_b, tools=tools)
        self._judge = JudgeAgent(model=judge_model)

        self.graph = self.build_graph()

    def build_graph(self) -> CompiledStateGraph[Any, Any, Any]:
        """Build the orchestrator's StateGraph."""
        graph = StateGraph(OrchestratorState)

        graph.add_node("init_debate", self.init_debate)
        graph.add_node("run_debater_a", self.run_debater_a)
        graph.add_node("run_debater_b", self.run_debater_b)
        graph.add_node("check_convergence", self.check_convergence)
        graph.add_node("run_judge", self.run_judge)

        graph.add_edge(START, "init_debate")
        graph.add_edge("init_debate", "run_debater_a")
        graph.add_edge("run_debater_a", "run_debater_b")
        graph.add_edge("run_debater_b", "check_convergence")
        graph.add_conditional_edges(
            "check_convergence",
            self._route_after_convergence,
            {"continue": "run_debater_a", "judge": "run_judge"},
        )
        graph.add_edge("run_judge", END)

        return graph.compile()

    def visualize(self) -> bytes:
        """Render the orchestrator graph as PNG bytes.

        Uses the Mermaid.Ink API (requires network access).

        Returns:
            PNG image bytes.
        """
        return self.graph.get_graph().draw_mermaid_png()

    def visualize_ascii(self) -> str:
        """Render the orchestrator graph as ASCII art (no network needed)."""
        return self.graph.get_graph().draw_ascii()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    @staticmethod
    def init_debate(state: OrchestratorState) -> dict[str, Any]:
        """Initialize debate state for the first round."""
        logger.info("Debate started: %s", state["config"].topic[:80])
        return {
            "current_round": 1,
            "current_agent": "a",
            "convergence_history": [],
            "status": DebateStatus.RUNNING,
            "synthesis": None,
            "error": None,
        }

    async def run_debater_a(self, state: OrchestratorState) -> dict[str, Any]:
        """Invoke debater A's subgraph."""
        return await self._run_debater(state, "a", self._debater_a)

    async def run_debater_b(self, state: OrchestratorState) -> dict[str, Any]:
        """Invoke debater B's subgraph."""
        return await self._run_debater(state, "b", self._debater_b)

    async def _run_debater(
        self,
        state: OrchestratorState,
        role: Literal["a", "b"],
        debater: DebaterAgent,
    ) -> dict[str, Any]:
        """Build messages, invoke debater agent, extract structured response."""
        config: DebateConfig = state["config"]
        instructions = config.instructions_a if role == "a" else config.instructions_b
        side = "A" if role == "a" else "B"
        current_round: int = state["current_round"]

        logger.info("Running debater %s, round %d", role.upper(), current_round)

        # Build debate history for context
        history_dicts = [
            {
                "agent": str(t.get("agent", "")),
                "round_number": str(t.get("round_number", "")),
                "argument": _get_turn_argument(t),
            }
            for t in state["turns"]
        ]
        formatted_history = format_debate_history(history_dicts)

        agent_turns = [t for t in state["turns"] if t.get("agent") == role]
        is_first_turn = len(agent_turns) == 0

        user_content = build_debater_user_message(
            topic=config.topic,
            side=side,
            instructions=instructions,
            debate_history=formatted_history,
            round_number=current_round,
            is_first_turn=is_first_turn,
        )

        # Invoke the debater's ReAct agent with messages
        result = await debater.graph.ainvoke({"messages": [HumanMessage(content=user_content)]})

        # Extract structured response from create_agent output
        response = cast("AgentResponse", result["structured_response"])

        # Build DebateTurn
        turn = DebateTurn(
            agent=role,
            round_number=current_round,
            response=response,
            timestamp=datetime.now(UTC),
        )

        logger.info(
            "Debater %s done — convergence: %d",
            role.upper(),
            response.convergence_score,
        )

        return {"turns": [turn.model_dump(mode="python")]}

    def check_convergence(self, state: OrchestratorState) -> dict[str, Any]:
        """Check if debate has converged or hit max rounds."""
        config: DebateConfig = state["config"]
        current_round: int = state["current_round"]
        turns = state["turns"]

        # Extract convergence scores from current round
        round_turns = [t for t in turns if t.get("round_number") == current_round]

        score_a = 0
        score_b = 0
        for t in round_turns:
            response = t.get("response")
            if isinstance(response, dict):
                score = response.get("convergence_score", 0)
            else:
                score = getattr(response, "convergence_score", 0)
            score = int(score) if isinstance(score, (int, float)) else 0

            if t.get("agent") == "a":
                score_a = score
            elif t.get("agent") == "b":
                score_b = score

        history = [*state["convergence_history"], (score_a, score_b)]

        logger.info(
            "Round %d convergence — A: %d, B: %d",
            current_round,
            score_a,
            score_b,
        )

        is_converged = check_convergence(
            history=history,
            threshold=config.convergence_threshold,
            required_rounds=config.convergence_rounds,
        )

        if is_converged:
            logger.info("Debate converged after %d rounds", current_round)
            return {
                "convergence_history": history,
                "status": DebateStatus.CONVERGED,
            }

        if current_round >= config.max_rounds:
            logger.info("Max rounds (%d) reached", config.max_rounds)
            return {
                "convergence_history": history,
                "status": DebateStatus.MAX_ROUNDS,
            }

        return {
            "convergence_history": history,
            "current_round": current_round + 1,
        }

    async def run_judge(self, state: OrchestratorState) -> dict[str, Any]:
        """Invoke judge agent to produce synthesis."""
        logger.info("Running judge synthesis")

        # Format full transcript for the judge
        history_dicts = [
            {
                "agent": str(t.get("agent", "")),
                "round_number": str(t.get("round_number", "")),
                "argument": _get_turn_argument(t),
            }
            for t in state["turns"]
        ]
        transcript = format_debate_history(history_dicts)

        user_message = build_judge_user_message(
            topic=state["config"].topic,
            transcript=transcript,
            total_rounds=state["current_round"],
        )

        result = await self._judge.graph.ainvoke({"messages": [HumanMessage(content=user_message)]})

        synthesis = cast("JudgeSynthesis", result["structured_response"])

        logger.info(
            "Judge done — winner: %s",
            synthesis.winner or "not declared",
        )

        return {"synthesis": synthesis}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_convergence(
        state: OrchestratorState,
    ) -> Literal["continue", "judge"]:
        """Route to judge if converged/max_rounds, otherwise continue debating."""
        status = state["status"]
        if status in (DebateStatus.CONVERGED, DebateStatus.MAX_ROUNDS):
            return "judge"
        return "continue"
