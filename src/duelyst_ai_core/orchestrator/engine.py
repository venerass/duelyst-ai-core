"""Debate orchestrator — outer LangGraph graph managing the full debate lifecycle.

Graph topology:
    START → init_debate → run_debater_a → run_debater_b → check_convergence
                          ↑                                       |
                          |_____ continue ________________________|
                                                                  |
                                                converged/max → run_judge → END
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
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
from duelyst_ai_core.orchestrator.callbacks import DebateEventCallback, NullCallback
from duelyst_ai_core.orchestrator.convergence import check_convergence
from duelyst_ai_core.orchestrator.events import (
    ConvergenceUpdate,
    DebateCompleted,
    DebateError,
    DebateEvent,
    DebateStarted,
    RoundStarted,
    SynthesisCompleted,
    SynthesisStarted,
    TurnCompleted,
    TurnStarted,
)

# Runtime import: LangGraph resolves node function annotations at runtime
from duelyst_ai_core.orchestrator.state import (
    DebateConfig,
    DebateStatus,
    OrchestratorState,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def _get_turn_argument(turn: dict[str, object]) -> str:
    """Safely extract the argument text from a serialized debate turn."""
    response = turn.get("response")
    if isinstance(response, dict):
        return str(response.get("argument", ""))
    return ""


def _get_turn_evidence(turn: dict[str, object]) -> list[dict[str, str | None]]:
    """Safely extract evidence list from a serialized debate turn."""
    response = turn.get("response")
    if isinstance(response, dict):
        evidence = response.get("evidence")
        if isinstance(evidence, list):
            return [
                {"claim": str(e.get("claim", "")), "source": e.get("source")}
                for e in evidence
                if isinstance(e, dict)
            ]
    return []


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
        callback: DebateEventCallback | None = None,
        langchain_callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        self._config = config
        self._callback: DebateEventCallback = callback or NullCallback()
        self._langchain_callbacks = langchain_callbacks or []

        # Create models and sub-agents
        model_a = create_model(config.model_a)
        model_b = create_model(config.model_b)

        judge_config = config.judge_model or get_judge_model(config.model_a, config.model_b)
        self._judge_config = judge_config
        judge_model = create_model(judge_config)

        logger.info(
            "Orchestrator init — A: %s/%s, B: %s/%s, Judge: %s/%s, topic: %s",
            config.model_a.provider,
            config.model_a.model_id,
            config.model_b.provider,
            config.model_b.model_id,
            judge_config.provider,
            judge_config.model_id,
            config.topic[:80],
        )

        self._debater_a = DebaterAgent(model=model_a, tools=tools, agent_label="Debater A")
        self._debater_b = DebaterAgent(model=model_b, tools=tools, agent_label="Debater B")
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

    async def init_debate(self, state: OrchestratorState) -> dict[str, Any]:
        """Initialize debate state for the first round."""
        logger.info("Debate started: %s", state["config"].topic[:80])
        await self._emit(DebateStarted(config=state["config"]))
        await self._emit(RoundStarted(round_number=1))
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
        history_dicts: list[dict[str, object]] = [
            {
                "agent": str(t.get("agent", "")),
                "round_number": str(t.get("round_number", "")),
                "argument": _get_turn_argument(t),
                "evidence": _get_turn_evidence(t),
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

        await self._emit(TurnStarted(agent=role, round_number=current_round))

        # Invoke the debater's ReAct agent with messages
        invoke_config: RunnableConfig | None = (
            RunnableConfig(callbacks=self._langchain_callbacks)
            if self._langchain_callbacks
            else None
        )
        result = await debater.graph.ainvoke(
            {"messages": [HumanMessage(content=user_content)]}, config=invoke_config
        )

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

        await self._emit(TurnCompleted(agent=role, round_number=current_round, response=response))

        return {"turns": [turn.model_dump(mode="python")]}

    async def check_convergence(self, state: OrchestratorState) -> dict[str, Any]:
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

        await self._emit(
            ConvergenceUpdate(
                round_number=current_round,
                score_a=score_a,
                score_b=score_b,
                is_converged=is_converged,
            )
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

        await self._emit(RoundStarted(round_number=current_round + 1))

        return {
            "convergence_history": history,
            "current_round": current_round + 1,
        }

    async def run_judge(self, state: OrchestratorState) -> dict[str, Any]:
        """Invoke judge agent to produce synthesis."""
        logger.info(
            "Running judge synthesis — model: %s/%s, rounds: %d, turns: %d",
            self._judge_config.provider,
            self._judge_config.model_id,
            state["current_round"],
            len(state["turns"]),
        )
        await self._emit(SynthesisStarted())

        # Format full transcript for the judge
        history_dicts: list[dict[str, object]] = [
            {
                "agent": str(t.get("agent", "")),
                "round_number": str(t.get("round_number", "")),
                "argument": _get_turn_argument(t),
                "evidence": _get_turn_evidence(t),
            }
            for t in state["turns"]
        ]
        transcript = format_debate_history(history_dicts)

        user_message = build_judge_user_message(
            topic=state["config"].topic,
            transcript=transcript,
            total_rounds=state["current_round"],
        )

        invoke_config: RunnableConfig | None = (
            RunnableConfig(callbacks=self._langchain_callbacks)
            if self._langchain_callbacks
            else None
        )
        result = await self._judge.graph.ainvoke(
            {"messages": [HumanMessage(content=user_message)]}, config=invoke_config
        )

        synthesis = cast("JudgeSynthesis", result["structured_response"])

        logger.info(
            "Judge done — winner: %s",
            synthesis.winner or "not declared",
        )

        await self._emit(SynthesisCompleted(synthesis=synthesis))

        return {"synthesis": synthesis}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def _emit(self, event: DebateEvent) -> None:
        """Emit an event to the callback, swallowing any callback errors."""
        try:
            await self._callback.on_event(event)
        except Exception:
            logger.exception("Callback error for %s", type(event).__name__)

    async def arun_with_events(self) -> AsyncGenerator[DebateEvent, None]:
        """Run the debate and yield events as they occur.

        This is the primary streaming interface for programmatic consumers.
        Internally uses an ``asyncio.Queue`` bridging event callbacks to the
        async generator.

        Yields:
            Debate events in emission order, ending with ``DebateCompleted``
            or ``DebateError``.
        """
        queue: asyncio.Queue[DebateEvent | None] = asyncio.Queue()

        class _QueueCallback:
            async def on_event(self, event: DebateEvent) -> None:
                await queue.put(event)

        # Swap in the queue-backed callback
        self._callback = _QueueCallback()

        initial_state: dict[str, object] = {
            "config": self._config,
            "turns": [],
            "current_round": 0,
            "current_agent": "a",
            "convergence_history": [],
            "status": "running",
            "synthesis": None,
            "error": None,
        }

        async def _run_graph() -> None:
            try:
                result = await self.graph.ainvoke(initial_state)
                from duelyst_ai_core.agents.schemas import DebateMetadata, DebateResult, DebateTurn

                turns = [
                    DebateTurn.model_validate(t) if isinstance(t, dict) else t
                    for t in result["turns"]
                ]
                debate_result = DebateResult(
                    config=self._config,
                    turns=turns,
                    synthesis=result["synthesis"],
                    status="converged" if result["status"] == "converged" else "max_rounds",
                    total_rounds=result["current_round"],
                    metadata=DebateMetadata(
                        started_at=datetime.now(UTC),
                        finished_at=datetime.now(UTC),
                        duration_seconds=0.0,
                    ),
                )
                await queue.put(DebateCompleted(result=debate_result))
            except Exception as exc:
                logger.exception(
                    "Debate graph failed — topic: %s, error: %s: %s",
                    self._config.topic[:80],
                    type(exc).__name__,
                    exc,
                )
                await queue.put(DebateError(error_type=type(exc).__name__, error_message=str(exc)))
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(_run_graph())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @staticmethod
    def _route_after_convergence(
        state: OrchestratorState,
    ) -> Literal["continue", "judge"]:
        """Route to judge if converged/max_rounds, otherwise continue debating."""
        status = state["status"]
        if status in (DebateStatus.CONVERGED, DebateStatus.MAX_ROUNDS):
            return "judge"
        return "continue"
