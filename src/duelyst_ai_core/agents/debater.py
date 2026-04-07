"""Debater agent — ReAct agent for a single debate participant.

Each debater is a `create_agent` graph that:
1. Receives the debate context as messages
2. Optionally uses tools (web search, code execution) to gather evidence
3. Produces a structured AgentResponse with argument and convergence score

The agent decides autonomously when to use tools and when to respond.
Both agents in a debate share the same class, differing only in model
and configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from duelyst_ai_core.agents.prompts import DEBATER_SYSTEM_PROMPT, build_debater_system_prompt
from duelyst_ai_core.agents.schemas import AgentResponse

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class DebaterAgent:
    """A single debate participant as a LangGraph agent.

    Wraps `create_agent` with the debater system prompt and
    structured output (AgentResponse). The model decides when to use
    tools for research and when to formulate its final argument.

    Args:
        model: A LangChain chat model (already configured).
        tools: Optional tools for evidence gathering (search, code).
        agent_label: Identity label for the agent (e.g. "Debater A").
            When provided, the system prompt includes this identity.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[BaseTool] | None = None,
        agent_label: str | None = None,
    ) -> None:
        self._model = model
        self._tools = tools or []
        system_prompt = (
            build_debater_system_prompt(agent_label) if agent_label else DEBATER_SYSTEM_PROMPT
        )
        self.graph: CompiledStateGraph[Any, Any, Any] = create_agent(
            model,
            tools=self._tools,
            system_prompt=system_prompt,
            response_format=AgentResponse,
        )

        logger.info(
            "DebaterAgent created (tools: %d)",
            len(self._tools),
        )
