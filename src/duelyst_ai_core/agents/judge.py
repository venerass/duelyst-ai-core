"""Judge agent — produces a balanced synthesis of the complete debate.

The judge is a `create_agent` graph (no tools) that receives the
full debate transcript and generates a structured JudgeSynthesis.
It always uses a different model than both debaters to avoid bias.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from duelyst_ai_core.agents.prompts import JUDGE_SYSTEM_PROMPT
from duelyst_ai_core.agents.schemas import JudgeSynthesis

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class JudgeAgent:
    """Judge agent that synthesizes debate results.

    Wraps `create_agent` with the judge system prompt and
    structured output (JudgeSynthesis). No tools — pure analysis.

    Args:
        model: A LangChain chat model (must differ from debaters).
    """

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model
        self.graph: CompiledStateGraph[Any, Any, Any] = create_agent(
            model,
            tools=[],
            system_prompt=JUDGE_SYSTEM_PROMPT,
            response_format=JudgeSynthesis,
        )

        logger.info("JudgeAgent created")
