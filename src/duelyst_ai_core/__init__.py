"""Duelyst.ai Core — Open-source AI debate engine.

Run multi-model AI debates with LangGraph orchestration.

Basic usage::

    import asyncio
    from duelyst_ai_core import DebateConfig, DebateOrchestrator, ModelConfig

    config = DebateConfig(
        topic="Should startups use microservices or monoliths?",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
    )

    orchestrator = DebateOrchestrator(config)
    result = asyncio.run(orchestrator.graph.ainvoke({
        "config": config,
        "turns": [],
        "current_round": 0,
        "current_agent": "a",
        "convergence_history": [],
        "status": "running",
        "synthesis": None,
        "error": None,
    }))
"""

__version__ = "0.1.0"

from duelyst_ai_core.agents.schemas import (
    AgentResponse,
    DebateResult,
    DebateTurn,
    Evidence,
    JudgeSynthesis,
)
from duelyst_ai_core.formatters import JsonFormatter, MarkdownFormatter, RichTerminalFormatter
from duelyst_ai_core.models.registry import create_model, get_judge_model, resolve_alias
from duelyst_ai_core.orchestrator.engine import DebateOrchestrator
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig

__all__ = [
    "AgentResponse",
    "DebateConfig",
    "DebateOrchestrator",
    "DebateResult",
    "DebateTurn",
    "Evidence",
    "JsonFormatter",
    "JudgeSynthesis",
    "MarkdownFormatter",
    "ModelConfig",
    "RichTerminalFormatter",
    "create_model",
    "get_judge_model",
    "resolve_alias",
]
