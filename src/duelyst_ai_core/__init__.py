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

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from duelyst_ai_core._runtime import suppress_known_warnings

suppress_known_warnings()

__version__ = "0.1.0"

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import (
        AgentResponse,
        DebateResult,
        DebateTurn,
        Evidence,
        JudgeSynthesis,
    )
    from duelyst_ai_core.formatters import (
        JsonFormatter,
        MarkdownFormatter,
        RichTerminalFormatter,
    )
    from duelyst_ai_core.models.registry import create_model, get_judge_model, resolve_alias
    from duelyst_ai_core.orchestrator.callbacks import (
        CollectorCallback,
        DebateEventCallback,
    )
    from duelyst_ai_core.orchestrator.engine import DebateOrchestrator
    from duelyst_ai_core.orchestrator.events import (
        ConvergenceUpdate,
        DebateCompleted,
        DebateError,
        DebateEvent,
        DebateStarted,
        RoundStarted,
        SynthesisCompleted,
        TurnCompleted,
    )
    from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig

__all__ = [
    "AgentResponse",
    "CollectorCallback",
    "ConvergenceUpdate",
    "DebateCompleted",
    "DebateConfig",
    "DebateError",
    "DebateEvent",
    "DebateEventCallback",
    "DebateOrchestrator",
    "DebateResult",
    "DebateStarted",
    "DebateTurn",
    "Evidence",
    "JsonFormatter",
    "JudgeSynthesis",
    "MarkdownFormatter",
    "ModelConfig",
    "RichTerminalFormatter",
    "RoundStarted",
    "SynthesisCompleted",
    "TurnCompleted",
    "create_model",
    "get_judge_model",
    "resolve_alias",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentResponse": ("duelyst_ai_core.agents.schemas", "AgentResponse"),
    "CollectorCallback": ("duelyst_ai_core.orchestrator.callbacks", "CollectorCallback"),
    "ConvergenceUpdate": ("duelyst_ai_core.orchestrator.events", "ConvergenceUpdate"),
    "DebateCompleted": ("duelyst_ai_core.orchestrator.events", "DebateCompleted"),
    "DebateConfig": ("duelyst_ai_core.orchestrator.state", "DebateConfig"),
    "DebateError": ("duelyst_ai_core.orchestrator.events", "DebateError"),
    "DebateEvent": ("duelyst_ai_core.orchestrator.events", "DebateEvent"),
    "DebateEventCallback": ("duelyst_ai_core.orchestrator.callbacks", "DebateEventCallback"),
    "DebateOrchestrator": ("duelyst_ai_core.orchestrator.engine", "DebateOrchestrator"),
    "DebateResult": ("duelyst_ai_core.agents.schemas", "DebateResult"),
    "DebateStarted": ("duelyst_ai_core.orchestrator.events", "DebateStarted"),
    "DebateTurn": ("duelyst_ai_core.agents.schemas", "DebateTurn"),
    "Evidence": ("duelyst_ai_core.agents.schemas", "Evidence"),
    "JsonFormatter": ("duelyst_ai_core.formatters", "JsonFormatter"),
    "JudgeSynthesis": ("duelyst_ai_core.agents.schemas", "JudgeSynthesis"),
    "MarkdownFormatter": ("duelyst_ai_core.formatters", "MarkdownFormatter"),
    "ModelConfig": ("duelyst_ai_core.orchestrator.state", "ModelConfig"),
    "RichTerminalFormatter": ("duelyst_ai_core.formatters", "RichTerminalFormatter"),
    "RoundStarted": ("duelyst_ai_core.orchestrator.events", "RoundStarted"),
    "SynthesisCompleted": ("duelyst_ai_core.orchestrator.events", "SynthesisCompleted"),
    "TurnCompleted": ("duelyst_ai_core.orchestrator.events", "TurnCompleted"),
    "create_model": ("duelyst_ai_core.models.registry", "create_model"),
    "get_judge_model": ("duelyst_ai_core.models.registry", "get_judge_model"),
    "resolve_alias": ("duelyst_ai_core.models.registry", "resolve_alias"),
}


def __getattr__(name: str) -> Any:
    """Lazily load public exports to keep import-time side effects minimal."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the available public exports for interactive use."""
    return sorted(list(globals()) + __all__)
