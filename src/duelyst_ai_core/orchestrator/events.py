"""Streaming event types for real-time debate progress."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import AgentResponse, DebateResult, JudgeSynthesis
    from duelyst_ai_core.orchestrator.state import DebateConfig


class _BaseEvent(BaseModel):
    """Base class for all debate events."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DebateStarted(_BaseEvent):
    """Emitted when a debate begins."""

    event: Literal["debate_started"] = "debate_started"
    config: DebateConfig


class RoundStarted(_BaseEvent):
    """Emitted at the beginning of each debate round."""

    event: Literal["round_started"] = "round_started"
    round_number: int = Field(ge=1)


class TurnStarted(_BaseEvent):
    """Emitted when an agent begins its turn."""

    event: Literal["turn_started"] = "turn_started"
    agent: Literal["a", "b"]
    round_number: int = Field(ge=1)


class TurnCompleted(_BaseEvent):
    """Emitted when an agent finishes its turn."""

    event: Literal["turn_completed"] = "turn_completed"
    agent: Literal["a", "b"]
    round_number: int = Field(ge=1)
    response: AgentResponse


class ConvergenceUpdate(_BaseEvent):
    """Emitted after both agents complete a round with convergence scores."""

    event: Literal["convergence_update"] = "convergence_update"
    round_number: int = Field(ge=1)
    score_a: int = Field(ge=0, le=10)
    score_b: int = Field(ge=0, le=10)
    is_converged: bool


class SynthesisStarted(_BaseEvent):
    """Emitted when the judge begins synthesis."""

    event: Literal["synthesis_started"] = "synthesis_started"


class SynthesisCompleted(_BaseEvent):
    """Emitted when the judge finishes synthesis."""

    event: Literal["synthesis_completed"] = "synthesis_completed"
    synthesis: JudgeSynthesis


class DebateCompleted(_BaseEvent):
    """Emitted when the debate finishes successfully."""

    event: Literal["debate_completed"] = "debate_completed"
    result: DebateResult


class DebateError(_BaseEvent):
    """Emitted when the debate encounters an unrecoverable error."""

    event: Literal["debate_error"] = "debate_error"
    error_type: str
    error_message: str
    round_number: int | None = None
    agent: Literal["a", "b"] | None = None


DebateEvent = (
    DebateStarted
    | RoundStarted
    | TurnStarted
    | TurnCompleted
    | ConvergenceUpdate
    | SynthesisStarted
    | SynthesisCompleted
    | DebateCompleted
    | DebateError
)
"""Union type of all possible debate events for streaming."""


def _rebuild_forward_refs() -> None:
    """Rebuild forward references for event models."""
    from duelyst_ai_core.agents.schemas import (
        AgentResponse,
        DebateResult,
        JudgeSynthesis,
    )
    from duelyst_ai_core.orchestrator.state import DebateConfig

    ns = {
        "AgentResponse": AgentResponse,
        "DebateResult": DebateResult,
        "JudgeSynthesis": JudgeSynthesis,
        "DebateConfig": DebateConfig,
    }
    DebateStarted.model_rebuild(_types_namespace=ns)
    TurnCompleted.model_rebuild(_types_namespace=ns)
    SynthesisCompleted.model_rebuild(_types_namespace=ns)
    DebateCompleted.model_rebuild(_types_namespace=ns)


_rebuild_forward_refs()
