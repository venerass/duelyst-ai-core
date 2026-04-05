"""Debate state schemas — configuration and orchestrator state.

The orchestrator state is the single source of truth during a debate.
Agent subgraphs (debater, judge) use MessagesState internally via
create_react_agent — no custom agent-level state needed.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

# Runtime import: LangGraph resolves TypedDict annotations via get_type_hints()
from duelyst_ai_core.agents.schemas import (
    JudgeSynthesis,
    rebuild_debate_result_forward_refs,
)


class ToolType(StrEnum):
    """Available tool types for debate agents."""

    SEARCH = "search"
    CODE = "code"  # Phase 4 — defined now for forward compatibility


class ModelConfig(BaseModel):
    """Configuration for a single LLM model.

    Args:
        provider: The model provider — one of anthropic, openai, google.
        model_id: Provider-specific model identifier (e.g. "claude-haiku-4-5").
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens in model response.
    """

    model_config = ConfigDict(frozen=True)

    provider: Literal["anthropic", "openai", "google"]
    model_id: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0, le=128_000)


class DebateConfig(BaseModel):
    """Immutable configuration for a debate session.

    Args:
        topic: The debate topic or question.
        model_a: Model configuration for debater A.
        model_b: Model configuration for debater B.
        judge_model: Model for synthesis. Auto-selected from a different provider if None.
        instructions_a: Optional custom instructions for debater A.
        instructions_b: Optional custom instructions for debater B.
        max_rounds: Maximum number of debate rounds before forced termination.
        convergence_threshold: Minimum convergence score (0-10) for both agents
            to consider converged.
        convergence_rounds: Number of consecutive rounds both agents must meet threshold.
        tools_enabled: List of tool types available to agents during the debate.
    """

    model_config = ConfigDict(frozen=True)

    topic: str = Field(min_length=1, max_length=2000)
    model_a: ModelConfig
    model_b: ModelConfig
    judge_model: ModelConfig | None = None
    instructions_a: str | None = Field(default=None, max_length=5000)
    instructions_b: str | None = Field(default=None, max_length=5000)
    max_rounds: int = Field(default=5, ge=1, le=20)
    convergence_threshold: int = Field(default=7, ge=1, le=10)
    convergence_rounds: int = Field(default=2, ge=1, le=5)
    tools_enabled: list[ToolType] = Field(default_factory=list)


class DebateStatus(StrEnum):
    """Current status of a debate."""

    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ROUNDS = "max_rounds"
    ERROR = "error"


rebuild_debate_result_forward_refs(DebateConfig)


# ---------------------------------------------------------------------------
# Orchestrator state (outer graph)
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict):
    """Top-level state for the debate orchestrator graph.

    The turns field uses operator.add so each node appends to the list
    rather than replacing it.
    """

    config: DebateConfig
    turns: Annotated[list[dict[str, object]], operator.add]
    current_round: int
    current_agent: Literal["a", "b"]
    convergence_history: list[tuple[int, int]]
    status: DebateStatus
    synthesis: JudgeSynthesis | None
    error: str | None
