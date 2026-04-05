"""Agent I/O schemas — structured models for debater and judge outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from duelyst_ai_core.orchestrator.state import DebateConfig


class Evidence(BaseModel):
    """A piece of evidence cited by an agent during a debate turn.

    Args:
        claim: The factual claim being supported.
        source: URL, citation, or None if derived from reasoning.
        source_type: How the evidence was obtained.
    """

    model_config = ConfigDict(frozen=True)

    claim: str = Field(min_length=1)
    source: str | None = None
    source_type: Literal["web", "code", "reasoning"] = "reasoning"


class Reflection(BaseModel):
    """Internal reflection before formulating a response.

    The agent analyzes the opponent's last arguments to identify strengths,
    weaknesses, and plan a strategy for the current turn.

    Args:
        opponent_strong_points: Arguments from the opponent that are well-supported.
        opponent_weak_points: Arguments from the opponent that can be challenged.
        strategy: The agent's plan for this turn.
    """

    model_config = ConfigDict(frozen=True)

    opponent_strong_points: list[str] = Field(default_factory=list)
    opponent_weak_points: list[str] = Field(default_factory=list)
    strategy: str = Field(min_length=1)


class ToolCallRecord(BaseModel):
    """Record of a tool invocation made during an agent's turn.

    Args:
        tool_name: Name of the tool that was called.
        query: The input/query passed to the tool.
        result_summary: Brief summary of what the tool returned.
        success: Whether the tool call completed successfully.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    query: str
    result_summary: str
    success: bool = True


class AgentResponse(BaseModel):
    """Structured output from a debater agent turn.

    This is the primary output schema that the LLM must produce via structured
    output. It contains the argument, supporting evidence, and a convergence
    score indicating how much the agent agrees with its opponent.

    Args:
        argument: The main argument text for this turn.
        key_points: Bullet-point summary of the main claims.
        evidence: Supporting evidence for the argument.
        convergence_score: Agreement level with opponent
            (0=total disagreement, 10=full agreement).
        convergence_reasoning: Explanation of why this convergence score.
    """

    model_config = ConfigDict(frozen=True)

    argument: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    convergence_score: int = Field(ge=0, le=10)
    convergence_reasoning: str = Field(min_length=1)


class DebateTurn(BaseModel):
    """A single turn in the debate history.

    Args:
        agent: Which agent produced this turn.
        round_number: The debate round (1-indexed).
        response: The agent's structured response.
        reflection: The agent's internal reflection (None for first turn).
        tool_calls: Record of any tool invocations made during this turn.
        timestamp: When this turn was completed.
    """

    model_config = ConfigDict(frozen=True)

    agent: Literal["a", "b"]
    round_number: int = Field(ge=1)
    response: AgentResponse
    reflection: Reflection | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class JudgeSynthesis(BaseModel):
    """Structured output from the judge agent.

    The judge analyzes the complete debate transcript and produces an
    impartial synthesis covering both sides' positions.

    Args:
        summary_side_a: Summary of debater A's overall position.
        summary_side_b: Summary of debater B's overall position.
        key_evidence_a: Most significant evidence cited by debater A.
        key_evidence_b: Most significant evidence cited by debater B.
        points_of_agreement: Topics where both debaters reached consensus.
        points_of_disagreement: Topics where debaters remained opposed.
        conclusion: The judge's balanced conclusion.
        winner: Optional explicit winner determination.
    """

    model_config = ConfigDict(frozen=True)

    summary_side_a: str = Field(min_length=1)
    summary_side_b: str = Field(min_length=1)
    key_evidence_a: list[Evidence] = Field(default_factory=list)
    key_evidence_b: list[Evidence] = Field(default_factory=list)
    points_of_agreement: list[str] = Field(default_factory=list)
    points_of_disagreement: list[str] = Field(default_factory=list)
    conclusion: str = Field(min_length=1)
    winner: Literal["a", "b", "draw"] | None = None


class DebateMetadata(BaseModel):
    """Metadata about the debate execution.

    Args:
        started_at: When the debate began.
        finished_at: When the debate completed.
        duration_seconds: Total wall-clock time in seconds.
        total_tokens_used: Approximate total tokens consumed
            across all providers (None if unavailable).
    """

    model_config = ConfigDict(frozen=True)

    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0.0)
    total_tokens_used: int | None = Field(default=None, ge=0)


class DebateResult(BaseModel):
    """Complete result of a finished debate.

    This is the top-level output returned by the orchestrator after a debate
    completes. It contains the full configuration, transcript, synthesis,
    and metadata.

    Args:
        config: The debate configuration used.
        turns: Complete ordered list of debate turns.
        synthesis: The judge's synthesis of the debate.
        status: How the debate ended — converged or hit max rounds.
        total_rounds: Number of rounds completed.
        metadata: Timing and usage metadata.
    """

    model_config = ConfigDict(frozen=True)

    config: DebateConfig
    turns: list[DebateTurn]
    synthesis: JudgeSynthesis
    status: Literal["converged", "max_rounds"]
    total_rounds: int = Field(ge=1)
    metadata: DebateMetadata


def rebuild_debate_result_forward_refs(debate_config_type: type[object]) -> None:
    """Rebuild DebateResult once DebateConfig is available at runtime."""
    DebateResult.model_rebuild(_types_namespace={"DebateConfig": debate_config_type})
