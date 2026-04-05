"""Tests for agent I/O schemas — validation, serialization, edge cases."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from duelyst_ai_core.agents.schemas import (
    AgentResponse,
    DebateMetadata,
    DebateResult,
    DebateTurn,
    Evidence,
    JudgeSynthesis,
    Reflection,
    ToolCallRecord,
)
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig, ToolType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_config_anthropic() -> ModelConfig:
    return ModelConfig(provider="anthropic", model_id="claude-haiku-4-5")


@pytest.fixture
def model_config_openai() -> ModelConfig:
    return ModelConfig(provider="openai", model_id="gpt-5.4-mini")


@pytest.fixture
def debate_config(
    model_config_anthropic: ModelConfig, model_config_openai: ModelConfig
) -> DebateConfig:
    return DebateConfig(
        topic="Should startups use microservices or monoliths?",
        model_a=model_config_anthropic,
        model_b=model_config_openai,
    )


@pytest.fixture
def sample_evidence() -> Evidence:
    return Evidence(
        claim="Microservices increase deployment complexity",
        source="https://example.com/study",
        source_type="web",
    )


@pytest.fixture
def sample_response(sample_evidence: Evidence) -> AgentResponse:
    return AgentResponse(
        argument="Monoliths are better for startups because they reduce operational overhead.",
        key_points=["Simpler deployment", "Lower infrastructure cost"],
        evidence=[sample_evidence],
        convergence_score=3,
        convergence_reasoning="Opponent makes valid points about scalability "
        "but ignores startup constraints.",
    )


@pytest.fixture
def sample_turn(sample_response: AgentResponse) -> DebateTurn:
    return DebateTurn(
        agent="a",
        round_number=1,
        response=sample_response,
    )


@pytest.fixture
def sample_synthesis() -> JudgeSynthesis:
    return JudgeSynthesis(
        summary_side_a="Debater A argued for monoliths citing simplicity.",
        summary_side_b="Debater B argued for microservices citing scalability.",
        points_of_agreement=["Both agree early-stage speed matters"],
        points_of_disagreement=["Disagree on long-term maintenance cost"],
        conclusion="For most startups, starting with a monolith is pragmatic.",
        winner="draw",
    )


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_valid_creation(self, model_config_anthropic: ModelConfig) -> None:
        assert model_config_anthropic.provider == "anthropic"
        assert model_config_anthropic.model_id == "claude-haiku-4-5"
        assert model_config_anthropic.temperature == 0.7
        assert model_config_anthropic.max_tokens == 4096

    def test_custom_temperature(self) -> None:
        config = ModelConfig(provider="openai", model_id="gpt-5.4-mini", temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model_id="gpt-5.4-mini", temperature=-0.1)
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model_id="gpt-5.4-mini", temperature=2.1)

    def test_max_tokens_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model_id="gpt-5.4-mini", max_tokens=0)
        with pytest.raises(ValidationError):
            ModelConfig(provider="openai", model_id="gpt-5.4-mini", max_tokens=200_000)

    def test_invalid_provider(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(provider="mistral", model_id="mistral-large")

    def test_frozen(self, model_config_anthropic: ModelConfig) -> None:
        with pytest.raises(ValidationError):
            model_config_anthropic.temperature = 0.5

    def test_json_roundtrip(self, model_config_anthropic: ModelConfig) -> None:
        json_str = model_config_anthropic.model_dump_json()
        restored = ModelConfig.model_validate_json(json_str)
        assert restored == model_config_anthropic


# ---------------------------------------------------------------------------
# DebateConfig
# ---------------------------------------------------------------------------


class TestDebateConfig:
    def test_valid_creation(self, debate_config: DebateConfig) -> None:
        assert debate_config.topic == "Should startups use microservices or monoliths?"
        assert debate_config.max_rounds == 5
        assert debate_config.judge_model is None
        assert debate_config.tools_enabled == []

    def test_defaults(self, debate_config: DebateConfig) -> None:
        assert debate_config.convergence_threshold == 7
        assert debate_config.convergence_rounds == 2
        assert debate_config.instructions_a is None
        assert debate_config.instructions_b is None

    def test_empty_topic_rejected(
        self, model_config_anthropic: ModelConfig, model_config_openai: ModelConfig
    ) -> None:
        with pytest.raises(ValidationError):
            DebateConfig(
                topic="",
                model_a=model_config_anthropic,
                model_b=model_config_openai,
            )

    def test_max_rounds_bounds(
        self, model_config_anthropic: ModelConfig, model_config_openai: ModelConfig
    ) -> None:
        with pytest.raises(ValidationError):
            DebateConfig(
                topic="Test",
                model_a=model_config_anthropic,
                model_b=model_config_openai,
                max_rounds=0,
            )
        with pytest.raises(ValidationError):
            DebateConfig(
                topic="Test",
                model_a=model_config_anthropic,
                model_b=model_config_openai,
                max_rounds=21,
            )

    def test_with_tools(
        self, model_config_anthropic: ModelConfig, model_config_openai: ModelConfig
    ) -> None:
        config = DebateConfig(
            topic="Test",
            model_a=model_config_anthropic,
            model_b=model_config_openai,
            tools_enabled=[ToolType.SEARCH],
        )
        assert config.tools_enabled == [ToolType.SEARCH]

    def test_with_instructions(
        self, model_config_anthropic: ModelConfig, model_config_openai: ModelConfig
    ) -> None:
        config = DebateConfig(
            topic="Test",
            model_a=model_config_anthropic,
            model_b=model_config_openai,
            instructions_a="Defend microservices",
            instructions_b="Defend monoliths",
        )
        assert config.instructions_a == "Defend microservices"

    def test_frozen(self, debate_config: DebateConfig) -> None:
        with pytest.raises(ValidationError):
            debate_config.topic = "New topic"

    def test_json_roundtrip(self, debate_config: DebateConfig) -> None:
        json_str = debate_config.model_dump_json()
        restored = DebateConfig.model_validate_json(json_str)
        assert restored == debate_config


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_reasoning_default(self) -> None:
        e = Evidence(claim="Earth is round")
        assert e.source is None
        assert e.source_type == "reasoning"

    def test_web_evidence(self, sample_evidence: Evidence) -> None:
        assert sample_evidence.source_type == "web"
        assert sample_evidence.source is not None

    def test_empty_claim_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(claim="")

    def test_invalid_source_type(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(claim="Test", source_type="wikipedia")


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


class TestReflection:
    def test_valid(self) -> None:
        r = Reflection(
            opponent_strong_points=["Good data on scalability"],
            opponent_weak_points=["Ignores cost"],
            strategy="Focus on operational cost for small teams",
        )
        assert len(r.opponent_strong_points) == 1

    def test_empty_strategy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Reflection(strategy="")

    def test_empty_points_allowed(self) -> None:
        r = Reflection(strategy="Open with strong thesis")
        assert r.opponent_strong_points == []
        assert r.opponent_weak_points == []


# ---------------------------------------------------------------------------
# ToolCallRecord
# ---------------------------------------------------------------------------


class TestToolCallRecord:
    def test_valid(self) -> None:
        record = ToolCallRecord(
            tool_name="web_search",
            query="microservices vs monolith statistics 2026",
            result_summary="Found 3 relevant studies",
        )
        assert record.success is True

    def test_failed_call(self) -> None:
        record = ToolCallRecord(
            tool_name="web_search",
            query="test",
            result_summary="API timeout",
            success=False,
        )
        assert record.success is False


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


class TestAgentResponse:
    def test_valid(self, sample_response: AgentResponse) -> None:
        assert sample_response.convergence_score == 3
        assert len(sample_response.key_points) == 2

    def test_convergence_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AgentResponse(
                argument="Test",
                key_points=["Point"],
                convergence_score=-1,
                convergence_reasoning="Reason",
            )
        with pytest.raises(ValidationError):
            AgentResponse(
                argument="Test",
                key_points=["Point"],
                convergence_score=11,
                convergence_reasoning="Reason",
            )

    def test_empty_key_points_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentResponse(
                argument="Test",
                key_points=[],
                convergence_score=5,
                convergence_reasoning="Reason",
            )

    def test_empty_evidence_allowed(self) -> None:
        resp = AgentResponse(
            argument="Test",
            key_points=["Point"],
            convergence_score=5,
            convergence_reasoning="Reason",
        )
        assert resp.evidence == []


# ---------------------------------------------------------------------------
# DebateTurn
# ---------------------------------------------------------------------------


class TestDebateTurn:
    def test_valid(self, sample_turn: DebateTurn) -> None:
        assert sample_turn.agent == "a"
        assert sample_turn.round_number == 1
        assert sample_turn.reflection is None
        assert sample_turn.tool_calls == []

    def test_timestamp_auto_set(self, sample_turn: DebateTurn) -> None:
        assert sample_turn.timestamp is not None
        assert sample_turn.timestamp.tzinfo == UTC

    def test_invalid_round_number(self, sample_response: AgentResponse) -> None:
        with pytest.raises(ValidationError):
            DebateTurn(agent="a", round_number=0, response=sample_response)

    def test_invalid_agent(self, sample_response: AgentResponse) -> None:
        with pytest.raises(ValidationError):
            DebateTurn(agent="c", round_number=1, response=sample_response)

    def test_with_reflection(self, sample_response: AgentResponse) -> None:
        reflection = Reflection(
            opponent_strong_points=["Good point"],
            strategy="Counter with data",
        )
        turn = DebateTurn(
            agent="b",
            round_number=2,
            response=sample_response,
            reflection=reflection,
        )
        assert turn.reflection is not None

    def test_json_roundtrip(self, sample_turn: DebateTurn) -> None:
        json_str = sample_turn.model_dump_json()
        restored = DebateTurn.model_validate_json(json_str)
        assert restored.agent == sample_turn.agent
        assert restored.round_number == sample_turn.round_number


# ---------------------------------------------------------------------------
# JudgeSynthesis
# ---------------------------------------------------------------------------


class TestJudgeSynthesis:
    def test_valid(self, sample_synthesis: JudgeSynthesis) -> None:
        assert sample_synthesis.winner == "draw"
        assert len(sample_synthesis.points_of_agreement) == 1

    def test_winner_optional(self) -> None:
        synthesis = JudgeSynthesis(
            summary_side_a="A argued X",
            summary_side_b="B argued Y",
            conclusion="Both have merit",
        )
        assert synthesis.winner is None

    def test_invalid_winner(self) -> None:
        with pytest.raises(ValidationError):
            JudgeSynthesis(
                summary_side_a="A",
                summary_side_b="B",
                conclusion="C",
                winner="c",
            )


# ---------------------------------------------------------------------------
# DebateMetadata
# ---------------------------------------------------------------------------


class TestDebateMetadata:
    def test_valid(self) -> None:
        now = datetime.now(UTC)
        meta = DebateMetadata(
            started_at=now,
            finished_at=now,
            duration_seconds=120.5,
        )
        assert meta.total_tokens_used is None
        assert meta.duration_seconds == 120.5

    def test_negative_duration_rejected(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            DebateMetadata(
                started_at=now,
                finished_at=now,
                duration_seconds=-1.0,
            )

    def test_negative_tokens_rejected(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            DebateMetadata(
                started_at=now,
                finished_at=now,
                duration_seconds=10.0,
                total_tokens_used=-100,
            )


# ---------------------------------------------------------------------------
# DebateResult
# ---------------------------------------------------------------------------


class TestDebateResult:
    def test_valid(
        self,
        debate_config: DebateConfig,
        sample_turn: DebateTurn,
        sample_synthesis: JudgeSynthesis,
    ) -> None:
        now = datetime.now(UTC)
        result = DebateResult(
            config=debate_config,
            turns=[sample_turn],
            synthesis=sample_synthesis,
            status="converged",
            total_rounds=1,
            metadata=DebateMetadata(
                started_at=now,
                finished_at=now,
                duration_seconds=60.0,
            ),
        )
        assert result.status == "converged"
        assert result.total_rounds == 1

    def test_invalid_status(
        self,
        debate_config: DebateConfig,
        sample_turn: DebateTurn,
        sample_synthesis: JudgeSynthesis,
    ) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            DebateResult(
                config=debate_config,
                turns=[sample_turn],
                synthesis=sample_synthesis,
                status="cancelled",
                total_rounds=1,
                metadata=DebateMetadata(
                    started_at=now,
                    finished_at=now,
                    duration_seconds=60.0,
                ),
            )

    def test_json_roundtrip(
        self,
        debate_config: DebateConfig,
        sample_turn: DebateTurn,
        sample_synthesis: JudgeSynthesis,
    ) -> None:
        now = datetime.now(UTC)
        result = DebateResult(
            config=debate_config,
            turns=[sample_turn],
            synthesis=sample_synthesis,
            status="max_rounds",
            total_rounds=5,
            metadata=DebateMetadata(
                started_at=now,
                finished_at=now,
                duration_seconds=300.0,
                total_tokens_used=15000,
            ),
        )
        json_str = result.model_dump_json()
        restored = DebateResult.model_validate_json(json_str)
        assert restored.status == result.status
        assert restored.total_rounds == result.total_rounds
        assert restored.metadata.total_tokens_used == 15000
