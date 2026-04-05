"""Tests for the JSON formatter."""

from __future__ import annotations

import json

from duelyst_ai_core.agents.schemas import DebateResult
from duelyst_ai_core.formatters.json_fmt import JsonFormatter


class TestJsonFormatter:
    def test_valid_json(self, sample_debate_result: DebateResult) -> None:
        output = JsonFormatter().format(sample_debate_result)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_roundtrip(self, sample_debate_result: DebateResult) -> None:
        """JSON output should deserialize back to the same model."""
        output = JsonFormatter().format(sample_debate_result)
        restored = DebateResult.model_validate_json(output)
        assert restored.config.topic == sample_debate_result.config.topic
        assert len(restored.turns) == len(sample_debate_result.turns)
        assert restored.synthesis.winner == sample_debate_result.synthesis.winner

    def test_contains_all_fields(self, sample_debate_result: DebateResult) -> None:
        output = JsonFormatter().format(sample_debate_result)
        parsed = json.loads(output)
        assert "config" in parsed
        assert "turns" in parsed
        assert "synthesis" in parsed
        assert "status" in parsed
        assert "metadata" in parsed

    def test_pretty_printed(self, sample_debate_result: DebateResult) -> None:
        output = JsonFormatter().format(sample_debate_result)
        assert "\n" in output
        assert "  " in output
