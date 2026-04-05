"""Tests for the Markdown formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from duelyst_ai_core.formatters.markdown import MarkdownFormatter

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import DebateResult


class TestMarkdownFormatter:
    def test_contains_topic(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "microservices or monoliths" in output

    def test_contains_model_ids(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "claude-haiku-4-5" in output
        assert "gpt-5.4-mini" in output

    def test_contains_round_headers(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "### Round 1" in output
        assert "### Round 2" in output

    def test_contains_arguments(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "Monoliths reduce operational complexity" in output
        assert "Microservices enable horizontal scaling" in output

    def test_contains_key_points(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "Simpler deployment" in output
        assert "Horizontal scaling" in output

    def test_contains_evidence(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "Netflix" in output

    def test_contains_convergence_scores(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "3/10" in output
        assert "7/10" in output

    def test_contains_synthesis(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "Judge's Synthesis" in output
        assert "monolith-first approach" in output

    def test_contains_winner(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "Draw" in output

    def test_contains_agreement_points(self, sample_debate_result: DebateResult) -> None:
        output = MarkdownFormatter().format(sample_debate_result)
        assert "Start with a monolith" in output

    def test_is_valid_markdown(self, sample_debate_result: DebateResult) -> None:
        """Output should start with a top-level heading."""
        output = MarkdownFormatter().format(sample_debate_result)
        assert output.startswith("# Debate:")
