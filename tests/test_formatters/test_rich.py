"""Tests for the Rich terminal formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from duelyst_ai_core.formatters.rich_terminal import RichTerminalFormatter

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import DebateResult


class TestRichTerminalFormatter:
    def test_produces_output(self, sample_debate_result: DebateResult) -> None:
        output = RichTerminalFormatter(width=120).format(sample_debate_result)
        assert len(output) > 0

    def test_contains_topic(self, sample_debate_result: DebateResult) -> None:
        output = RichTerminalFormatter(width=120).format(sample_debate_result)
        assert "microservices or monoliths" in output

    def test_contains_arguments(self, sample_debate_result: DebateResult) -> None:
        output = RichTerminalFormatter(width=120).format(sample_debate_result)
        assert "Monoliths reduce" in output
        assert "Microservices enable" in output

    def test_contains_conclusion(self, sample_debate_result: DebateResult) -> None:
        output = RichTerminalFormatter(width=120).format(sample_debate_result)
        assert "monolith-first" in output

    def test_contains_winner(self, sample_debate_result: DebateResult) -> None:
        output = RichTerminalFormatter(width=120).format(sample_debate_result)
        assert "Draw" in output
