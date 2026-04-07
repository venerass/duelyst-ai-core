"""Markdown formatter for debate results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from duelyst_ai_core.formatters.base import BaseFormatter

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import DebateResult, DebateTurn, JudgeSynthesis


class MarkdownFormatter(BaseFormatter):
    """Formats a DebateResult as clean Markdown.

    Produces a document with headers, quoted arguments, evidence lists,
    and the judge's synthesis.
    """

    def format(self, result: DebateResult) -> str:
        """Format a debate result as Markdown.

        Args:
            result: The complete debate result.

        Returns:
            Markdown string.
        """
        parts: list[str] = []

        # Title and metadata
        parts.append(f"# Debate: {result.config.topic}\n")
        parts.append(
            f"**Models:** {result.config.model_a.model_id} (A) "
            f"vs {result.config.model_b.model_id} (B)  "
        )
        parts.append(f"**Rounds:** {result.total_rounds} | **Status:** {result.status}\n")

        # Turns
        parts.append("---\n")
        parts.append("## Debate Transcript\n")

        current_round = 0
        for turn in result.turns:
            if turn.round_number != current_round:
                current_round = turn.round_number
                parts.append(f"### Round {current_round}\n")
            parts.append(self._format_turn(turn))

        # Synthesis
        parts.append("---\n")
        parts.append("## Judge's Synthesis\n")
        parts.append(self._format_synthesis(result.synthesis))

        return "\n".join(parts)

    @staticmethod
    def _format_turn(turn: DebateTurn) -> str:
        """Format a single debate turn."""
        label = f"Debater {turn.agent.upper()}"
        lines = [f"**{label}:**\n"]
        lines.append(f"{turn.response.argument}\n")

        if turn.response.evidence:
            lines.append("**Evidence:**")
            for ev in turn.response.evidence:
                if ev.source:
                    source = f" ([{ev.source_type}]({ev.source}))"
                else:
                    source = f" ({ev.source_type})"
                lines.append(f"- {ev.claim}{source}")
            lines.append("")

        lines.append(
            f"*Convergence: {turn.response.convergence_score}/10 "
            f"— {turn.response.convergence_reasoning}*\n"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_synthesis(synthesis: JudgeSynthesis) -> str:
        """Format the judge's synthesis."""
        lines: list[str] = []

        lines.append(f"**Side A:** {synthesis.summary_side_a}\n")
        lines.append(f"**Side B:** {synthesis.summary_side_b}\n")

        if synthesis.points_of_agreement:
            lines.append("**Points of Agreement:**")
            for point in synthesis.points_of_agreement:
                lines.append(f"- {point}")
            lines.append("")

        if synthesis.points_of_disagreement:
            lines.append("**Points of Disagreement:**")
            for point in synthesis.points_of_disagreement:
                lines.append(f"- {point}")
            lines.append("")

        lines.append(f"**Conclusion:** {synthesis.conclusion}\n")

        if synthesis.winner:
            winner_label = (
                "Draw" if synthesis.winner == "draw" else f"Debater {synthesis.winner.upper()}"
            )
            lines.append(f"**Winner:** {winner_label}\n")

        return "\n".join(lines)
