"""Rich terminal formatter for debate results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from duelyst_ai_core.formatters.base import BaseFormatter

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import DebateResult, DebateTurn, JudgeSynthesis


# Agent colors for terminal output
_AGENT_STYLES = {"a": "cyan", "b": "magenta"}


class RichTerminalFormatter(BaseFormatter):
    """Formats a DebateResult using Rich for terminal display.

    Produces colored panels, tables, and styled text.
    """

    def __init__(self, width: int | None = None) -> None:
        self._console = Console(width=width, force_terminal=True)

    def format(self, result: DebateResult) -> str:
        """Format a debate result as Rich-rendered terminal output.

        Args:
            result: The complete debate result.

        Returns:
            ANSI-formatted string for terminal display.
        """
        with self._console.capture() as capture:
            self._render(result)
        return capture.get()

    def _render(self, result: DebateResult) -> None:
        """Render the full debate to the console."""
        # Header
        self._console.print()
        self._console.print(
            Panel(
                f"[bold]{result.config.topic}[/bold]\n\n"
                f"[dim]{result.config.model_a.model_id} (A) vs "
                f"{result.config.model_b.model_id} (B)[/dim]\n"
                f"[dim]Rounds: {result.total_rounds} | Status: {result.status}[/dim]",
                title="[bold]Debate[/bold]",
                border_style="blue",
            )
        )

        # Turns
        for turn in result.turns:
            self._render_turn(turn)

        # Synthesis
        self._render_synthesis(result.synthesis)

    def _render_turn(self, turn: DebateTurn) -> None:
        """Render a single debate turn as a panel."""
        style = _AGENT_STYLES.get(turn.agent, "white")
        label = f"Debater {turn.agent.upper()}"

        content = Text()
        content.append(turn.response.argument)
        content.append("\n\n")

        content.append(
            f"Convergence: {turn.response.convergence_score}/10",
            style="dim",
        )

        self._console.print(
            Panel(
                content,
                title=f"[bold {style}]{label}[/bold {style}] — Round {turn.round_number}",
                border_style=style,
            )
        )

    def _render_synthesis(self, synthesis: JudgeSynthesis) -> None:
        """Render the judge's synthesis."""
        # Summary table
        table = Table(title="Positions", show_header=True, header_style="bold")
        table.add_column("Side A", style="cyan")
        table.add_column("Side B", style="magenta")
        table.add_row(synthesis.summary_side_a, synthesis.summary_side_b)

        self._console.print()
        self._console.print(table)

        # Agreement / Disagreement
        if synthesis.points_of_agreement:
            self._console.print(
                Panel(
                    "\n".join(f"  [green]+[/green] {p}" for p in synthesis.points_of_agreement),
                    title="[bold green]Agreement[/bold green]",
                    border_style="green",
                )
            )

        if synthesis.points_of_disagreement:
            self._console.print(
                Panel(
                    "\n".join(f"  [red]-[/red] {p}" for p in synthesis.points_of_disagreement),
                    title="[bold red]Disagreement[/bold red]",
                    border_style="red",
                )
            )

        # Conclusion
        winner_text = ""
        if synthesis.winner:
            winner_label = (
                "Draw" if synthesis.winner == "draw" else f"Debater {synthesis.winner.upper()}"
            )
            winner_text = f"\n\n[bold]Winner:[/bold] {winner_label}"

        self._console.print(
            Panel(
                f"{synthesis.conclusion}{winner_text}",
                title="[bold yellow]Conclusion[/bold yellow]",
                border_style="yellow",
            )
        )
