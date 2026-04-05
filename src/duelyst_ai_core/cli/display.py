"""Rich live display for real-time debate progress."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.status import Status

from duelyst_ai_core.agents.schemas import DebateMetadata, DebateResult
from duelyst_ai_core.orchestrator.engine import DebateOrchestrator
from duelyst_ai_core.orchestrator.state import ToolType
from duelyst_ai_core.tools.search import create_search_tool, is_search_available

if TYPE_CHECKING:
    from rich.console import Console

    from duelyst_ai_core.orchestrator.state import DebateConfig


class DebateDisplay:
    """Manages Rich terminal output during a debate.

    Shows the debate configuration, runs the orchestrator, and
    displays progress with spinners.
    """

    def __init__(self, console: Console) -> None:
        self._console = console

    def show_config(self, config: DebateConfig) -> None:
        """Display the debate configuration before starting."""
        self._console.print()
        self._console.print(
            Panel(
                f"[bold]{config.topic}[/bold]\n\n"
                f"[cyan]Model A:[/cyan] {config.model_a.provider}/{config.model_a.model_id}\n"
                f"[magenta]Model B:[/magenta] {config.model_b.provider}/{config.model_b.model_id}\n"
                f"[dim]Max rounds: {config.max_rounds} | "
                f"Convergence: {config.convergence_threshold}/10 "
                f"for {config.convergence_rounds} rounds[/dim]",
                title="[bold blue]Debate Configuration[/bold blue]",
                border_style="blue",
            )
        )

    async def run_debate(self, config: DebateConfig) -> DebateResult:
        """Run the debate with live progress display.

        Args:
            config: The debate configuration.

        Returns:
            The complete DebateResult.
        """
        started_at = datetime.now(UTC)

        # Build tools list
        tools = []
        if ToolType.SEARCH in config.tools_enabled:
            if is_search_available():
                tools.append(create_search_tool())
            else:
                self._console.print(
                    "[yellow]Warning: Search tool requested but "
                    "TAVILY_API_KEY not set. Continuing without "
                    "search.[/yellow]"
                )

        # Create orchestrator
        orchestrator = DebateOrchestrator(config, tools=tools or None)

        # Run with status spinner
        with Status("[bold]Running debate...[/bold]", console=self._console, spinner="dots"):
            initial_state = {
                "config": config,
                "turns": [],
                "current_round": 0,
                "current_agent": "a",
                "convergence_history": [],
                "status": "running",
                "synthesis": None,
                "error": None,
            }

            result = await orchestrator.graph.ainvoke(initial_state)

        finished_at = datetime.now(UTC)
        duration = (finished_at - started_at).total_seconds()

        self._console.print(
            f"\n[green]Debate completed![/green] "
            f"Status: {result['status']} | "
            f"Rounds: {result['current_round']} | "
            f"Duration: {duration:.1f}s"
        )

        # Build DebateResult from graph output
        from duelyst_ai_core.agents.schemas import DebateTurn

        turns = [
            DebateTurn.model_validate(t) if isinstance(t, dict) else t for t in result["turns"]
        ]

        metadata = DebateMetadata(
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
        )

        return DebateResult(
            config=config,
            turns=turns,
            synthesis=result["synthesis"],
            status="converged" if result["status"] == "converged" else "max_rounds",
            total_rounds=result["current_round"],
            metadata=metadata,
        )
