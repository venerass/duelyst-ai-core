"""Rich live display for real-time debate progress."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rich.live import Live
from rich.panel import Panel
from rich.status import Status

from duelyst_ai_core.agents.schemas import DebateMetadata, DebateResult
from duelyst_ai_core.cli.live_panel import RichDisplayCallback
from duelyst_ai_core.exceptions import ConfigError, ToolError
from duelyst_ai_core.orchestrator.engine import DebateOrchestrator
from duelyst_ai_core.orchestrator.state import ToolType
from duelyst_ai_core.tools.search import create_search_tool

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

    async def run_debate(
        self,
        config: DebateConfig,
        *,
        live: bool = True,
    ) -> DebateResult:
        """Run the debate with live progress display.

        Args:
            config: The debate configuration.
            live: If True, show a real-time Rich Live display.
                  If False, use a simple spinner (for markdown/json output).

        Returns:
            The complete DebateResult.
        """
        started_at = datetime.now(UTC)

        # Build tools list
        tools = []
        if ToolType.SEARCH in config.tools_enabled:
            try:
                tools.append(create_search_tool())
            except (ConfigError, ToolError) as exc:
                self._console.print(
                    "[yellow]Warning: Search tool requested but unavailable: "
                    f"{exc}. Continuing without search.[/yellow]"
                )

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

        if live:
            result = await self._run_live(config, tools, initial_state)
        else:
            result = await self._run_simple(config, tools, initial_state)

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

    async def _run_live(
        self,
        config: DebateConfig,
        tools: list[Any],
        initial_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the debate with Rich Live display driven by event callbacks."""
        callback = RichDisplayCallback()
        orchestrator = DebateOrchestrator(config, tools=tools or None, callback=callback)

        with Live(callback.build(), console=self._console, refresh_per_second=8) as live_display:

            async def _updating_callback(event: object) -> None:
                await callback.on_event(event)  # type: ignore[arg-type]
                live_display.update(callback.build())

            # Replace the orchestrator's callback with one that also refreshes Live
            orchestrator._callback = type(
                "_LiveUpdatingCallback",
                (),
                {"on_event": staticmethod(_updating_callback)},
            )()

            return await orchestrator.graph.ainvoke(initial_state)

    async def _run_simple(
        self,
        config: DebateConfig,
        tools: list[Any],
        initial_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the debate with a simple spinner (no live updating)."""
        orchestrator = DebateOrchestrator(config, tools=tools or None)

        with Status("[bold]Running debate...[/bold]", console=self._console, spinner="dots"):
            return await orchestrator.graph.ainvoke(initial_state)
