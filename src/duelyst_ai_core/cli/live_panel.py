"""Rich live display components for real-time debate progress.

Provides ``RichDisplayCallback`` — a ``DebateEventCallback`` implementation
that drives a ``rich.live.Live`` display, updating in real-time as debate
events are emitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import RenderableType

    from duelyst_ai_core.orchestrator.events import DebateEvent

# Agent colors for terminal output
_AGENT_STYLES = {"a": "cyan", "b": "magenta"}


class RichDisplayCallback:
    """Callback that builds Rich renderables from debate events.

    Feed events via ``on_event()``, then call ``build()`` to get the
    current renderable for ``rich.live.Live``.

    Args:
        topic: The debate topic, used in the header.
    """

    def __init__(self) -> None:
        self._sections: list[RenderableType] = []
        self._spinner: RenderableType | None = None

    async def on_event(self, event: DebateEvent) -> None:
        """Process a debate event and update the display state.

        Args:
            event: The debate event to process.
        """
        # Import event types at call time to avoid circular imports at module level
        from duelyst_ai_core.orchestrator.events import (
            ConvergenceUpdate,
            DebateError,
            DebateStarted,
            RoundStarted,
            SynthesisCompleted,
            SynthesisStarted,
            TurnCompleted,
            TurnStarted,
        )

        if isinstance(event, DebateStarted):
            config = event.config
            self._sections.append(
                Panel(
                    f"[bold]{config.topic}[/bold]\n\n"
                    f"[cyan]Model A:[/cyan] "
                    f"{config.model_a.provider}/{config.model_a.model_id}\n"
                    f"[magenta]Model B:[/magenta] "
                    f"{config.model_b.provider}/{config.model_b.model_id}\n"
                    f"[dim]Max rounds: {config.max_rounds} | "
                    f"Convergence: {config.convergence_threshold}/10 "
                    f"for {config.convergence_rounds} rounds[/dim]",
                    title="[bold blue]Debate[/bold blue]",
                    border_style="blue",
                )
            )
        elif isinstance(event, RoundStarted):
            self._sections.append(Text(f"\n  Round {event.round_number}", style="bold yellow"))
        elif isinstance(event, TurnStarted):
            label = f"Agent {event.agent.upper()}"
            style = _AGENT_STYLES.get(event.agent, "white")
            self._spinner = Spinner("dots", text=f"[{style}]{label} is thinking...[/{style}]")
        elif isinstance(event, TurnCompleted):
            self._spinner = None
            style = _AGENT_STYLES.get(event.agent, "white")
            label = f"Debater {event.agent.upper()}"
            resp = event.response

            content = Text()
            content.append(resp.argument)
            content.append("\n")
            content.append(f"Convergence: {resp.convergence_score}/10", style="dim")

            self._sections.append(
                Panel(
                    content,
                    title=f"[bold {style}]{label}[/bold {style}]",
                    border_style=style,
                    padding=(0, 1),
                )
            )
        elif isinstance(event, ConvergenceUpdate):
            status = "[green]converged[/green]" if event.is_converged else "not converged"
            self._sections.append(
                Text(
                    f"  Convergence: A={event.score_a}, B={event.score_b} — {status}",
                    style="dim",
                )
            )
        elif isinstance(event, SynthesisStarted):
            self._spinner = Spinner("dots", text="[bold]Judge is synthesizing...[/bold]")
        elif isinstance(event, SynthesisCompleted):
            self._spinner = None
            synthesis = event.synthesis
            self._sections.append(
                Panel(
                    synthesis.conclusion,
                    title="[bold green]Synthesis[/bold green]",
                    border_style="green",
                )
            )
        elif isinstance(event, DebateError):
            self._spinner = None
            self._sections.append(
                Panel(
                    f"[red]{event.error_type}: {event.error_message}[/red]",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )

    def build(self) -> RenderableType:
        """Build the current renderable from accumulated sections.

        Returns:
            A Rich renderable suitable for ``Live.update()``.
        """
        parts: list[RenderableType] = list(self._sections)
        if self._spinner is not None:
            parts.append(self._spinner)
        return Group(*parts)
