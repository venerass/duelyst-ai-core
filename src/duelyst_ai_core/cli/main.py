"""CLI entry point for duelyst-ai-core.

Usage:
    duelyst debate "Should startups use microservices or monoliths?"
    duelyst debate "Rust vs Go" --model-a claude-haiku --model-b gpt-mini
    duelyst debate "AI replacing jobs" --rounds 5 --output markdown
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer
from rich.console import Console

from duelyst_ai_core.cli.display import DebateDisplay
from duelyst_ai_core.exceptions import ConfigError, DuelystError
from duelyst_ai_core.formatters import JsonFormatter, MarkdownFormatter, RichTerminalFormatter
from duelyst_ai_core.models.registry import resolve_alias
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig, ToolType

app = typer.Typer(
    name="duelyst",
    help="Duelyst.ai — AI-powered debate engine",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


def _parse_tools(tools_str: str | None) -> list[ToolType]:
    """Parse comma-separated tool names into ToolType list."""
    if not tools_str:
        return []
    result = []
    for name in tools_str.split(","):
        name = name.strip().lower()
        try:
            result.append(ToolType(name))
        except ValueError:
            valid = ", ".join(t.value for t in ToolType)
            msg = f"Unknown tool '{name}'. Valid tools: {valid}"
            raise typer.BadParameter(msg) from None
    return result


def _build_config(
    topic: str,
    model_a: str,
    model_b: str,
    judge: str | None,
    instructions_a: str | None,
    instructions_b: str | None,
    rounds: int,
    convergence_threshold: int,
    convergence_rounds: int,
    tools_str: str | None,
) -> DebateConfig:
    """Build a DebateConfig from CLI arguments."""
    provider_a, model_id_a = resolve_alias(model_a)
    provider_b, model_id_b = resolve_alias(model_b)

    judge_config = None
    if judge:
        provider_j, model_id_j = resolve_alias(judge)
        judge_config = ModelConfig(provider=provider_j, model_id=model_id_j)

    return DebateConfig(
        topic=topic,
        model_a=ModelConfig(provider=provider_a, model_id=model_id_a),
        model_b=ModelConfig(provider=provider_b, model_id=model_id_b),
        judge_model=judge_config,
        instructions_a=instructions_a,
        instructions_b=instructions_b,
        max_rounds=rounds,
        convergence_threshold=convergence_threshold,
        convergence_rounds=convergence_rounds,
        tools_enabled=_parse_tools(tools_str),
    )


@app.command()
def debate(
    topic: Annotated[str, typer.Argument(help="The debate topic or question")],
    model_a: Annotated[
        str, typer.Option("--model-a", "-a", help="Model for side A")
    ] = "claude-haiku",
    model_b: Annotated[str, typer.Option("--model-b", "-b", help="Model for side B")] = "gpt-mini",
    judge: Annotated[str | None, typer.Option("--judge", "-j", help="Judge model")] = None,
    instructions_a: Annotated[
        str | None, typer.Option("--instructions-a", help="Instructions for A")
    ] = None,
    instructions_b: Annotated[
        str | None, typer.Option("--instructions-b", help="Instructions for B")
    ] = None,
    rounds: Annotated[int, typer.Option("--rounds", "-r", help="Maximum debate rounds")] = 5,
    convergence_threshold: Annotated[
        int, typer.Option("--threshold", help="Convergence threshold (1-10)")
    ] = 7,
    convergence_rounds: Annotated[
        int, typer.Option("--convergence-rounds", help="Consecutive converged rounds")
    ] = 2,
    tools: Annotated[str | None, typer.Option("--tools", "-t", help="Tools: search,code")] = None,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Format: rich, markdown, json")
    ] = "rich",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logging")] = False,
) -> None:
    """Run an AI debate on the given topic."""
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s %(levelname)s: %(message)s")

    try:
        config = _build_config(
            topic=topic,
            model_a=model_a,
            model_b=model_b,
            judge=judge,
            instructions_a=instructions_a,
            instructions_b=instructions_b,
            rounds=rounds,
            convergence_threshold=convergence_threshold,
            convergence_rounds=convergence_rounds,
            tools_str=tools,
        )
    except (ConfigError, typer.BadParameter) as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(code=1) from None

    display = DebateDisplay(console)
    display.show_config(config)

    try:
        result = asyncio.run(display.run_debate(config))
    except DuelystError as e:
        console.print(f"\n[red]Debate error:[/red] {e}")
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Debate interrupted.[/yellow]")
        raise typer.Exit(code=130) from None

    # Format output
    if output == "markdown":
        console.print(MarkdownFormatter().format(result))
    elif output == "json":
        console.print(JsonFormatter().format(result))
    else:
        console.print(RichTerminalFormatter().format(result))


if __name__ == "__main__":
    app()
