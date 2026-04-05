"""CLI entry point for duelyst-ai-core.

Usage:
    duelyst debate "Should startups use microservices or monoliths?"
    duelyst debate "Rust vs Go" --model-a claude-haiku --model-b gpt-mini
    duelyst debate "AI replacing jobs" --rounds 5 --output markdown
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from duelyst_ai_core.exceptions import ConfigError, DuelystError
from duelyst_ai_core.models.registry import resolve_alias
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig, ToolType

_ROOT_HELP = """Run structured AI debates between two models and synthesize the outcome.

Quick start:
1. Put provider keys in `.env` or export them in your shell.
2. Run `duelyst debate "Your topic"`.
3. Run `duelyst debate --help` for guided examples and option details.

Common model aliases: `claude-haiku`, `claude-sonnet`, `gpt-mini`, `gpt-5`, `gemini-flash`.
"""

_DEBATE_HELP = """Run a structured multi-round debate on TOPIC.

The CLI auto-loads `.env` from the current working directory before resolving providers.
You usually want API keys for at least two providers across Anthropic, OpenAI, and Google.

Examples:
- `duelyst debate "Biscoito ou bolacha?"`
- `duelyst debate "Rust vs Go for backend" --model-a claude-sonnet --model-b gpt-5`
- `duelyst debate "Bitcoin price prediction 2026" --tools search --output markdown`

Common aliases:
- Anthropic: `claude-haiku`, `claude-sonnet`, `claude-opus`
- OpenAI: `gpt-mini`, `gpt-5`, `gpt-nano`
- Google: `gemini-flash`, `gemini-flash-lite`, `gemini-pro`
"""


class OutputFormat(StrEnum):
    """Supported CLI output formats."""

    RICH = "rich"
    MARKDOWN = "markdown"
    JSON = "json"

app = typer.Typer(
    name="duelyst",
    help=_ROOT_HELP,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
)

console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def main() -> None:
    """Duelyst.ai — AI-powered debate engine."""


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


@app.command("debate", help=_DEBATE_HELP)
def debate(
    topic: Annotated[
        str,
        typer.Argument(help="Question or proposition to debate. Quote multi-word topics."),
    ],
    model_a: Annotated[
        str,
        typer.Option(
            "--model-a",
            "-a",
            metavar="ALIAS_OR_MODEL",
            help=(
                "Model for side A. Use a friendly alias such as claude-haiku, gpt-mini, "
                "or gemini-flash, or pass a full model ID starting with claude/gpt/gemini."
            ),
            rich_help_panel="Model Selection",
        ),
    ] = "claude-haiku",
    model_b: Annotated[
        str,
        typer.Option(
            "--model-b",
            "-b",
            metavar="ALIAS_OR_MODEL",
            help=(
                "Model for side B. Defaults to gpt-mini for cheap local smoke tests."
            ),
            rich_help_panel="Model Selection",
        ),
    ] = "gpt-mini",
    judge: Annotated[
        str | None,
        typer.Option(
            "--judge",
            "-j",
            metavar="ALIAS_OR_MODEL",
            help=(
                "Optional judge model. If omitted, Duelyst auto-picks a low-cost judge from "
                "a different provider when possible."
            ),
            rich_help_panel="Model Selection",
        ),
    ] = None,
    instructions_a: Annotated[
        str | None,
        typer.Option(
            "--instructions-a",
            help=(
                "Optional stance or extra guidance for side A. Example: Defend the monolith "
                "position even if it seems weaker."
            ),
            rich_help_panel="Debate Setup",
        ),
    ] = None,
    instructions_b: Annotated[
        str | None,
        typer.Option(
            "--instructions-b",
            help=(
                "Optional stance or extra guidance for side B. Use this when you want a "
                "specific framing instead of a generic opposing side."
            ),
            rich_help_panel="Debate Setup",
        ),
    ] = None,
    rounds: Annotated[
        int,
        typer.Option(
            "--rounds",
            "-r",
            help="Maximum rounds before forcing synthesis.",
            rich_help_panel="Debate Setup",
        ),
    ] = 5,
    convergence_threshold: Annotated[
        int,
        typer.Option(
            "--threshold",
            help=(
                "Minimum convergence score both agents must reach in the same round. "
                "Higher values make the debate push harder before stopping."
            ),
            rich_help_panel="Debate Setup",
        ),
    ] = 7,
    convergence_rounds: Annotated[
        int,
        typer.Option(
            "--convergence-rounds",
            help=(
                "How many consecutive converged rounds are required before the judge runs."
            ),
            rich_help_panel="Debate Setup",
        ),
    ] = 2,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            "-t",
            metavar="CSV",
            help=(
                "Comma-separated tools. Use search for Tavily-backed web search after "
                "installing duelyst-ai-core[search] and setting TAVILY_API_KEY. code is "
                "reserved for a future phase and is not available yet."
            ),
            rich_help_panel="Debate Setup",
        ),
    ] = None,
    output: Annotated[
        OutputFormat,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Final output format. rich is best for interactive reading, markdown for "
                "files, and json for automation or piping into other tools."
            ),
            rich_help_panel="Output & Diagnostics",
        ),
    ] = OutputFormat.RICH,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show debug logging from orchestration and provider setup.",
            rich_help_panel="Output & Diagnostics",
        ),
    ] = False,
) -> None:
    """Run an AI debate on the given topic."""
    load_dotenv(dotenv_path=".env", override=False)

    from duelyst_ai_core.cli.display import DebateDisplay
    from duelyst_ai_core.formatters import JsonFormatter, MarkdownFormatter, RichTerminalFormatter

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
    if output == OutputFormat.MARKDOWN:
        console.print(MarkdownFormatter().format(result))
    elif output == OutputFormat.JSON:
        console.print(JsonFormatter().format(result))
    else:
        console.print(RichTerminalFormatter().format(result))


if __name__ == "__main__":
    app()
