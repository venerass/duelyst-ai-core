"""CLI entry point for duelyst-ai-core."""

import typer

app = typer.Typer(
    name="duelyst",
    help="Duelyst.ai — AI-powered debate engine",
    no_args_is_help=True,
)


@app.command()
def debate(
    topic: str = typer.Argument(..., help="The debate topic"),
) -> None:
    """Run an AI debate on the given topic."""
    typer.echo(f"Debate engine coming soon. Topic: {topic}")


if __name__ == "__main__":
    app()
