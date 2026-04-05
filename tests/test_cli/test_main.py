"""Tests for the CLI."""

from __future__ import annotations

import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import duelyst_ai_core.cli.main as cli_main
from duelyst_ai_core.cli.display import DebateDisplay
from duelyst_ai_core.cli.main import _build_config, _parse_tools, app
from duelyst_ai_core.exceptions import ConfigError
from duelyst_ai_core.formatters import RichTerminalFormatter
from duelyst_ai_core.orchestrator.state import ToolType

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestParseTools:
    def test_none(self) -> None:
        assert _parse_tools(None) == []

    def test_empty_string(self) -> None:
        assert _parse_tools("") == []

    def test_single_tool(self) -> None:
        assert _parse_tools("search") == [ToolType.SEARCH]

    def test_multiple_tools(self) -> None:
        result = _parse_tools("search,code")
        assert ToolType.SEARCH in result
        assert ToolType.CODE in result

    def test_whitespace_handling(self) -> None:
        result = _parse_tools(" search , code ")
        assert ToolType.SEARCH in result
        assert ToolType.CODE in result

    def test_invalid_tool(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter, match="Unknown tool"):
            _parse_tools("invalid")


class TestBuildConfig:
    def test_basic_config(self) -> None:
        config = _build_config(
            topic="Test topic",
            model_a="claude-haiku",
            model_b="gpt-mini",
            judge=None,
            instructions_a=None,
            instructions_b=None,
            rounds=3,
            convergence_threshold=7,
            convergence_rounds=2,
            tools_str=None,
        )
        assert config.topic == "Test topic"
        assert config.model_a.provider == "anthropic"
        assert config.model_b.provider == "openai"
        assert config.model_a.model_id == "claude-haiku-4-5"
        assert config.model_b.model_id == "gpt-5.4-mini"
        assert config.max_rounds == 3
        assert config.judge_model is None

    def test_with_judge(self) -> None:
        config = _build_config(
            topic="Test",
            model_a="claude-haiku",
            model_b="gpt-mini",
            judge="gemini-flash",
            instructions_a=None,
            instructions_b=None,
            rounds=5,
            convergence_threshold=7,
            convergence_rounds=2,
            tools_str=None,
        )
        assert config.judge_model is not None
        assert config.judge_model.provider == "google"
        assert config.judge_model.model_id == "gemini-2.5-flash"

    def test_with_instructions(self) -> None:
        config = _build_config(
            topic="Test",
            model_a="claude-haiku",
            model_b="gpt-mini",
            judge=None,
            instructions_a="Defend position X",
            instructions_b="Defend position Y",
            rounds=5,
            convergence_threshold=7,
            convergence_rounds=2,
            tools_str=None,
        )
        assert config.instructions_a == "Defend position X"
        assert config.instructions_b == "Defend position Y"

    def test_invalid_model(self) -> None:
        with pytest.raises(ConfigError):
            _build_config(
                topic="Test",
                model_a="invalid-model",
                model_b="gpt-mini",
                judge=None,
                instructions_a=None,
                instructions_b=None,
                rounds=5,
                convergence_threshold=7,
                convergence_rounds=2,
                tools_str=None,
            )


class TestCLIDebateCommand:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "commands" in plain.lower()
        assert "debate" in plain.lower()
        assert "quick start" in plain.lower()
        assert "duelyst debate --help" in plain
        assert "pydantic v1" not in plain.lower()

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["debate", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "topic" in plain.lower()
        assert "--model-a" in plain
        assert "--model-b" in plain
        assert "--output" in plain
        assert "auto-loads .env" in plain.lower()
        assert "claude-haiku" in plain
        assert "search" in plain.lower()
        assert "duelyst-ai-core[search]" in plain
        assert "pydantic v1" not in plain.lower()

    def test_debate_subcommand_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_run_debate(self: DebateDisplay, config: object) -> object:
            return SimpleNamespace(status="converged")

        def fake_load_dotenv(
            dotenv_path: str | None = None,
            override: bool = False,
        ) -> None:
            dotenv_calls.append((dotenv_path, override))

        dotenv_calls: list[tuple[str | None, bool]] = []

        monkeypatch.setattr(cli_main, "load_dotenv", fake_load_dotenv)
        monkeypatch.setattr(DebateDisplay, "show_config", lambda self, config: None)
        monkeypatch.setattr(DebateDisplay, "run_debate", fake_run_debate)
        monkeypatch.setattr(
            RichTerminalFormatter,
            "format",
            lambda self, result: "formatted debate output",
        )

        result = runner.invoke(app, ["debate", "Test topic"])

        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "formatted debate output" in plain
        assert dotenv_calls == [(".env", False)]

    def test_import_suppresses_python_314_warning(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import duelyst_ai_core; import langchain_core; print('ok')"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        combined = f"{result.stdout}\n{result.stderr}".lower()
        assert "pydantic v1" not in combined

    def test_module_help_runs_in_subprocess(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "duelyst_ai_core.cli.main", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        combined = f"{result.stdout}\n{result.stderr}"
        assert "Quick start" in combined
