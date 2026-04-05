"""Tests for the CLI."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from duelyst_ai_core.cli.main import _build_config, _parse_tools, app
from duelyst_ai_core.exceptions import ConfigError
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
            model_a="claude-sonnet",
            model_b="gpt-4o",
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
        assert config.max_rounds == 3
        assert config.judge_model is None

    def test_with_judge(self) -> None:
        config = _build_config(
            topic="Test",
            model_a="claude-sonnet",
            model_b="gpt-4o",
            judge="gemini-pro",
            instructions_a=None,
            instructions_b=None,
            rounds=5,
            convergence_threshold=7,
            convergence_rounds=2,
            tools_str=None,
        )
        assert config.judge_model is not None
        assert config.judge_model.provider == "google"

    def test_with_instructions(self) -> None:
        config = _build_config(
            topic="Test",
            model_a="claude-sonnet",
            model_b="gpt-4o",
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
                model_b="gpt-4o",
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
        assert "debate" in result.output.lower()

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["debate", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.output)
        assert "topic" in plain.lower()
        assert "--model-a" in plain
        assert "--model-b" in plain
        assert "--output" in plain
