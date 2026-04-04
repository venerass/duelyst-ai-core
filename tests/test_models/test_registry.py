"""Tests for model registry — alias resolution, factory, judge selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from duelyst_ai_core.exceptions import ConfigError
from duelyst_ai_core.models.registry import (
    create_adapter,
    get_judge_model,
    resolve_alias,
)
from duelyst_ai_core.orchestrator.state import ModelConfig

# ---------------------------------------------------------------------------
# resolve_alias
# ---------------------------------------------------------------------------


class TestResolveAlias:
    def test_known_alias(self) -> None:
        provider, model_id = resolve_alias("claude-sonnet")
        assert provider == "anthropic"
        assert model_id == "claude-sonnet-4-20250514"

    def test_openai_alias(self) -> None:
        provider, model_id = resolve_alias("gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"

    def test_google_alias(self) -> None:
        provider, model_id = resolve_alias("gemini-pro")
        assert provider == "google"
        assert model_id == "gemini-2.5-pro"

    def test_full_model_id_claude(self) -> None:
        provider, model_id = resolve_alias("claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert model_id == "claude-sonnet-4-20250514"

    def test_full_model_id_gpt(self) -> None:
        provider, model_id = resolve_alias("gpt-4.1-nano")
        assert provider == "openai"
        assert model_id == "gpt-4.1-nano"

    def test_full_model_id_gemini(self) -> None:
        provider, model_id = resolve_alias("gemini-2.0-flash")
        assert provider == "google"
        assert model_id == "gemini-2.0-flash"

    def test_unknown_model(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model"):
            resolve_alias("llama-3")


# ---------------------------------------------------------------------------
# create_adapter
# ---------------------------------------------------------------------------


class TestCreateAdapter:
    def test_create_anthropic(self) -> None:
        config = ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514")
        mock_cls = MagicMock()
        with patch.dict(
            "duelyst_ai_core.models.registry._ADAPTER_CLASSES",
            {"anthropic": mock_cls},
        ):
            adapter = create_adapter(config)
        mock_cls.assert_called_once_with(config)
        assert adapter is mock_cls.return_value

    def test_create_openai(self) -> None:
        config = ModelConfig(provider="openai", model_id="gpt-4o")
        mock_cls = MagicMock()
        with patch.dict(
            "duelyst_ai_core.models.registry._ADAPTER_CLASSES",
            {"openai": mock_cls},
        ):
            adapter = create_adapter(config)
        mock_cls.assert_called_once_with(config)
        assert adapter is mock_cls.return_value

    def test_create_google(self) -> None:
        config = ModelConfig(provider="google", model_id="gemini-2.5-pro")
        mock_cls = MagicMock()
        with patch.dict(
            "duelyst_ai_core.models.registry._ADAPTER_CLASSES",
            {"google": mock_cls},
        ):
            adapter = create_adapter(config)
        mock_cls.assert_called_once_with(config)
        assert adapter is mock_cls.return_value

    def test_unsupported_provider(self) -> None:
        config = MagicMock()
        config.provider = "mistral"
        with pytest.raises(ConfigError, match="Unsupported provider"):
            create_adapter(config)


# ---------------------------------------------------------------------------
# get_judge_model
# ---------------------------------------------------------------------------


class TestGetJudgeModel:
    def test_both_same_provider(self) -> None:
        """When both debaters use the same provider, judge picks another."""
        a = ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514")
        b = ModelConfig(provider="anthropic", model_id="claude-opus-4-20250514")
        judge = get_judge_model(a, b)
        assert judge.provider != "anthropic"
        assert judge.provider in ("openai", "google")

    def test_different_providers(self) -> None:
        """When debaters use different providers, judge picks the remaining one."""
        a = ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514")
        b = ModelConfig(provider="openai", model_id="gpt-4o")
        judge = get_judge_model(a, b)
        assert judge.provider == "google"

    def test_anthropic_google_gives_openai(self) -> None:
        a = ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514")
        b = ModelConfig(provider="google", model_id="gemini-2.5-pro")
        judge = get_judge_model(a, b)
        assert judge.provider == "openai"

    def test_openai_google_gives_anthropic(self) -> None:
        a = ModelConfig(provider="openai", model_id="gpt-4o")
        b = ModelConfig(provider="google", model_id="gemini-2.5-pro")
        judge = get_judge_model(a, b)
        assert judge.provider == "anthropic"

    def test_returns_valid_model_config(self) -> None:
        a = ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514")
        b = ModelConfig(provider="openai", model_id="gpt-4o")
        judge = get_judge_model(a, b)
        assert isinstance(judge, ModelConfig)
        assert judge.model_id  # non-empty
        assert judge.temperature == 0.7  # default
