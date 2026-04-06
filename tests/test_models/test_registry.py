"""Tests for model registry — alias resolution, factory, judge, and tier classification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from duelyst_ai_core.exceptions import ConfigError
from duelyst_ai_core.models.registry import (
    ModelAlias,
    create_model,
    get_judge_model,
    get_model_tier,
    list_all_models,
    list_models_by_tier,
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
        assert model_id == "claude-sonnet-4-6"

    def test_openai_alias(self) -> None:
        provider, model_id = resolve_alias("gpt-mini")
        assert provider == "openai"
        assert model_id == "gpt-5.4-mini"

    def test_legacy_openai_alias(self) -> None:
        provider, model_id = resolve_alias("gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"

    def test_google_alias(self) -> None:
        provider, model_id = resolve_alias("gemini-flash")
        assert provider == "google"
        assert model_id == "gemini-3.0-flash"

    def test_full_model_id_claude(self) -> None:
        provider, model_id = resolve_alias("claude-sonnet-4-6")
        assert provider == "anthropic"
        assert model_id == "claude-sonnet-4-6"

    def test_full_model_id_gpt(self) -> None:
        provider, model_id = resolve_alias("gpt-5.4-nano")
        assert provider == "openai"
        assert model_id == "gpt-5.4-nano"

    def test_full_model_id_gemini(self) -> None:
        provider, model_id = resolve_alias("gemini-2.5-flash")
        assert provider == "google"
        assert model_id == "gemini-2.5-flash"

    def test_unknown_model(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model"):
            resolve_alias("llama-3")


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------


class TestCreateModel:
    def test_create_anthropic(self) -> None:
        config = ModelConfig(provider="anthropic", model_id="claude-haiku-4-5")
        mock_chat = MagicMock()
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("langchain_anthropic.ChatAnthropic", mock_chat),
        ):
            model = create_model(config)
        assert model is mock_chat.return_value

    def test_create_openai(self) -> None:
        config = ModelConfig(provider="openai", model_id="gpt-5.4-mini")
        mock_chat = MagicMock()
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("langchain_openai.ChatOpenAI", mock_chat),
        ):
            model = create_model(config)
        assert model is mock_chat.return_value

    def test_create_google(self) -> None:
        config = ModelConfig(provider="google", model_id="gemini-2.5-flash")
        mock_chat = MagicMock()
        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_chat),
        ):
            model = create_model(config)
        assert model is mock_chat.return_value

    def test_unsupported_provider(self) -> None:
        config = MagicMock()
        config.provider = "mistral"
        with pytest.raises(ConfigError, match="Unsupported provider"):
            create_model(config)


# ---------------------------------------------------------------------------
# get_judge_model
# ---------------------------------------------------------------------------


class TestGetJudgeModel:
    def test_both_same_provider(self) -> None:
        """When both debaters use the same provider, judge picks another."""
        a = ModelConfig(provider="anthropic", model_id="claude-haiku-4-5")
        b = ModelConfig(provider="anthropic", model_id="claude-opus-4-6")
        judge = get_judge_model(a, b)
        assert judge.provider != "anthropic"
        assert judge.provider in ("openai", "google")

    def test_different_providers(self) -> None:
        """When debaters use different providers, judge picks the remaining one."""
        a = ModelConfig(provider="anthropic", model_id="claude-haiku-4-5")
        b = ModelConfig(provider="openai", model_id="gpt-5.4-mini")
        judge = get_judge_model(a, b)
        assert judge.provider == "google"
        assert judge.model_id == "gemini-3.0-flash"

    def test_anthropic_google_gives_openai(self) -> None:
        a = ModelConfig(provider="anthropic", model_id="claude-haiku-4-5")
        b = ModelConfig(provider="google", model_id="gemini-3.0-flash")
        judge = get_judge_model(a, b)
        assert judge.provider == "openai"
        assert judge.model_id == "gpt-5.4-mini"

    def test_openai_google_gives_anthropic(self) -> None:
        a = ModelConfig(provider="openai", model_id="gpt-5.4-mini")
        b = ModelConfig(provider="google", model_id="gemini-3.0-flash")
        judge = get_judge_model(a, b)
        assert judge.provider == "anthropic"
        assert judge.model_id == "claude-haiku-4-5"

    def test_returns_valid_model_config(self) -> None:
        a = ModelConfig(provider="anthropic", model_id="claude-haiku-4-5")
        b = ModelConfig(provider="openai", model_id="gpt-5.4-mini")
        judge = get_judge_model(a, b)
        assert isinstance(judge, ModelConfig)
        assert judge.model_id  # non-empty
        assert judge.temperature == 0.7  # default


# ---------------------------------------------------------------------------
# get_model_tier
# ---------------------------------------------------------------------------


class TestGetModelTier:
    def test_free_tier(self) -> None:
        assert get_model_tier("claude-haiku") == "free"
        assert get_model_tier("gpt-mini") == "free"
        assert get_model_tier("gpt-nano") == "free"
        assert get_model_tier("gemini-flash-lite") == "free"

    def test_standard_tier(self) -> None:
        assert get_model_tier("claude-sonnet") == "standard"
        assert get_model_tier("gpt-5") == "standard"
        assert get_model_tier("gpt-4o") == "standard"
        assert get_model_tier("gemini-flash") == "standard"
        assert get_model_tier("gemini-pro") == "standard"

    def test_pro_tier(self) -> None:
        assert get_model_tier("claude-opus") == "pro"
        assert get_model_tier("gpt-5-high") == "pro"
        assert get_model_tier("gemini-pro-high") == "pro"

    def test_unknown_alias(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model alias"):
            get_model_tier("llama-3")


# ---------------------------------------------------------------------------
# list_models_by_tier
# ---------------------------------------------------------------------------


class TestListModelsByTier:
    def test_free_models(self) -> None:
        free = list_models_by_tier("free")
        assert "claude-haiku" in free
        assert "gpt-mini" in free
        assert "gpt-nano" in free
        assert "gemini-flash-lite" in free
        assert "claude-opus" not in free

    def test_pro_models(self) -> None:
        pro = list_models_by_tier("pro")
        assert "claude-opus" in pro
        assert "gpt-5-high" in pro
        assert "gemini-pro-high" in pro
        assert len(pro) == 3

    def test_standard_models(self) -> None:
        standard = list_models_by_tier("standard")
        assert "claude-sonnet" in standard
        assert "gemini-pro" in standard


# ---------------------------------------------------------------------------
# list_all_models
# ---------------------------------------------------------------------------


class TestListAllModels:
    def test_returns_all_aliases(self) -> None:
        models = list_all_models()
        aliases = {m["alias"] for m in models}
        assert "claude-haiku" in aliases
        assert "claude-opus" in aliases
        assert "gpt-mini" in aliases

    def test_entry_structure(self) -> None:
        models = list_all_models()
        entry = next(m for m in models if m["alias"] == "claude-haiku")
        assert entry["provider"] == "anthropic"
        assert entry["model_id"] == "claude-haiku-4-5"
        assert entry["tier"] == "free"


# ---------------------------------------------------------------------------
# ModelAlias dataclass
# ---------------------------------------------------------------------------


class TestModelAlias:
    def test_is_frozen(self) -> None:
        alias = ModelAlias("anthropic", "claude-haiku-4-5", "free")
        with pytest.raises(AttributeError):
            alias.tier = "pro"  # type: ignore[misc]
