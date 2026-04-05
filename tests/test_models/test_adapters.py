"""Tests for model creation — verify correct LangChain models are instantiated."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from duelyst_ai_core.exceptions import ConfigError
from duelyst_ai_core.models.registry import create_model
from duelyst_ai_core.orchestrator.state import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_config() -> ModelConfig:
    return ModelConfig(provider="anthropic", model_id="claude-haiku-4-5")


@pytest.fixture
def openai_config() -> ModelConfig:
    return ModelConfig(provider="openai", model_id="gpt-5.4-mini")


@pytest.fixture
def google_config() -> ModelConfig:
    return ModelConfig(provider="google", model_id="gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestCreateAnthropic:
    def test_missing_api_key(self, anthropic_config: ModelConfig) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"),
        ):
            create_model(anthropic_config)

    def test_creates_chat_anthropic(
        self,
        anthropic_config: ModelConfig,
    ) -> None:
        mock_chat = MagicMock()
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("langchain_anthropic.ChatAnthropic", mock_chat),
        ):
            model = create_model(anthropic_config)

        assert model is mock_chat.return_value
        mock_chat.assert_called_once_with(
            model="claude-haiku-4-5",
            temperature=0.7,
            max_tokens=4096,
            api_key="test-key",
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class TestCreateOpenAI:
    def test_missing_api_key(self, openai_config: ModelConfig) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="OPENAI_API_KEY"),
        ):
            create_model(openai_config)

    def test_creates_chat_openai(self, openai_config: ModelConfig) -> None:
        mock_chat = MagicMock()
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("langchain_openai.ChatOpenAI", mock_chat),
        ):
            model = create_model(openai_config)

        assert model is mock_chat.return_value
        mock_chat.assert_called_once_with(
            model="gpt-5.4-mini",
            temperature=0.7,
            max_tokens=4096,
            api_key="test-key",
        )


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


class TestCreateGoogle:
    def test_missing_api_key(self, google_config: ModelConfig) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="GOOGLE_API_KEY"),
        ):
            create_model(google_config)

    def test_creates_chat_google(self, google_config: ModelConfig) -> None:
        mock_chat = MagicMock()
        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_chat),
        ):
            model = create_model(google_config)

        assert model is mock_chat.return_value
        mock_chat.assert_called_once_with(
            model="gemini-2.5-flash",
            temperature=0.7,
            max_output_tokens=4096,
            google_api_key="test-key",
        )


# ---------------------------------------------------------------------------
# Unsupported provider
# ---------------------------------------------------------------------------


class TestUnsupportedProvider:
    def test_unsupported_provider(self) -> None:
        config = MagicMock()
        config.provider = "mistral"
        with pytest.raises(ConfigError, match="Unsupported provider"):
            create_model(config)
