"""Tests for model adapters — Anthropic, OpenAI, Google."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from duelyst_ai_core.agents.schemas import Evidence
from duelyst_ai_core.exceptions import ConfigError, ModelError
from duelyst_ai_core.models.anthropic import AnthropicAdapter
from duelyst_ai_core.models.google import GoogleAdapter
from duelyst_ai_core.models.openai import OpenAIAdapter
from duelyst_ai_core.orchestrator.state import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_config() -> ModelConfig:
    return ModelConfig(provider="anthropic", model_id="claude-sonnet-4-20250514")


@pytest.fixture
def openai_config() -> ModelConfig:
    return ModelConfig(provider="openai", model_id="gpt-4o")


@pytest.fixture
def google_config() -> ModelConfig:
    return ModelConfig(provider="google", model_id="gemini-2.5-pro")


@pytest.fixture
def sample_messages() -> list[HumanMessage]:
    return [HumanMessage(content="What is the meaning of life?")]


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------


class TestAnthropicAdapter:
    def test_missing_api_key(self, anthropic_config: ModelConfig) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"),
        ):
            AnthropicAdapter(anthropic_config)

    @patch("duelyst_ai_core.models.anthropic.ChatAnthropic")
    def test_creation(
        self,
        mock_chat: MagicMock,
        anthropic_config: ModelConfig,
    ) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter(anthropic_config)
        assert adapter.provider == "anthropic"
        assert adapter.model_id == "claude-sonnet-4-20250514"
        mock_chat.assert_called_once()

    @patch("duelyst_ai_core.models.anthropic.ChatAnthropic")
    async def test_generate(
        self,
        mock_chat_cls: MagicMock,
        anthropic_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="42"),
        )
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter(anthropic_config)

        result = await adapter.generate(sample_messages, "You are helpful.")
        assert result.content == "42"
        mock_llm.ainvoke.assert_awaited_once()

    @patch("duelyst_ai_core.models.anthropic.ChatAnthropic")
    async def test_generate_structured(
        self,
        mock_chat_cls: MagicMock,
        anthropic_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        expected = Evidence(claim="Test claim", source_type="reasoning")

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected)

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter(anthropic_config)

        result = await adapter.generate_structured(sample_messages, "System", Evidence)
        assert result == expected
        mock_llm.with_structured_output.assert_called_once_with(Evidence)

    @patch("duelyst_ai_core.models.anthropic.ChatAnthropic")
    async def test_generate_api_error(
        self,
        mock_chat_cls: MagicMock,
        anthropic_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API timeout"))
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter(anthropic_config)

        with pytest.raises(ModelError, match="Anthropic API call failed"):
            await adapter.generate(sample_messages, "System")


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------


class TestOpenAIAdapter:
    def test_missing_api_key(self, openai_config: ModelConfig) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="OPENAI_API_KEY"),
        ):
            OpenAIAdapter(openai_config)

    @patch("duelyst_ai_core.models.openai.ChatOpenAI")
    def test_creation(
        self,
        mock_chat: MagicMock,
        openai_config: ModelConfig,
    ) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIAdapter(openai_config)
        assert adapter.provider == "openai"
        assert adapter.model_id == "gpt-4o"

    @patch("duelyst_ai_core.models.openai.ChatOpenAI")
    async def test_generate(
        self,
        mock_chat_cls: MagicMock,
        openai_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Hello"),
        )
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIAdapter(openai_config)

        result = await adapter.generate(sample_messages, "System")
        assert result.content == "Hello"

    @patch("duelyst_ai_core.models.openai.ChatOpenAI")
    async def test_generate_structured(
        self,
        mock_chat_cls: MagicMock,
        openai_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        expected = Evidence(claim="Test", source_type="web", source="https://x.com")

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected)

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIAdapter(openai_config)

        result = await adapter.generate_structured(sample_messages, "System", Evidence)
        assert result == expected

    @patch("duelyst_ai_core.models.openai.ChatOpenAI")
    async def test_generate_api_error(
        self,
        mock_chat_cls: MagicMock,
        openai_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("Rate limited"))
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIAdapter(openai_config)

        with pytest.raises(ModelError, match="OpenAI API call failed"):
            await adapter.generate(sample_messages, "System")


# ---------------------------------------------------------------------------
# GoogleAdapter
# ---------------------------------------------------------------------------


class TestGoogleAdapter:
    def test_missing_api_key(self, google_config: ModelConfig) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="GOOGLE_API_KEY"),
        ):
            GoogleAdapter(google_config)

    @patch("duelyst_ai_core.models.google.ChatGoogleGenerativeAI")
    def test_creation(
        self,
        mock_chat: MagicMock,
        google_config: ModelConfig,
    ) -> None:
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            adapter = GoogleAdapter(google_config)
        assert adapter.provider == "google"
        assert adapter.model_id == "gemini-2.5-pro"

    @patch("duelyst_ai_core.models.google.ChatGoogleGenerativeAI")
    async def test_generate(
        self,
        mock_chat_cls: MagicMock,
        google_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=AIMessage(content="Gemini says hi"),
        )
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            adapter = GoogleAdapter(google_config)

        result = await adapter.generate(sample_messages, "System")
        assert result.content == "Gemini says hi"

    @patch("duelyst_ai_core.models.google.ChatGoogleGenerativeAI")
    async def test_generate_structured(
        self,
        mock_chat_cls: MagicMock,
        google_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        expected = Evidence(claim="Test", source_type="reasoning")

        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected)

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            adapter = GoogleAdapter(google_config)

        result = await adapter.generate_structured(sample_messages, "System", Evidence)
        assert result == expected

    @patch("duelyst_ai_core.models.google.ChatGoogleGenerativeAI")
    async def test_generate_api_error(
        self,
        mock_chat_cls: MagicMock,
        google_config: ModelConfig,
        sample_messages: list[HumanMessage],
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("Quota exceeded"))
        mock_chat_cls.return_value = mock_llm

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            adapter = GoogleAdapter(google_config)

        with pytest.raises(ModelError, match="Google API call failed"):
            await adapter.generate(sample_messages, "System")
