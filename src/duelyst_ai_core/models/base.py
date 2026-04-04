"""Abstract base for all LLM model adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from pydantic import BaseModel

    from duelyst_ai_core.orchestrator.state import ModelConfig

T = TypeVar("T", bound="BaseModel")

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """Uniform async interface for calling LLM providers.

    All model adapters implement this interface so the orchestrator and agents
    are model-agnostic. Concrete adapters wrap LangChain chat models and handle
    provider-specific quirks internally.

    Args:
        config: The model configuration (provider, model_id, temperature, etc.).
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def config(self) -> ModelConfig:
        """The model configuration."""
        return self._config

    @property
    def provider(self) -> str:
        """The provider name (anthropic, openai, google)."""
        return self._config.provider

    @property
    def model_id(self) -> str:
        """The provider-specific model identifier."""
        return self._config.model_id

    @abstractmethod
    async def generate(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
    ) -> AIMessage:
        """Generate a response from the model.

        Args:
            messages: The conversation history as LangChain message objects.
            system_prompt: The system prompt (static, no user input).

        Returns:
            The model's response as an AIMessage.

        Raises:
            ModelError: If the API call fails.
        """

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
        response_model: type[T],
    ) -> T:
        """Generate a structured response matching a Pydantic model.

        Uses the provider's native structured output when available,
        falls back to JSON mode + Pydantic parsing otherwise.

        Args:
            messages: The conversation history as LangChain message objects.
            system_prompt: The system prompt (static, no user input).
            response_model: The Pydantic model class to parse the response into.

        Returns:
            An instance of response_model populated from the model's response.

        Raises:
            ModelError: If the API call or parsing fails.
        """
