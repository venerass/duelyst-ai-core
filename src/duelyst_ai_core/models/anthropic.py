"""Anthropic (Claude) model adapter."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, TypeVar, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from duelyst_ai_core.exceptions import ConfigError, ModelError
from duelyst_ai_core.models.base import BaseModelAdapter

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from pydantic import BaseModel

    from duelyst_ai_core.orchestrator.state import ModelConfig

T = TypeVar("T", bound="BaseModel")

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-20250506",
    "claude-sonnet-4-5-20250514",
}


class AnthropicAdapter(BaseModelAdapter):
    """Model adapter for Anthropic Claude models.

    Wraps LangChain's ChatAnthropic to provide the uniform adapter interface.

    Args:
        config: Model configuration with provider="anthropic".

    Raises:
        ConfigError: If ANTHROPIC_API_KEY is not set or model is unsupported.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            msg = "ANTHROPIC_API_KEY environment variable is not set"
            raise ConfigError(msg)

        if config.model_id not in SUPPORTED_MODELS:
            logger.warning(
                "Model '%s' is not in the known supported list, proceeding anyway",
                config.model_id,
            )

        self._llm = ChatAnthropic(
            model=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key,
        )

    async def generate(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
    ) -> AIMessage:
        """Generate a response from Claude."""
        try:
            full_messages = [SystemMessage(content=system_prompt), *messages]
            result = await self._llm.ainvoke(full_messages)
            return cast("AIMessage", result)
        except Exception as e:
            msg = f"Anthropic API call failed for model '{self.model_id}': {e}"
            raise ModelError(msg) from e

    async def generate_structured(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
        response_model: type[T],
    ) -> T:
        """Generate a structured response from Claude.

        Uses Anthropic's native structured output support via
        `.with_structured_output()`.
        """
        try:
            structured_llm = self._llm.with_structured_output(response_model)
            full_messages = [SystemMessage(content=system_prompt), *messages]
            result = await structured_llm.ainvoke(full_messages)
        except Exception as e:
            msg = f"Anthropic structured output failed for model '{self.model_id}': {e}"
            raise ModelError(msg) from e

        if not isinstance(result, response_model):
            msg = f"Expected {response_model.__name__}, got {type(result).__name__}"
            raise ModelError(msg)

        return result
