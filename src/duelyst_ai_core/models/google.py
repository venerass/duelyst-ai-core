"""Google (Gemini) model adapter."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, TypeVar, cast

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from duelyst_ai_core.exceptions import ConfigError, ModelError
from duelyst_ai_core.models.base import BaseModelAdapter

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from pydantic import BaseModel

    from duelyst_ai_core.orchestrator.state import ModelConfig

T = TypeVar("T", bound="BaseModel")

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
}


class GoogleAdapter(BaseModelAdapter):
    """Model adapter for Google Gemini models.

    Wraps LangChain's ChatGoogleGenerativeAI to provide the uniform
    adapter interface.

    Args:
        config: Model configuration with provider="google".

    Raises:
        ConfigError: If GOOGLE_API_KEY is not set.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            msg = "GOOGLE_API_KEY environment variable is not set"
            raise ConfigError(msg)

        if config.model_id not in SUPPORTED_MODELS:
            logger.warning(
                "Model '%s' is not in the known supported list, proceeding anyway",
                config.model_id,
            )

        self._llm = ChatGoogleGenerativeAI(
            model=config.model_id,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            google_api_key=api_key,
        )

    async def generate(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
    ) -> AIMessage:
        """Generate a response from Gemini."""
        try:
            full_messages = [SystemMessage(content=system_prompt), *messages]
            result = await self._llm.ainvoke(full_messages)
            return cast("AIMessage", result)
        except Exception as e:
            msg = f"Google API call failed for model '{self.model_id}': {e}"
            raise ModelError(msg) from e

    async def generate_structured(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
        response_model: type[T],
    ) -> T:
        """Generate a structured response from Gemini.

        Uses LangChain's `.with_structured_output()` which handles
        Google's structured output support.
        """
        try:
            structured_llm = self._llm.with_structured_output(response_model)
            full_messages = [SystemMessage(content=system_prompt), *messages]
            result = await structured_llm.ainvoke(full_messages)
        except Exception as e:
            msg = f"Google structured output failed for model '{self.model_id}': {e}"
            raise ModelError(msg) from e

        if not isinstance(result, response_model):
            msg = f"Expected {response_model.__name__}, got {type(result).__name__}"
            raise ModelError(msg)

        return result
