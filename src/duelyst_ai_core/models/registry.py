"""Model registry — factory for creating LangChain chat models and judge auto-selection.

This module replaces the adapter pattern with direct LangChain chat model
creation. Consumer code receives a BaseChatModel and uses LangChain's native
methods (with_structured_output, bind_tools, etc.) directly.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from duelyst_ai_core.exceptions import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

    from duelyst_ai_core.orchestrator.state import ModelConfig

logger = logging.getLogger(__name__)

# Maps short names to (provider, model_id) tuples for CLI convenience.
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # Anthropic
    "claude-opus": ("anthropic", "claude-opus-4-20250514"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-haiku": ("anthropic", "claude-haiku-4-20250506"),
    # OpenAI
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4.1": ("openai", "gpt-4.1"),
    "gpt-4.1-mini": ("openai", "gpt-4.1-mini"),
    # Google
    "gemini-pro": ("google", "gemini-2.5-pro"),
    "gemini-flash": ("google", "gemini-2.5-flash"),
}

# Default judge models per provider, ordered by preference
_JUDGE_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "google": "gemini-2.5-pro",
}


def _create_anthropic(config: ModelConfig) -> BaseChatModel:
    """Create an Anthropic (Claude) chat model."""
    from langchain_anthropic import ChatAnthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        msg = "ANTHROPIC_API_KEY environment variable is not set"
        raise ConfigError(msg)

    return ChatAnthropic(
        model=config.model_id,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=api_key,
    )


def _create_openai(config: ModelConfig) -> BaseChatModel:
    """Create an OpenAI (GPT) chat model."""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY environment variable is not set"
        raise ConfigError(msg)

    return ChatOpenAI(
        model=config.model_id,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=api_key,
    )


def _create_google(config: ModelConfig) -> BaseChatModel:
    """Create a Google (Gemini) chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        msg = "GOOGLE_API_KEY environment variable is not set"
        raise ConfigError(msg)

    return ChatGoogleGenerativeAI(
        model=config.model_id,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        google_api_key=api_key,
    )


_PROVIDER_FACTORIES: dict[str, Callable[[ModelConfig], BaseChatModel]] = {
    "anthropic": _create_anthropic,
    "openai": _create_openai,
    "google": _create_google,
}


def resolve_alias(name: str) -> tuple[str, str]:
    """Resolve a model alias to (provider, model_id).

    If the name is not an alias, attempts to infer the provider from the
    model_id prefix.

    Args:
        name: Model alias (e.g. "claude-sonnet") or full model_id.

    Returns:
        Tuple of (provider, model_id).

    Raises:
        ConfigError: If the model name cannot be resolved.
    """
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]

    # Try to infer provider from model_id prefix
    if name.startswith("claude"):
        return ("anthropic", name)
    if name.startswith("gpt"):
        return ("openai", name)
    if name.startswith("gemini"):
        return ("google", name)

    msg = (
        f"Unknown model '{name}'. Use a known alias "
        f"({', '.join(sorted(MODEL_ALIASES))}) or a full model ID "
        f"prefixed with claude/gpt/gemini."
    )
    raise ConfigError(msg)


def create_model(config: ModelConfig) -> BaseChatModel:
    """Create a LangChain chat model from a ModelConfig.

    Returns the native LangChain chat model (ChatAnthropic, ChatOpenAI,
    ChatGoogleGenerativeAI) directly. Consumer code uses LangChain's
    built-in methods like with_structured_output() and bind_tools().

    Args:
        config: The model configuration.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ConfigError: If the provider is not supported or API key is missing.
    """
    factory = _PROVIDER_FACTORIES.get(config.provider)
    if factory is None:
        msg = (
            f"Unsupported provider '{config.provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDER_FACTORIES))}"
        )
        raise ConfigError(msg)

    return factory(config)


def get_judge_model(model_a: ModelConfig, model_b: ModelConfig) -> ModelConfig:
    """Auto-select a judge model from a different provider than both debaters.

    The judge must use a different provider to avoid bias. If both debaters
    use the same provider, any other provider is selected. If they use
    different providers, the remaining provider is chosen.

    Args:
        model_a: Configuration for debater A.
        model_b: Configuration for debater B.

    Returns:
        A ModelConfig for the judge.

    Raises:
        ConfigError: If no suitable judge provider can be found.
    """
    from duelyst_ai_core.orchestrator.state import ModelConfig as ModelConfigCls

    used_providers = {model_a.provider, model_b.provider}
    available_providers = set(_JUDGE_DEFAULTS) - used_providers

    if not available_providers:
        # Both debaters use different providers and all 3 are taken;
        # this shouldn't happen with 3 providers and 2 debaters, but
        # fall back to picking a provider different from at least one.
        available_providers = set(_JUDGE_DEFAULTS) - {model_a.provider}

    judge_provider = sorted(available_providers)[0]
    judge_model_id = _JUDGE_DEFAULTS[judge_provider]

    logger.info(
        "Auto-selected judge model: %s/%s (debaters: %s, %s)",
        judge_provider,
        judge_model_id,
        model_a.provider,
        model_b.provider,
    )

    return ModelConfigCls(provider=judge_provider, model_id=judge_model_id)
