"""Model registry — factory for creating LangChain chat models and judge auto-selection.

This module replaces the adapter pattern with direct LangChain chat model
creation. Consumer code receives a BaseChatModel and uses LangChain's native
methods (with_structured_output, bind_tools, etc.) directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from duelyst_ai_core.exceptions import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

    from duelyst_ai_core.orchestrator.state import ModelConfig

logger = logging.getLogger(__name__)

ModelTier = Literal["free", "standard", "pro"]


@dataclass(frozen=True)
class ModelAlias:
    """Metadata for a model alias including provider, model ID, and access tier."""

    provider: str
    model_id: str
    tier: ModelTier


# Maps current friendly aliases to ModelAlias entries for CLI convenience and tier gating.
MODEL_ALIASES: dict[str, ModelAlias] = {
    # Anthropic current generation
    "claude-opus": ModelAlias("anthropic", "claude-opus-4-6", "pro"),
    "claude-sonnet": ModelAlias("anthropic", "claude-sonnet-4-6", "standard"),
    "claude-haiku": ModelAlias("anthropic", "claude-haiku-4-5", "free"),
    # OpenAI current generation
    "gpt-5": ModelAlias("openai", "gpt-5.4", "pro"),
    "gpt-mini": ModelAlias("openai", "gpt-5.4-mini", "free"),
    "gpt-nano": ModelAlias("openai", "gpt-5.4-nano", "free"),
    # Legacy compatibility aliases
    "gpt-4o": ModelAlias("openai", "gpt-4o", "standard"),
    "gpt-4o-mini": ModelAlias("openai", "gpt-4o-mini", "standard"),
    "gpt-4.1": ModelAlias("openai", "gpt-4.1", "standard"),
    "gpt-4.1-mini": ModelAlias("openai", "gpt-4.1-mini", "standard"),
    # Google stable/current
    "gemini-pro": ModelAlias("google", "gemini-2.5-pro", "standard"),
    "gemini-flash": ModelAlias("google", "gemini-2.5-flash", "standard"),
    "gemini-flash-lite": ModelAlias("google", "gemini-2.5-flash-lite", "free"),
}

# Low-cost defaults per provider for auto-selected judges.
_JUDGE_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5.4-mini",
    "google": "gemini-2.5-flash",
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
        name: Model alias (e.g. "claude-haiku" or "gpt-mini") or full model_id.

    Returns:
        Tuple of (provider, model_id).

    Raises:
        ConfigError: If the model name cannot be resolved.
    """
    entry = MODEL_ALIASES.get(name)
    if entry is not None:
        return (entry.provider, entry.model_id)

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


def get_model_tier(alias: str) -> ModelTier:
    """Return the access tier for a known model alias.

    Args:
        alias: A model alias (e.g. "claude-haiku").

    Returns:
        The tier: "free", "standard", or "pro".

    Raises:
        ConfigError: If the alias is not recognised.
    """
    entry = MODEL_ALIASES.get(alias)
    if entry is None:
        msg = f"Unknown model alias: {alias!r}"
        raise ConfigError(msg)
    return entry.tier


def list_models_by_tier(tier: ModelTier) -> list[str]:
    """Return all aliases matching the given tier.

    Args:
        tier: One of "free", "standard", or "pro".

    Returns:
        Sorted list of matching alias names.
    """
    return sorted(alias for alias, meta in MODEL_ALIASES.items() if meta.tier == tier)


def list_all_models() -> list[dict[str, str]]:
    """Return metadata for all registered model aliases.

    Returns:
        List of dicts with keys: alias, provider, model_id, tier.
    """
    return [
        {
            "alias": alias,
            "provider": meta.provider,
            "model_id": meta.model_id,
            "tier": meta.tier,
        }
        for alias, meta in MODEL_ALIASES.items()
    ]
