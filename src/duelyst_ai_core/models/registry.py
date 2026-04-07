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
# The API maintains a separate product catalog with display labels, reasoning labels,
# and ordering — this registry is purely for alias resolution and model creation.
MODEL_ALIASES: dict[str, ModelAlias] = {
    # ── Anthropic ──────────────────────────────────────────────────────────
    "claude-haiku": ModelAlias("anthropic", "claude-haiku-4-5", "free"),
    "claude-sonnet": ModelAlias("anthropic", "claude-sonnet-4-6", "standard"),
    "claude-opus": ModelAlias("anthropic", "claude-opus-4-6", "pro"),
    # ── OpenAI ─────────────────────────────────────────────────────────────
    "gpt-nano": ModelAlias("openai", "gpt-5.4-nano", "free"),
    "gpt-mini": ModelAlias("openai", "gpt-5.4-mini", "free"),
    "gpt-5": ModelAlias("openai", "gpt-5.4", "standard"),
    "gpt-5-high": ModelAlias("openai", "gpt-5.4", "pro"),
    # ── Google ─────────────────────────────────────────────────────────────
    "gemini-flash-lite": ModelAlias("google", "gemini-2.5-flash-lite", "free"),
    "gemini-flash": ModelAlias("google", "gemini-2.5-flash", "standard"),
    "gemini-pro": ModelAlias("google", "gemini-2.5-pro", "standard"),
    "gemini-pro-high": ModelAlias("google", "gemini-2.5-pro", "pro"),
    # ── Legacy compatibility ───────────────────────────────────────────────
    # Kept so existing debates still resolve. Not shown in the product catalog.
    "gpt-4o": ModelAlias("openai", "gpt-4o", "standard"),
    "gpt-4o-mini": ModelAlias("openai", "gpt-4o-mini", "standard"),
    "gpt-4.1": ModelAlias("openai", "gpt-4.1", "standard"),
    "gpt-4.1-mini": ModelAlias("openai", "gpt-4.1-mini", "standard"),
    "gemini-2.5-pro": ModelAlias("google", "gemini-2.5-pro", "standard"),
    "gemini-2.5-flash": ModelAlias("google", "gemini-2.5-flash", "standard"),
    "gemini-2.5-flash-lite": ModelAlias("google", "gemini-2.5-flash-lite", "free"),
    "gemini-3.1-pro": ModelAlias("google", "gemini-3.1-pro-preview", "standard"),
    "gemini-3-flash": ModelAlias("google", "gemini-3-flash-preview", "standard"),
    "gemini-3.1-flash-lite": ModelAlias("google", "gemini-3.1-flash-lite-preview", "free"),
}

# Judge selection: pick the best model from the provider absent from the debate.
# Priority: anthropic (sonnet) > openai (gpt-5.4) > google (gemini-3.1-pro).
_JUDGE_BY_MISSING_PROVIDER: list[tuple[str, str, str]] = [
    ("anthropic", "anthropic", "claude-sonnet-4-6"),
    ("openai", "openai", "gpt-5.4"),
    ("google", "google", "gemini-2.5-pro"),
]


# Default resilience settings for LLM calls.
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TIMEOUT = 120.0


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
        max_retries=_DEFAULT_MAX_RETRIES,
        timeout=_DEFAULT_TIMEOUT,
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
        max_retries=_DEFAULT_MAX_RETRIES,
        timeout=_DEFAULT_TIMEOUT,
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
        max_retries=_DEFAULT_MAX_RETRIES,
        timeout=_DEFAULT_TIMEOUT,
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
    """Auto-select a judge model from a provider not used by either debater.

    Priority order when multiple providers are absent:
      1. Anthropic  → ``claude-sonnet-4-6``
      2. OpenAI     → ``gpt-5.4``
      3. Google     → ``gemini-2.5-pro``

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

    for missing_provider, judge_provider, judge_model_id in _JUDGE_BY_MISSING_PROVIDER:
        if missing_provider not in used_providers:
            logger.info(
                "Auto-selected judge model: %s/%s (debaters: %s, %s)",
                judge_provider,
                judge_model_id,
                model_a.provider,
                model_b.provider,
            )
            return ModelConfigCls(provider=judge_provider, model_id=judge_model_id)

    # Should be unreachable with 3 providers and 2 debaters, but guard anyway.
    msg = f"Cannot auto-select judge: all providers in use ({model_a.provider}, {model_b.provider})"
    raise ConfigError(msg)


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
