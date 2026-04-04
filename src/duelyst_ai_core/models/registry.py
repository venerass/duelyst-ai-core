"""Model registry — factory for creating adapters and judge auto-selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from duelyst_ai_core.exceptions import ConfigError
from duelyst_ai_core.models.anthropic import AnthropicAdapter
from duelyst_ai_core.models.google import GoogleAdapter
from duelyst_ai_core.models.openai import OpenAIAdapter

if TYPE_CHECKING:
    from duelyst_ai_core.models.base import BaseModelAdapter
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

# Provider → adapter class mapping
_ADAPTER_CLASSES: dict[str, type[BaseModelAdapter]] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "google": GoogleAdapter,
}

# Default judge models per provider, ordered by preference
_JUDGE_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "google": "gemini-2.5-pro",
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


def create_adapter(config: ModelConfig) -> BaseModelAdapter:
    """Create a model adapter from a ModelConfig.

    Args:
        config: The model configuration.

    Returns:
        An instantiated model adapter.

    Raises:
        ConfigError: If the provider is not supported.
    """
    adapter_cls = _ADAPTER_CLASSES.get(config.provider)
    if adapter_cls is None:
        msg = (
            f"Unsupported provider '{config.provider}'. "
            f"Supported: {', '.join(sorted(_ADAPTER_CLASSES))}"
        )
        raise ConfigError(msg)

    return adapter_cls(config)


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
