"""Custom exception hierarchy for duelyst-ai-core."""


class DuelystError(Exception):
    """Base exception for all duelyst-ai-core errors."""


class ModelError(DuelystError):
    """LLM provider error — authentication, rate limit, timeout, or API failure."""


class ConfigError(DuelystError):
    """Invalid debate configuration."""


class ToolError(DuelystError):
    """Tool execution failure — search, code execution, etc."""


class ConvergenceError(DuelystError):
    """Unexpected convergence state during debate."""
