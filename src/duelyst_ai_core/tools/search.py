"""Web search tool using Tavily API.

Provides a LangChain-compatible tool that agents can use for real-time
evidence gathering during debates. Gracefully unavailable when the
TAVILY_API_KEY is not set or the optional search dependency is missing.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from duelyst_ai_core.exceptions import ConfigError, ToolError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_SEARCH_INSTALL_HINT = 'Install them with: pip install "duelyst-ai-core[search]"'


def get_search_unavailable_reason() -> str | None:
    """Return the reason web search is unavailable, if any."""
    if not os.environ.get("TAVILY_API_KEY"):
        return "TAVILY_API_KEY environment variable is not set"

    try:
        from langchain_tavily import TavilySearch  # noqa: F401
    except ImportError:
        return (
            "Search dependencies are not installed "
            f"(missing langchain-tavily). {_SEARCH_INSTALL_HINT}"
        )

    return None


def create_search_tool() -> BaseTool:
    """Create a Tavily web search tool.

    Returns a LangChain BaseTool ready to be passed to create_agent's
    tools parameter.

    Returns:
        A configured TavilySearch tool.

    Raises:
        ConfigError: If TAVILY_API_KEY is not set.
        ToolError: If search dependencies are not installed.
    """
    unavailable_reason = get_search_unavailable_reason()
    if unavailable_reason is not None:
        if "TAVILY_API_KEY" in unavailable_reason:
            raise ConfigError(unavailable_reason)
        raise ToolError(unavailable_reason)

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        msg = (
            "Search dependencies are not installed "
            f"(missing langchain-tavily). {_SEARCH_INSTALL_HINT}"
        )
        raise ToolError(msg) from None

    tool = TavilySearch(
        max_results=5,
        topic="general",
    )

    logger.info("Web search tool created (Tavily via langchain-tavily)")
    return tool


def is_search_available() -> bool:
    """Check if the web search tool can be created.

    Returns:
        True if TAVILY_API_KEY is set and search dependencies are installed.
    """
    return get_search_unavailable_reason() is None
