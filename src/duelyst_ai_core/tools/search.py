"""Web search tool using Tavily API.

Provides a LangChain-compatible tool that agents can use for real-time
evidence gathering during debates. Gracefully unavailable when the
TAVILY_API_KEY is not set.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from duelyst_ai_core.exceptions import ConfigError, ToolError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def create_search_tool() -> BaseTool:
    """Create a Tavily web search tool.

    Returns a LangChain BaseTool ready to be passed to create_agent's
    tools parameter.

    Returns:
        A configured TavilySearchResults tool.

    Raises:
        ConfigError: If TAVILY_API_KEY is not set.
        ToolError: If tavily-python is not installed.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        msg = (
            "TAVILY_API_KEY environment variable is not set. "
            "Install tavily-python and set the key to enable web search."
        )
        raise ConfigError(msg)

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except ImportError:
        msg = "tavily-python is not installed. Install it with: pip install duelyst-ai-core[search]"
        raise ToolError(msg) from None

    tool = TavilySearchResults(
        max_results=5,
        tavily_api_key=api_key,
    )

    logger.info("Web search tool created (Tavily)")
    return tool  # type: ignore[no-any-return]


def is_search_available() -> bool:
    """Check if the web search tool can be created.

    Returns:
        True if TAVILY_API_KEY is set and tavily-python is installed.
    """
    if not os.environ.get("TAVILY_API_KEY"):
        return False
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # noqa: F401
    except ImportError:
        return False
    return True
