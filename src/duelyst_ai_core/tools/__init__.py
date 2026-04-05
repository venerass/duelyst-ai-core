"""Tool integrations — web search, code execution.

Tools are optional — debates work without them. Each tool is a standard
LangChain BaseTool that integrates directly with create_agent.
"""

from duelyst_ai_core.tools.search import (
    create_search_tool,
    get_search_unavailable_reason,
    is_search_available,
)

__all__ = ["create_search_tool", "get_search_unavailable_reason", "is_search_available"]
