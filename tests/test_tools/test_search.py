"""Tests for the web search tool."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from duelyst_ai_core.exceptions import ConfigError, ToolError
from duelyst_ai_core.tools.search import (
    create_search_tool,
    get_search_unavailable_reason,
    is_search_available,
)


class TestCreateSearchTool:
    def test_missing_api_key(self) -> None:
        """Raises ConfigError when TAVILY_API_KEY is not set."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="TAVILY_API_KEY"),
        ):
            create_search_tool()

    def test_missing_package(self) -> None:
        """Raises ToolError when search dependencies are not installed."""
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {"langchain_tavily": None},
            ),
            pytest.raises(ToolError, match=r"duelyst-ai-core\[search\]"),
        ):
            create_search_tool()

    def test_creates_tool_with_api_key(self) -> None:
        """Creates a tool when API key and package are available."""
        tool_kwargs: dict[str, object] = {}

        class MockTool:
            def __init__(self, **kwargs: object) -> None:
                tool_kwargs.update(kwargs)

        mock_module = type(sys)("langchain_tavily")
        mock_module.TavilySearch = MockTool  # type: ignore[attr-defined]

        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {"langchain_tavily": mock_module},
            ),
        ):
            tool = create_search_tool()
            assert tool is not None
            assert tool_kwargs == {"max_results": 5, "topic": "general"}


class TestIsSearchAvailable:
    def test_unavailable_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert is_search_available() is False

    def test_unavailable_without_package(self) -> None:
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {"langchain_tavily": None},
            ),
        ):
            assert is_search_available() is False


class TestSearchUnavailableReason:
    def test_reports_missing_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            reason = get_search_unavailable_reason()
            assert reason is not None
            assert "TAVILY_API_KEY" in reason

    def test_reports_missing_dependencies(self) -> None:
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {"langchain_tavily": None},
            ),
        ):
            reason = get_search_unavailable_reason()
            assert reason is not None
            assert "langchain-tavily" in reason
            assert "duelyst-ai-core[search]" in reason
