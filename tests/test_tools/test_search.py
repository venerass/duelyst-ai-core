"""Tests for the web search tool."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from duelyst_ai_core.exceptions import ConfigError, ToolError
from duelyst_ai_core.tools.search import create_search_tool, is_search_available


class TestCreateSearchTool:
    def test_missing_api_key(self) -> None:
        """Raises ConfigError when TAVILY_API_KEY is not set."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigError, match="TAVILY_API_KEY"),
        ):
            create_search_tool()

    def test_missing_package(self) -> None:
        """Raises ToolError when tavily-python is not installed."""
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {"langchain_community.tools.tavily_search": None},
            ),
            pytest.raises(ToolError, match="tavily-python"),
        ):
            create_search_tool()

    def test_creates_tool_with_api_key(self) -> None:
        """Creates a tool when API key and package are available."""
        mock_tool_cls = type("MockTool", (), {"__init__": lambda self, **kw: None})

        mock_module = type(sys)("langchain_community.tools.tavily_search")
        mock_module.TavilySearchResults = mock_tool_cls  # type: ignore[attr-defined]

        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {"langchain_community.tools.tavily_search": mock_module},
            ),
        ):
            tool = create_search_tool()
            assert tool is not None


class TestIsSearchAvailable:
    def test_unavailable_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert is_search_available() is False

    def test_unavailable_without_package(self) -> None:
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {"langchain_community.tools.tavily_search": None},
            ),
        ):
            assert is_search_available() is False
