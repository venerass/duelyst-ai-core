"""Integration test configuration — real API calls."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Automatically mark all tests in this directory as integration tests."""
    for item in items:
        item.add_marker(pytest.mark.integration)
