"""Abstract base formatter for debate results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import DebateResult


class BaseFormatter(ABC):
    """Abstract base for all output formatters.

    Subclasses transform a DebateResult into a specific format
    (Markdown, JSON, Rich terminal).
    """

    @abstractmethod
    def format(self, result: DebateResult) -> str:
        """Format a debate result into a string.

        Args:
            result: The complete debate result.

        Returns:
            Formatted string representation.
        """
