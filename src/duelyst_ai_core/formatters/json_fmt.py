"""JSON formatter for debate results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from duelyst_ai_core.formatters.base import BaseFormatter

if TYPE_CHECKING:
    from duelyst_ai_core.agents.schemas import DebateResult


class JsonFormatter(BaseFormatter):
    """Formats a DebateResult as indented JSON.

    Uses Pydantic's model_dump_json for reliable serialization.
    """

    def format(self, result: DebateResult) -> str:
        """Format a debate result as JSON.

        Args:
            result: The complete debate result.

        Returns:
            Pretty-printed JSON string.
        """
        return result.model_dump_json(indent=2)
