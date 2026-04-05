"""Event callback protocol and built-in implementations.

Defines the interface for consuming debate events emitted by the orchestrator.
Both the CLI (Rich Live display) and external consumers (e.g., FastAPI SSE
endpoints) implement this protocol.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from duelyst_ai_core.orchestrator.events import DebateEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class DebateEventCallback(Protocol):
    """Protocol for receiving debate events from the orchestrator.

    Implement ``on_event`` to process events as they occur during a debate.
    """

    async def on_event(self, event: DebateEvent) -> None:
        """Handle a single debate event.

        Args:
            event: The debate event that occurred.
        """
        ...  # pragma: no cover


class NullCallback:
    """No-op callback used when no consumer is provided.

    Avoids ``if callback:`` guards in every orchestrator node.
    """

    async def on_event(self, event: DebateEvent) -> None:
        """Silently discard the event."""


class CollectorCallback:
    """Callback that buffers events in a list for testing and batch processing.

    Attributes:
        events: List of collected events in emission order.
    """

    def __init__(self) -> None:
        self.events: list[DebateEvent] = []

    async def on_event(self, event: DebateEvent) -> None:
        """Append the event to the internal list.

        Args:
            event: The debate event to collect.
        """
        self.events.append(event)
