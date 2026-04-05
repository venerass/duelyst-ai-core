"""Tests for the event callback protocol and built-in implementations."""

from __future__ import annotations

import pytest

from duelyst_ai_core.orchestrator.callbacks import (
    CollectorCallback,
    DebateEventCallback,
    NullCallback,
)
from duelyst_ai_core.orchestrator.events import (
    DebateStarted,
    RoundStarted,
    TurnStarted,
)
from duelyst_ai_core.orchestrator.state import DebateConfig, ModelConfig


@pytest.fixture
def debate_config() -> DebateConfig:
    return DebateConfig(
        topic="Test topic",
        model_a=ModelConfig(provider="anthropic", model_id="claude-haiku-4-5"),
        model_b=ModelConfig(provider="openai", model_id="gpt-5.4-mini"),
    )


class TestDebateEventCallbackProtocol:
    def test_null_callback_satisfies_protocol(self) -> None:
        assert isinstance(NullCallback(), DebateEventCallback)

    def test_collector_callback_satisfies_protocol(self) -> None:
        assert isinstance(CollectorCallback(), DebateEventCallback)

    def test_custom_class_satisfies_protocol(self) -> None:
        class MyCallback:
            async def on_event(self, event: object) -> None:
                pass

        assert isinstance(MyCallback(), DebateEventCallback)

    def test_missing_method_fails_protocol(self) -> None:
        class NotACallback:
            pass

        assert not isinstance(NotACallback(), DebateEventCallback)


class TestNullCallback:
    async def test_on_event_is_noop(self, debate_config: DebateConfig) -> None:
        cb = NullCallback()
        event = DebateStarted(config=debate_config)
        # Should not raise
        await cb.on_event(event)


class TestCollectorCallback:
    async def test_collects_events_in_order(self, debate_config: DebateConfig) -> None:
        cb = CollectorCallback()

        e1 = DebateStarted(config=debate_config)
        e2 = RoundStarted(round_number=1)
        e3 = TurnStarted(agent="a", round_number=1)

        await cb.on_event(e1)
        await cb.on_event(e2)
        await cb.on_event(e3)

        assert len(cb.events) == 3
        assert cb.events[0] is e1
        assert cb.events[1] is e2
        assert cb.events[2] is e3

    async def test_starts_empty(self) -> None:
        cb = CollectorCallback()
        assert cb.events == []
