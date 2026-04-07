"""Tests for prompt templates and message builders."""

from __future__ import annotations

from duelyst_ai_core.agents.prompts import (
    DEBATER_SYSTEM_PROMPT,
    DEBATER_SYSTEM_PROMPT_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    build_debater_system_prompt,
    build_debater_user_message,
    build_judge_user_message,
    format_debate_history,
)


class TestSystemPrompts:
    def test_debater_prompt_is_static(self) -> None:
        assert isinstance(DEBATER_SYSTEM_PROMPT, str)
        assert len(DEBATER_SYSTEM_PROMPT) > 100

    def test_judge_prompt_is_static(self) -> None:
        assert isinstance(JUDGE_SYSTEM_PROMPT, str)
        assert len(JUDGE_SYSTEM_PROMPT) > 100

    def test_no_user_input_in_system_prompts(self) -> None:
        """System prompts must not contain format placeholders."""
        # The static DEBATER_SYSTEM_PROMPT (default) should have no placeholders
        assert "{" not in DEBATER_SYSTEM_PROMPT
        assert "}" not in DEBATER_SYSTEM_PROMPT
        for prompt in [JUDGE_SYSTEM_PROMPT]:
            assert "{" not in prompt
            assert "}" not in prompt

    def test_debater_template_has_placeholder(self) -> None:
        assert "{agent_label}" in DEBATER_SYSTEM_PROMPT_TEMPLATE

    def test_build_debater_system_prompt_with_identity(self) -> None:
        prompt = build_debater_system_prompt("Debater A")
        assert "Debater A" in prompt
        assert "arrive as close to the truth" in prompt
        assert "{" not in prompt
        assert "}" not in prompt

    def test_build_debater_system_prompt_different_agents(self) -> None:
        prompt_a = build_debater_system_prompt("Debater A")
        prompt_b = build_debater_system_prompt("Debater B")
        assert "Debater A" in prompt_a
        assert "Debater B" in prompt_b
        assert "Debater B" not in prompt_a
        assert "Debater A" not in prompt_b

    def test_debater_prompt_contains_purpose(self) -> None:
        prompt = build_debater_system_prompt("Debater A")
        assert "truth" in prompt
        assert "evidence and reasoning" in prompt


class TestBuildDebaterUserMessage:
    def test_first_turn(self) -> None:
        msg = build_debater_user_message(
            topic="Monoliths vs microservices",
            side="A",
            instructions="Defend monoliths",
            debate_history="",
            round_number=1,
            is_first_turn=True,
        )
        assert "Monoliths vs microservices" in msg
        assert "Defend monoliths" in msg
        assert "opening argument" in msg
        assert "Round 1" in msg

    def test_subsequent_turn(self) -> None:
        msg = build_debater_user_message(
            topic="Test topic",
            side="B",
            instructions=None,
            debate_history="Previous debate content...",
            round_number=3,
            is_first_turn=False,
        )
        assert "Test topic" in msg
        assert "Previous debate content..." in msg
        assert "Respond to your opponent" in msg
        assert "Instructions" not in msg

    def test_no_instructions(self) -> None:
        msg = build_debater_user_message(
            topic="Topic",
            side="A",
            instructions=None,
            debate_history="",
            round_number=1,
            is_first_turn=True,
        )
        assert "Instructions" not in msg


class TestBuildJudgeUserMessage:
    def test_contains_topic_and_transcript(self) -> None:
        msg = build_judge_user_message(
            topic="AI debate",
            transcript="Full transcript here...",
            total_rounds=5,
        )
        assert "AI debate" in msg
        assert "Full transcript here..." in msg
        assert "5 rounds" in msg
        assert "balanced synthesis" in msg


class TestFormatDebateHistory:
    def test_empty_turns(self) -> None:
        result = format_debate_history([])
        assert "No previous turns" in result

    def test_single_turn(self) -> None:
        turns = [
            {"agent": "a", "round_number": "1", "argument": "Monoliths are better."},
        ]
        result = format_debate_history(turns)
        assert "Debater A" in result
        assert "Round 1" in result
        assert "Monoliths are better." in result

    def test_multiple_turns(self) -> None:
        turns = [
            {"agent": "a", "round_number": "1", "argument": "Arg A1"},
            {"agent": "b", "round_number": "1", "argument": "Arg B1"},
            {"agent": "a", "round_number": "2", "argument": "Arg A2"},
        ]
        result = format_debate_history(turns)
        assert "Debater A" in result
        assert "Debater B" in result
        assert "Arg A1" in result
        assert "Arg B1" in result
        assert "Arg A2" in result

    def test_with_evidence(self) -> None:
        turns = [
            {
                "agent": "a",
                "round_number": "1",
                "argument": "My argument",
                "evidence": [
                    {"claim": "Fact one", "source": "https://example.com"},
                    {"claim": "Fact two", "source": None},
                ],
            },
        ]
        result = format_debate_history(turns)
        assert "My argument" in result
        assert "Sources cited" in result
        assert "Fact one" in result
        assert "https://example.com" in result
        assert "Fact two" in result

    def test_without_evidence(self) -> None:
        turns = [
            {"agent": "a", "round_number": "1", "argument": "No sources"},
        ]
        result = format_debate_history(turns)
        assert "No sources" in result
        assert "Sources cited" not in result

    def test_empty_evidence_list(self) -> None:
        turns = [
            {"agent": "a", "round_number": "1", "argument": "Arg", "evidence": []},
        ]
        result = format_debate_history(turns)
        assert "Sources cited" not in result
