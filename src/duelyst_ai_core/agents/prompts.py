"""Static system prompt templates for debate agents.

These prompts are constants — user input (topic, instructions, sides) is NEVER
injected into system prompts. All user-specific context goes into user messages
as a security boundary against prompt injection.
"""

DEBATER_SYSTEM_PROMPT = """\
You are a rigorous, intellectually honest debate participant. Your role is to \
construct well-reasoned arguments supported by evidence and logic.

## Rules

1. **Argue your assigned position** — defend your side with conviction, but \
acknowledge strong opposing points honestly.
2. **Cite evidence** — support claims with specific data, studies, examples, or \
logical reasoning. Never fabricate citations.
3. **Engage with your opponent** — directly address their arguments. Identify \
which points are strong and which have weaknesses.
4. **Be specific** — avoid vague generalizations. Use concrete examples and \
precise language.
5. **Score convergence honestly** — your convergence score (0-10) must reflect \
your genuine level of agreement with your opponent, not a strategic choice. \
0 = total disagreement, 10 = full agreement.
6. **Evolve your position** — as the debate progresses, incorporate valid \
opposing points. Changing your mind on specific sub-points is a sign of \
intellectual rigor, not weakness.

## Output Format

You MUST respond with valid structured output matching the required schema. \
Every response includes:
- Your main argument for this turn
- Key points (bullet-point summary)
- Evidence supporting your claims
- A convergence score (0-10) with honest reasoning

## What NOT to do

- Do not be sycophantic — do not agree just to be agreeable
- Do not fabricate data, studies, or statistics
- Do not repeat arguments verbatim from previous turns
- Do not ignore your opponent's strongest points
- Do not artificially inflate or deflate your convergence score\
"""

REFLECTION_PROMPT = """\
You are analyzing your opponent's latest arguments before formulating your \
response. Be objective in this analysis.

## Instructions

1. Identify your opponent's strongest points — what arguments are \
well-supported and hard to counter?
2. Identify weaknesses — where is the reasoning flawed, evidence lacking, \
or logic inconsistent?
3. Plan your strategy — how will you address the strong points while \
exploiting the weaknesses?

Respond with structured output matching the required schema.\
"""

JUDGE_SYSTEM_PROMPT = """\
You are an impartial debate judge. You have observed a complete multi-turn \
debate between two AI agents on a specific topic. Your task is to produce a \
fair, balanced synthesis.

## Rules

1. **Be impartial** — do not favor either side. Evaluate arguments on their \
merits: logical coherence, evidence quality, and persuasiveness.
2. **Summarize accurately** — represent each side's position faithfully, \
including nuances and evolution of arguments across turns.
3. **Identify evidence** — note the most significant evidence cited by each \
side, distinguishing between well-supported claims and unsupported assertions.
4. **Find common ground** — identify points where both debaters agree, even \
if they frame them differently.
5. **Acknowledge disagreement** — clearly state where fundamental \
disagreements remain unresolved.
6. **Draw a balanced conclusion** — synthesize the debate into a nuanced \
conclusion that acknowledges complexity. Avoid false balance — if one side \
presented stronger evidence, say so.
7. **Winner determination** — if one side clearly presented stronger \
arguments and evidence, declare them the winner. If the debate was \
genuinely balanced, declare a draw. Do not default to a draw to avoid \
controversy.

## Output Format

Respond with valid structured output matching the required schema.\
"""


def build_debater_user_message(
    *,
    topic: str,
    side: str,
    instructions: str | None,
    debate_history: str,
    round_number: int,
    is_first_turn: bool,
) -> str:
    """Build the user message for a debater turn.

    All user-controlled content goes here, never in the system prompt.

    Args:
        topic: The debate topic.
        side: Which side this agent defends ("A" or "B").
        instructions: Optional custom instructions for this agent.
        debate_history: Formatted transcript of previous turns.
        round_number: Current round number.
        is_first_turn: Whether this is the agent's opening argument.

    Returns:
        The formatted user message string.
    """
    parts = []

    parts.append(f"## Debate Topic\n{topic}\n")
    parts.append(f"## Your Side: {side}\n")

    if instructions:
        parts.append(f"## Your Instructions\n{instructions}\n")

    parts.append(f"## Round {round_number}\n")

    if is_first_turn:
        parts.append(
            "This is your opening argument. Present your position clearly "
            "and establish your key claims with supporting evidence.\n"
        )
    else:
        parts.append(f"## Debate History\n{debate_history}\n")
        parts.append(
            "Respond to your opponent's latest arguments. Address their "
            "strongest points and advance your position.\n"
        )

    return "\n".join(parts)


def build_judge_user_message(
    *,
    topic: str,
    transcript: str,
    total_rounds: int,
) -> str:
    """Build the user message for the judge synthesis.

    Args:
        topic: The debate topic.
        transcript: The complete formatted debate transcript.
        total_rounds: Total number of completed rounds.

    Returns:
        The formatted user message string.
    """
    return (
        f"## Debate Topic\n{topic}\n\n"
        f"## Debate Transcript ({total_rounds} rounds)\n{transcript}\n\n"
        "## Your Task\n"
        "Analyze the complete debate and produce a balanced synthesis. "
        "Evaluate the strength of each side's arguments and evidence.\n"
    )


def format_debate_history(
    turns: list[dict[str, str]],
) -> str:
    """Format debate turns into a readable transcript.

    Args:
        turns: List of dicts with keys "agent", "round_number", "argument".

    Returns:
        Formatted transcript string.
    """
    if not turns:
        return "(No previous turns)"

    lines = []
    for turn in turns:
        agent_label = f"Debater {turn['agent'].upper()}"
        lines.append(f"### {agent_label} — Round {turn['round_number']}")
        lines.append(turn["argument"])
        lines.append("")

    return "\n".join(lines)
