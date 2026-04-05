"""Convergence detection — pure function for checking debate convergence."""

from __future__ import annotations


def check_convergence(
    history: list[tuple[int, int]],
    threshold: int,
    required_rounds: int,
) -> bool:
    """Check if both agents have converged for the required number of rounds.

    Convergence means both agents score at or above the threshold for
    `required_rounds` consecutive rounds.

    Args:
        history: List of (score_a, score_b) tuples, one per completed round.
        threshold: Minimum score (0-10) for convergence.
        required_rounds: Number of consecutive rounds both must meet threshold.

    Returns:
        True if converged, False otherwise.
    """
    if len(history) < required_rounds:
        return False

    return all(a >= threshold and b >= threshold for a, b in history[-required_rounds:])
