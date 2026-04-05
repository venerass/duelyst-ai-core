"""Tests for convergence detection — pure function, no mocking needed."""

from __future__ import annotations

from duelyst_ai_core.orchestrator.convergence import check_convergence


class TestCheckConvergence:
    def test_empty_history(self) -> None:
        assert check_convergence([], threshold=7, required_rounds=2) is False

    def test_insufficient_rounds(self) -> None:
        assert check_convergence([(8, 8)], threshold=7, required_rounds=2) is False

    def test_both_above_threshold(self) -> None:
        history = [(8, 9), (7, 8)]
        assert check_convergence(history, threshold=7, required_rounds=2) is True

    def test_one_below_threshold(self) -> None:
        history = [(8, 9), (6, 8)]
        assert check_convergence(history, threshold=7, required_rounds=2) is False

    def test_only_recent_rounds_matter(self) -> None:
        """Early rounds below threshold don't matter if recent ones converge."""
        history = [(2, 3), (4, 5), (8, 9), (7, 8)]
        assert check_convergence(history, threshold=7, required_rounds=2) is True

    def test_recent_regression(self) -> None:
        """If the last round dips below, not converged."""
        history = [(8, 9), (7, 8), (6, 5)]
        assert check_convergence(history, threshold=7, required_rounds=2) is False

    def test_exact_threshold(self) -> None:
        history = [(7, 7), (7, 7)]
        assert check_convergence(history, threshold=7, required_rounds=2) is True

    def test_single_round_required(self) -> None:
        history = [(8, 9)]
        assert check_convergence(history, threshold=7, required_rounds=1) is True

    def test_three_rounds_required(self) -> None:
        history = [(8, 8), (9, 9), (7, 8)]
        assert check_convergence(history, threshold=7, required_rounds=3) is True

    def test_three_rounds_not_met(self) -> None:
        history = [(8, 8), (6, 9), (7, 8)]
        assert check_convergence(history, threshold=7, required_rounds=3) is False
