"""Unit tests for match summary metrics."""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import DRAW, GameOutcome, LOSS, WIN, summarize
from src.utils.config import BLACK, RED


def test_summarize_basic_counts_and_rates():
    outcomes = [
        GameOutcome(WIN, plies=50, a_color=RED),
        GameOutcome(LOSS, plies=60, a_color=BLACK),
        GameOutcome(DRAW, plies=300, a_color=RED),
        GameOutcome(WIN, plies=40, a_color=BLACK),
    ]
    stats = summarize(outcomes)
    assert stats.games == 4
    assert stats.wins == 2 and stats.losses == 1 and stats.draws == 1
    assert math.isclose(stats.score, (2 + 0.5) / 4)         # 0.625
    assert math.isclose(stats.win_rate, 0.5)
    assert math.isclose(stats.draw_rate, 0.25)
    assert math.isclose(stats.decisiveness, 0.75)
    assert math.isclose(stats.avg_game_length, (50 + 60 + 300 + 40) / 4)


def test_per_color_win_rates():
    outcomes = [
        GameOutcome(WIN, plies=10, a_color=RED),    # red: 1 win
        GameOutcome(LOSS, plies=10, a_color=RED),   # red: 1 loss
        GameOutcome(WIN, plies=10, a_color=BLACK),  # black: 1 win
    ]
    stats = summarize(outcomes)
    assert math.isclose(stats.win_rate_as_red, 0.5)
    assert math.isclose(stats.win_rate_as_black, 1.0)


def test_empty_match_raises():
    with pytest.raises(ValueError):
        summarize([])
