"""Unit tests for the tournament manager."""

from __future__ import annotations

from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.evaluation import tournament
from src.utils.config import BLACK, RED


def test_play_game_returns_valid_winner_and_length():
    winner, plies = tournament.play_game(
        RandomAgent(seed=1), RandomAgent(seed=2), seed=0
    )
    assert winner in {RED, BLACK, None}
    assert plies > 0


def test_play_match_balances_colours():
    outcomes = tournament.play_match(
        RandomAgent(seed=1), RandomAgent(seed=2), n_games=4, seed=0
    )
    assert len(outcomes) == 4
    reds = sum(1 for o in outcomes if o.a_color == RED)
    blacks = sum(1 for o in outcomes if o.a_color == BLACK)
    assert reds == 2 and blacks == 2  # colours split evenly


def test_estimate_elo_structure():
    est = tournament.estimate_elo(
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        opponent_rating=1200.0,
        n_games=4,
        seed=0,
    )
    assert est.stats.games == 4
    assert est.confidence_interval > 0
    assert isinstance(est.rating, float)


def test_minimax_beats_random():
    # A depth-2 Minimax should score well above a random opponent.
    est = tournament.estimate_elo(
        MinimaxAgent(depth=2),
        RandomAgent(seed=3),
        opponent_rating=1000.0,
        n_games=6,
        seed=10,
    )
    assert est.stats.wins >= 1              # wins at least one game
    assert est.stats.score > 0.5           # clearly stronger than random
    assert est.rating > 1000.0             # estimated above the random baseline


def test_round_robin_covers_all_pairs():
    agents = {
        "rand_a": RandomAgent(seed=1),
        "rand_b": RandomAgent(seed=2),
        "rand_c": RandomAgent(seed=3),
    }
    results = tournament.round_robin(agents, n_games=2, seed=0)
    # 3 agents ⇒ 3 unordered pairs.
    assert len(results) == 3
    assert ("rand_a", "rand_b") in results
    for stats in results.values():
        assert stats.games == 2
