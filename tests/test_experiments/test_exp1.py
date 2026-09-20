"""Integration test for Experiment 1 (main comparison), on quick agents."""

from __future__ import annotations

import os

from experiments.exp1_main_comparison import run_exp1
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent


def test_exp1_ranks_minimax_above_random_and_writes_tables(tmp_path):
    agents = {
        "minimax": MinimaxAgent(depth=1),
        "random_a": RandomAgent(seed=1),
        "random_b": RandomAgent(seed=2),
    }
    result = run_exp1(agents, n_games=4, seed=0, out_dir=str(tmp_path))

    # Every agent got a rating; Minimax should top the field.
    assert set(result.elo) == set(agents)
    assert result.elo["minimax"] == max(result.elo.values())

    # Result tables were written.
    assert os.path.exists(tmp_path / "exp1_elo.csv")
    assert os.path.exists(tmp_path / "exp1_head_to_head.csv")


def test_exp1_requires_two_agents():
    import pytest

    with pytest.raises(ValueError):
        run_exp1({"only": RandomAgent(seed=0)}, n_games=2)
