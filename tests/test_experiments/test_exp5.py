"""Integration test for Experiment 5 (MCTS simulation-count ablation)."""

from __future__ import annotations

import os

from experiments.exp5_mcts_ablation import run_exp5
from src.agents.random_agent import RandomAgent
from src.models.network import PolicyValueNetwork


def _tiny_net() -> PolicyValueNetwork:
    return PolicyValueNetwork(channels=8, num_blocks=1)


def test_exp5_produces_one_record_per_sim_count_and_writes_csv(tmp_path):
    records = run_exp5(
        _tiny_net(),
        sim_counts=(2, 8),
        opponent=RandomAgent(seed=1),
        n_games=2,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert [r.n_simulations for r in records] == [2, 8]
    assert all(r.games == 2 for r in records)
    assert all(r.avg_move_time_s > 0 for r in records)
    # More simulations should cost more time per move.
    assert records[1].avg_move_time_s >= records[0].avg_move_time_s
    assert os.path.exists(tmp_path / "exp5_mcts_ablation.csv")
