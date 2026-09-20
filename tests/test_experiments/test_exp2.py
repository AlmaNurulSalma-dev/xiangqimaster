"""Integration smoke test for Experiment 2 (training efficiency / RQ1)."""

from __future__ import annotations

import os

from experiments.exp2_training_efficiency import run_exp2
from src.models.network import PolicyValueNetwork


def test_exp2_produces_both_learning_curves(tmp_path):
    il = PolicyValueNetwork(channels=8, num_blocks=1)
    curves = run_exp2(
        total_timesteps=64,
        il_network=il,
        eval_freq=32,
        channels=8,
        num_blocks=1,
        n_steps=32,
        batch_size=16,
        n_eval_games=2,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert set(curves) == {"scratch", "il_init"}
    assert all(len(points) > 0 for points in curves.values())
    # Each recorded point is (timesteps, elo).
    for points in curves.values():
        for timesteps, elo in points:
            assert timesteps > 0 and isinstance(elo, float)
    assert os.path.exists(tmp_path / "exp2_learning_curves.csv")
