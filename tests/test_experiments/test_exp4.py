"""Integration smoke test for Experiment 4 (curriculum training)."""

from __future__ import annotations

import os

from experiments.exp4_curriculum import run_exp4
from src.agents.random_agent import RandomAgent


def test_exp4_runs_curriculum_and_writes_csv(tmp_path):
    # Two "stages" (random opponents of increasing seed) exercise the loop.
    stages = [RandomAgent(seed=1, name="stage1"), RandomAgent(seed=2, name="stage2")]
    history = run_exp4(
        stages,
        channels=8,
        num_blocks=1,
        chunk_timesteps=32,
        max_timesteps_per_stage=32,   # one chunk per stage → advances immediately
        win_threshold=2.0,            # unreachable → always uses the full budget
        n_eval_games=2,
        n_steps=32,
        batch_size=16,
        seed=0,
        out_dir=str(tmp_path),
    )
    # One record per stage (budget = one chunk, threshold never met).
    assert [r.stage for r in history] == [0, 1]
    assert all(0.0 <= r.win_rate <= 1.0 for r in history)
    assert history[0].opponent == "stage1"
    assert os.path.exists(tmp_path / "exp4_curriculum.csv")


def test_exp4_rejects_empty_curriculum():
    import pytest

    with pytest.raises(ValueError):
        run_exp4([], channels=8, num_blocks=1, chunk_timesteps=32,
                 max_timesteps_per_stage=32)
