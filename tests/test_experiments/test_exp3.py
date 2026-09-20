"""Integration test for Experiment 3 (opening analysis), on quick agents."""

from __future__ import annotations

import math
import os

from experiments.exp3_opening_theory import run_exp3
from src.agents.random_agent import RandomAgent


def test_exp3_produces_reports_and_writes_csv(tmp_path):
    agents = {"rand_a": RandomAgent(seed=1), "rand_b": RandomAgent(seed=2)}
    # A tiny reference distribution (one "book" opening).
    reference = {((3, 0, 4, 0),): 1.0}

    reports = run_exp3(
        agents,
        reference_distribution=reference,
        opponent=RandomAgent(seed=3),
        n_games=4,
        n_plies=4,
        seed=0,
        out_dir=str(tmp_path),
    )

    assert set(reports) == set(agents)
    for report in reports.values():
        assert math.isclose(sum(report.distribution.values()), 1.0) or not report.distribution
        assert report.kl_vs_reference is not None
        assert report.most_common_class  # some class label
    assert os.path.exists(tmp_path / "exp3_openings.csv")
