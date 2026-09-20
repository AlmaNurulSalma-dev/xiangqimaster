"""Experiment 1 — Main Performance Comparison (docs/09-EXPERIMENTS.md section 2).

Runs a colour-balanced round-robin among all supplied agents, fits relative Elo
ratings to the results, and writes the thesis tables:

* ``exp1_elo.csv``          — each agent's fitted Elo (+ score, games),
* ``exp1_head_to_head.csv`` — the agent-vs-agent win-rate matrix.

Absolute anchoring to ElephantEye/Pikafish is added later by including the
engine (wrapped as a BaseAgent) in the ``agents`` dict and re-centring on its
known rating; the code path is identical.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent
from src.evaluation import elo
from src.evaluation.metrics import MatchStats
from src.evaluation.tournament import round_robin


@dataclass
class Exp1Result:
    """Fitted Elo per agent plus the raw head-to-head statistics."""

    elo: dict[str, float]
    head_to_head: dict[tuple[str, str], MatchStats]


def run_exp1(
    agents: dict[str, BaseAgent],
    n_games: int = 400,
    *,
    seed: int | None = 0,
    anchor_mean: float = 1500.0,
    out_dir: str | None = None,
) -> Exp1Result:
    """Play the round-robin, fit Elo, and (optionally) write result tables."""
    if len(agents) < 2:
        raise ValueError("need at least two agents for a comparison")

    head_to_head = round_robin(agents, n_games, seed=seed)
    pairwise = [
        (name_a, name_b, stats.score, stats.games)
        for (name_a, name_b), stats in head_to_head.items()
    ]
    ratings = elo.fit_ratings(pairwise, anchor_mean=anchor_mean)
    result = Exp1Result(elo=ratings, head_to_head=head_to_head)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _write_elo_table(result, os.path.join(out_dir, "exp1_elo.csv"))
        _write_head_to_head(result, os.path.join(out_dir, "exp1_head_to_head.csv"))
    return result


def _aggregate_score(name: str, result: Exp1Result) -> tuple[float, int]:
    """A player's total score fraction and game count across the round-robin."""
    score_sum = 0.0
    games = 0
    for (a, b), stats in result.head_to_head.items():
        if a == name:
            score_sum += stats.score * stats.games
            games += stats.games
        elif b == name:
            score_sum += (1.0 - stats.score) * stats.games
            games += stats.games
    return (score_sum / games if games else 0.0), games


def _write_elo_table(result: Exp1Result, path: str) -> None:
    rows = []
    for name, rating in sorted(result.elo.items(), key=lambda kv: -kv[1]):
        score, games = _aggregate_score(name, result)
        ci = elo.elo_confidence_interval(score, games)
        rows.append((name, round(rating, 1), round(ci, 1), round(score, 3), games))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "elo", "elo_ci95", "score", "games"])
        writer.writerows(rows)


def _write_head_to_head(result: Exp1Result, path: str) -> None:
    names = sorted(result.elo)
    win_rate: dict[tuple[str, str], float] = {}
    for (a, b), stats in result.head_to_head.items():
        win_rate[(a, b)] = stats.win_rate
        win_rate[(b, a)] = 1.0 - stats.score  # B's score share
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent \\ opponent", *names])
        for a in names:
            row = [a]
            for b in names:
                row.append("" if a == b else round(win_rate.get((a, b), float("nan")), 3))
            writer.writerow(row)


__all__ = ["Exp1Result", "run_exp1"]
