"""Experiment 5 — MCTS Simulation-Count Ablation (docs/09-EXPERIMENTS.md 6).

Measures how MCTS strength and speed trade off against the number of
simulations per move. For each simulation count we run an MCTS agent against a
fixed opponent, recording its score / Elo and its average thinking time per
move — revealing the "sweet spot" for practical deployment (e.g. in the
dashboard, where a huge simulation budget would be too slow).

Writes ``exp5_mcts_ablation.csv``: (n_simulations, score, elo, avg_move_time_s).
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.environment.board import Board
from src.evaluation import elo
from src.evaluation.metrics import summarize
from src.evaluation.tournament import play_match
from src.models.network import PolicyValueNetwork


class _TimedAgent(BaseAgent):
    """Wraps an agent to record how long each ``select_move`` takes."""

    def __init__(self, agent: BaseAgent) -> None:
        self._agent = agent
        self.name = agent.name
        self.total_time = 0.0
        self.calls = 0

    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        start = time.perf_counter()
        action = self._agent.select_move(board, legal_mask)
        self.total_time += time.perf_counter() - start
        self.calls += 1
        return action

    def avg_time(self) -> float:
        return self.total_time / self.calls if self.calls else 0.0


@dataclass
class Exp5Record:
    """One row of the ablation: strength and speed at a simulation count."""

    n_simulations: int
    score: float
    elo: float
    avg_move_time_s: float
    games: int


def run_exp5(
    network: PolicyValueNetwork,
    *,
    sim_counts: tuple[int, ...] = (100, 400, 800, 1600),
    opponent: BaseAgent | None = None,
    opponent_rating: float = 1500.0,
    n_games: int = 40,
    seed: int | None = 0,
    out_dir: str | None = None,
) -> list[Exp5Record]:
    """Run the ablation over ``sim_counts`` and return one record per count."""
    opponent = opponent or RandomAgent(seed=seed)
    records: list[Exp5Record] = []

    for n_sims in sim_counts:
        agent = _TimedAgent(MCTSAgent(network, n_simulations=n_sims))
        outcomes = play_match(agent, opponent, n_games, seed=seed)
        stats = summarize(outcomes)
        rating = elo.estimate_rating_from_score(opponent_rating, stats.score)
        records.append(
            Exp5Record(
                n_simulations=n_sims,
                score=stats.score,
                elo=rating,
                avg_move_time_s=agent.avg_time(),
                games=n_games,
            )
        )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(records, os.path.join(out_dir, "exp5_mcts_ablation.csv"))
    return records


def _write_csv(records: list[Exp5Record], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n_simulations", "score", "elo", "avg_move_time_s", "games"])
        for r in records:
            writer.writerow(
                [r.n_simulations, round(r.score, 3), round(r.elo, 1),
                 round(r.avg_move_time_s, 4), r.games]
            )


__all__ = ["Exp5Record", "run_exp5"]
