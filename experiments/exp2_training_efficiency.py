"""Experiment 2 — Training Efficiency, RQ1 (docs/09-EXPERIMENTS.md section 3).

**The thesis's central experiment.** Trains two PPO agents while tracking Elo
over training:

* ``scratch``  — PPO from random init (Agent 1),
* ``il_init``  — PPO initialized from the imitation-learning body (Agent 2
  Phase 2).

Plots/records both learning curves so the two can be compared: does IL
pre-training start higher and reach a given Elo in fewer training steps?

Writes ``exp2_learning_curves.csv``: (variant, timesteps, elo).
"""

from __future__ import annotations

import csv
import os

from src.agents.random_agent import RandomAgent
from src.models.network import PolicyValueNetwork
from src.training.callbacks import EloEvalCallback
from src.training.ppo_train import build_model


def run_exp2(
    total_timesteps: int,
    *,
    il_network: PolicyValueNetwork | None,
    eval_freq: int,
    channels: int,
    num_blocks: int,
    n_steps: int = 256,
    batch_size: int = 64,
    n_eval_games: int = 10,
    opponent_rating: float = 1000.0,
    seed: int | None = 0,
    out_dir: str | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Train scratch-PPO and IL-initialized PPO, returning both Elo curves.

    ``il_network`` must share ``channels``/``num_blocks`` with the PPO body.
    If it is None, only the ``scratch`` curve is produced.
    """
    variants: list[tuple[str, PolicyValueNetwork | None]] = [("scratch", None)]
    if il_network is not None:
        variants.append(("il_init", il_network))

    curves: dict[str, list[tuple[int, float]]] = {}
    for label, il in variants:
        model = build_model(
            channels=channels,
            num_blocks=num_blocks,
            n_steps=n_steps,
            batch_size=batch_size,
            seed=seed,
            il_network=il,
        )
        callback = EloEvalCallback(
            RandomAgent(seed=seed),
            opponent_rating=opponent_rating,
            eval_freq=eval_freq,
            n_games=n_eval_games,
            seed=seed,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        curves[label] = callback.history

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(curves, os.path.join(out_dir, "exp2_learning_curves.csv"))
    return curves


def _write_csv(curves: dict[str, list[tuple[int, float]]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "timesteps", "elo"])
        for variant, points in curves.items():
            for timesteps, elo in points:
                writer.writerow([variant, timesteps, round(elo, 1)])


__all__ = ["run_exp2"]
