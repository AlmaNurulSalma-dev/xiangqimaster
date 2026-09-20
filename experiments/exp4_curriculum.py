"""Experiment 4 — Curriculum vs Self-Play, RQ4 (docs/09 5, docs/07 5).

Trains a PPO agent against a *curriculum* of progressively stronger fixed
opponents instead of pure self-play. The agent advances to the next stage once
it reaches a win-rate threshold against the current opponent (or a per-stage
step budget is exhausted). Comparing the curriculum agent's final Elo and
training efficiency to standard self-play (Experiment 2's ``scratch`` run)
answers RQ4.

A graded opponent is required; ElephantEye at increasing depths is ideal, but
MinimaxAgent at increasing depth works with no external engine. Writes
``exp4_curriculum.csv``: (stage, opponent, timesteps, win_rate).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent
from src.agents.ppo_agent import PPOAgent
from src.evaluation.metrics import summarize
from src.evaluation.tournament import play_match
from src.training.ppo_train import build_model
from src.training.self_play_env import SelfPlayEnv


@dataclass
class StageRecord:
    """One evaluation checkpoint during curriculum training."""

    stage: int
    opponent: str
    timesteps: int
    win_rate: float


def _win_rate(model, opponent: BaseAgent, n_games: int, seed: int | None) -> float:
    outcomes = play_match(PPOAgent(model), opponent, n_games, seed=seed)
    return summarize(outcomes).win_rate


def run_exp4(
    stages: list[BaseAgent],
    *,
    channels: int,
    num_blocks: int,
    chunk_timesteps: int,
    max_timesteps_per_stage: int,
    win_threshold: float = 0.6,
    n_eval_games: int = 10,
    n_steps: int = 256,
    batch_size: int = 64,
    seed: int | None = 0,
    out_dir: str | None = None,
) -> list[StageRecord]:
    """Train PPO through a curriculum of opponents, recording progress."""
    if not stages:
        raise ValueError("curriculum needs at least one stage")

    model = build_model(
        env=SelfPlayEnv(opponent=stages[0]),
        channels=channels,
        num_blocks=num_blocks,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
    )

    history: list[StageRecord] = []
    total_timesteps = 0
    for index, opponent in enumerate(stages):
        model.set_env(SelfPlayEnv(opponent=opponent))
        stage_timesteps = 0
        while stage_timesteps < max_timesteps_per_stage:
            model.learn(chunk_timesteps, reset_num_timesteps=False)
            stage_timesteps += chunk_timesteps
            total_timesteps += chunk_timesteps
            win_rate = _win_rate(model, opponent, n_eval_games, seed)
            history.append(
                StageRecord(index, opponent.name, total_timesteps, win_rate)
            )
            if win_rate >= win_threshold:
                break  # mastered this stage → advance

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(history, os.path.join(out_dir, "exp4_curriculum.csv"))
    return history


def _write_csv(history: list[StageRecord], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "opponent", "timesteps", "win_rate"])
        for r in history:
            writer.writerow([r.stage, r.opponent, r.timesteps, round(r.win_rate, 3)])


__all__ = ["StageRecord", "run_exp4"]
