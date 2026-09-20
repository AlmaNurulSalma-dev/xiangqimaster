"""Experiment 3 — Opening Theory Analysis (docs/09-EXPERIMENTS.md section 4).

For each agent, play games, collect their opening moves, and compare the
resulting opening distribution to a reference (professional) distribution via
KL divergence — answering RQ3 (do agents rediscover classical openings?).

Writes ``exp3_openings.csv``: (agent, kl_vs_reference, most_common_opening_class).
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.opening_analysis import (
    Signature,
    build_distribution,
    classify_opening,
    collect_agent_openings,
    kl_divergence,
)


@dataclass
class AgentOpeningReport:
    """One agent's opening behaviour."""

    distribution: dict[Signature, float]
    class_counts: dict[str, int]
    kl_vs_reference: float | None = None

    @property
    def most_common_class(self) -> str:
        if not self.class_counts:
            return "none"
        return max(self.class_counts, key=self.class_counts.get)


def run_exp3(
    agents: dict[str, BaseAgent],
    *,
    reference_distribution: dict[Signature, float] | None = None,
    opponent: BaseAgent | None = None,
    n_games: int = 100,
    n_plies: int = 6,
    seed: int | None = 0,
    out_dir: str | None = None,
) -> dict[str, AgentOpeningReport]:
    """Analyse each agent's openings and (optionally) write the result table."""
    opponent = opponent or RandomAgent(seed=seed)
    reports: dict[str, AgentOpeningReport] = {}

    for name, agent in agents.items():
        signatures = collect_agent_openings(
            agent, opponent, n_games=n_games, n_plies=n_plies, seed=seed
        )
        distribution = build_distribution(signatures)
        class_counts = dict(Counter(classify_opening(sig) for sig in signatures))
        kl = (
            kl_divergence(distribution, reference_distribution)
            if reference_distribution is not None
            else None
        )
        reports[name] = AgentOpeningReport(
            distribution=distribution,
            class_counts=class_counts,
            kl_vs_reference=kl,
        )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(reports, os.path.join(out_dir, "exp3_openings.csv"))
    return reports


def _write_csv(reports: dict[str, AgentOpeningReport], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "kl_vs_reference", "most_common_opening_class"])
        for name, report in reports.items():
            kl = "" if report.kl_vs_reference is None else round(report.kl_vs_reference, 4)
            writer.writerow([name, kl, report.most_common_class])


__all__ = ["AgentOpeningReport", "run_exp3"]
