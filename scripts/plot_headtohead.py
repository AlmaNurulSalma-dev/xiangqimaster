"""Plot the head-to-head win-rate matrix as a heatmap (for the proposal/thesis).

Reads a round-robin head-to-head CSV (agent x opponent, cell = row agent's mean
score vs the column opponent) and renders a diverging heatmap, in the style of
the payoff-matrix figures used by recent multi-agent Xiangqi papers.

Usage::

    <venv-python> scripts/plot_headtohead.py \
        results/exp1_agent3/exp1_head_to_head.csv \
        research/proposal/figures/headtohead.png
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/exp1_agent3/exp1_head_to_head.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "research/proposal/figures/headtohead.png"

SHORT = {"MCTS_a3": "MCTS", "MCTS_v2": "MCTS", "IL_only": "IL",
         "ILPPO_rand": "IL+PPO", "Minimax_d2": "Minimax", "Random": "Random"}

with open(CSV) as f:
    reader = csv.reader(f)
    header = next(reader)
    opponents = header[1:]
    agents, matrix = [], []
    for row in reader:
        agents.append(row[0])
        matrix.append([np.nan if c == "" else float(c) for c in row[1:]])

M = np.array(matrix)
rlab = [SHORT.get(a, a) for a in agents]
clab = [SHORT.get(a, a) for a in opponents]

plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.3, 3.5))
im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=1.0)
ax.set_xticks(range(len(clab)))
ax.set_xticklabels(clab, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(rlab)))
ax.set_yticklabels(rlab, fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j] - 0.5) > 0.35 else "black")
ax.set_xlabel("opponent", fontsize=9)
ax.set_ylabel("agent", fontsize=9)
cb = fig.colorbar(im, fraction=0.046, pad=0.04)
cb.set_label("win rate (row vs column)", fontsize=8)
cb.ax.tick_params(labelsize=7)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
