"""Method-flow diagrams and result charts for the consultation document."""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_consult_figures import (  # noqa: E402
    OUT, INK, ACC, RED, TEAL, AMB, GREY, VIO, LIGHT, save, blank, box, arrow, plt)
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_elo(rel):
    rows = []
    with open(os.path.join(ROOT, rel), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["agent"], float(r["elo"]), float(r["elo_ci95"])))
    return rows


# ============================================================ M1 Minimax ====
def fig_m1():
    fig, ax = plt.subplots(figsize=(7.4, 3.5)); blank(ax)
    ax.text(0.0, 0.95, "M1   MINIMAX WITH ALPHA-BETA PRUNING", fontsize=9.5,
            fontweight="bold", color=INK)
    ax.text(0.0, 0.875, "Pure search. No learning, no data, no network.",
            fontsize=8, color=GREY)
    nodes = {"root": (0.5, 0.74), "a": (0.26, 0.50), "b": (0.74, 0.50),
             "a1": (0.14, 0.26), "a2": (0.38, 0.26), "b1": (0.62, 0.26),
             "b2": (0.86, 0.26)}
    for k, (x, y) in nodes.items():
        pruned = k == "b2"
        c = "#d9dde3" if pruned else (LIGHT if k in ("root", "a1", "a2", "b1") else "#fdecea")
        ax.add_patch(Circle((x, y), 0.045, fc=c, ec=GREY if pruned else INK, lw=1.1, zorder=3))
    ax.text(0.5, 0.74, "MAX", ha="center", va="center", fontsize=7.2, zorder=4, fontweight="bold")
    for k in ("a", "b"):
        ax.text(nodes[k][0], nodes[k][1], "MIN", ha="center", va="center",
                fontsize=7.2, zorder=4, fontweight="bold", color=RED)
    for k, v in (("a1", "+2"), ("a2", "-1"), ("b1", "+5")):
        ax.text(nodes[k][0], nodes[k][1], v, ha="center", va="center", fontsize=7.4, zorder=4)
    ax.text(nodes["b2"][0], nodes["b2"][1], "cut", ha="center", va="center",
            fontsize=6.8, zorder=4, color=GREY)
    for p, c in (("a", "root"), ("b", "root"), ("a1", "a"), ("a2", "a"),
                 ("b1", "b"), ("b2", "b")):
        x1, y1 = nodes[p]; x2, y2 = nodes[c]
        dashed = p == "b2"
        ax.plot([x1, x2], [y1, y2], color=GREY if dashed else INK,
                lw=1.0, ls=":" if dashed else "-", zorder=1)
    ax.plot([nodes["root"][0], nodes["a"][0]], [nodes["root"][1], nodes["a"][1]],
            color=TEAL, lw=2.4, zorder=2)
    ax.plot([nodes["a"][0], nodes["a1"][0]], [nodes["a"][1], nodes["a1"][1]],
            color=TEAL, lw=2.4, zorder=2)
    ax.text(0.5, 0.60, "chosen line\n(best worst-case)", fontsize=6.9, color=TEAL,
            ha="center", va="center", linespacing=1.4)
    ax.text(0.86, 0.145, "alpha-beta prunes\nbranches that cannot\nchange the result",
            fontsize=6.8, color=GREY, ha="center", va="top", linespacing=1.4)
    box(ax, 0.02, 0.015, 0.30, 0.095,
        "Hand-written evaluation:\nmaterial balance only", fc="white", ec=RED, fs=7.2)
    box(ax, 0.355, 0.015, 0.30, 0.095,
        "Depth d = 1, 2, 3 plies\n(the only tunable knob)", fc="white", ec=RED, fs=7.2)
    box(ax, 0.69, 0.015, 0.29, 0.095,
        "Strength comes entirely\nfrom search depth", fc="white", ec=RED, fs=7.2)
    save(fig, "f05_m1_minimax.png")


# ================================================= M2 Imitation Learning ====
def fig_m2():
    fig, ax = plt.subplots(figsize=(8.8, 2.9)); blank(ax)
    ax.text(0.0, 0.94, "M2   IMITATION LEARNING  (supervised; learns from human experts)",
            fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 0.005, 0.34, 0.145, 0.40,
        "141,356 human\ngames\n11.6 M positions", fc="#e8edfb", ec=ACC, bold=True, fs=7.7)
    box(ax, 0.175, 0.34, 0.125, 0.40, "Encode each\nposition\n14 x 10 x 9", fc=LIGHT, fs=7.7)
    box(ax, 0.325, 0.34, 0.145, 0.40, "Policy-value\nnetwork\n(C64 / N10)", fc="#e8edfb", fs=7.7)
    box(ax, 0.495, 0.56, 0.145, 0.19, "Predicted move\ndistribution", fc="#e6f4f1", ec=TEAL, fs=7.7)
    box(ax, 0.495, 0.34, 0.145, 0.19, "Predicted\noutcome", fc="#fdf0e3", ec=AMB, fs=7.7)
    box(ax, 0.675, 0.56, 0.15, 0.19, "Cross-entropy\nvs. human move", fc="white", ec=TEAL, fs=7.7)
    box(ax, 0.675, 0.34, 0.15, 0.19, "MSE vs. game\nresult", fc="white", ec=AMB, fs=7.7)
    box(ax, 0.855, 0.45, 0.14, 0.19, "Gradient update\n(Adam, lr 1e-3)", fc="#f3f4f6", ec=INK, bold=True, fs=7.7)
    for p in ((0.15, 0.54, 0.175, 0.54), (0.30, 0.54, 0.325, 0.54)):
        arrow(ax, (p[0], p[1]), (p[2], p[3]))
    arrow(ax, (0.47, 0.58), (0.495, 0.655), c=TEAL)
    arrow(ax, (0.47, 0.50), (0.495, 0.435), c=AMB)
    arrow(ax, (0.64, 0.655), (0.675, 0.655), c=TEAL)
    arrow(ax, (0.64, 0.435), (0.675, 0.435), c=AMB)
    arrow(ax, (0.825, 0.655), (0.855, 0.58)); arrow(ax, (0.825, 0.435), (0.855, 0.50))
    ax.add_patch(FancyArrowPatch((0.925, 0.45), (0.40, 0.30), connectionstyle="arc3,rad=0.30",
                 arrowstyle="-|>", mutation_scale=11, color=GREY, lw=1.1, ls="--"))
    ax.text(0.66, 0.235, "weights updated; repeat for 30 epochs over 9.3 M training positions",
            fontsize=7.2, color=GREY, ha="center")
    ax.text(0.0, 0.07, "The objective is to reproduce the move a human master actually played. "
            "The agent therefore inherits human opening theory and\nstrategic habits, but it can never "
            "exceed the quality of the games it was shown.",
            fontsize=7.4, color="#444a52", linespacing=1.5)
    save(fig, "f06_m2_imitation.png")


# ============================================== M3 Self-play RL with PPO ====
def fig_m3():
    fig, ax = plt.subplots(figsize=(7.6, 3.5)); blank(ax)
    ax.text(0.0, 0.95, "M3   SELF-PLAY REINFORCEMENT LEARNING (PPO)  --  no human data",
            fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 0.30, 0.70, 0.40, 0.17,
        "Agent plays a full game against a copy of itself\n(illegal moves removed by the action mask)",
        fc="#fdf2e6", ec=AMB, fs=7.7)
    box(ax, 0.72, 0.42, 0.27, 0.17, "Terminal reward\n+1 win / -1 loss / 0 draw", fc="white", ec=AMB, bold=True, fs=7.7)
    box(ax, 0.30, 0.17, 0.40, 0.19,
        "PPO update (clipped surrogate objective)\nlr 3e-4, n_steps 2048, batch 512,\n4 epochs, clip 0.2, gamma 0.99",
        fc="#fdf2e6", ec=AMB, fs=7.7)
    box(ax, 0.01, 0.42, 0.25, 0.17, "Updated policy\nand value function", fc="white", ec=AMB, fs=7.7)
    ax.add_patch(FancyArrowPatch((0.70, 0.78), (0.855, 0.59), connectionstyle="arc3,rad=-0.3",
                 arrowstyle="-|>", mutation_scale=12, color=AMB, lw=1.4))
    ax.add_patch(FancyArrowPatch((0.855, 0.42), (0.70, 0.26), connectionstyle="arc3,rad=-0.3",
                 arrowstyle="-|>", mutation_scale=12, color=AMB, lw=1.4))
    ax.add_patch(FancyArrowPatch((0.30, 0.26), (0.135, 0.42), connectionstyle="arc3,rad=-0.3",
                 arrowstyle="-|>", mutation_scale=12, color=AMB, lw=1.4))
    ax.add_patch(FancyArrowPatch((0.135, 0.59), (0.30, 0.78), connectionstyle="arc3,rad=-0.3",
                 arrowstyle="-|>", mutation_scale=12, color=AMB, lw=1.4))
    ax.text(0.5, 0.52, "repeat for\n500,000 steps", ha="center", va="center",
            fontsize=8.2, color=AMB, fontweight="bold", linespacing=1.4)
    ax.text(0.0, 0.035, "Two initialisations are compared: M3a starts from random weights (tabula rasa), "
            "M3b starts from the imitation-learned\nnetwork (M2). The difference isolates whether human "
            "pretraining helps or hinders subsequent self-play.",
            fontsize=7.4, color="#444a52", linespacing=1.5)
    save(fig, "f07_m3_selfplay.png")


# ================================================= M4 MCTS plus network ====
def fig_m4():
    fig, ax = plt.subplots(figsize=(8.8, 3.6)); blank(ax)
    ax.text(0.0, 0.95, "M4   MONTE CARLO TREE SEARCH GUIDED BY THE NETWORK  (AlphaZero recipe)",
            fontsize=9.5, fontweight="bold", color=INK)
    tree = {"r": (0.20, 0.72), "c1": (0.10, 0.50), "c2": (0.22, 0.50), "c3": (0.33, 0.50),
            "g1": (0.185, 0.29), "g2": (0.265, 0.29)}
    for k, (x, y) in tree.items():
        ax.add_patch(Circle((x, y), 0.028, fc=LIGHT if k != "g2" else "#efe9f8",
                            ec=VIO, lw=1.1, zorder=3))
    for p, c, w in (("r", "c1", 1.0), ("r", "c2", 2.8), ("r", "c3", 1.0),
                    ("c2", "g1", 2.4), ("c2", "g2", 1.0)):
        x1, y1 = tree[p]; x2, y2 = tree[c]
        ax.plot([x1, x2], [y1, y2], color=VIO, lw=w, zorder=1, alpha=0.85)
    ax.text(0.20, 0.80, "root position", fontsize=7.2, ha="center", color=INK)
    ax.text(0.265, 0.235, "new leaf", fontsize=6.8, ha="center", color=VIO)
    ax.text(0.20, 0.145, "edge thickness = visit count", fontsize=6.9, ha="center", color=GREY)
    steps = [("1  SELECT", "descend the tree by PUCT:\nQ + c * P * sqrt(N) / (1 + n),\nwith c_puct = 1.5"),
             ("2  EXPAND", "add the leaf node reached\nand list its legal moves"),
             ("3  EVALUATE", "the network returns move\npriors P and a value v for\nthe leaf (no rollout)"),
             ("4  BACK UP", "propagate v up the path;\ngood branches accumulate\nvisits")]
    for i, (t, d) in enumerate(steps):
        y = 0.70 - i * 0.175
        box(ax, 0.42, y, 0.265, 0.145, d, fc="#f6f1fc", ec=VIO, fs=7.0)
        ax.text(0.425, y + 0.165, t, fontsize=7.6, fontweight="bold", color=VIO)
    box(ax, 0.72, 0.52, 0.27, 0.19,
        "Repeat for n simulations\n(100 in training, 48 in the\nlive demo)", fc="white", ec=VIO, fs=7.4)
    box(ax, 0.72, 0.26, 0.27, 0.19,
        "Play the MOST VISITED move\n(not the highest-value one)", fc="#efe9f8", ec=VIO, bold=True, fs=7.4)
    arrow(ax, (0.855, 0.52), (0.855, 0.455), c=VIO)
    arrow(ax, (0.355, 0.50), (0.42, 0.55), c=VIO)
    ax.text(0.0, 0.055, "The network supplies prior intuition and a position evaluation; the tree search "
            "verifies that intuition by look-ahead. During training the\nvisit distribution itself becomes the "
            "policy target, and Dirichlet noise (alpha = 0.3, epsilon = 0.25) at the root preserves exploration.",
            fontsize=7.4, color="#444a52", linespacing=1.5)
    save(fig, "f08_m4_mcts.png")


# ============================================================ Elo charts ====
LABEL = {"MCTS_v2": "M4  MCTS + network", "MCTS_a3": "M4  MCTS + network",
         "IL_only": "M2  Imitation Learning", "IL_human": "M2  Imitation Learning",
         "Minimax_d2": "M1  Minimax (d = 2)", "ILPPO_rand": "M3b  IL then PPO",
         "ILPPO_self": "M3b  IL then PPO (self)", "PPO_s_rand": "M3a  PPO from scratch",
         "PPO_s_self": "M3a  PPO from scratch (self)", "Random": "Random baseline",
         "Pikafish_d8": "Pikafish engine (depth 8)", "Distill_pika": "Engine-distilled net",
         "Distill_small": "Distilled net (1.15 M)", "Distill_big": "Distilled net (4.85 M)"}


def _bars(ax, rows, title, highlight=None):
    rows = sorted(rows, key=lambda r: r[1])
    names = [LABEL.get(r[0], r[0]) for r in rows]
    vals = [r[1] for r in rows]
    cis = [r[2] for r in rows]
    cols = []
    for r in rows:
        if highlight and r[0] in highlight:
            cols.append(AMB)
        elif r[0].startswith("MCTS"):
            cols.append(VIO)
        elif r[0].startswith("IL_"):
            cols.append(ACC)
        elif r[0].startswith("Minimax"):
            cols.append(RED)
        elif "PPO" in r[0]:
            cols.append(TEAL)
        else:
            cols.append(GREY)
    y = np.arange(len(rows))
    ax.barh(y, vals, xerr=cis, color=cols, height=0.62, alpha=0.9,
            error_kw=dict(ecolor="#555b63", lw=1.0, capsize=3))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v + cis[i] + 14, i, f"{v:.0f}", va="center", fontsize=7.8, color=INK)
    ax.axvline(1500, color=GREY, lw=0.9, ls="--")
    ax.set_xlabel("Relative Elo (anchor = 1500, error bars = 95% CI)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
    ax.set_xlim(1050, max(vals) + max(cis) + 150)
    ax.grid(axis="x", color="#e8eaee", lw=0.7)
    ax.set_axisbelow(True)


def fig_elo():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    _bars(axes[0], read_elo("results/exp1_agent3_v2/exp1_elo.csv"),
          "(a)  Main league: the four paradigms")
    rows = read_elo("results/exp_distill/exp1_elo.csv")
    _bars(axes[1], rows, "(b)  Engine-anchored league", highlight={"Pikafish_d8"})
    axes[1].annotate("", xy=(2062, 4.46), xytext=(1490, 4.46),
                     arrowprops=dict(arrowstyle="<->", color=AMB, lw=1.2))
    axes[1].text(1776, 4.30, "about 570 Elo", ha="center", va="top", fontsize=7.6,
                 color=AMB, fontweight="bold")
    fig.tight_layout()
    save(fig, "f09_elo.png")


# ====================================================== ceiling evidence ====
def fig_ceiling():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    ax = axes[0]
    sims, score = [], []
    with open(os.path.join(ROOT, "results/exp5_mcts_v2/exp5_mcts_ablation.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sims.append(int(r["n_simulations"])); score.append(float(r["score_vs_IL"]))
    ax.plot(sims, score, "o-", color=VIO, lw=1.8, ms=7)
    ax.axhline(0.5, color=GREY, ls="--", lw=0.9)
    ax.text(820, 0.515, "parity with M2", fontsize=7.2, color=GREY, ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(sims)
    ax.set_xticklabels([str(s) for s in sims])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("MCTS simulations per move (log scale)", fontsize=8)
    ax.set_ylabel("Score vs. the M2 network", fontsize=8)
    ax.set_title("(a)  Eight-fold more search does not help", fontsize=9,
                 fontweight="bold", loc="left")
    ax.grid(color="#e8eaee", lw=0.7); ax.set_axisbelow(True)
    ax = axes[1]
    rows = {a: (e, c) for a, e, c in read_elo("results/exp_bignet/exp1_elo.csv")}
    keys = ["Distill_small", "Distill_big", "IL_human", "Minimax_d2"]
    labs = ["Distilled\n1.15 M params", "Distilled\n4.85 M params",
            "Imitation\n1.15 M params", "Minimax\n(no network)"]
    vals = [rows[k][0] for k in keys]; cis = [rows[k][1] for k in keys]
    cols = [GREY, AMB, ACC, RED]
    ax.bar(range(4), vals, yerr=cis, color=cols, width=0.6, alpha=0.9,
           error_kw=dict(ecolor="#555b63", lw=1.0, capsize=3))
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7.4)
    ax.set_ylim(1400, 1720)
    ax.set_ylabel("Relative Elo (same tournament)", fontsize=8)
    ax.set_title("(b)  Four times the parameters: no significant gain", fontsize=9,
                 fontweight="bold", loc="left")
    for i, v in enumerate(vals):
        ax.text(i, v + cis[i] + 6, f"{v:.0f}", ha="center", fontsize=7.8)
    ax.grid(axis="y", color="#e8eaee", lw=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "f10_ceiling.png")


# ========================================================== compute gap ====
def fig_compute():
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    labs = ["AlphaZero (Silver et al., 2018)\n5,000 + 64 TPUs",
            "ELF OpenGo (Tian et al., 2019)\napprox. 2,000 GPUs",
            "This project\n1 laptop GPU (RTX 4060, 8 GB)"]
    vals = [5064, 2000, 1]
    cols = [GREY, GREY, ACC]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=cols, height=0.55, alpha=0.92)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
    for yy, v in zip(y, vals):
        ax.text(v * 1.4, yy, f"{v:,}", va="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xlim(0.5, 40000)
    ax.set_xlabel("Hardware accelerators used (log scale)", fontsize=8)
    ax.set_title("The compute gap is three to four orders of magnitude",
                 fontsize=9, fontweight="bold", loc="left")
    ax.grid(axis="x", color="#e8eaee", lw=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "f11_compute.png")


# ============================================================= accuracy ====
def fig_accuracy():
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    phases = ["Overall", "Opening\n(ply < 20)", "Midgame\n(20-59)", "Endgame\n(>= 60)"]
    top1 = [42.3, 63.2, 37.7, 32.3]
    top3 = [66.5, 87.8, 61.8, 56.5]
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, top1, w, label="Top-1 (legal-masked)", color=ACC, alpha=0.92)
    ax.bar(x + w / 2, top3, w, label="Top-3 (legal-masked)", color=TEAL, alpha=0.92)
    for xi, v in zip(x - w / 2, top1):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.4)
    for xi, v in zip(x + w / 2, top3):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.4)
    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=8)
    ax.set_ylabel("Human-move prediction accuracy (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    ax.set_title("M2 on the held-out test split (163,416 positions)",
                 fontsize=9, fontweight="bold", loc="left")
    ax.grid(axis="y", color="#e8eaee", lw=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "f12_accuracy.png")


# ============================================================= timeline ====
def fig_timeline():
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    months = ["Sep\n2026", "Oct", "Nov", "Dec", "Jan\n2027", "Feb", "Mar", "Apr", "May"]
    tasks = [
        ("Environment, rules, encoder, action space", 0, 1.0, True),
        ("Four agents implemented", 0.5, 1.5, True),
        ("Training runs: IL, PPO, MCTS", 1.0, 1.4, True),
        ("Engine integration and distillation", 1.6, 0.8, True),
        ("Experiments and Elo league (six studies)", 1.3, 1.3, True),
        ("Proposal, progress review, live demo", 2.0, 1.0, True),
        ("Advisor consultation and protocol freeze", 2.7, 0.8, False),
        ("Scale-up study: capacity and self-play budget", 3.2, 2.0, False),
        ("Absolute-strength calibration vs. engine ladder", 4.6, 1.3, False),
        ("Interactive platform: human vs. model + explanations", 4.2, 2.0, False),
        ("Thesis writing", 5.6, 2.3, False),
        ("Revision, rehearsal, defense", 7.4, 1.4, False),
    ]
    for i, (name, start, dur, done) in enumerate(tasks):
        y = len(tasks) - i - 1
        ax.barh(y, dur, left=start, height=0.56,
                color=TEAL if done else ACC, alpha=0.85 if done else 0.62,
                edgecolor="white")
        ax.text(start + dur + 0.08, y, "completed" if done else "planned",
                va="center", fontsize=6.8, color=TEAL if done else ACC)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks][::-1], fontsize=7.8)
    ax.set_xticks(range(9)); ax.set_xticklabels(months, fontsize=7.8)
    ax.set_xlim(-0.1, 10.1)
    ax.axvline(2.15, color=RED, lw=1.4, ls="--")
    ax.text(2.22, len(tasks) - 0.35, "today\n5 Nov 2026", fontsize=7.4, color=RED,
            fontweight="bold", va="top")
    ax.axvline(8.9, color=INK, lw=1.4)
    ax.text(8.82, len(tasks) - 0.35, "defense\nend of May 2027", fontsize=7.4,
            color=INK, fontweight="bold", va="top", ha="right")
    ax.grid(axis="x", color="#e8eaee", lw=0.7); ax.set_axisbelow(True)
    ax.set_title("Project schedule: completed work and the plan to the defense",
                 fontsize=9.5, fontweight="bold", loc="left")
    fig.tight_layout()
    save(fig, "f13_timeline.png")


if __name__ == "__main__":
    fig_m1(); fig_m2(); fig_m3(); fig_m4()
    fig_elo(); fig_ceiling(); fig_compute(); fig_accuracy(); fig_timeline()
