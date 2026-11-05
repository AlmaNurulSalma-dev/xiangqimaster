"""Column-width figures and composites for the two-column IEEE build."""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_consult_figures import (  # noqa: E402
    OUT, INK, ACC, RED, TEAL, AMB, GREY, VIO, LIGHT, save, blank, box, arrow, plt)
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Circle  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PF = os.path.join(ROOT, "research", "figures", "proposal_src")


# -------------------------------------------------------- composites ------
def hcombine(paths, out_name, pad=18, bg=(255, 255, 255), target_h=None):
    ims = [PILImage.open(os.path.join(ROOT, p)).convert("RGB") for p in paths]
    h = target_h or max(i.height for i in ims)
    sc = [i.resize((max(1, int(i.width * h / i.height)), h), PILImage.LANCZOS) for i in ims]
    W = sum(i.width for i in sc) + pad * (len(sc) - 1)
    canvas = PILImage.new("RGB", (W, h), bg)
    x = 0
    for i in sc:
        canvas.paste(i, (x, 0)); x += i.width + pad
    p = os.path.join(OUT, out_name)
    canvas.save(p, quality=95)
    print("  wrote", out_name, canvas.size)


# ----------------------------------------------------- M1 column ---------
def c_m1():
    fig, ax = plt.subplots(figsize=(3.4, 3.0)); blank(ax)
    nodes = {"r": (0.50, 0.86), "a": (0.26, 0.60), "b": (0.74, 0.60),
             "a1": (0.12, 0.34), "a2": (0.40, 0.34), "b1": (0.62, 0.34), "b2": (0.88, 0.34)}
    for k, (x, y) in nodes.items():
        pr = k == "b2"
        fc = "#dfe3e9" if pr else (LIGHT if k not in ("a", "b") else "#fdecea")
        ax.add_patch(Circle((x, y), 0.055, fc=fc, ec=GREY if pr else INK, lw=1.0, zorder=3))
    ax.text(0.50, 0.86, "MAX", ha="center", va="center", fontsize=6.4, fontweight="bold", zorder=4)
    for k in ("a", "b"):
        ax.text(*nodes[k], "MIN", ha="center", va="center", fontsize=6.4,
                fontweight="bold", color=RED, zorder=4)
    for k, v in (("a1", "+2"), ("a2", "-1"), ("b1", "+5")):
        ax.text(*nodes[k], v, ha="center", va="center", fontsize=6.6, zorder=4)
    ax.text(*nodes["b2"], "cut", ha="center", va="center", fontsize=6.0, color=GREY, zorder=4)
    for p, c in (("a", "r"), ("b", "r"), ("a1", "a"), ("a2", "a"), ("b1", "b"), ("b2", "b")):
        x1, y1 = nodes[p]; x2, y2 = nodes[c]
        d = p == "b2"
        ax.plot([x1, x2], [y1, y2], color=GREY if d else INK, lw=0.9,
                ls=":" if d else "-", zorder=1)
    for p, c in (("a", "r"), ("a1", "a")):
        x1, y1 = nodes[p]; x2, y2 = nodes[c]
        ax.plot([x1, x2], [y1, y2], color=TEAL, lw=2.2, zorder=2)
    ax.text(0.50, 0.715, "selected line", fontsize=6.3, color=TEAL, ha="center")
    ax.text(0.88, 0.255, "pruned", fontsize=6.2, color=GREY, ha="center")
    box(ax, 0.02, 0.135, 0.96, 0.085, "Leaf score = hand-written material balance", fc="white", ec=RED, fs=6.8)
    box(ax, 0.02, 0.025, 0.96, 0.085, "Only tunable knob: search depth d = 1, 2, 3", fc="white", ec=RED, fs=6.8)
    save(fig, "c05_m1.png")


# ----------------------------------------------------- M2 column ---------
def c_m2():
    fig, ax = plt.subplots(figsize=(3.4, 3.9)); blank(ax)
    ys = [0.86, 0.70, 0.54, 0.345, 0.185, 0.045]
    box(ax, 0.06, ys[0], 0.70, 0.105, "141,356 human games\n11.6 M positions", fc="#e8edfb", ec=ACC, fs=7.0, bold=True)
    box(ax, 0.06, ys[1], 0.70, 0.105, "Encode position\n14 x 10 x 9 tensor", fc=LIGHT, ec=ACC, fs=7.0)
    box(ax, 0.06, ys[2], 0.70, 0.105, "Shared policy-value network\n(C = 64, N = 10)", fc="#e8edfb", ec=ACC, fs=7.0)
    box(ax, 0.02, ys[3], 0.44, 0.115, "Policy output\n2086 logits", fc="#e6f4f1", ec=TEAL, fs=6.9)
    box(ax, 0.52, ys[3], 0.44, 0.115, "Value output\nscalar", fc="#fdf0e3", ec=AMB, fs=6.9)
    box(ax, 0.02, ys[4], 0.44, 0.115, "Cross-entropy vs.\nthe human move", fc="white", ec=TEAL, fs=6.9)
    box(ax, 0.52, ys[4], 0.44, 0.115, "MSE vs. the\ngame result", fc="white", ec=AMB, fs=6.9)
    box(ax, 0.18, ys[5], 0.64, 0.095, "Gradient update (Adam, lr 1e-3)", fc="#f3f4f6", ec=INK, fs=7.0, bold=True)
    arrow(ax, (0.41, ys[0]), (0.41, ys[1] + 0.105))
    arrow(ax, (0.41, ys[1]), (0.41, ys[2] + 0.105))
    arrow(ax, (0.33, ys[2]), (0.24, ys[3] + 0.115), c=TEAL)
    arrow(ax, (0.49, ys[2]), (0.74, ys[3] + 0.115), c=AMB)
    arrow(ax, (0.24, ys[3]), (0.24, ys[4] + 0.115), c=TEAL)
    arrow(ax, (0.74, ys[3]), (0.74, ys[4] + 0.115), c=AMB)
    arrow(ax, (0.24, ys[4]), (0.36, ys[5] + 0.095))
    arrow(ax, (0.74, ys[4]), (0.62, ys[5] + 0.095))
    ax.set_xlim(0, 1.16)
    ax.add_patch(FancyArrowPatch((1.00, 0.092), (1.00, 0.592),
                 connectionstyle="arc3,rad=-0.55", arrowstyle="-|>",
                 mutation_scale=9, color=GREY, lw=1.0, ls="--"))
    ax.text(1.13, 0.342, "repeat 30 epochs", fontsize=6.4, color=GREY, rotation=90,
            ha="center", va="center")
    save(fig, "c06_m2.png")


# ----------------------------------------------------- M3 column ---------
def c_m3():
    fig, ax = plt.subplots(figsize=(3.4, 3.2)); blank(ax)
    box(ax, 0.10, 0.80, 0.80, 0.135, "Play one game against\na copy of itself\n(illegal moves masked)",
        fc="#fdf2e6", ec=AMB, fs=7.0)
    box(ax, 0.10, 0.565, 0.80, 0.115, "Terminal reward\n+1 win / -1 loss / 0 draw", fc="white", ec=AMB, fs=7.0, bold=True)
    box(ax, 0.10, 0.315, 0.80, 0.155, "PPO clipped update\nlr 3e-4, 2048 steps, batch 512,\n4 epochs, clip 0.2, gamma 0.99",
        fc="#fdf2e6", ec=AMB, fs=7.0)
    box(ax, 0.10, 0.125, 0.80, 0.105, "Updated policy and value", fc="white", ec=AMB, fs=7.0)
    arrow(ax, (0.50, 0.80), (0.50, 0.68), c=AMB)
    arrow(ax, (0.50, 0.565), (0.50, 0.47), c=AMB)
    arrow(ax, (0.50, 0.315), (0.50, 0.23), c=AMB)
    ax.add_patch(FancyArrowPatch((0.09, 0.178), (0.09, 0.868),
                 connectionstyle="arc3,rad=0.52", arrowstyle="-|>",
                 mutation_scale=11, color=AMB, lw=1.3))
    ax.text(0.015, 0.52, "repeat for 500,000 steps", fontsize=6.6, color=AMB,
            rotation=90, ha="center", va="center", fontweight="bold")
    ax.text(0.5, 0.045, "M3a starts from random weights; M3b starts from M2",
            fontsize=6.6, color="#444a52", ha="center")
    save(fig, "c07_m3.png")


# ----------------------------------------------------- M4 column ---------
def c_m4():
    fig, ax = plt.subplots(figsize=(3.4, 3.9)); blank(ax)
    tree = {"r": (0.50, 0.93), "c1": (0.28, 0.80), "c2": (0.50, 0.80), "c3": (0.72, 0.80),
            "g1": (0.42, 0.67), "g2": (0.60, 0.67)}
    for k, (x, y) in tree.items():
        ax.add_patch(Circle((x, y), 0.030, fc="#efe9f8" if k == "g2" else LIGHT,
                            ec=VIO, lw=1.0, zorder=3))
    for p, c, w in (("r", "c1", 0.9), ("r", "c2", 2.6), ("r", "c3", 0.9),
                    ("c2", "g1", 2.2), ("c2", "g2", 0.9)):
        x1, y1 = tree[p]; x2, y2 = tree[c]
        ax.plot([x1, x2], [y1, y2], color=VIO, lw=w, zorder=1, alpha=0.85)
    ax.text(0.5, 0.615, "edge thickness = visit count", fontsize=6.3, color=GREY, ha="center")
    steps = [("1  SELECT", "descend by PUCT: Q + c P sqrt(N)/(1+n),  c = 1.5"),
             ("2  EXPAND", "add the leaf and list its legal moves"),
             ("3  EVALUATE", "network returns priors P and value v (no rollout)"),
             ("4  BACK UP", "propagate v along the path; increment visits")]
    y = 0.47
    for t, d in steps:
        box(ax, 0.03, y, 0.94, 0.093, d, fc="#f6f1fc", ec=VIO, fs=6.5)
        ax.text(0.055, y + 0.103, t, fontsize=6.6, fontweight="bold", color=VIO)
        y -= 0.125
    box(ax, 0.03, 0.015, 0.94, 0.078, "Repeat n simulations, then play the most-visited move",
        fc="#efe9f8", ec=VIO, fs=6.8, bold=True)
    save(fig, "c08_m4.png")


# ------------------------------------------------- compute (column) ------
def c_compute():
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    labs = ["AlphaZero [11]\n5,000 + 64 TPUs", "ELF OpenGo [18]\nabout 2,000 GPUs",
            "This project\n1 laptop GPU"]
    vals = [5064, 2000, 1]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=[GREY, GREY, ACC], height=0.55, alpha=0.92)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.8)
    for yy, v in zip(y, vals):
        ax.text(v * 1.5, yy, f"{v:,}", va="center", fontsize=7.0, fontweight="bold")
    ax.set_xlim(0.5, 60000)
    ax.set_xlabel("Accelerators (log scale)", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.grid(axis="x", color="#e8eaee", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "c11_compute.png")


# ------------------------------------------------ accuracy (column) ------
def c_accuracy():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    phases = ["Overall", "Opening", "Midgame", "Endgame"]
    top1 = [42.3, 63.2, 37.7, 32.3]; top3 = [66.5, 87.8, 61.8, 56.5]
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, top1, w, label="Top-1", color=ACC, alpha=0.92)
    ax.bar(x + w / 2, top3, w, label="Top-3", color=TEAL, alpha=0.92)
    for xi, v in zip(x - w / 2, top1):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.4)
    for xi, v in zip(x + w / 2, top3):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=7.0)
    ax.set_ylabel("Accuracy (%)", fontsize=7.2)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.6, frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="y", color="#e8eaee", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "c12_accuracy.png")


if __name__ == "__main__":
    c_m1(); c_m2(); c_m3(); c_m4(); c_compute(); c_accuracy()
    hcombine(["research/proposal/figures/gfig1_board.png",
              "research/proposal/figures/gfig2_moves.png"],
             "x01_board_moves.png", target_h=941)
    hcombine(["research/figures/sources/src_cglemon_repo.jpg",
              "research/figures/sources/src_cglemon_readme.jpg"],
             "x09_sources.png", target_h=455)
