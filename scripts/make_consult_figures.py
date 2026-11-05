"""Publication-quality figures for the advisor consultation document.

Run:  python scripts/make_consult_figures.py
Out:  research/figures/consult/*.png
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = os.path.join("research", "figures", "consult")
os.makedirs(OUT, exist_ok=True)

INK = "#1f2430"; ACC = "#2b5fd9"; RED = "#c0392b"; TEAL = "#0d9488"
AMB = "#b8860b"; GREY = "#9aa0a8"; VIO = "#6d4aa7"; LIGHT = "#eef2fb"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "savefig.dpi": 220, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
CJK = {"fontname": "Microsoft YaHei"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("  wrote", name)


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(ax, x, y, w, h, text, fc=LIGHT, ec=ACC, fs=8.2, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.012",
                 fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc or INK,
            linespacing=1.5, zorder=3)


def arrow(ax, p1, p2, c=None, lw=1.25, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=10,
                 color=c or "#6b7280", lw=lw, linestyle=ls,
                 shrinkA=1, shrinkB=1, zorder=1))


# ---------------------------------------------------------------- board ----
BACK_R = "車馬相仕帥仕相馬車"
BACK_B = "車馬象士將士象馬車"
START = []
for _c, _ch in enumerate(BACK_R):
    START.append((_c, 0, _ch, True))
for _c, _ch in enumerate(BACK_B):
    START.append((_c, 9, _ch, False))
for _c in (1, 7):
    START.append((_c, 2, "炮", True))
    START.append((_c, 7, "砲", False))
for _c in (0, 2, 4, 6, 8):
    START.append((_c, 3, "兵", True))
    START.append((_c, 6, "卒", False))


def board_grid(ax, river_text=True):
    for r in range(10):
        ax.plot([0, 8], [r, r], color=INK, lw=0.8, zorder=1)
    for c in range(9):
        if c in (0, 8):
            ax.plot([c, c], [0, 9], color=INK, lw=0.8, zorder=1)
        else:
            ax.plot([c, c], [0, 4], color=INK, lw=0.8, zorder=1)
            ax.plot([c, c], [5, 9], color=INK, lw=0.8, zorder=1)
    for y0 in (0, 7):
        ax.plot([3, 5], [y0, y0 + 2], color=INK, lw=0.8, zorder=1)
        ax.plot([5, 3], [y0, y0 + 2], color=INK, lw=0.8, zorder=1)
    ax.add_patch(Rectangle((-0.22, -0.22), 8.44, 9.44, fill=False, ec=INK, lw=1.6))
    if river_text:
        ax.text(2, 4.5, "楚  河", ha="center", va="center", fontsize=12,
                color="#555b63", **CJK)
        ax.text(6, 4.5, "漢  界", ha="center", va="center", fontsize=12,
                color="#555b63", **CJK)


def piece(ax, c, r, ch, is_red, rad=0.40, fs=10):
    col = RED if is_red else "#20303f"
    ax.add_patch(Circle((c, r), rad, fc="#fdf6e6", ec=col, lw=1.5, zorder=4))
    ax.text(c, r, ch, ha="center", va="center", fontsize=fs, color=col,
            fontweight="bold", zorder=5, **CJK)


def fig_board():
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    board_grid(ax)
    for c, r, ch, red in START:
        piece(ax, c, r, ch, red)
    ax.add_patch(Rectangle((3, 0), 2, 2, fc=AMB, alpha=0.11, zorder=0))
    ax.add_patch(Rectangle((3, 7), 2, 2, fc=AMB, alpha=0.11, zorder=0))
    ax.add_patch(Rectangle((-0.22, 4), 8.44, 1, fc=ACC, alpha=0.07, zorder=0))
    ax.annotate("Palace: the General and the two\nAdvisors never leave these 9 points",
                xy=(5.1, 1.0), xytext=(9.1, 1.1), fontsize=7.4, color=AMB,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=AMB, lw=1))
    ax.annotate("The river: a Soldier that crosses it\ngains a sideways step; an Elephant\nmay never cross",
                xy=(8.3, 4.5), xytext=(9.1, 5.0), fontsize=7.4, color=ACC,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1))
    ax.annotate("Pieces stand ON the intersections,\nnot inside the squares\n(9 x 10 = 90 playable points)",
                xy=(0, 6.0), xytext=(-5.6, 7.0), fontsize=7.4, color=INK,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
    ax.text(4, -0.95, "RED  (moves first)", ha="center", fontsize=8.6,
            color=RED, fontweight="bold")
    ax.text(4, 9.9, "BLACK", ha="center", fontsize=8.6, color="#20303f",
            fontweight="bold")
    ax.set_xlim(-8.6, 14.8); ax.set_ylim(-1.5, 10.4)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "f01_board.png")


# ---------------------------------------------------------- piece moves ----
def mini(ax, title, rule, dots, piece_ch, is_red=True, extra=None, cross=None,
         lines=None, n=6):
    for i in range(n + 1):
        ax.plot([0, n], [i, i], color="#c9ced6", lw=0.7, zorder=1)
        ax.plot([i, i], [0, n], color="#c9ced6", lw=0.7, zorder=1)
    cx = cy = n // 2
    if lines:
        for (dx, dy, ln) in lines:
            ax.plot([cx, cx + dx * ln], [cy, cy + dy * ln], color=TEAL, lw=1.8,
                    alpha=0.7, zorder=2)
    for (dx, dy) in dots:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.17, fc=TEAL, ec="none",
                            alpha=0.95, zorder=3))
    if cross:
        for (dx, dy) in cross:
            ax.plot([cx + dx - 0.2, cx + dx + 0.2], [cy + dy - 0.2, cy + dy + 0.2],
                    color=RED, lw=1.6, zorder=6)
            ax.plot([cx + dx - 0.2, cx + dx + 0.2], [cy + dy + 0.2, cy + dy - 0.2],
                    color=RED, lw=1.6, zorder=6)
    if extra:
        for (dx, dy, ch, red) in extra:
            piece(ax, cx + dx, cy + dy, ch, red, rad=0.33, fs=8)
    piece(ax, cx, cy, piece_ch, is_red, rad=0.36, fs=9)
    ax.set_title(title, fontsize=8.6, fontweight="bold", pad=4, **CJK)
    ax.text(n / 2, -0.75, rule, ha="center", va="top", fontsize=7.0,
            color="#444a52", linespacing=1.4)
    ax.set_xlim(-0.6, n + 0.6); ax.set_ylim(-2.9, n + 0.6)
    ax.set_aspect("equal"); ax.axis("off")


def fig_moves():
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.4))
    a = axes.ravel()
    mini(a[0], "General  帥 / 將",
         "One step orthogonally, and it never\nleaves the palace. The two Generals\nmay not face each other down an\nopen file.",
         [(0, 1), (0, -1), (1, 0), (-1, 0)], "帥")
    mini(a[1], "Advisor  仕 / 士",
         "One step diagonally, also confined\nto the palace. A purely defensive\npiece.",
         [(1, 1), (1, -1), (-1, 1), (-1, -1)], "仕")
    mini(a[2], "Elephant  相 / 象",
         "Exactly two points diagonally, and\nblocked if the midpoint is occupied.\nIt may never cross the river, so it\ndefends only its own half.",
         [(2, 2), (2, -2), (-2, -2)], "相",
         extra=[(-1, 1, "兵", True)], cross=[(-2, 2)])
    mini(a[3], "Horse  馬 / 僌",
         "One step orthogonally then one\ndiagonally outward. Unlike the chess\nknight it is blocked if the orthogonal\nleg is occupied.",
         [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1)], "馬",
         extra=[(0, 1, "卒", False)], cross=[(-1, 2)])
    mini(a[4], "Chariot  車 / 俵",
         "Any distance orthogonally until\nblocked. The strongest piece, and\nthe direct equivalent of the rook.",
         [], "車", lines=[(1, 0, 3), (-1, 0, 3), (0, 1, 3), (0, -1, 3)])
    mini(a[5], "Cannon  炮 / 砲",
         "Moves like a Chariot, but captures\nONLY by jumping exactly one\nintervening piece (the screen).\nThis has no analogue in chess.",
         [], "炮", lines=[(-1, 0, 3), (0, -1, 3)],
         extra=[(0, 1, "兵", True), (0, 2, "卒", False)])
    mini(a[6], "Soldier  兵 (own half)",
         "One step forward only, and it can\nnever retreat.",
         [(0, 1)], "兵")
    mini(a[7], "Soldier  兵 (past the river)",
         "After crossing the river it also gains\na sideways step. This single rule is\nwhat makes river control strategically\ndecisive.",
         [(0, 1), (1, 0), (-1, 0)], "兵")
    fig.subplots_adjust(hspace=0.62, wspace=0.10)
    save(fig, "f02_moves.png")


# -------------------------------------------------------- architecture ----
def fig_arch():
    fig, ax = plt.subplots(figsize=(8.8, 2.6)); blank(ax)
    box(ax, 0.005, 0.34, 0.145, 0.42,
        "Board state\n14 x 10 x 9\nbinary planes", fc="#f3f4f6", ec=GREY, bold=True, fs=7.9)
    box(ax, 0.195, 0.34, 0.125, 0.42, "Input conv\n3 x 3, C = 64", fc=LIGHT, fs=7.9)
    box(ax, 0.365, 0.28, 0.195, 0.54,
        "Residual tower\nN = 10 blocks\n(conv-BN-ReLU x2\nplus skip connection)", fc="#e8edfb", fs=7.9)
    box(ax, 0.615, 0.56, 0.185, 0.26, "Policy head\n2086 logits", fc="#e6f4f1", ec=TEAL, fs=7.9)
    box(ax, 0.615, 0.24, 0.185, 0.26, "Value head\nscalar in [-1, 1]", fc="#fdf0e3", ec=AMB, fs=7.9)
    box(ax, 0.845, 0.56, 0.15, 0.26, "Move\nprobabilities", fc="white", ec=TEAL, fs=7.9)
    box(ax, 0.845, 0.24, 0.15, 0.26, "Position\nevaluation", fc="white", ec=AMB, fs=7.9)
    arrow(ax, (0.15, 0.55), (0.195, 0.55)); arrow(ax, (0.32, 0.55), (0.365, 0.55))
    arrow(ax, (0.56, 0.60), (0.615, 0.69), c=TEAL)
    arrow(ax, (0.56, 0.50), (0.615, 0.37), c=AMB)
    arrow(ax, (0.80, 0.69), (0.845, 0.69), c=TEAL)
    arrow(ax, (0.80, 0.37), (0.845, 0.37), c=AMB)
    ax.text(0.5, 0.07, "Approximately 1.15 M parameters (C = 64, N = 10). The identical network is shared by "
            "Imitation Learning, PPO self-play and MCTS,\nso that the comparison isolates the learning "
            "algorithm rather than the architecture.",
            ha="center", fontsize=7.5, color="#444a52", linespacing=1.5)
    save(fig, "f03_architecture.png")


# ------------------------------------------------------------ pipeline ----
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.4, 4.4)); blank(ax)
    for xx, lab in ((0.005, "SOURCE"), (0.245, "ACQUISITION"),
                    (0.455, "PROCESSING"), (0.715, "RESULTING DATA"),
                    (0.915, "USED BY")):
        ax.text(xx, 0.955, lab, fontsize=7.2, color=GREY, fontweight="bold")
    box(ax, 0.005, 0.70, 0.195, 0.17,
        "D1a  WXF archive\n41,743 games", fc="#e8edfb", ec=ACC, bold=True, fs=7.9)
    box(ax, 0.005, 0.50, 0.195, 0.17,
        "D1b  DPXQ (dpxq.com)\n99,813 games", fc="#e8edfb", ec=ACC, bold=True, fs=7.9)
    box(ax, 0.005, 0.27, 0.195, 0.17,
        "D2  Pikafish engine\n(open-source, NNUE)", fc="#fdf0e3", ec=AMB, bold=True, fs=7.9)
    box(ax, 0.005, 0.05, 0.195, 0.17,
        "D3  The agent itself\n(MCTS self-play)", fc="#efe9f8", ec=VIO, bold=True, fs=7.9)
    box(ax, 0.235, 0.56, 0.16, 0.26,
        "Download the two\nGoogle Drive folders\n(gdown); ICCS .pgns", fc=LIGHT, ec=ACC, fs=7.6)
    box(ax, 0.235, 0.27, 0.16, 0.17,
        "Generate 5,000\nself-play games,\ndepth 8, 6 random\nopening plies", fc="#fdf6ec", ec=AMB, fs=7.6)
    box(ax, 0.235, 0.05, 0.16, 0.17,
        "Generate online:\n20 iterations\nx 25 games", fc="#f6f1fc", ec=VIO, fs=7.6)
    box(ax, 0.435, 0.49, 0.235, 0.39,
        "1  Parse movetext (WXF / ICCS)\n2  Replay-validate every game\n     against the rules engine\n"
        "3  Filter length to 10-300 plies\n4  Encode to 14 x 10 x 9 tensor\n     plus policy and value labels\n"
        "5  Game-level split 80 / 10 / 10", fc="white", ec=INK, fs=7.4)
    box(ax, 0.435, 0.27, 0.235, 0.17,
        "Written in the same JSONL\nschema, so the identical\nsupervised pipeline is reused", fc="white", ec=AMB, fs=7.4)
    box(ax, 0.435, 0.05, 0.235, 0.17,
        "Store (state, MCTS visit\ndistribution, game outcome)\nin a persistent replay buffer", fc="white", ec=VIO, fs=7.4)
    box(ax, 0.705, 0.56, 0.175, 0.26,
        "141,356 games\n11,624,921 positions\ntrain 9.30 M / val 1.17 M\ntest 1.16 M", fc="#e6f4f1", ec=TEAL, bold=True, fs=7.5)
    box(ax, 0.705, 0.27, 0.175, 0.17, "815 k positions\n(about 59% draws)", fc="#e6f4f1", ec=TEAL, fs=7.6)
    box(ax, 0.705, 0.05, 0.175, 0.17, "Replay buffer\n(about 269 MB)", fc="#e6f4f1", ec=TEAL, fs=7.6)
    box(ax, 0.905, 0.56, 0.09, 0.26, "M2\nImitation\nLearning", fc="#eef2fb", ec=ACC, bold=True, fs=7.6)
    box(ax, 0.905, 0.27, 0.09, 0.17, "Distilled\nnetwork", fc="#fdf0e3", ec=AMB, bold=True, fs=7.6)
    box(ax, 0.905, 0.05, 0.09, 0.17, "M4\nMCTS", fc="#efe9f8", ec=VIO, bold=True, fs=7.6)
    arrow(ax, (0.20, 0.785), (0.235, 0.74)); arrow(ax, (0.20, 0.585), (0.235, 0.64))
    arrow(ax, (0.20, 0.355), (0.235, 0.355), c=AMB)
    arrow(ax, (0.20, 0.135), (0.235, 0.135), c=VIO)
    arrow(ax, (0.395, 0.69), (0.435, 0.69))
    arrow(ax, (0.395, 0.355), (0.435, 0.355), c=AMB)
    arrow(ax, (0.395, 0.135), (0.435, 0.135), c=VIO)
    arrow(ax, (0.67, 0.69), (0.705, 0.69)); arrow(ax, (0.67, 0.355), (0.705, 0.355), c=AMB)
    arrow(ax, (0.67, 0.135), (0.705, 0.135), c=VIO)
    arrow(ax, (0.88, 0.69), (0.905, 0.69)); arrow(ax, (0.88, 0.355), (0.905, 0.355), c=AMB)
    arrow(ax, (0.88, 0.135), (0.905, 0.135), c=VIO)
    save(fig, "f04_data_pipeline.png")


if __name__ == "__main__":
    print("figures ->", OUT)
    fig_board(); fig_moves(); fig_arch(); fig_pipeline()
