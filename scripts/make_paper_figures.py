"""Formal, journal-style figures for the IEEE consultation paper.

Design rules: Times New Roman throughout, hairline rules, square corners,
white/light-grey fills, a restrained four-colour print-safe palette, no
gradients or shadows. Output: research/figures/paper/*.png
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "figures", "paper")
os.makedirs(OUT, exist_ok=True)

BK = "#000000"
NAVY = "#1f3864"
GREY = "#808080"
LGREY = "#d9d9d9"
FILL = "#f2f2f2"
DRED = "#9e2a2b"
DGRN = "#44633f"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7.6,
    "axes.linewidth": 0.8,
    "axes.edgecolor": BK,
    "axes.labelcolor": BK,
    "text.color": BK,
    "xtick.color": BK, "ytick.color": BK,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "savefig.dpi": 400,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
CJK = {"fontname": "Microsoft YaHei"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  ", name)


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(ax, x, y, w, h, text, fc=WHITE, ec=BK, fs=6.6, lw=0.8, bold=False,
        tc=BK, ha="center", style="square"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=2))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.42, zorder=3)


def arr(ax, p1, p2, c=BK, lw=0.8, ls="-", ms=7, rad=None):
    cs = "arc3,rad=%.2f" % rad if rad else None
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, linestyle=ls, shrinkA=0.5,
                                 shrinkB=0.5, zorder=1,
                                 connectionstyle=cs or "arc3,rad=0"))


def read_elo(rel):
    rows = []
    with open(os.path.join(ROOT, rel), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["agent"], float(r["elo"]), float(r["elo_ci95"])))
    return rows


# =========================================================== architecture ==
def f_arch():
    fig, ax = plt.subplots(figsize=(3.30, 4.35)); blank(ax)
    W, X = 0.56, 0.22
    rows = [
        (0.930, 0.052, "Board state  $s_t$", FILL, True, "14 x 10 x 9"),
        (0.855, 0.050, "Conv 3x3, 64  +  BN  +  ReLU", WHITE, False, "64 x 10 x 9"),
    ]
    for y, h, t, fc, b, shape in rows:
        box(ax, X, y, W, h, t, fc=fc, bold=b, fs=6.5)
        ax.text(X + W + 0.016, y + h / 2, shape, fontsize=5.8, color=GREY, va="center")
    ax.add_patch(Rectangle((X - 0.025, 0.520), W + 0.05, 0.295, fc="#fafafa",
                           ec=BK, lw=0.8, ls=(0, (3, 2)), zorder=1))
    ax.text(X + W / 2, 0.793, "Residual tower,  N = 10 blocks", fontsize=6.4,
            ha="center", fontweight="bold")
    inner = [(0.730, "Conv 3x3, 64  +  BN  +  ReLU"),
             (0.672, "Conv 3x3, 64  +  BN"),
             (0.614, "$\\oplus$  skip connection  +  ReLU")]
    for y, t in inner:
        box(ax, X, y, W, 0.046, t, fc=WHITE, fs=6.1)
    ax.text(X + W / 2, 0.572, "repeated 10 times", fontsize=6.0, ha="center", style="italic")
    arr(ax, (0.50, 0.730), (0.50, 0.718), lw=0.7, ms=5)
    arr(ax, (0.50, 0.672), (0.50, 0.660), lw=0.7, ms=5)
    arr(ax, (0.50, 0.930), (0.50, 0.906))
    arr(ax, (0.50, 0.855), (0.50, 0.817))
    arr(ax, (0.50, 0.520), (0.50, 0.478))
    box(ax, 0.030, 0.372, 0.43, 0.100,
        "Policy head\nConv 1x1, 2  +  BN  +  ReLU\nFC  180 $\\rightarrow$ 2086",
        fc=WHITE, fs=6.0)
    box(ax, 0.520, 0.372, 0.45, 0.100,
        "Value head\nConv 1x1, 1  +  BN  +  ReLU\nFC 90 $\\rightarrow$ 64 $\\rightarrow$ 1",
        fc=WHITE, fs=6.0)
    ax.plot([0.245, 0.745], [0.500, 0.500], color=BK, lw=0.8, zorder=1)
    arr(ax, (0.245, 0.500), (0.245, 0.474))
    arr(ax, (0.745, 0.500), (0.745, 0.474))
    box(ax, 0.030, 0.262, 0.43, 0.078,
        "Legal-move mask\nillegal logits $\\rightarrow -\\infty$", fc=FILL, fs=6.0)
    box(ax, 0.520, 0.262, 0.45, 0.078, "tanh\n$v \\in [-1, 1]$", fc=FILL, fs=6.0)
    arr(ax, (0.245, 0.372), (0.245, 0.340))
    arr(ax, (0.745, 0.372), (0.745, 0.340))
    box(ax, 0.030, 0.158, 0.43, 0.072, "softmax\n$\\mathbf{p} \\in \\Delta^{2085}$", fc=WHITE, fs=6.0)
    box(ax, 0.520, 0.158, 0.45, 0.072, "Position value\nfor search and RL", fc=WHITE, fs=6.0)
    arr(ax, (0.245, 0.262), (0.245, 0.230))
    arr(ax, (0.745, 0.262), (0.745, 0.230))
    ax.plot([0.03, 0.97], [0.112, 0.112], color=BK, lw=0.8)
    ax.text(0.5, 0.055, "Total 1.15 M parameters.  The identical network is shared by\n"
            "M2, M3 and M4, so the comparison isolates the learning rule.",
            fontsize=6.0, ha="center", va="center", linespacing=1.5)
    save(fig, "p05_architecture.png")


# ================================================================ M1 =======
def f_m1():
    fig, ax = plt.subplots(figsize=(3.30, 2.95)); blank(ax)
    L = {"r": (0.50, 0.90), "a": (0.26, 0.655), "b": (0.76, 0.655),
         "a1": (0.11, 0.405), "a2": (0.40, 0.405), "b1": (0.63, 0.405), "b2": (0.90, 0.405)}
    for k, (x, y) in L.items():
        cut = k == "b2"
        ax.add_patch(Circle((x, y), 0.052, fc=LGREY if cut else WHITE,
                            ec=GREY if cut else BK, lw=0.8, zorder=3))
    ax.text(*L["r"], "max", ha="center", va="center", fontsize=6.0, zorder=4, style="italic")
    for k in ("a", "b"):
        ax.text(*L[k], "min", ha="center", va="center", fontsize=6.0, zorder=4, style="italic")
    for k, v in (("a1", "+2"), ("a2", "$-$1"), ("b1", "+5")):
        ax.text(*L[k], v, ha="center", va="center", fontsize=6.2, zorder=4)
    ax.text(*L["b2"], "--", ha="center", va="center", fontsize=6.2, color=GREY, zorder=4)
    for p, c in (("r", "a"), ("r", "b"), ("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b2")):
        x1, y1 = L[p]; x2, y2 = L[c]
        cut = c == "b2"
        ax.plot([x1, x2], [y1, y2], color=GREY if cut else BK, lw=0.8,
                ls=(0, (2, 2)) if cut else "-", zorder=1)
    for p, c in (("r", "a"), ("a", "a1")):
        x1, y1 = L[p]; x2, y2 = L[c]
        ax.plot([x1, x2], [y1, y2], color=NAVY, lw=2.0, zorder=2)
    ax.text(0.555, 0.79, "$v = -1$", fontsize=5.8, color=NAVY)
    ax.text(0.085, 0.545, "principal\nvariation", fontsize=5.8, color=NAVY,
            ha="center", linespacing=1.4)
    ax.annotate("$\\beta$ cut-off:  subtree cannot\nchange the root value",
                xy=(0.90, 0.455), xytext=(0.60, 0.245), fontsize=5.9,
                ha="center", linespacing=1.45, color=GREY,
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7))
    ax.plot([0.02, 0.98], [0.175, 0.175], color=BK, lw=0.8)
    ax.text(0.02, 0.118, "Evaluation  $h(s)=\\sum_i w_i\\,(n_i^{R}-n_i^{B})$, with "
            "$w$ = (9, 4.5, 2, 2, 1, 1) for\nchariot, cannon, horse, elephant, advisor "
            "and soldier; depth $d \\in \\{1,2,3\\}$.\n"
            "No parameters are learned, so strength is a pure function of $d$ and the\n"
            "agent serves as the learning-free control condition for the whole study.",
            fontsize=5.9, va="top", linespacing=1.55)
    save(fig, "p06_m1.png")


# ================================================================ M2 =======
def f_m2():
    fig, ax = plt.subplots(figsize=(3.30, 4.15)); blank(ax)
    cx, W, X = 0.52, 0.72, 0.16
    box(ax, X, 0.902, W, 0.080,
        "141,356 games  $\\rightarrow$  11,624,921 positions\ngame-level split  80 / 10 / 10",
        fc=FILL, bold=True, fs=6.0)
    box(ax, X, 0.782, W, 0.080,
        "Sample $(s_t, a_t^{h}, z)$ and encode $s_t$ lazily\nby replaying the game to ply $t$",
        fs=6.0)
    box(ax, X, 0.690, W, 0.054, "Shared policy-value network  $f_\\theta$", fc=FILL,
        bold=True, fs=6.0)
    arr(ax, (cx, 0.902), (cx, 0.866))
    arr(ax, (cx, 0.782), (cx, 0.748))
    ax.plot([0.275, 0.765], [0.655, 0.655], color=BK, lw=0.8)
    arr(ax, (cx, 0.690), (cx, 0.656))
    arr(ax, (0.275, 0.655), (0.275, 0.622))
    arr(ax, (0.765, 0.655), (0.765, 0.622))
    box(ax, 0.075, 0.542, 0.40, 0.080, "Policy $\\mathbf{p}_\\theta(s_t)$\n2086 logits", fs=6.0)
    box(ax, 0.565, 0.542, 0.40, 0.080, "Value $v_\\theta(s_t)$\nscalar", fs=6.0)
    arr(ax, (0.275, 0.542), (0.275, 0.506))
    arr(ax, (0.765, 0.542), (0.765, 0.506))
    box(ax, 0.075, 0.424, 0.40, 0.082,
        "$\\mathcal{L}_{\\pi}=-\\log p_\\theta(a^{h}_t | s_t)$", fs=6.2)
    box(ax, 0.565, 0.424, 0.40, 0.082, "$\\mathcal{L}_{v}=(v_\\theta(s_t)-z)^2$", fs=6.2)
    arr(ax, (0.275, 0.424), (0.37, 0.378))
    arr(ax, (0.765, 0.424), (0.67, 0.378))
    box(ax, 0.17, 0.286, 0.70, 0.092,
        "$\\mathcal{L}=\\mathcal{L}_{\\pi}+\\mathcal{L}_{v}$  (equal weighting)\n"
        "Adam,  lr $10^{-3}$,  batch 512,  30 epochs", fc=FILL, fs=6.0)
    ax.plot([0.17, 0.085], [0.332, 0.332], color=BK, lw=0.8, ls=(0, (3, 2)))
    ax.plot([0.085, 0.085], [0.332, 0.717], color=BK, lw=0.8, ls=(0, (3, 2)))
    arr(ax, (0.085, 0.717), (0.158, 0.717), lw=0.8, ms=7, ls=(0, (3, 2)))
    ax.text(0.052, 0.525, "training loop: $\\theta \\leftarrow \\theta - \\eta\\nabla L$",
            fontsize=5.7, rotation=90, ha="center", va="center")
    ax.plot([0.02, 0.98], [0.238, 0.238], color=BK, lw=0.8)
    ax.text(0.02, 0.196, "The target is the move a human master actually played, so the agent\n"
            "reproduces human play but cannot exceed the quality of the demonstra-\n"
            "tions. Theory predicts degradation away from the demonstrated state\n"
            "distribution; Section VIII-C reports exactly that pattern across the\n"
            "three phases of the game.",
            fontsize=5.9, va="top", linespacing=1.55)
    save(fig, "p07_m2.png")


# ================================================================ M3 =======
def f_m3():
    fig, ax = plt.subplots(figsize=(3.30, 4.20)); blank(ax)
    box(ax, 0.17, 0.902, 0.74, 0.078,
        "Roll out 2048 steps of self-play:\n$\\pi_{\\theta_{old}}$ against a frozen copy of itself",
        fc=FILL, fs=6.0)
    box(ax, 0.17, 0.794, 0.74, 0.078,
        "Terminal reward  $r_T \\in \\{+1, 0, -1\\}$;\nno intermediate reward is given", fs=6.0)
    box(ax, 0.17, 0.686, 0.74, 0.078,
        "Generalised advantage estimate $\\hat{A}_t$\n$\\gamma = 0.99$,  $\\lambda = 0.95$", fs=6.0)
    box(ax, 0.11, 0.530, 0.80, 0.126,
        "Clipped surrogate objective\n"
        "$L^{CLIP}=E_t[\\min(r_t\\hat{A}_t,\\ "
        "\\mathrm{clip}(r_t,1\\!-\\!\\epsilon,1\\!+\\!\\epsilon)\\hat{A}_t)]$\n"
        "$r_t = \\pi_\\theta(a_t|s_t)\\,/\\,\\pi_{\\theta_{old}}(a_t|s_t)$,   $\\epsilon = 0.2$",
        fc=FILL, fs=5.9)
    box(ax, 0.17, 0.418, 0.74, 0.078,
        "4 epochs over the batch, minibatch 512,\nlr $3\\times10^{-4}$, entropy bonus 0.01", fs=6.0)
    for y1, y2 in ((0.902, 0.874), (0.794, 0.766), (0.686, 0.658), (0.530, 0.498)):
        arr(ax, (0.54, y1), (0.54, y2))
    ax.plot([0.17, 0.085], [0.457, 0.457], color=BK, lw=0.9)
    ax.plot([0.085, 0.085], [0.457, 0.941], color=BK, lw=0.9)
    arr(ax, (0.085, 0.941), (0.158, 0.941), lw=0.9, ms=8)
    ax.text(0.050, 0.700, "iteration loop: repeat to 500,000 steps", fontsize=5.7,
            rotation=90, ha="center", va="center")
    ax.plot([0.02, 0.98], [0.372, 0.372], color=BK, lw=0.8)
    ax.text(0.02, 0.340, "Two initialisations are compared under an identical budget:",
            fontsize=6.0, va="center")
    box(ax, 0.04, 0.198, 0.44, 0.102, "M3a\n$\\theta_0$ random\n(tabula rasa)", fs=6.0)
    box(ax, 0.52, 0.198, 0.44, 0.102, "M3b\n$\\theta_0$ initialised from M2\n(imitation warm start)",
        fc=FILL, fs=6.0)
    ax.text(0.02, 0.150, "The contrast between M3a and M3b is the direct test of RQ2: whether\n"
            "imitation pretraining helps or constrains the self-play that follows.\n"
            "Illegal actions are masked inside the trainer, so no gradient is ever\n"
            "spent on an unplayable move.",
            fontsize=5.9, va="top", linespacing=1.55)
    save(fig, "p08_m3.png")


# ================================================================ M4 =======
def f_m4():
    fig, ax = plt.subplots(figsize=(3.30, 4.05)); blank(ax)
    T = {"r": (0.50, 0.955), "a": (0.26, 0.855), "b": (0.50, 0.855), "c": (0.74, 0.855),
         "d": (0.40, 0.752), "e": (0.60, 0.752)}
    wid = {"a": 0.8, "b": 2.4, "c": 0.8, "d": 2.0, "e": 0.8}
    par = {"a": "r", "b": "r", "c": "r", "d": "b", "e": "b"}
    for k, (x, y) in T.items():
        ax.add_patch(Circle((x, y), 0.027, fc=FILL if k == "e" else WHITE,
                            ec=BK, lw=0.8, zorder=3))
    for k, p in par.items():
        x1, y1 = T[p]; x2, y2 = T[k]
        ax.plot([x1, x2], [y1, y2], color=BK, lw=wid[k], zorder=1)
    ax.text(0.885, 0.905, "each edge stores\n$N, W, Q, P$", fontsize=5.6, color=GREY,
            ha="center", va="center", linespacing=1.4)
    ax.text(0.665, 0.726, "new leaf", fontsize=5.6, color=GREY, ha="left")
    ax.text(0.105, 0.905, "edge width\n$\\propto N(s,a)$", fontsize=5.6, color=GREY,
            ha="center", va="center", linespacing=1.4)
    ax.plot([0.02, 0.98], [0.690, 0.690], color=BK, lw=0.8)
    steps = [
        ("1)  Select", "descend by  $a^{*}=\\arg\\max_a [\\,Q(s,a)+c_{puct}P(s,a)"
                       "\\frac{\\sqrt{\\sum_b N(s,b)}}{1+N(s,a)}]$,  $c_{puct}=1.5$"),
        ("2)  Expand", "add the leaf $s_L$ and enumerate its legal moves"),
        ("3)  Evaluate", "one network pass: $(\\mathbf{p}, v)=f_\\theta(s_L)$ "
                         "replaces the random rollout"),
        ("4)  Back up", "for every edge on the path: $N\\!\\leftarrow\\!N+1$, "
                        "$W\\!\\leftarrow\\!W+v$, $Q\\!=\\!W/N$"),
    ]
    gaps = [0.108, 0.072, 0.072, 0.072]
    y = 0.650
    for (t, d), g in zip(steps, gaps):
        ax.text(0.03, y, t, fontsize=6.3, fontweight="bold", va="center")
        ax.text(0.03, y - 0.036, d, fontsize=5.9, va="center", linespacing=1.45)
        y -= g
    ax.plot([0.02, 0.98], [0.302, 0.302], color=BK, lw=0.8)
    box(ax, 0.03, 0.182, 0.45, 0.098,
        "Training self-play\n$n = 100$ simulations\nDirichlet $\\alpha=0.3$, $\\varepsilon=0.25$", fs=5.8)
    box(ax, 0.52, 0.182, 0.45, 0.098,
        "Move selection\nvisit sampling for 30 plies,\nthen the most-visited move", fc=FILL, fs=5.8)
    ax.text(0.02, 0.145, "The visit distribution $\\pi(a)\\propto N(s,a)^{1/\\tau}$ is itself the "
            "policy target during\ntraining, so search quality is distilled back into $f_\\theta$. "
            "Twenty iterations of\ntwenty-five games, with two network epochs per iteration.",
            fontsize=5.9, va="top", linespacing=1.5)
    save(fig, "p09_m4.png")


# ========================================================== data pipeline ==
def f_pipeline():
    fig, ax = plt.subplots(figsize=(7.10, 3.30)); blank(ax)
    heads = [(0.005, "Source"), (0.215, "Acquisition"), (0.420, "Processing"),
             (0.685, "Resulting corpus"), (0.885, "Consumed by")]
    for x, t in heads:
        ax.text(x, 0.955, t.upper(), fontsize=6.4, color=BK, fontweight="bold")
    ax.plot([0.0, 1.0], [0.935, 0.935], color=BK, lw=0.8)
    rows = [
        (0.690, 0.195, "D1  Human corpus\nWXF 41,743 games\nDPXQ 99,813 games",
         "Download two cloud\nfolders (gdown);\nPGN in ICCS notation",
         "1) parse movetext (WXF / ICCS)\n2) replay-validate every game\n"
         "3) keep 10-300 plies\n4) encode 14 x 10 x 9 + labels\n5) split by game, 80/10/10",
         "141,356 games\n11,624,921 positions\n9.30 M / 1.17 M / 1.16 M", "M2\nImitation"),
        (0.420, 0.175, "D2  Pikafish engine\nUCI, NNUE, depth 8",
         "Generate 5,000 self-\nplay games; first 6\nplies random for\nopening diversity",
         "Written in the identical JSONL\nschema, so the same supervised\n"
         "pipeline is reused unchanged;\nonly the teacher differs",
         "815 k positions\napprox. 59% draws", "Distilled\nnetwork"),
        (0.175, 0.175, "D3  The agent itself\nMCTS self-play",
         "Generated online:\n20 iterations of\n25 games",
         "Store $(s,\\ \\pi_{MCTS},\\ z)$ per move;\n"
         "persisted between sessions to\navoid the regression of Sec. VIII-E",
         "Replay buffer\napprox. 269 MB", "M4\nMCTS"),
    ]
    xs = [0.005, 0.215, 0.420, 0.685, 0.885]
    ws = [0.185, 0.175, 0.250, 0.175, 0.110]
    for y, h, a, b, c, d, e in rows:
        for k, (txt, fc) in enumerate(((a, FILL), (b, WHITE), (c, WHITE), (d, FILL), (e, FILL))):
            box(ax, xs[k], y, ws[k], h, txt, fc=fc, fs=5.7,
                bold=(k in (0, 4)))
        for k in range(4):
            arr(ax, (xs[k] + ws[k], y + h / 2), (xs[k + 1], y + h / 2), lw=0.7, ms=6)
    ax.plot([0.0, 1.0], [0.118, 0.118], color=BK, lw=0.8)
    ax.text(0.0, 0.055, "All three corpora share one JSON-lines schema, so a single loader and "
            "encoder serves every paradigm. PPO (M3) also generates experience by self-play, "
            "but it is consumed\nimmediately by the on-policy update and is therefore not "
            "retained as a dataset.", fontsize=5.9, va="center", linespacing=1.5)
    save(fig, "p10_pipeline.png")


# ================================================================ Elo ======
LAB = {"MCTS_v2": "M4  MCTS + network", "IL_only": "M2  Imitation learning",
       "Minimax_d2": "M1  Minimax, $d=2$", "ILPPO_rand": "M3b  IL then PPO",
       "Random": "Random baseline", "Pikafish_d8": "Pikafish engine, $d=8$",
       "Distill_pika": "Engine-distilled net", "IL_human": "M2  Imitation learning",
       "Distill_small": "Distilled, 1.15 M", "Distill_big": "Distilled, 4.85 M"}


def _elo_panel(ax, rows, hi=None, xlo=1100):
    rows = sorted(rows, key=lambda r: r[1])
    names = [LAB.get(r[0], r[0]) for r in rows]
    v = [r[1] for r in rows]; e = [r[2] for r in rows]
    y = np.arange(len(rows))
    cols = [NAVY if (hi and r[0] in hi) else "#9fb0c9" for r in rows]
    ax.barh(y, v, xerr=e, height=0.58, color=cols, edgecolor=BK, linewidth=0.7,
            error_kw=dict(ecolor=BK, lw=0.7, capsize=2.2))
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.6)
    for i, vv in enumerate(v):
        ax.text(vv + e[i] + 12, i, "%.0f" % vv, va="center", fontsize=6.3)
    ax.axvline(1500, color=GREY, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlabel("Relative Elo (anchor 1500, bars show 95% CI)", fontsize=6.8)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_xlim(xlo, max(v) + max(e) + 130)
    ax.grid(axis="x", color=LGREY, lw=0.5)
    ax.set_axisbelow(True)


def f_elo_main():
    fig, ax = plt.subplots(figsize=(3.30, 1.85))
    _elo_panel(ax, read_elo("results/exp1_agent3_v2/exp1_elo.csv"))
    fig.tight_layout(pad=0.2)
    save(fig, "p13_elo_main.png")


def f_elo_engine():
    fig, ax = plt.subplots(figsize=(3.30, 2.05))
    rows = read_elo("results/exp_distill/exp1_elo.csv")
    _elo_panel(ax, rows, hi={"Pikafish_d8"}, xlo=1050)
    ax.annotate("", xy=(2062, 4.44), xytext=(1502, 4.44),
                arrowprops=dict(arrowstyle="<->", color=BK, lw=0.7))
    ax.text(1782, 4.33, "570 Elo", ha="center", va="top", fontsize=6.3)
    fig.tight_layout(pad=0.2)
    save(fig, "p14_elo_engine.png")


# =========================================================== accuracy ======
def f_accuracy():
    fig, ax = plt.subplots(figsize=(3.30, 1.95))
    ph = ["Overall", "Opening\n($t<20$)", "Midgame\n($20\\!-\\!59$)", "Endgame\n($t\\geq60$)"]
    t1 = [42.3, 63.2, 37.7, 32.3]; t3 = [66.5, 87.8, 61.8, 56.5]
    x = np.arange(4); w = 0.34
    ax.bar(x - w / 2, t1, w, label="Top-1", color=NAVY, edgecolor=BK, lw=0.7)
    ax.bar(x + w / 2, t3, w, label="Top-3", color="#c7d0de", edgecolor=BK, lw=0.7)
    for xi, vv in zip(x - w / 2, t1):
        ax.text(xi, vv + 1.8, "%.1f" % vv, ha="center", fontsize=6.0)
    for xi, vv in zip(x + w / 2, t3):
        ax.text(xi, vv + 1.8, "%.1f" % vv, ha="center", fontsize=6.0)
    ax.set_xticks(x); ax.set_xticklabels(ph, fontsize=6.4)
    ax.set_ylabel("Human-move prediction accuracy (%)", fontsize=6.8)
    ax.tick_params(axis="y", labelsize=6.3)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.4, loc="upper right", ncol=2, handlelength=1.1)
    ax.grid(axis="y", color=LGREY, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2)
    save(fig, "p15_accuracy.png")


# ======================================================== ablations ========
def f_abl_search():
    fig, ax = plt.subplots(figsize=(3.30, 1.90))
    sims, sc = [], []
    with open(os.path.join(ROOT, "results/exp5_mcts_v2/exp5_mcts_ablation.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sims.append(int(r["n_simulations"])); sc.append(float(r["score_vs_IL"]))
    ax.plot(sims, sc, "o-", color=NAVY, lw=1.0, ms=4.2, mfc=WHITE, mec=NAVY, mew=1.0)
    ax.axhline(0.5, color=GREY, lw=0.7, ls=(0, (3, 2)))
    ax.text(820, 0.525, "parity with M2", fontsize=6.1, ha="right", color=GREY)
    ax.set_xscale("log", base=2); ax.set_xticks(sims)
    ax.set_xticklabels([str(s) for s in sims], fontsize=6.3)
    ax.tick_params(axis="y", labelsize=6.3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("MCTS simulations per move (log$_2$ scale)", fontsize=6.8)
    ax.set_ylabel("Score against M2", fontsize=6.8)
    ax.grid(color=LGREY, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2)
    save(fig, "p16_abl_search.png")


def f_abl_capacity():
    fig, ax = plt.subplots(figsize=(3.30, 1.95))
    rows = {a: (e, c) for a, e, c in read_elo("results/exp_bignet/exp1_elo.csv")}
    keys = ["Distill_small", "Distill_big", "IL_human", "Minimax_d2"]
    labs = ["Distilled\n1.15 M", "Distilled\n4.85 M", "Imitation\n1.15 M", "Minimax\nno network"]
    v = [rows[k][0] for k in keys]; e = [rows[k][1] for k in keys]
    cols = ["#c7d0de", NAVY, "#9fb0c9", "#e0e0e0"]
    ax.bar(range(4), v, yerr=e, color=cols, edgecolor=BK, lw=0.7, width=0.58,
           error_kw=dict(ecolor=BK, lw=0.7, capsize=2.2))
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=6.3)
    ax.set_ylim(1400, 1730)
    ax.tick_params(axis="y", labelsize=6.3)
    ax.set_ylabel("Relative Elo (one tournament)", fontsize=6.8)
    for i, vv in enumerate(v):
        ax.text(i, vv + e[i] + 7, "%.0f" % vv, ha="center", fontsize=6.1)
    ax.set_ylim(1400, 1760)
    ax.annotate("", xy=(1, 1690), xytext=(0, 1690),
                arrowprops=dict(arrowstyle="<->", color=BK, lw=0.7))
    ax.text(0.5, 1700, "+43 Elo (CI $\\pm$78)", ha="center", fontsize=6.1)
    ax.grid(axis="y", color=LGREY, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2)
    save(fig, "p17_abl_capacity.png")


# ========================================================== compute ========
def f_compute():
    fig, ax = plt.subplots(figsize=(3.30, 1.62))
    labs = ["AlphaZero [11]\n5,000 + 64 TPUs", "ELF OpenGo [18]\napprox. 2,000 GPUs",
            "This project\n1 laptop GPU"]
    v = [5064, 2000, 1]
    y = np.arange(3)[::-1]
    ax.barh(y, v, color=["#d6d6d6", "#d6d6d6", NAVY], edgecolor=BK, lw=0.7, height=0.52)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.3)
    for yy, vv in zip(y, v):
        ax.text(vv * 1.6, yy, "{:,}".format(vv), va="center", fontsize=6.3)
    ax.set_xlim(0.5, 90000)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_xlabel("Hardware accelerators (log scale)", fontsize=6.8)
    ax.grid(axis="x", color=LGREY, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.2)
    save(fig, "p18_compute.png")


# ========================================================== timeline =======
def f_timeline():
    fig, ax = plt.subplots(figsize=(7.10, 2.65))
    months = ["Sep\n2026", "Oct", "Nov", "Dec", "Jan\n2027", "Feb", "Mar", "Apr", "May"]
    tasks = [
        ("Environment, rules, encoder, action space", 0, 1.0, 1),
        ("Four paradigms implemented", 0.5, 1.5, 1),
        ("Training runs: M2, M3, M4", 1.0, 1.4, 1),
        ("Engine integration and distillation", 1.6, 0.8, 1),
        ("Evaluation league and six studies", 1.3, 1.3, 1),
        ("Proposal, progress review, demonstrator", 2.0, 1.0, 1),
        ("Supervisor consultation, protocol freeze", 2.7, 0.8, 0),
        ("Scale-up study: capacity and self-play budget", 3.2, 2.0, 0),
        ("Absolute calibration against engine ladder", 4.6, 1.3, 0),
        ("Interactive platform (subject to Sec. X)", 4.2, 2.0, 0),
        ("Thesis writing", 5.6, 2.3, 0),
        ("Revision, rehearsal, defense", 7.4, 1.4, 0),
    ]
    for i, (nm, st, du, done) in enumerate(tasks):
        y = len(tasks) - i - 1
        ax.barh(y, du, left=st, height=0.52,
                color=NAVY if done else "#ffffff",
                edgecolor=BK, lw=0.8,
                hatch="" if done else "///")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks][::-1], fontsize=6.5)
    ax.set_xticks(range(9)); ax.set_xticklabels(months, fontsize=6.4)
    ax.set_xlim(-0.05, 9.05)
    ax.axvline(2.15, color=BK, lw=1.0, ls=(0, (4, 2)))
    ax.text(2.22, len(tasks) - 0.45, "5 Nov 2026", fontsize=6.3, va="top")
    ax.axvline(8.95, color=BK, lw=1.0)
    ax.text(8.88, len(tasks) - 0.45, "defense", fontsize=6.3, va="top", ha="right")
    ax.grid(axis="x", color=LGREY, lw=0.5); ax.set_axisbelow(True)
    h1 = plt.Rectangle((0, 0), 1, 1, fc=NAVY, ec=BK, lw=0.8)
    h2 = plt.Rectangle((0, 0), 1, 1, fc=WHITE, ec=BK, lw=0.8, hatch="///")
    ax.legend([h1, h2], ["completed", "planned"], fontsize=6.4, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.11), handlelength=1.3)
    fig.tight_layout(pad=0.25)
    save(fig, "p21_timeline.png")


if __name__ == "__main__":
    print("formal figures ->", OUT)
    f_arch(); f_m1(); f_m2(); f_m3(); f_m4(); f_pipeline()
    f_elo_main(); f_elo_engine(); f_accuracy()
    f_abl_search(); f_abl_capacity(); f_compute(); f_timeline()
