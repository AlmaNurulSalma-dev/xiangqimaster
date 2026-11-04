"""Generate accurate, editable SVG figures for the thesis / defence slides.

Produces, in research/figures/:
  fig1_board.svg        - Xiangqi board with the starting position
  fig2_moves.svg        - how each of the seven piece types moves
  fig3_tensor.svg       - the 14x10x9 game-state tensor encoding
  fig4_framework.svg    - the four-paradigm comparison pipeline (our method)
  fig5_mechanisms.svg   - the four agent mechanisms (Minimax / PPO / IL / MCTS)
  fig6_selfplay.svg     - the AlphaZero-style self-play training loop

These are hand-built vector diagrams (no external deps), so the geometry and the
labels are exactly correct -- unlike image-generation output.  Run:

    python scripts/make_thesis_figures.py
"""
from __future__ import annotations
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "research", "figures")

# palette
BG, INK = "#ffffff", "#1e293b"
RED, BLK = "#dc2626", "#334155"
TEAL, GREEN = "#0d9488", "#16a34a"
GREY = "#94a3b8"

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'viewBox="0 0 {w} {h}" font-family="Segoe UI, Arial, sans-serif">\n'
        '<rect width="{w}" height="{h}" fill="{bg}"/>\n'
        '<defs>'
        '<marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{ink}"/></marker>'
        '<marker id="ag" markerWidth="9" markerHeight="9" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="%s"/></marker>'
        '<marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="%s"/></marker>'
        '</defs>\n') % (GREEN, RED)


def ln(x1, y1, x2, y2, c=INK, w=1.5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{c}" stroke-width="{w}"{d}/>\n')


def arrow(x1, y1, x2, y2, c=INK, w=2, mk="ah", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{c}" stroke-width="{w}" marker-end="url(#{mk})"{d}/>\n')


def box(x, y, w, h, fill="#f8fafc", stroke=INK, rx=8, sw=1.6):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n')


def circ(cx, cy, r, fill, stroke=INK, sw=1.5):
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>\n')


def txt(x, y, s, size=13, c=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{c}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'dominant-baseline="middle">{s}</text>\n')


def save(name, w, h, body):
    svg = HEAD.format(w=w, h=h, bg=BG, ink=INK) + body + "</svg>\n"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", name)


# piece codes -> full name
NAMES = {"G": "General", "A": "Advisor", "E": "Elephant", "H": "Horse",
         "R": "Chariot", "C": "Cannon", "S": "Soldier"}


def board_grid(ox, oy, cell):
    """Return SVG for an empty Xiangqi board; x(c), y(r) helpers."""
    def X(c): return ox + c * cell
    def Y(r): return oy + (9 - r) * cell
    s = ""
    for r in range(10):                       # horizontals
        s += ln(X(0), Y(r), X(8), Y(r))
    for c in range(9):                        # verticals (river gap inside)
        if c in (0, 8):
            s += ln(X(c), Y(0), X(c), Y(9))
        else:
            s += ln(X(c), Y(0), X(c), Y(4))
            s += ln(X(c), Y(5), X(c), Y(9))
    for (r0, r1) in [(0, 2), (7, 9)]:         # palace diagonals
        s += ln(X(3), Y(r0), X(5), Y(r1))
        s += ln(X(5), Y(r0), X(3), Y(r1))
    s += txt((X(0) + X(8)) / 2, (Y(4) + Y(5)) / 2, "River",
             size=int(cell * 0.42), c=GREY, weight="bold")
    return s, X, Y


def piece(cx, cy, code, side, r=20):
    fill = RED if side == "r" else BLK
    return circ(cx, cy, r, fill, stroke="#ffffff", sw=2) + \
        txt(cx, cy + 1, code, size=int(r * 0.9), c="#ffffff", weight="bold")


# ---- Figure 1: board + starting position -----------------------------------
def fig1():
    cell = 52; ox = 46; oy = 36
    w = ox * 2 + 8 * cell; h = oy + 9 * cell + 120
    s, X, Y = board_grid(ox, oy, cell)
    back = ["R", "H", "E", "A", "G", "A", "E", "H", "R"]
    layout = []
    for c, code in enumerate(back):
        layout += [(c, 0, code, "r"), (c, 9, code, "b")]
    for c in (1, 7):
        layout += [(c, 2, "C", "r"), (c, 7, "C", "b")]
    for c in (0, 2, 4, 6, 8):
        layout += [(c, 3, "S", "r"), (c, 6, "S", "b")]
    for (c, r, code, side) in layout:
        s += piece(X(c), Y(r), code, side)
    # legend (4-column grid so it never overflows the canvas)
    ly = oy + 9 * cell + 34
    s += txt(ox, ly - 10, "Red (bottom) vs Black (top). Pieces sit on "
             "intersections; 9 files x 10 ranks.", size=12, c=INK, anchor="start")
    colw = (w - 2 * ox) / 4
    for idx, code in enumerate(["G", "A", "E", "H", "R", "C", "S"]):
        cx = ox + (idx % 4) * colw
        cy = ly + 18 + (idx // 4) * 26
        s += piece(cx + 11, cy, code, "r", r=11)
        s += txt(cx + 26, cy, NAMES[code], size=12, c=INK, anchor="start")
    save("fig1_board.svg", w, h, s)


# ---- Figure 2: piece movement rules ----------------------------------------
def fig2():
    # 7 small panels in a 4x2 grid; each a local grid with move/capture arrows
    pw, ph = 196, 150; gap = 16; cols = 4
    pieces = [
        ("General", [(0, 1), (0, -1), (1, 0), (-1, 0)], None, "1 step, in palace"),
        ("Advisor", [(1, 1), (1, -1), (-1, 1), (-1, -1)], None, "diagonal, in palace"),
        ("Elephant", [(2, 2), (2, -2), (-2, 2), (-2, -2)], None, "2 diag, not past river"),
        ("Horse", [(1, 2), (2, 1), (2, -1), (1, -2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)], None, "L-shape, can be blocked"),
        ("Chariot", [(0, 3), (0, -3), (3, 0), (-3, 0)], None, "straight, any distance"),
        ("Cannon", [(0, 3), (3, 0), (-3, 0)], (0, -3), "move straight; jump to capture"),
        ("Soldier", [(0, 1), (1, 0), (-1, 0)], None, "forward; sideways after river"),
    ]
    rows = (len(pieces) + cols - 1) // cols
    w = cols * pw + (cols + 1) * gap
    h = rows * ph + (rows + 1) * gap + 30
    s = txt(w / 2, 20, "How each piece moves  (green = move, red dashed = capture)",
            size=15, c=INK, weight="bold")
    cs = 22  # cell size inside a panel
    for i, (name, moves, cap, note) in enumerate(pieces):
        r, c = divmod(i, cols)
        px = gap + c * (pw + gap); py = 30 + gap + r * (ph + gap)
        s += box(px, py, pw, ph, fill="#f8fafc")
        s += txt(px + pw / 2, py + 16, name, size=13, c=INK, weight="bold")
        cx0, cy0 = px + pw / 2, py + ph / 2 + 8
        # faint local grid
        for k in range(-3, 4):
            s += ln(cx0 + k * cs, cy0 - 3 * cs, cx0 + k * cs, cy0 + 3 * cs, c="#e2e8f0", w=1)
            s += ln(cx0 - 3 * cs, cy0 + k * cs, cx0 + 3 * cs, cy0 + k * cs, c="#e2e8f0", w=1)
        for (dx, dy) in moves:
            s += arrow(cx0, cy0, cx0 + dx * cs, cy0 - dy * cs, c=GREEN, w=2, mk="ag")
        if cap:
            dx, dy = cap
            s += arrow(cx0, cy0, cx0 + dx * cs, cy0 - dy * cs, c=RED, w=2, mk="ar", dash="4 3")
        s += circ(cx0, cy0, 11, INK, stroke="#ffffff", sw=2)
        s += txt(px + pw / 2, py + ph - 10, note, size=10, c=GREY)
    save("fig2_moves.svg", w, h, s)


# ---- Figure 3: 14x10x9 tensor ----------------------------------------------
def fig3():
    w, h = 640, 300
    s = txt(w / 2, 24, "Game-state encoding:  14 x 10 x 9 tensor",
            size=15, c=INK, weight="bold")
    # small board hint
    s += box(30, 90, 90, 100, fill="#f8fafc")
    s += txt(75, 140, "board", size=12, c=GREY)
    s += arrow(125, 140, 165, 140, c=INK)
    s += txt(145, 126, "encode", size=10, c=GREY)
    # stack of 14 planes (offset parallelograms)
    bx, by, pw, ph, off = 175, 70, 120, 150, 7
    for i in range(14):
        x = bx + (13 - i) * off; y = by + (13 - i) * off
        col = "#ccfbf1" if i < 13 else "#5eead4"
        s += (f'<rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="3" '
              f'fill="{col}" stroke="{TEAL}" stroke-width="1.2"/>\n')
    s += txt(bx + pw / 2 + 10, by - 6, "14 planes (7 types x 2 sides)", size=12, c=INK)
    s += arrow(bx + pw + 20, by + ph + 20, bx + pw + 70, by + ph / 2, c=INK)
    # pulled-out 10x9 grid with a few filled cells
    gx, gy, cs = bx + pw + 80, 70, 15
    for r in range(11):
        s += ln(gx, gy + r * cs, gx + 9 * cs, gy + r * cs, c="#cbd5e1", w=1)
    for c in range(10):
        s += ln(gx + c * cs, gy, gx + c * cs, gy + 10 * cs, c="#cbd5e1", w=1)
    s += box(gx, gy, 9 * cs, 10 * cs, fill="none", stroke=INK, sw=1.4, rx=0)
    for (cc, rr) in [(1, 7), (7, 7), (1, 2), (7, 2)]:
        s += (f'<rect x="{gx + cc * cs}" y="{gy + rr * cs}" width="{cs}" '
              f'height="{cs}" fill="{RED}" opacity="0.8"/>\n')
    s += txt(gx + 4.5 * cs, gy + 10 * cs + 18, "one plane: Red Cannons",
             size=11, c=INK)
    s += txt(gx + 4.5 * cs, gy + 10 * cs + 34, "(1 = piece here, else 0)",
             size=10, c=GREY)
    save("fig3_tensor.svg", w, h, s)


# ---- Figure 4: four-paradigm framework -------------------------------------
def fig4():
    w, h = 720, 320
    s = txt(w / 2, 22, "Controlled comparison: only the decision function changes",
            size=15, c=INK, weight="bold")
    # left shared core
    s += box(20, 70, 150, 180, fill="#ecfeff", stroke=TEAL)
    s += txt(95, 95, "Shared", size=13, c=INK, weight="bold")
    for i, t in enumerate(["Xiangqi engine", "2086-move", "action space",
                           "policy-value", "network", "one Elo scale"]):
        s += txt(95, 120 + i * 20, t, size=12, c=INK)
    # four agents
    agents = ["Minimax  (alpha-beta search)",
              "PPO  (self-play reinforcement)",
              "Imitation learning  ->  PPO",
              "MCTS over the network  (AlphaZero)"]
    acol = [BLK, "#2563eb", "#7c3aed", TEAL]
    for i, (a, c) in enumerate(zip(agents, acol)):
        y = 70 + i * 46
        s += box(230, y, 250, 36, fill="#f8fafc", stroke=c)
        s += txt(355, y + 18, a, size=12, c=c, weight="bold")
        s += arrow(170, 160, 228, y + 18, c=GREY, w=1.6)
    # right evaluation
    s += box(540, 110, 160, 100, fill="#fef2f2", stroke=RED)
    s += txt(620, 135, "Evaluation", size=13, c=INK, weight="bold")
    s += txt(620, 158, "league:", size=12, c=INK)
    s += txt(620, 176, "round-robin ->", size=12, c=INK)
    s += txt(620, 192, "Elo + 95% CI", size=12, c=INK)
    for i in range(4):
        s += arrow(482, 88 + i * 46, 538, 160, c=GREY, w=1.6)
    save("fig4_framework.svg", w, h, s)


# ---- Figure 5: four agent mechanisms ---------------------------------------
def fig5():
    w, h = 720, 440; pw, ph = 340, 190; gap = 20
    s = txt(w / 2, 20, "The four decision mechanisms", size=15, c=INK, weight="bold")
    def panel(px, py, title, c):
        return box(px, py, pw, ph, fill="#f8fafc", stroke=c) + \
            txt(px + 14, py + 20, title, size=13, c=c, anchor="start", weight="bold")
    # (a) Minimax tree
    px, py = gap, 34
    s += panel(px, py, "(a) Minimax: alpha-beta search", BLK)
    root = (px + pw / 2, py + 46)
    kids = [(px + 70, py + 110), (px + pw / 2, py + 110), (px + pw - 70, py + 110)]
    for i, k in enumerate(kids):
        s += ln(root[0], root[1], k[0], k[1], c=INK)
        for j in range(2):
            gx = k[0] - 22 + j * 44; gy = py + 160
            dash = "4 3" if (i == 2 and j == 1) else ""
            s += ln(k[0], k[1], gx, gy, c=("#cbd5e1" if dash else INK), w=1.4, dash=dash)
            s += circ(gx, gy, 7, "#e2e8f0" if dash else "#fff")
        s += circ(k[0], k[1], 9, "#fff")
    s += circ(root[0], root[1], 10, INK)
    s += txt(px + pw - 54, py + 150, "pruned", size=10, c=GREY)
    # (b) PPO loop
    px = gap * 2 + pw
    s += panel(px, py, "(b) PPO: self-play reinforcement", "#2563eb")
    cx, cy, rr = px + pw / 2, py + 110, 54
    pts = [(cx, cy - rr, "self-play"), (cx + rr + 10, cy, "advantage"),
           (cx, cy + rr, "clipped update"), (cx - rr - 14, cy, "better policy")]
    for (x, y, t) in pts:
        s += circ(x, y, 5, "#2563eb")
        s += txt(x, y + (18 if y > cy else -14), t, size=11, c=INK)
    s += (f'<path d="M {cx+rr} {cy} A {rr} {rr} 0 1 1 {cx-rr} {cy}" fill="none" '
          f'stroke="#2563eb" stroke-width="1.8" marker-end="url(#ah)"/>')
    # (c) Imitation
    px, py = gap, 34 + ph + gap
    s += panel(px, py, "(c) Imitation learning (supervised)", "#7c3aed")
    s += box(px + 20, py + 70, 80, 40, fill="#fff", stroke="#7c3aed")
    s += txt(px + 60, py + 90, "board", size=12, c=INK)
    s += arrow(px + 104, py + 90, px + 150, py + 90, c=INK)
    s += box(px + 152, py + 70, 90, 40, fill="#fff", stroke="#7c3aed")
    s += txt(px + 197, py + 90, "network", size=12, c=INK)
    s += arrow(px + 246, py + 90, px + 292, py + 90, c=INK)
    s += txt(px + pw - 36, py + 90, "human", size=11, c=INK)
    s += txt(px + pw - 36, py + 104, "move", size=11, c=INK)
    s += txt(px + pw / 2, py + 140, "loss = cross-entropy(move) + MSE(value)",
             size=11, c=GREY)
    # (d) MCTS
    px = gap * 2 + pw
    s += panel(px, py, "(d) MCTS over the network (AlphaZero)", TEAL)
    for i, t in enumerate(["1. select (PUCT)", "2. expand",
                           "3. evaluate (value head)", "4. backpropagate"]):
        s += txt(px + 24, py + 48 + i * 26, t, size=12, c=INK, anchor="start")
    s += txt(px + pw / 2, py + ph - 14, "move = most-visited child", size=11, c=GREY)
    save("fig5_mechanisms.svg", w, h, s)


# ---- Figure 6: self-play training loop -------------------------------------
def fig6():
    w, h = 560, 300
    s = txt(w / 2, 24, "Self-play training loop (AlphaZero-style)",
            size=15, c=INK, weight="bold")
    cx, cy, rr = w / 2, 160, 95
    nodes = [(cx, cy - rr, "self-play games\n(MCTS)"),
             (cx + rr, cy, "data:\nstate, policy, outcome"),
             (cx, cy + rr, "train the\npolicy-value network"),
             (cx - rr, cy, "a stronger\nnetwork")]
    for (x, y, t) in nodes:
        s += box(x - 78, y - 24, 156, 48, fill="#ecfeff", stroke=TEAL)
        for j, line in enumerate(t.split("\n")):
            s += txt(x, y - 6 + j * 16, line, size=12, c=INK)
    seq = nodes + [nodes[0]]
    import math
    for a, b in zip(seq, seq[1:]):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        x1 = a[0] + math.cos(ang) * 86; y1 = a[1] + math.sin(ang) * 30
        x2 = b[0] - math.cos(ang) * 86; y2 = b[1] - math.sin(ang) * 30
        s += arrow(x1, y1, x2, y2, c=TEAL, w=2)
    save("fig6_selfplay.svg", w, h, s)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done ->", OUT)
