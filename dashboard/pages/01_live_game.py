"""Live Game viewer + demo advantage bar (docs/10-DASHBOARD.md section 4).

Built for the thesis-defence demo: an AlphaGo-style framing, an AI-vs-AI
showcase, difficulty labels on the agent ladder, a chess.com-style advantage
bar so a non-Xiangqi audience can see who is winning, and last-move highlighting.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st  # noqa: E402

from dashboard.components.advantage import bar_html, evaluate_snapshots  # noqa: E402
from dashboard.components.agents_ui import (  # noqa: E402
    AGENT_NAMES,
    make_agent,
    pikafish_available,
)
from dashboard.components.game_playback import record_game  # noqa: E402
from src.utils.render import board_to_plotly  # noqa: E402

# Agent ladder → difficulty, shown next to each name so the audience can map
# "which AI" to "how strong" (Random → Minimax → IL → MCTS, plus the engine).
_DIFFICULTY = {
    "Random": "Easy",
    "Minimax d1": "Easy",
    "Minimax d2": "Medium",
    "Minimax d3": "Medium",
    "IL (human)": "Medium",
    "Distill (Pikafish)": "Medium",
    "MCTS v2": "Hard",
    "Pikafish MAX": "Max (engine)",
}


def _fmt(name: str) -> str:
    tag = _DIFFICULTY.get(name)
    return f"{name}  ·  {tag}" if tag else name


def _play(red_name: str, black_name: str, seed: int) -> None:
    """Play a game and score every position for the advantage bar."""
    with st.spinner(f"Playing {red_name} (Red) vs {black_name} (Black)…"):
        moves, snapshots = record_game(make_agent(red_name), make_agent(black_name), seed=seed)
    with st.spinner("Scoring each position for the advantage bar…"):
        advs = evaluate_snapshots(snapshots)
    st.session_state.update(
        snapshots=snapshots, moves=moves, advs=advs, matchup=(red_name, black_name)
    )


st.title("♟️ Live Game")
st.markdown(
    "> **Like AlphaGo — but for Chinese Chess.** These agents learned to play from "
    "millions of human moves and self-play games. Watch two of them play, and "
    "follow the **advantage bar** to see who's winning — no Xiangqi knowledge needed."
)

if pikafish_available():
    st.caption("Advantage bar powered by the **Pikafish** engine (centipawn score).")
else:
    st.caption("Advantage bar powered by a trained **value network** (Pikafish not found).")

# ─── Pick the match ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
default_red = AGENT_NAMES.index("MCTS v2") if "MCTS v2" in AGENT_NAMES else len(AGENT_NAMES) - 1
red_name = col1.selectbox("Red (moves first)", AGENT_NAMES, index=default_red, format_func=_fmt)
black_name = col2.selectbox("Black", AGENT_NAMES, index=0, format_func=_fmt)
seed = col3.number_input("Seed", min_value=0, value=0, step=1)

btn1, btn2 = st.columns(2)
if btn1.button("▶ Play game", use_container_width=True):
    _play(red_name, black_name, int(seed))

# One-click showcase: our strongest learned agent vs a classical baseline.
show_red = "MCTS v2" if "MCTS v2" in AGENT_NAMES else red_name
show_black = "Minimax d2" if "Minimax d2" in AGENT_NAMES else black_name
if btn2.button(f"🎬 Showcase: {show_red} vs {show_black}", use_container_width=True):
    _play(show_red, show_black, int(seed))

# ─── Playback ────────────────────────────────────────────────────────────────
snapshots = st.session_state.get("snapshots")
if not snapshots:
    st.info("Pick agents and click **Play game**, or hit **Showcase**.")
else:
    moves = st.session_state["moves"]
    advs = st.session_state["advs"]
    red_name, black_name = st.session_state.get("matchup", (red_name, black_name))
    max_ply = len(snapshots) - 1
    st.markdown(f"**Red:** {red_name}  **vs  Black:** {black_name} — {max_ply} plies played")

    bar_ph = st.empty()
    board_ph = st.empty()

    def _render(ply: int) -> None:
        last_move = moves[ply - 1] if ply > 0 else None
        bar_ph.markdown(bar_html(advs[ply]), unsafe_allow_html=True)
        board_ph.plotly_chart(
            board_to_plotly(snapshots[ply], title=f"After {ply} of {max_ply} plies",
                            last_move=last_move),
            use_container_width=True,
        )

    if st.button("⏯ Auto-play"):
        for k in range(max_ply + 1):
            _render(k)
            time.sleep(0.45)
        st.session_state["ply"] = max_ply

    ply = st.slider("Ply", 0, max_ply, st.session_state.get("ply", max_ply))
    st.session_state["ply"] = ply
    _render(ply)
