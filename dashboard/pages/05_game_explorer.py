"""Game Explorer — step through a game move by move (docs/10-DASHBOARD.md 8)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st  # noqa: E402

from dashboard.components.agents_ui import AGENT_NAMES, make_agent  # noqa: E402
from dashboard.components.game_playback import record_game  # noqa: E402
from src.utils.render import board_to_plotly  # noqa: E402

st.title("🔎 Game Explorer")
st.caption("Generate a game between two agents and inspect it position by position.")

col1, col2, col3 = st.columns(3)
red_name = col1.selectbox("Red", AGENT_NAMES, index=2, key="ge_red")
black_name = col2.selectbox("Black", AGENT_NAMES, index=1, key="ge_black")
seed = col3.number_input("Seed", min_value=0, value=7, step=1, key="ge_seed")

if st.button("Generate game"):
    with st.spinner("Playing…"):
        moves, snapshots = record_game(
            make_agent(red_name), make_agent(black_name), seed=int(seed)
        )
    st.session_state["ge_snapshots"] = snapshots
    st.session_state["ge_moves"] = moves

snapshots = st.session_state.get("ge_snapshots")
if not snapshots:
    st.info("Pick agents and click **Generate game**.")
else:
    moves = st.session_state["ge_moves"]
    board_col, list_col = st.columns([3, 1])
    ply = board_col.slider("Ply", 0, len(snapshots) - 1, 0)
    board_col.plotly_chart(
        board_to_plotly(snapshots[ply], title=f"Ply {ply}"),
        use_container_width=True,
    )
    with list_col:
        st.write("**Moves**")
        st.write(
            "\n".join(
                f"{i + 1}. {m}" for i, m in enumerate(moves[:ply])
            )
            or "_(start)_"
        )
