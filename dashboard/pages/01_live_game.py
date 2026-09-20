"""Live Game viewer (docs/10-DASHBOARD.md section 4)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st  # noqa: E402

from dashboard.components.agents_ui import AGENT_NAMES, make_agent  # noqa: E402
from dashboard.components.game_playback import record_game  # noqa: E402
from src.utils.render import board_to_plotly  # noqa: E402

st.title("♟️ Live Game")
st.caption("Watch two agents play, then scrub through the game move by move.")

col1, col2, col3 = st.columns(3)
red_name = col1.selectbox("Red (moves first)", AGENT_NAMES, index=2)
black_name = col2.selectbox("Black", AGENT_NAMES, index=0)
seed = col3.number_input("Seed", min_value=0, value=0, step=1)

if st.button("▶ Play game"):
    with st.spinner("Playing…"):
        moves, snapshots = record_game(
            make_agent(red_name), make_agent(black_name), seed=int(seed)
        )
    st.session_state["snapshots"] = snapshots
    st.session_state["moves"] = moves

snapshots = st.session_state.get("snapshots")
if not snapshots:
    st.info("Pick agents and click **Play game**.")
else:
    max_ply = len(snapshots) - 1
    ply = st.slider("Ply", 0, max_ply, max_ply)
    st.plotly_chart(
        board_to_plotly(snapshots[ply], title=f"After {ply} of {max_ply} plies"),
        use_container_width=True,
    )
    st.write(f"Total plies played: **{max_ply}**")
