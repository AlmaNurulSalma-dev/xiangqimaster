"""XiangqiMaster dashboard — home / overview (docs/10-DASHBOARD.md section 9).

Run with::

    streamlit run dashboard/app.py

Read-only: it displays saved results and plays agents live; it never trains.
"""

from __future__ import annotations

import os
import sys

# Make the repo root importable when Streamlit runs this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st  # noqa: E402

from dashboard.components.results_io import default_results_dir, load_table  # noqa: E402

st.set_page_config(page_title="XiangqiMaster", page_icon="♟️", layout="wide")

st.title("♟️ XiangqiMaster · 象棋大师")
st.caption(
    "A Comparative Study of Deep Reinforcement Learning Approaches for Xiangqi "
    "(Chinese Chess) — from Imitation Learning to Self-Play Mastery."
)

st.markdown(
    """
**What this is.** Xiangqi (Chinese Chess) is a two-player board game on a
9×10 grid. This project trains and compares four approaches to playing it:
**PPO** (self-play RL), **IL+RL** (learn from human games, then self-play),
**MCTS+NN** (AlphaZero-style), and a classical **Minimax** baseline — all
measured by **Elo rating**.

Use the pages in the sidebar to watch agents play, compare their strength,
inspect training curves, and study their openings.
"""
)

st.subheader("Headline results")
elo_rows = load_table(os.path.join(default_results_dir(), "exp1_elo.csv"))
if not elo_rows:
    st.info(
        "No results yet. Run Experiment 1 to populate the Elo table "
        "(`results/tables/exp1_elo.csv`)."
    )
else:
    top = max(elo_rows, key=lambda r: float(r["elo"]))
    cols = st.columns(3)
    cols[0].metric("Strongest agent", top["agent"])
    cols[1].metric("Best Elo", f"{float(top['elo']):.0f}")
    cols[2].metric("Agents compared", str(len(elo_rows)))

st.divider()
st.markdown(
    "Pages: **Live Game** · **Elo Comparison** · **Learning Curves** · "
    "**Opening Analysis** · **Game Explorer**"
)
