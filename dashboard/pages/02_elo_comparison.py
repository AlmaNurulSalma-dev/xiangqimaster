"""Elo comparison — Experiment 1 results (docs/10-DASHBOARD.md section 5)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.results_io import default_results_dir, load_table  # noqa: E402

st.title("📊 Elo Comparison")
st.caption("Agent strength from the Experiment 1 round-robin, with 95% CIs.")

tables = default_results_dir()
elo_rows = load_table(os.path.join(tables, "exp1_elo.csv"))
if not elo_rows:
    st.info("Run Experiment 1 first (`results/tables/exp1_elo.csv`).")
else:
    elo_rows = sorted(elo_rows, key=lambda r: float(r["elo"]), reverse=True)
    agents = [r["agent"] for r in elo_rows]
    elos = [float(r["elo"]) for r in elo_rows]
    cis = [float(r.get("elo_ci95", 0) or 0) for r in elo_rows]
    fig = go.Figure(
        go.Bar(x=agents, y=elos, error_y=dict(type="data", array=cis),
               marker_color="#2980b9")
    )
    fig.update_layout(yaxis_title="Elo", xaxis_title="Agent", height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(elo_rows, use_container_width=True)

    h2h = load_table(os.path.join(tables, "exp1_head_to_head.csv"))
    if h2h:
        st.subheader("Head-to-head win rates")
        st.dataframe(h2h, use_container_width=True)
