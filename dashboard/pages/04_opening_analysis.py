"""Opening analysis — Experiment 3 / RQ3 (docs/10-DASHBOARD.md section 7)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.results_io import default_results_dir, load_table  # noqa: E402

st.title("🧭 Opening Analysis")
st.caption(
    "How far each agent's opening style sits from professional play "
    "(KL divergence), and its most common opening class."
)

rows = load_table(os.path.join(default_results_dir(), "exp3_openings.csv"))
if not rows:
    st.info("Run Experiment 3 first (`results/tables/exp3_openings.csv`).")
else:
    have_kl = any(r.get("kl_vs_reference") not in (None, "") for r in rows)
    if have_kl:
        agents = [r["agent"] for r in rows]
        kls = [float(r["kl_vs_reference"] or 0) for r in rows]
        fig = go.Figure(go.Bar(x=agents, y=kls, marker_color="#8e44ad"))
        fig.update_layout(
            yaxis_title="KL divergence vs reference", xaxis_title="Agent", height=440
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Most common opening per agent")
    st.dataframe(rows, use_container_width=True)
