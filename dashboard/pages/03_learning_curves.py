"""Learning curves — Experiment 2 / RQ1 (docs/10-DASHBOARD.md section 6)."""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.results_io import default_results_dir, load_table  # noqa: E402

st.title("📈 Learning Curves")
st.caption(
    "Elo vs training steps — pure PPO (scratch) vs IL-initialized PPO. "
    "The central RQ1 comparison: does human pre-training converge faster?"
)

rows = load_table(os.path.join(default_results_dir(), "exp2_learning_curves.csv"))
if not rows:
    st.info("Run Experiment 2 first (`results/tables/exp2_learning_curves.csv`).")
else:
    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        curves[r["variant"]].append((int(r["timesteps"]), float(r["elo"])))
    fig = go.Figure()
    for variant, points in curves.items():
        points.sort()
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points], y=[p[1] for p in points],
                mode="lines+markers", name=variant,
            )
        )
    fig.update_layout(xaxis_title="Training steps", yaxis_title="Elo", height=480)
    st.plotly_chart(fig, use_container_width=True)
