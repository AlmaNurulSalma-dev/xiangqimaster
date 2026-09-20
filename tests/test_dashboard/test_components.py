"""Unit tests for dashboard components (results IO, playback, agent factory)."""

from __future__ import annotations

from dashboard.components.agents_ui import AGENT_NAMES, make_agent
from dashboard.components.game_playback import record_game
from dashboard.components.results_io import load_table
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.environment.board import Board


# ─── results_io ─────────────────────────────────────────────────────────────
def test_load_table_reads_rows(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("agent,elo\nA,1600\nB,1400\n", encoding="utf-8")
    rows = load_table(str(path))
    assert rows == [{"agent": "A", "elo": "1600"}, {"agent": "B", "elo": "1400"}]


def test_load_table_missing_returns_none(tmp_path):
    assert load_table(str(tmp_path / "nope.csv")) is None


# ─── game_playback ──────────────────────────────────────────────────────────
def test_record_game_snapshots_align_with_moves():
    moves, snapshots = record_game(
        RandomAgent(seed=1), RandomAgent(seed=2), seed=0, max_plies=20
    )
    assert len(snapshots) == len(moves) + 1  # snapshot 0 is the start position


def test_record_game_snapshots_are_independent():
    _, snapshots = record_game(
        RandomAgent(seed=1), RandomAgent(seed=2), seed=0, max_plies=6
    )
    # The start snapshot must not have been mutated by later moves.
    assert (snapshots[0].grid == Board().grid).all()


# ─── agents_ui ──────────────────────────────────────────────────────────────
def test_make_agent_types():
    assert isinstance(make_agent("Random"), RandomAgent)
    mm = make_agent("Minimax d3")
    assert isinstance(mm, MinimaxAgent) and mm.depth == 3


def test_all_listed_agents_are_constructable():
    for name in AGENT_NAMES:
        assert make_agent(name) is not None
