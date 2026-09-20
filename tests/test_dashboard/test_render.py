"""Unit tests for the Plotly board renderer."""

from __future__ import annotations

import plotly.graph_objects as go

from src.environment.board import Board
from src.utils.config import GENERAL, RED
from src.utils.render import board_to_plotly, piece_label


def test_returns_a_figure():
    assert isinstance(board_to_plotly(Board()), go.Figure)


def test_all_32_pieces_are_drawn_at_start():
    fig = board_to_plotly(Board())
    piece_points = sum(
        len(trace.x) for trace in fig.data if trace.name in ("Red", "Black")
    )
    assert piece_points == 32


def test_empty_board_draws_no_pieces():
    fig = board_to_plotly(Board(empty=True))
    piece_points = sum(
        len(trace.x) for trace in fig.data if trace.name in ("Red", "Black")
    )
    assert piece_points == 0


def test_piece_label_differs_by_colour():
    assert piece_label(GENERAL, RED) != piece_label(GENERAL, -RED)
