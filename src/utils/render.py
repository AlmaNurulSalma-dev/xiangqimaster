"""Xiangqi board rendering to a Plotly figure (docs/10-DASHBOARD.md section 10).

Shared by the dashboard's live-game and game-explorer pages. Given a
:class:`Board`, ``board_to_plotly`` returns an interactive figure with the grid,
the river, the palace diagonals, and the pieces drawn as labelled discs (Red
uppercase-style characters, Black the other form).
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.environment.board import Board
from src.utils.config import (
    ADVISOR,
    BOARD_COLS,
    BOARD_ROWS,
    CANNON,
    CHARIOT,
    ELEPHANT,
    GENERAL,
    HORSE,
    RED,
    SOLDIER,
)

_RED_LABELS = {
    GENERAL: "帥", ADVISOR: "仕", ELEPHANT: "相", HORSE: "傌",
    CHARIOT: "俥", CANNON: "炮", SOLDIER: "兵",
}
_BLACK_LABELS = {
    GENERAL: "將", ADVISOR: "士", ELEPHANT: "象", HORSE: "馬",
    CHARIOT: "車", CANNON: "砲", SOLDIER: "卒",
}

_RED_COLOR = "#c0392b"
_BLACK_COLOR = "#2c3e50"


def piece_label(piece_type: int, color: int) -> str:
    """The character used to draw a piece of the given type and colour."""
    labels = _RED_LABELS if color == RED else _BLACK_LABELS
    return labels[piece_type]


def _grid_shapes() -> list[dict]:
    shapes: list[dict] = []
    # Horizontal rank lines (full width).
    for row in range(BOARD_ROWS):
        shapes.append(dict(type="line", x0=0, y0=row, x1=BOARD_COLS - 1, y1=row,
                           line=dict(color="#8d6e63", width=1)))
    # Vertical file lines — broken at the river (between rows 4 and 5) except
    # the two outer files.
    for col in range(BOARD_COLS):
        if col in (0, BOARD_COLS - 1):
            shapes.append(dict(type="line", x0=col, y0=0, x1=col, y1=BOARD_ROWS - 1,
                               line=dict(color="#8d6e63", width=1)))
        else:
            shapes.append(dict(type="line", x0=col, y0=0, x1=col, y1=4,
                               line=dict(color="#8d6e63", width=1)))
            shapes.append(dict(type="line", x0=col, y0=5, x1=col, y1=BOARD_ROWS - 1,
                               line=dict(color="#8d6e63", width=1)))
    # Palace diagonals (both palaces).
    for y0, y1 in ((0, 2), (7, 9)):
        shapes.append(dict(type="line", x0=3, y0=y0, x1=5, y1=y1,
                           line=dict(color="#8d6e63", width=1)))
        shapes.append(dict(type="line", x0=5, y0=y0, x1=3, y1=y1,
                           line=dict(color="#8d6e63", width=1)))
    return shapes


def board_to_plotly(board: Board, title: str | None = None) -> go.Figure:
    """Render ``board`` as an interactive Plotly figure."""
    fig = go.Figure()

    for color, name, disc in ((RED, "Red", _RED_COLOR), (-RED, "Black", _BLACK_COLOR)):
        xs, ys, texts = [], [], []
        for row, col, piece_type in board.pieces_of(color):
            xs.append(col)
            ys.append(row)
            texts.append(piece_label(piece_type, color))
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, text=texts, mode="markers+text", name=name,
                textfont=dict(color="white", size=16),
                marker=dict(size=28, color=disc, line=dict(color="white", width=1)),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        shapes=_grid_shapes(),
        xaxis=dict(range=[-0.5, BOARD_COLS - 0.5], showgrid=False,
                   zeroline=False, visible=False),
        yaxis=dict(range=[-0.5, BOARD_ROWS - 0.5], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#f5deb3",
        showlegend=False,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=560,
    )
    return fig


__all__ = ["board_to_plotly", "piece_label"]
