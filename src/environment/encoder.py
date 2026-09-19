"""Board → tensor encoding for the neural network.

Produces the ``14 x 10 x 9`` (channels x rows x cols) tensor described in
docs/03-ENVIRONMENT.md section 2 and docs/05-NEURAL-NETWORK.md.

This encoder is deliberately part of Layer 1 (``environment/``) even though
docs/02-ARCHITECTURE.md sketches it under ``data/``: the environment itself must
emit observations, and the dependency rules forbid ``environment`` importing
``data``. The data pipeline imports THIS encoder so training data and self-play
observations are guaranteed identical (docs/04-DATA-PIPELINE.md section 4).

Player-relative encoding (the key design decision, docs/03 section 2.2)
----------------------------------------------------------------------
The board is always encoded from the perspective of the side to move:

* channels 0–6  → the current player's 7 piece types (General … Soldier)
* channels 7–13 → the opponent's 7 piece types

and, when Black is to move, the board is rotated 180° so the current player's
pieces sit at the bottom and "move upward". This lets a single network play
both colours — it always sees "me at the bottom vs. opponent at the top".
"""

from __future__ import annotations

import numpy as np

from src.environment.board import Board
from src.utils.config import (
    BLACK,
    BOARD_COLS,
    BOARD_ROWS,
    NUM_CHANNELS,
    RED,
)


def _orient(row: int, col: int, perspective: int) -> tuple[int, int]:
    """Map absolute ``(row, col)`` into the perspective's canonical frame.

    Red's frame is the board as stored. Black's frame is rotated 180° so Black
    also "moves upward".
    """
    if perspective == RED:
        return row, col
    return BOARD_ROWS - 1 - row, BOARD_COLS - 1 - col


def encode(board: Board, perspective: int | None = None) -> np.ndarray:
    """Encode ``board`` as a ``(14, 10, 9)`` float32 tensor.

    Args:
        board: the position to encode.
        perspective: whose perspective to encode from (``RED`` or ``BLACK``).
            Defaults to ``board.to_move`` — the normal case.

    Returns:
        A ``(NUM_CHANNELS, BOARD_ROWS, BOARD_COLS)`` array of 0.0/1.0 planes.
    """
    if perspective is None:
        perspective = board.to_move

    tensor = np.zeros((NUM_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            value = int(board.grid[row, col])
            if value == 0:
                continue
            color = RED if value > 0 else BLACK
            piece_type = abs(value)  # 1..7
            base = 0 if color == perspective else 7  # my planes vs opponent's
            channel = base + (piece_type - 1)
            r, c = _orient(row, col, perspective)
            tensor[channel, r, c] = 1.0
    return tensor


__all__ = ["encode"]
