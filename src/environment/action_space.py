"""Fixed action-space encoding for Xiangqi (2086 moves).

A neural-network policy head outputs one score per possible move, so every move
needs a stable integer index. This module builds a fixed, bidirectional map
between a move ``(from_row, from_col, to_row, to_col)`` and an integer
``action_index`` in ``[0, 2086)``.

The enumeration follows the widely-used "cchess-zero" label scheme so the
indexing is compatible with other Xiangqi AlphaZero-style codebases:

* all straight-line moves (any change of row within a file, any change of file
  within a rank) — covers Chariot, Cannon, General, and Soldier moves;
* all eight Horse jumps;
* the 16 Advisor diagonal steps inside the two palaces;
* the 32 Elephant two-step diagonals.

The map is built once at import time and never changes (docs/03-ENVIRONMENT.md
section 3.2). Not every indexed move is legal in a given position — that is what
the legal mask is for (:func:`legal_mask`).
"""

from __future__ import annotations

import numpy as np

from src.environment.board import Board
from src.utils.config import ACTION_SPACE_SIZE, BOARD_COLS, BOARD_ROWS

FullMove = tuple[int, int, int, int]  # (from_row, from_col, to_row, to_col)

# Horse jumps expressed as (d_col, d_row), matching the reference scheme.
_HORSE_DELTAS: tuple[tuple[int, int], ...] = (
    (-2, -1), (-1, -2), (-2, 1), (1, -2),
    (2, -1), (-1, 2), (2, 1), (1, 2),
)

# Advisor and Elephant moves are position-specific, listed here as
# "<col><row><col><row>" strings with columns a–i (0–8) and rows 0–9.
_ADVISOR_LABELS: tuple[str, ...] = (
    "d0e1", "e1d0", "e1f2", "f2e1", "d2e1", "e1d2", "f0e1", "e1f0",
    "d9e8", "e8d9", "e8f7", "f7e8", "d7e8", "e8d7", "f9e8", "e8f9",
)
_ELEPHANT_LABELS: tuple[str, ...] = (
    "a2c0", "c0a2", "a2c4", "c4a2", "c0e2", "e2c0", "c4e2", "e2c4",
    "e2g0", "g0e2", "e2g4", "g4e2", "g0i2", "i2g0", "g4i2", "i2g4",
    "a7c9", "c9a7", "a7c5", "c5a7", "c9e7", "e7c9", "c5e7", "e7c5",
    "e7g9", "g9e7", "e7g5", "g5e7", "g9i7", "i7g9", "g5i7", "i7g5",
)


def _label_to_move(label: str) -> FullMove:
    """Convert a 'd0e1'-style label to ``(from_row, from_col, to_row, to_col)``."""
    from_col = ord(label[0]) - ord("a")
    from_row = int(label[1])
    to_col = ord(label[2]) - ord("a")
    to_row = int(label[3])
    return (from_row, from_col, to_row, to_col)


def _build_moves() -> list[FullMove]:
    """Enumerate all 2086 possible moves in the fixed reference order."""
    moves: list[FullMove] = []
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS):
            destinations: list[tuple[int, int]] = (
                [(col, r) for r in range(BOARD_ROWS)]       # same file (vertical)
                + [(c, row) for c in range(BOARD_COLS)]      # same rank (horizontal)
                + [(col + dc, row + dr) for dc, dr in _HORSE_DELTAS]
            )
            for to_col, to_row in destinations:
                if (col, row) == (to_col, to_row):
                    continue
                if 0 <= to_col < BOARD_COLS and 0 <= to_row < BOARD_ROWS:
                    moves.append((row, col, to_row, to_col))

    moves.extend(_label_to_move(lbl) for lbl in _ADVISOR_LABELS)
    moves.extend(_label_to_move(lbl) for lbl in _ELEPHANT_LABELS)
    return moves


# ─── Build the map once, at import time ─────────────────────────────────────
INDEX_TO_MOVE: tuple[FullMove, ...] = tuple(_build_moves())
MOVE_TO_INDEX: dict[FullMove, int] = {m: i for i, m in enumerate(INDEX_TO_MOVE)}

# Fail loudly if the enumeration ever drifts from the documented size.
assert len(INDEX_TO_MOVE) == ACTION_SPACE_SIZE, (
    f"action space has {len(INDEX_TO_MOVE)} moves, expected {ACTION_SPACE_SIZE}"
)
assert len(MOVE_TO_INDEX) == ACTION_SPACE_SIZE, "duplicate moves in action space"


# ─── Public helpers ─────────────────────────────────────────────────────────
def move_to_index(move: FullMove) -> int:
    """Integer index for a move (raises KeyError if the move is not encodable)."""
    return MOVE_TO_INDEX[move]


def index_to_move(index: int) -> FullMove:
    """The move for a given action index."""
    return INDEX_TO_MOVE[index]


def legal_mask(board: Board, color: int | None = None) -> np.ndarray:
    """Boolean mask of length 2086: ``True`` at every legal move's index.

    This is the mandatory action mask (docs/03-ENVIRONMENT.md section 3.3): a
    policy must set illegal logits to -inf before sampling so an agent can never
    choose an illegal move.
    """
    # Local import avoids a load-time cycle (move_generator imports rules,
    # which lazily imports move_generator).
    from src.environment.move_generator import generate_legal_moves

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in generate_legal_moves(board, color):
        mask[MOVE_TO_INDEX[move]] = True
    return mask


__all__ = [
    "FullMove",
    "INDEX_TO_MOVE",
    "MOVE_TO_INDEX",
    "move_to_index",
    "index_to_move",
    "legal_mask",
]
