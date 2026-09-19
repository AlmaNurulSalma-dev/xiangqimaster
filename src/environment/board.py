"""Board representation for Xiangqi.

This is pure Layer-1 game state: it stores *where the pieces are* and offers
basic operations on that state. It deliberately knows NOTHING about how pieces
move or whether a position is check/checkmate — that logic lives in
``pieces.py``, ``move_generator.py`` and ``rules.py`` (see docs/02-ARCHITECTURE.md
dependency rules: ``environment/`` may import ``utils/`` only).

Storage model
-------------
The board is a single ``(10, 9)`` NumPy array of small integers. Each cell is
one intersection ("point") on the board:

* ``0``            → empty
* a positive value → a Red piece    (e.g. ``+CHARIOT``)
* a negative value → a Black piece   (e.g. ``-CHARIOT``)

So a cell's *magnitude* is the piece type and its *sign* is the owner. This
makes captures, ownership checks and cloning cheap and unambiguous.

Coordinates are always ``(row, col)``, 0-indexed, with row 0 = Red's back rank
at the bottom and row 9 = Black's back rank at the top (docs/01-GAME-RULES.md
section 2).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from src.utils import config
from src.utils.config import (
    ADVISOR,
    BLACK,
    BOARD_COLS,
    BOARD_ROWS,
    CANNON,
    CHARIOT,
    ELEPHANT,
    EMPTY,
    GENERAL,
    HORSE,
    PIECE_LETTERS,
    RED,
    SOLDIER,
)

# Back-rank piece layout, left → right, shared by both sides
# (docs/01-GAME-RULES.md section 4).
_BACK_RANK: tuple[int, ...] = (
    CHARIOT,
    HORSE,
    ELEPHANT,
    ADVISOR,
    GENERAL,
    ADVISOR,
    ELEPHANT,
    HORSE,
    CHARIOT,
)


class Board:
    """The Xiangqi board state and low-level operations on it."""

    __slots__ = ("grid", "to_move")

    def __init__(self, *, empty: bool = False) -> None:
        """Create a board.

        Args:
            empty: if True, start with a blank board (useful for tests that
                place pieces by hand). Otherwise start from the standard
                opening position with Red to move.
        """
        self.grid: np.ndarray = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        self.to_move: int = RED
        if not empty:
            self.reset()

    # ─── Setup ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Set the standard starting position, Red to move."""
        self.grid.fill(EMPTY)

        # Back ranks: Red (positive) on row 0, Black (negative) on row 9.
        for col, piece in enumerate(_BACK_RANK):
            self.grid[0, col] = RED * piece
            self.grid[9, col] = BLACK * piece

        # Cannons: row 2 for Red, row 7 for Black, on columns 1 and 7.
        for col in (1, 7):
            self.grid[2, col] = RED * CANNON
            self.grid[7, col] = BLACK * CANNON

        # Soldiers: row 3 for Red, row 6 for Black, on columns 0, 2, 4, 6, 8.
        for col in (0, 2, 4, 6, 8):
            self.grid[3, col] = RED * SOLDIER
            self.grid[6, col] = BLACK * SOLDIER

        self.to_move = RED

    # ─── Queries ──────────────────────────────────────────────────────────
    @staticmethod
    def is_inside(row: int, col: int) -> bool:
        """Is ``(row, col)`` a valid point on the board?"""
        return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS

    def piece_at(self, row: int, col: int) -> int:
        """Signed piece code at a point (``0`` if empty)."""
        return int(self.grid[row, col])

    def type_at(self, row: int, col: int) -> int:
        """Unsigned piece type at a point (``EMPTY`` if empty)."""
        return abs(int(self.grid[row, col]))

    def color_at(self, row: int, col: int) -> int:
        """Owner of the piece at a point: ``RED``, ``BLACK`` or ``0`` (empty)."""
        value = int(self.grid[row, col])
        if value > 0:
            return RED
        if value < 0:
            return BLACK
        return 0

    def is_empty(self, row: int, col: int) -> bool:
        return self.grid[row, col] == EMPTY

    def is_enemy(self, row: int, col: int, color: int) -> bool:
        """True if the point holds a piece owned by ``color``'s opponent."""
        return self.color_at(row, col) == -color

    def find_general(self, color: int) -> tuple[int, int] | None:
        """Return the ``(row, col)`` of ``color``'s General, or None if absent.

        (A missing General should only happen in hand-built test positions;
        normal play ends the moment a General would be captured.)
        """
        target = color * GENERAL
        hits = np.argwhere(self.grid == target)
        if len(hits) == 0:
            return None
        row, col = hits[0]
        return int(row), int(col)

    def pieces_of(self, color: int) -> Iterator[tuple[int, int, int]]:
        """Yield ``(row, col, piece_type)`` for every piece owned by ``color``."""
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                value = int(self.grid[row, col])
                if value != 0 and (value > 0) == (color > 0):
                    yield row, col, abs(value)

    # ─── Mutation ─────────────────────────────────────────────────────────
    def apply_move(self, from_r: int, from_c: int, to_r: int, to_c: int) -> int:
        """Move a piece and switch the side to move.

        Any enemy piece on the destination is captured (overwritten). No
        legality checking happens here — callers are responsible for supplying
        a legal move (``move_generator`` guarantees this in normal play).

        Returns:
            The signed code of the captured piece, or ``0`` if none.
        """
        captured = int(self.grid[to_r, to_c])
        self.grid[to_r, to_c] = self.grid[from_r, from_c]
        self.grid[from_r, from_c] = EMPTY
        self.to_move = -self.to_move
        return captured

    # ─── Copying ──────────────────────────────────────────────────────────
    def clone(self) -> "Board":
        """Return a fully independent copy.

        Required for MCTS and for legality testing (try a move on a clone,
        inspect the result, discard it). Mutating the clone must never affect
        the original — the ``.copy()`` on the grid is what guarantees that
        (docs/03-ENVIRONMENT.md section 6).
        """
        other = Board(empty=True)
        other.grid = self.grid.copy()
        other.to_move = self.to_move
        return other

    # ─── Hashing / repetition support ─────────────────────────────────────
    def position_key(self) -> tuple:
        """A hashable key identifying this position (pieces + side to move).

        Used by ``rules.py`` for threefold-repetition detection.
        """
        return (self.grid.tobytes(), self.to_move)

    # ─── Rendering ────────────────────────────────────────────────────────
    def to_ascii(self) -> str:
        """Human-readable board (Red uppercase, Black lowercase).

        Row 9 (Black's back rank) is printed on top, row 0 (Red) at the
        bottom, matching how the board is normally drawn.
        """
        lines: list[str] = []
        for row in range(BOARD_ROWS - 1, -1, -1):
            cells: list[str] = []
            for col in range(BOARD_COLS):
                value = int(self.grid[row, col])
                if value == EMPTY:
                    cells.append(".")
                else:
                    letter = PIECE_LETTERS[abs(value)]
                    cells.append(letter if value > 0 else letter.lower())
            lines.append(f"{row} " + " ".join(cells))
            if row == 5:  # draw the river between rows 5 and 4
                lines.append("  " + "- " * BOARD_COLS + "(river)")
        lines.append("  " + " ".join(str(c) for c in range(BOARD_COLS)))
        turn = "RED" if self.to_move == RED else "BLACK"
        lines.append(f"to move: {turn}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Board to_move={'RED' if self.to_move == RED else 'BLACK'}>"


# Re-export the config module so callers can do ``board.config`` if handy.
__all__ = ["Board", "config"]
