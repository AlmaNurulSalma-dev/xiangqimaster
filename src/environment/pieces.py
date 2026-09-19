"""Per-piece movement rules for Xiangqi.

Each function takes the board and the coordinates of a single piece and returns
the list of destination points that piece could move to, considering ONLY the
geometry of that piece and what is currently on the board. These are
*pseudo-legal* moves: they do not check whether making the move would leave
your own General in check — that filtering happens in ``move_generator.py``.

All the distinctive Xiangqi rules live here:

* Chariot stops at the first piece; captures an enemy, is blocked by a friend.
* Cannon moves like a Chariot over empty points, but captures by jumping
  exactly one "screen" piece.
* Horse can be blocked at the "leg" (蹩马腿) — unlike the chess knight.
* Elephant cannot cross the river and is blocked at its "eye" (塞象眼).
* Advisor and General are confined to the palace.
* Soldier moves only forward until it crosses the river, then also sideways,
  and never backward.

The full rulebook is docs/01-GAME-RULES.md section 3.
"""

from __future__ import annotations

from src.environment.board import Board
from src.utils.config import (
    ADVISOR,
    BLACK,
    CANNON,
    CHARIOT,
    ELEPHANT,
    GENERAL,
    HORSE,
    PALACE_COLS,
    RED,
    RED_PALACE_ROWS,
    BLACK_PALACE_ROWS,
    SOLDIER,
)

Move = tuple[int, int]  # a destination (row, col)

# The four orthogonal and four diagonal unit directions.
_ORTHOGONAL: tuple[Move, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL: tuple[Move, ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ─── Small geometry helpers ────────────────────────────────────────────────
def _in_palace(color: int, row: int, col: int) -> bool:
    """Is ``(row, col)`` inside ``color``'s 3x3 palace?"""
    if not (PALACE_COLS[0] <= col <= PALACE_COLS[1]):
        return False
    rows = RED_PALACE_ROWS if color == RED else BLACK_PALACE_ROWS
    return rows[0] <= row <= rows[1]


def _on_own_side(color: int, row: int) -> bool:
    """Is ``row`` on ``color``'s own side of the river (Elephant restriction)?"""
    return row <= 4 if color == RED else row >= 5


def _has_crossed_river(color: int, row: int) -> bool:
    """Has a soldier of ``color`` crossed the river (gaining sideways moves)?"""
    return row >= 5 if color == RED else row <= 4


def _forward(color: int) -> int:
    """Row delta for a forward step: Red moves up (+1), Black moves down (-1)."""
    return 1 if color == RED else -1


# ─── One function per piece type ───────────────────────────────────────────
def general_moves(board: Board, row: int, col: int) -> list[Move]:
    """General: one orthogonal step, confined to the palace.

    The "flying general" face-off rule is enforced during check detection
    (rules.py), not here, so this only produces the four palace-bounded steps.
    """
    color = board.color_at(row, col)
    moves: list[Move] = []
    for dr, dc in _ORTHOGONAL:
        r, c = row + dr, col + dc
        if _in_palace(color, r, c) and board.color_at(r, c) != color:
            moves.append((r, c))
    return moves


def advisor_moves(board: Board, row: int, col: int) -> list[Move]:
    """Advisor: one diagonal step, confined to the palace."""
    color = board.color_at(row, col)
    moves: list[Move] = []
    for dr, dc in _DIAGONAL:
        r, c = row + dr, col + dc
        if _in_palace(color, r, c) and board.color_at(r, c) != color:
            moves.append((r, c))
    return moves


def elephant_moves(board: Board, row: int, col: int) -> list[Move]:
    """Elephant: exactly two points diagonally, cannot cross the river,
    and is blocked if the midpoint ("eye") of the diagonal is occupied."""
    color = board.color_at(row, col)
    moves: list[Move] = []
    for dr, dc in _DIAGONAL:
        eye_r, eye_c = row + dr, col + dc          # midpoint of the jump
        r, c = row + 2 * dr, col + 2 * dc          # destination
        if not board.is_inside(r, c):
            continue
        if not _on_own_side(color, r):             # may not cross the river
            continue
        if not board.is_empty(eye_r, eye_c):       # eye blocked (塞象眼)
            continue
        if board.color_at(r, c) != color:          # empty or enemy
            moves.append((r, c))
    return moves


def horse_moves(board: Board, row: int, col: int) -> list[Move]:
    """Horse: an L-shape, but blocked at the "leg" (蹩马腿).

    For each L-move, the orthogonal point one step toward the target (the leg)
    must be empty; otherwise that direction is blocked and the horse cannot go.
    """
    color = board.color_at(row, col)
    moves: list[Move] = []
    # (leg offset) -> the two destinations reachable past that leg.
    leg_to_dests: tuple[tuple[Move, tuple[Move, Move]], ...] = (
        ((1, 0), ((2, 1), (2, -1))),    # leg down  → two forward-down L's
        ((-1, 0), ((-2, 1), (-2, -1))),  # leg up
        ((0, 1), ((1, 2), (-1, 2))),    # leg right
        ((0, -1), ((1, -2), (-1, -2))),  # leg left
    )
    for (leg_dr, leg_dc), dests in leg_to_dests:
        leg_r, leg_c = row + leg_dr, col + leg_dc
        if not board.is_inside(leg_r, leg_c) or not board.is_empty(leg_r, leg_c):
            continue  # leg is off-board or hobbled → this whole direction blocked
        for dr, dc in dests:
            r, c = row + dr, col + dc
            if board.is_inside(r, c) and board.color_at(r, c) != color:
                moves.append((r, c))
    return moves


def chariot_moves(board: Board, row: int, col: int) -> list[Move]:
    """Chariot: slides orthogonally until it hits a piece.

    It stops before a friendly piece and captures (then stops on) an enemy.
    """
    color = board.color_at(row, col)
    moves: list[Move] = []
    for dr, dc in _ORTHOGONAL:
        r, c = row + dr, col + dc
        while board.is_inside(r, c):
            occupant = board.color_at(r, c)
            if occupant == 0:
                moves.append((r, c))          # empty → keep sliding
            else:
                if occupant != color:
                    moves.append((r, c))      # enemy → capture, then stop
                break                          # any piece blocks further travel
            r, c = r + dr, c + dc
    return moves


def cannon_moves(board: Board, row: int, col: int) -> list[Move]:
    """Cannon: moves like a Chariot over empty points, but captures only by
    jumping exactly one "screen" piece and landing on an enemy beyond it."""
    color = board.color_at(row, col)
    moves: list[Move] = []
    for dr, dc in _ORTHOGONAL:
        r, c = row + dr, col + dc
        # Phase 1: travel over empty points (non-capturing moves).
        while board.is_inside(r, c) and board.is_empty(r, c):
            moves.append((r, c))
            r, c = r + dr, c + dc
        # Now (r, c) is the first occupied point in this direction: the screen.
        # Phase 2: look past the screen for the first piece — capture if enemy.
        r, c = r + dr, c + dc
        while board.is_inside(r, c):
            occupant = board.color_at(r, c)
            if occupant != 0:
                if occupant != color:
                    moves.append((r, c))      # enemy behind exactly one screen
                break                          # first piece past the screen only
            r, c = r + dr, c + dc
    return moves


def soldier_moves(board: Board, row: int, col: int) -> list[Move]:
    """Soldier: one point forward; also sideways after crossing the river;
    never backward and never diagonally."""
    color = board.color_at(row, col)
    moves: list[Move] = []
    candidates: list[Move] = [(row + _forward(color), col)]  # always: forward
    if _has_crossed_river(color, row):
        candidates.append((row, col + 1))   # sideways right
        candidates.append((row, col - 1))   # sideways left
    for r, c in candidates:
        if board.is_inside(r, c) and board.color_at(r, c) != color:
            moves.append((r, c))
    return moves


# ─── Dispatch ──────────────────────────────────────────────────────────────
_DISPATCH = {
    GENERAL: general_moves,
    ADVISOR: advisor_moves,
    ELEPHANT: elephant_moves,
    HORSE: horse_moves,
    CHARIOT: chariot_moves,
    CANNON: cannon_moves,
    SOLDIER: soldier_moves,
}


def piece_moves(board: Board, row: int, col: int) -> list[Move]:
    """Pseudo-legal destinations for whatever piece sits at ``(row, col)``.

    Returns an empty list if the point is empty.
    """
    piece_type = board.type_at(row, col)
    if piece_type == 0:
        return []
    return _DISPATCH[piece_type](board, row, col)


__all__ = [
    "Move",
    "general_moves",
    "advisor_moves",
    "elephant_moves",
    "horse_moves",
    "chariot_moves",
    "cannon_moves",
    "soldier_moves",
    "piece_moves",
]
