"""Game-state rules for Xiangqi: check, checkmate, stalemate, and results.

This module is the "referee". It answers three kinds of question:

* Is a given General under attack right now?  → :func:`is_in_check`
* Does the side to move have any legal reply?  → :func:`has_legal_moves`
* Has the game ended, and how?                 → :func:`get_game_result`

Two Xiangqi specifics are handled here and are easy to get wrong if you carry
over chess habits (docs/01-GAME-RULES.md section 6):

* **Stalemate is a LOSS**, not a draw. A side with no legal move loses whether
  or not it is in check.
* **Flying general (飞将):** the two Generals may not face each other down an
  open file. A position where they do is treated as check, so any move that
  would create it is illegal.

Import note: :func:`is_in_check` depends only on ``pieces``/``board``. The
checkmate/stalemate/result helpers need legal-move generation, which they pull
in lazily to avoid a circular import with ``move_generator``.
"""

from __future__ import annotations

from src.environment import pieces
from src.environment.board import Board
from src.utils.config import (
    BLACK,
    GENERAL,
    MAX_PLIES,
    RED,
    REPETITION_LIMIT,
    opponent,
)

# ─── Game-result sentinels ─────────────────────────────────────────────────
# A finished game returns the winner (RED or BLACK) or DRAW; an unfinished one
# returns ONGOING.
ONGOING: str = "ongoing"
DRAW: str = "draw"


# ─── Flying-general helper ─────────────────────────────────────────────────
def generals_face(board: Board) -> bool:
    """True if the two Generals sit on the same file with nothing between them.

    Such a position is forbidden (the "flying general" rule), so it is treated
    as check for whichever side is asked about.
    """
    red_pos = board.find_general(RED)
    black_pos = board.find_general(BLACK)
    if red_pos is None or black_pos is None:
        return False
    if red_pos[1] != black_pos[1]:
        return False  # different files → cannot face
    col = red_pos[1]
    low, high = sorted((red_pos[0], black_pos[0]))
    for row in range(low + 1, high):
        if not board.is_empty(row, col):
            return False  # a piece stands between them
    return True


# ─── Check detection ───────────────────────────────────────────────────────
def is_in_check(board: Board, color: int) -> bool:
    """Is ``color``'s General currently under attack?

    Returns True if any enemy piece can (pseudo-legally) reach the General's
    point, or if the flying-general rule applies. A missing General (only
    possible in hand-built positions) counts as "in check".
    """
    general_pos = board.find_general(color)
    if general_pos is None:
        return True

    if generals_face(board):
        return True

    enemy = opponent(color)
    for row, col, _piece_type in board.pieces_of(enemy):
        if general_pos in pieces.piece_moves(board, row, col):
            return True
    return False


# ─── Legal-move availability ───────────────────────────────────────────────
def has_legal_moves(board: Board, color: int) -> bool:
    """Does ``color`` have at least one legal move?"""
    # Lazy import breaks the rules <-> move_generator cycle.
    from src.environment.move_generator import generate_legal_moves

    return len(generate_legal_moves(board, color)) > 0


def is_checkmate(board: Board, color: int) -> bool:
    """In check AND no legal move to escape → ``color`` is checkmated (loses)."""
    return is_in_check(board, color) and not has_legal_moves(board, color)


def is_stalemate(board: Board, color: int) -> bool:
    """No legal move but NOT in check.

    In Xiangqi this is a LOSS for ``color`` (unlike chess, where it is a draw).
    """
    return not is_in_check(board, color) and not has_legal_moves(board, color)


# ─── Overall game result ───────────────────────────────────────────────────
def get_game_result(
    board: Board,
    *,
    repetition_count: int = 0,
    ply_count: int = 0,
) -> str | int:
    """Determine the game result from the perspective of the side to move.

    Args:
        board: current position (``board.to_move`` is the side to move).
        repetition_count: how many times the current position has occurred
            (the environment tracks this via ``board.position_key()``).
        ply_count: number of half-moves played so far.

    Returns:
        ``RED`` or ``BLACK`` if that side has won, :data:`DRAW`, or
        :data:`ONGOING` if the game continues.

    Note:
        Both checkmate and stalemate reduce to the same thing here — if the
        side to move has no legal move, it loses — so they need not be
        distinguished at this level.
    """
    color = board.to_move
    if not has_legal_moves(board, color):
        return opponent(color)  # side to move cannot move → it loses
    if repetition_count >= REPETITION_LIMIT:
        return DRAW
    if ply_count >= MAX_PLIES:
        return DRAW
    return ONGOING


__all__ = [
    "ONGOING",
    "DRAW",
    "generals_face",
    "is_in_check",
    "has_legal_moves",
    "is_checkmate",
    "is_stalemate",
    "get_game_result",
]
