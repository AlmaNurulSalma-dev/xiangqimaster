"""Legal move generation for Xiangqi.

``pieces.py`` produces *pseudo-legal* moves (correct geometry, but ignoring
whether your own General ends up in check). This module applies the final
filter: a move is legal only if, after making it, your own General is not in
check. That single rule also enforces the flying-general restriction and the
duty to escape check, because ``rules.is_in_check`` accounts for both.

A move is represented as a 4-tuple ``(from_row, from_col, to_row, to_col)``.
"""

from __future__ import annotations

from src.environment import pieces, rules
from src.environment.board import Board

FullMove = tuple[int, int, int, int]  # (from_row, from_col, to_row, to_col)


def generate_legal_moves(board: Board, color: int | None = None) -> list[FullMove]:
    """All fully-legal moves for ``color`` (defaults to the side to move).

    For each of ``color``'s pieces we take its pseudo-legal destinations, try
    each on an independent clone, and keep it only if ``color``'s General is
    safe afterwards.
    """
    if color is None:
        color = board.to_move

    legal: list[FullMove] = []
    for from_r, from_c, _piece_type in board.pieces_of(color):
        for to_r, to_c in pieces.piece_moves(board, from_r, from_c):
            trial = board.clone()
            trial.apply_move(from_r, from_c, to_r, to_c)
            if not rules.is_in_check(trial, color):
                legal.append((from_r, from_c, to_r, to_c))
    return legal


__all__ = ["FullMove", "generate_legal_moves"]
