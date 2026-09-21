"""Unit tests for UCCI helpers (FEN + coordinate conversion)."""

from __future__ import annotations

from src.environment.board import Board
from src.environment.elephanteye import board_to_fen, move_to_ucci, ucci_to_move


def test_move_ucci_roundtrip():
    move = (0, 0, 1, 0)
    assert move_to_ucci(move) == "a0a1"
    assert ucci_to_move("a0a1") == move


def test_ucci_conversion_uses_files_and_ranks():
    # (row 2, col 7) -> (row 2, col 4): the central-cannon move.
    assert move_to_ucci((2, 7, 2, 4)) == "h2e2"
    assert ucci_to_move("h2e2") == (2, 7, 2, 4)


def test_startpos_fen():
    assert board_to_fen(Board()) == (
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
    )


def test_fen_side_to_move_flips():
    board = Board()
    board.apply_move(2, 7, 2, 4)  # Red moves → Black to move
    assert board_to_fen(board).endswith(" b")
