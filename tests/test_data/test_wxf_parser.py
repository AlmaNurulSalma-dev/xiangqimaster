"""Unit tests for the WXF parser (docs/04-DATA-PIPELINE.md section 13)."""

from __future__ import annotations

import pytest

from src.data import wxf_parser
from src.data.wxf_parser import DRAW, WXFParseError, parse_game, parse_wxf_move
from src.environment.board import Board
from src.utils.config import BLACK, CHARIOT, GENERAL, RED


def test_central_cannon_for_red():
    # "C2.5": cannon on Red's file 2 traverses to file 5 (the central cannon).
    board = Board()
    assert parse_wxf_move(board, "C2.5") == (2, 7, 2, 4)


def test_red_file_numbering_is_from_the_right():
    board = Board()
    # File 1 is Red's rightmost = column 8; the corner chariot advancing 1 step.
    assert parse_wxf_move(board, "R1+1") == (0, 8, 1, 8)


def test_horse_forward_uses_destination_file():
    board = Board()
    # "H2+3": horse on file 2 (col 7) jumps forward to file 3 (col 6), up 2 rows.
    assert parse_wxf_move(board, "H2+3") == (0, 7, 2, 6)


def test_pawn_forward_one_step():
    board = Board()
    # "P3+1": pawn on file 3 (col 6) advances one row.
    assert parse_wxf_move(board, "P3+1") == (3, 6, 4, 6)


def test_black_move_mirrors_file_numbering():
    board = Board()
    board.apply_move(2, 7, 2, 4)  # Red plays central cannon; now Black to move
    # Black's file 2 is column 1; its central cannon.
    assert parse_wxf_move(board, "C2.5") == (7, 1, 7, 4)


def test_front_rear_disambiguation():
    board = Board(empty=True)
    board.grid[2, 4] = RED * CHARIOT   # rear (lower row for Red)
    board.grid[5, 4] = RED * CHARIOT   # front (higher row for Red)
    board.grid[0, 3] = RED * GENERAL
    board.grid[9, 5] = BLACK * GENERAL
    board.to_move = RED
    assert parse_wxf_move(board, "+R+1") == (5, 4, 6, 4)  # front chariot forward
    assert parse_wxf_move(board, "-R+1") == (2, 4, 3, 4)  # rear chariot forward


def test_parse_game_validates_and_returns_moves():
    game = parse_game(["C2.5", "C2.5", "H2+3", "H8+7"], outcome=DRAW)
    assert game.outcome == DRAW
    assert len(game.moves) == 4
    assert game.moves[0] == (2, 7, 2, 4)


def test_parse_game_rejects_illegal_move():
    # A chariot cannot slide 9 steps through its own back-rank pieces.
    with pytest.raises(WXFParseError):
        parse_game(["R1+9"], outcome=RED)


def test_unknown_piece_letter_raises():
    with pytest.raises(WXFParseError):
        parse_wxf_move(Board(), "Z2.5")
