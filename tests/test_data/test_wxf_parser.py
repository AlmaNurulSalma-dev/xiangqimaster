"""Unit tests for the WXF parser (docs/04-DATA-PIPELINE.md section 13)."""

from __future__ import annotations

import pytest

from src.data import wxf_parser
from src.data.wxf_parser import (
    DRAW,
    WXFParseError,
    detect_notation,
    parse_game,
    parse_iccs_move,
    parse_move,
    parse_wxf_move,
)
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


# ─── ICCS coordinate notation ──────────────────────────────────────────────


def test_iccs_move_maps_file_and_rank_to_row_col():
    # "h2e2": file h = col 7, rank 2 = row 2 → file e = col 4, rank 2. This is
    # the same central-cannon move as WXF "C2.5".
    assert parse_iccs_move("h2e2") == (2, 7, 2, 4)


def test_iccs_corner_move():
    # "a0a1": file a = col 0, rank 0 = row 0 → rank 1 (chariot advances one).
    assert parse_iccs_move("a0a1") == (0, 0, 1, 0)


def test_detect_notation_distinguishes_iccs_from_wxf():
    assert detect_notation("h2e2") == "iccs"
    assert detect_notation("C2.5") == "wxf"


def test_parse_move_auto_detects_iccs():
    assert parse_move(Board(), "h2e2") == (2, 7, 2, 4)


def test_parse_move_auto_detects_wxf():
    assert parse_move(Board(), "C2.5") == (2, 7, 2, 4)


def test_parse_move_rejects_unknown_notation():
    with pytest.raises(WXFParseError):
        parse_move(Board(), "h2e2", notation="pgn")


def test_iccs_rejects_malformed_token():
    with pytest.raises(WXFParseError):
        parse_iccs_move("h2e")  # too short


def test_parse_game_replays_iccs_moves():
    # Both sides open with the central cannon in ICCS coordinates.
    game = parse_game(["h2e2", "h7e7"], outcome=DRAW, notation="iccs")
    assert len(game.moves) == 2
    assert game.moves[0] == (2, 7, 2, 4)
    assert game.moves[1] == (7, 7, 7, 4)


def test_parse_game_auto_notation_handles_iccs():
    game = parse_game(["h2e2", "h7e7"], outcome=DRAW)
    assert game.moves[0] == (2, 7, 2, 4)


# ─── ICCS xqbase dialect: uppercase + dash (dpxq / WXF-Federation datasets) ──


def test_iccs_uppercase_dash_variant():
    # xqbase ICCS writes the central cannon as "H2-E2" (== UCCI "h2e2").
    assert parse_iccs_move("H2-E2") == (2, 7, 2, 4)
    assert parse_iccs_move("h2e2") == (2, 7, 2, 4)


def test_detect_notation_recognises_uppercase_dash_iccs():
    assert detect_notation("C3-C4") == "iccs"
    # A WXF backward move like "H2-3" must NOT be mistaken for a coordinate move.
    assert detect_notation("H2-3") == "wxf"


def test_parse_game_replays_full_xqbase_iccs_game():
    # First 8 plies of the CGLemon README sample game (xqbase ICCS, std start).
    tokens = ["C3-C4", "C9-E7", "B2-D2", "G6-G5", "B0-C2", "B9-C7", "A0-B0", "A9-B9"]
    game = parse_game(tokens, outcome=RED, notation="iccs")
    assert len(game.moves) == 8
    assert game.moves[0] == (3, 2, 4, 2)  # C3-C4
