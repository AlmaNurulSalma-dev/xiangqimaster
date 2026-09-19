"""Unit tests for check / checkmate / stalemate / results (docs/01 section 6)."""

from __future__ import annotations

from src.environment.board import Board
from src.environment import rules
from src.utils.config import (
    BLACK,
    CHARIOT,
    GENERAL,
    RED,
    SOLDIER,
)


def _empty() -> Board:
    return Board(empty=True)


# ─── Check detection ───────────────────────────────────────────────────────
def test_not_in_check_at_start():
    board = Board()
    assert not rules.is_in_check(board, RED)
    assert not rules.is_in_check(board, BLACK)


def test_chariot_gives_check_down_open_file():
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    board.grid[9, 4] = BLACK * GENERAL   # will be handled by facing too; move it
    board.grid[9, 4] = 0
    board.grid[7, 0] = BLACK * GENERAL   # park Black general off the file
    board.grid[5, 4] = BLACK * CHARIOT   # attacks straight down file 4
    assert rules.is_in_check(board, RED)


def test_blocking_piece_stops_check():
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    board.grid[7, 0] = BLACK * GENERAL
    board.grid[5, 4] = BLACK * CHARIOT
    board.grid[3, 4] = RED * SOLDIER     # blocks the file
    assert not rules.is_in_check(board, RED)


# ─── Flying general ─────────────────────────────────────────────────────────
def test_flying_general_is_check_for_both():
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    board.grid[2, 4] = BLACK * GENERAL   # same file, nothing between
    assert rules.generals_face(board)
    assert rules.is_in_check(board, RED)
    assert rules.is_in_check(board, BLACK)


def test_flying_general_blocked_by_piece_between():
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    board.grid[2, 4] = BLACK * GENERAL
    board.grid[1, 4] = RED * SOLDIER     # a piece stands between them
    assert not rules.generals_face(board)


# ─── Checkmate ───────────────────────────────────────────────────────────────
def test_checkmate_is_a_loss_for_the_mated_side():
    # Red general boxed on the back rank by three Black chariots controlling
    # files 3, 4 and 5. It is in check (file 4) with no escape.
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    board.grid[9, 3] = BLACK * CHARIOT
    board.grid[9, 4] = BLACK * CHARIOT
    board.grid[9, 5] = BLACK * CHARIOT
    board.grid[9, 0] = BLACK * GENERAL   # off-file, no interference
    board.to_move = RED

    assert rules.is_in_check(board, RED)
    assert rules.is_checkmate(board, RED)
    assert not rules.is_stalemate(board, RED)
    assert rules.get_game_result(board) == BLACK  # the mover loses


# ─── Stalemate = loss (Xiangqi rule, unlike chess) ──────────────────────────
def test_stalemate_is_also_a_loss():
    # Red general not in check but with no legal move.
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    board.grid[1, 3] = BLACK * CHARIOT   # covers file 3 (escape square 0,3)
    board.grid[1, 5] = BLACK * CHARIOT   # covers file 5 (escape square 0,5)
    board.grid[2, 4] = BLACK * SOLDIER   # covers (1,4) but not (0,4)
    board.grid[9, 0] = BLACK * GENERAL
    board.to_move = RED

    assert not rules.is_in_check(board, RED)     # not in check ...
    assert rules.is_stalemate(board, RED)        # ... but no legal move
    assert rules.get_game_result(board) == BLACK  # still a loss for the mover


# ─── Draw conditions ────────────────────────────────────────────────────────
def test_repetition_is_a_draw():
    board = Board()  # normal opening, plenty of legal moves
    assert rules.get_game_result(board, repetition_count=3) == rules.DRAW


def test_move_limit_is_a_draw():
    board = Board()
    assert rules.get_game_result(board, ply_count=300) == rules.DRAW


def test_ongoing_when_game_continues():
    board = Board()
    assert rules.get_game_result(board) == rules.ONGOING
