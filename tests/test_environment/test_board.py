"""Unit tests for the Board (docs/03-ENVIRONMENT.md section 9 checklist)."""

from __future__ import annotations

import numpy as np

from src.environment.board import Board
from src.utils.config import (
    BLACK,
    BOARD_COLS,
    BOARD_ROWS,
    CANNON,
    CHARIOT,
    EMPTY,
    GENERAL,
    RED,
    SOLDIER,
)


# ─── Starting position ─────────────────────────────────────────────────────
def test_starting_shape_and_turn():
    board = Board()
    assert board.grid.shape == (BOARD_ROWS, BOARD_COLS)
    assert board.to_move == RED  # Red always moves first


def test_starting_generals_in_place():
    board = Board()
    # Generals sit at the centre of each back rank, inside the palace.
    assert board.piece_at(0, 4) == RED * GENERAL
    assert board.piece_at(9, 4) == BLACK * GENERAL


def test_starting_chariots_and_cannons_and_soldiers():
    board = Board()
    # Chariots at the corners of the back ranks.
    assert board.piece_at(0, 0) == RED * CHARIOT
    assert board.piece_at(0, 8) == RED * CHARIOT
    assert board.piece_at(9, 0) == BLACK * CHARIOT
    # Cannons on row 2 (Red) / row 7 (Black), columns 1 and 7.
    assert board.piece_at(2, 1) == RED * CANNON
    assert board.piece_at(7, 7) == BLACK * CANNON
    # Soldiers on row 3 (Red) / row 6 (Black), even columns.
    assert board.piece_at(3, 0) == RED * SOLDIER
    assert board.piece_at(6, 4) == BLACK * SOLDIER
    # The point in front of a soldier gap is empty.
    assert board.piece_at(3, 1) == EMPTY


def test_starting_piece_counts():
    board = Board()
    red_pieces = list(board.pieces_of(RED))
    black_pieces = list(board.pieces_of(BLACK))
    assert len(red_pieces) == 16  # each side has 16 pieces
    assert len(black_pieces) == 16


# ─── Queries ───────────────────────────────────────────────────────────────
def test_color_at():
    board = Board()
    assert board.color_at(0, 0) == RED
    assert board.color_at(9, 0) == BLACK
    assert board.color_at(4, 4) == 0  # empty midfield


def test_is_inside_bounds():
    assert Board.is_inside(0, 0)
    assert Board.is_inside(9, 8)
    assert not Board.is_inside(-1, 0)
    assert not Board.is_inside(10, 0)
    assert not Board.is_inside(0, 9)


def test_find_general():
    board = Board()
    assert board.find_general(RED) == (0, 4)
    assert board.find_general(BLACK) == (9, 4)


def test_find_general_missing_returns_none():
    board = Board(empty=True)
    assert board.find_general(RED) is None


# ─── Mutation ────────────────────────────────────────────────────────────
def test_apply_move_moves_piece_and_switches_turn():
    board = Board()
    # Advance the Red soldier on column 0 from row 3 to row 4.
    captured = board.apply_move(3, 0, 4, 0)
    assert captured == EMPTY
    assert board.piece_at(3, 0) == EMPTY
    assert board.piece_at(4, 0) == RED * SOLDIER
    assert board.to_move == BLACK  # turn passed to the opponent


def test_apply_move_capture_returns_captured_piece():
    board = Board(empty=True)
    board.grid[0, 0] = RED * CHARIOT
    board.grid[5, 0] = BLACK * SOLDIER
    captured = board.apply_move(0, 0, 5, 0)
    assert captured == BLACK * SOLDIER
    assert board.piece_at(5, 0) == RED * CHARIOT
    assert board.piece_at(0, 0) == EMPTY


# ─── Cloning (critical for MCTS) ───────────────────────────────────────────
def test_clone_is_independent():
    board = Board()
    clone = board.clone()
    # Mutating the clone must never touch the original.
    clone.apply_move(3, 0, 4, 0)
    assert board.piece_at(3, 0) == RED * SOLDIER  # original unchanged
    assert board.piece_at(4, 0) == EMPTY
    assert board.to_move == RED
    assert clone.to_move == BLACK


def test_clone_grids_are_separate_arrays():
    board = Board()
    clone = board.clone()
    assert not np.shares_memory(board.grid, clone.grid)


# ─── Repetition key ────────────────────────────────────────────────────────
def test_position_key_distinguishes_turn():
    board = Board()
    key_red = board.position_key()
    board.to_move = BLACK
    key_black = board.position_key()
    assert key_red != key_black


def test_position_key_stable_for_same_position():
    a = Board()
    b = Board()
    assert a.position_key() == b.position_key()
