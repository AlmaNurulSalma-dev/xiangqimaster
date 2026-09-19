"""Unit tests for per-piece movement rules (docs/01-GAME-RULES.md section 9).

Each test builds a bare board and places just the pieces it needs, so the
tricky Xiangqi rules can be checked in isolation.
"""

from __future__ import annotations

from src.environment.board import Board
from src.environment import pieces
from src.utils.config import (
    BLACK,
    CANNON,
    CHARIOT,
    ELEPHANT,
    GENERAL,
    HORSE,
    RED,
    SOLDIER,
    ADVISOR,
)


def _empty() -> Board:
    return Board(empty=True)


# ─── Chariot ────────────────────────────────────────────────────────────────
def test_chariot_slides_on_empty_board():
    board = _empty()
    board.grid[0, 0] = RED * CHARIOT
    moves = set(pieces.chariot_moves(board, 0, 0))
    # Whole column 0 (rows 1-9) and whole row 0 (cols 1-8).
    expected = {(r, 0) for r in range(1, 10)} | {(0, c) for c in range(1, 9)}
    assert moves == expected


def test_chariot_stops_at_friend_and_captures_enemy():
    board = _empty()
    board.grid[0, 0] = RED * CHARIOT
    board.grid[0, 4] = RED * SOLDIER      # friend blocks further travel
    board.grid[5, 0] = BLACK * SOLDIER    # enemy is capturable
    moves = set(pieces.chariot_moves(board, 0, 0))
    assert (0, 1) in moves and (0, 2) in moves and (0, 3) in moves
    assert (0, 4) not in moves            # cannot land on a friend
    assert (0, 5) not in moves            # cannot pass the friend
    assert (5, 0) in moves                # captures the enemy
    assert (6, 0) not in moves            # cannot pass the captured piece


# ─── Cannon ─────────────────────────────────────────────────────────────────
def test_cannon_moves_over_empty_without_capturing():
    board = _empty()
    board.grid[0, 0] = RED * CANNON
    moves = set(pieces.cannon_moves(board, 0, 0))
    expected = {(r, 0) for r in range(1, 10)} | {(0, c) for c in range(1, 9)}
    assert moves == expected  # with no screen, behaves like a chariot on empties


def test_cannon_captures_over_exactly_one_screen():
    board = _empty()
    board.grid[0, 0] = RED * CANNON
    board.grid[0, 4] = BLACK * SOLDIER    # the screen (any colour works)
    board.grid[0, 7] = BLACK * CHARIOT    # enemy target behind the screen
    moves = set(pieces.cannon_moves(board, 0, 0))
    assert (0, 1) in moves and (0, 3) in moves   # empties before the screen
    assert (0, 4) not in moves                    # cannot capture the screen itself
    assert (0, 5) not in moves and (0, 6) not in moves  # no capture without a jump
    assert (0, 7) in moves                        # capture across one screen


def test_cannon_cannot_capture_over_two_screens():
    board = _empty()
    board.grid[0, 0] = RED * CANNON
    board.grid[0, 4] = BLACK * SOLDIER    # screen 1
    board.grid[0, 5] = BLACK * SOLDIER    # screen 2
    board.grid[0, 7] = BLACK * CHARIOT    # target is behind TWO screens
    moves = set(pieces.cannon_moves(board, 0, 0))
    assert (0, 7) not in moves            # two screens → no capture


# ─── Horse ──────────────────────────────────────────────────────────────────
def test_horse_full_moves_in_open_centre():
    board = _empty()
    board.grid[4, 4] = RED * HORSE
    moves = set(pieces.horse_moves(board, 4, 4))
    expected = {
        (6, 5), (6, 3), (2, 5), (2, 3),
        (5, 6), (3, 6), (5, 2), (3, 2),
    }
    assert moves == expected


def test_horse_blocked_at_the_leg():
    board = _empty()
    board.grid[4, 4] = RED * HORSE
    board.grid[5, 4] = BLACK * SOLDIER    # blocks the "leg" going up
    moves = set(pieces.horse_moves(board, 4, 4))
    # The two moves that pass through the blocked leg are removed.
    assert (6, 5) not in moves and (6, 3) not in moves
    # Other directions still available.
    assert (2, 5) in moves and (5, 6) in moves


# ─── Elephant ────────────────────────────────────────────────────────────────
def test_elephant_two_step_diagonal():
    board = _empty()
    board.grid[0, 2] = RED * ELEPHANT
    moves = set(pieces.elephant_moves(board, 0, 2))
    assert moves == {(2, 0), (2, 4)}


def test_elephant_cannot_cross_river():
    board = _empty()
    board.grid[4, 2] = RED * ELEPHANT     # on the river edge (Red side row 4)
    moves = set(pieces.elephant_moves(board, 4, 2))
    # Destinations at row 6 would be across the river → forbidden.
    assert (6, 0) not in moves and (6, 4) not in moves
    assert (2, 0) in moves and (2, 4) in moves


def test_elephant_blocked_at_the_eye():
    board = _empty()
    board.grid[0, 2] = RED * ELEPHANT
    board.grid[1, 3] = BLACK * SOLDIER    # occupies the "eye" toward (2,4)
    moves = set(pieces.elephant_moves(board, 0, 2))
    assert (2, 4) not in moves            # blocked eye
    assert (2, 0) in moves                # other diagonal is fine


# ─── Advisor & General (palace confinement) ──────────────────────────────────
def test_advisor_confined_to_palace():
    # From a palace corner, the advisor's only diagonal step is to the centre.
    board = _empty()
    board.grid[0, 3] = RED * ADVISOR
    assert set(pieces.advisor_moves(board, 0, 3)) == {(1, 4)}

    # From the palace centre, it reaches all four corners (fresh board so no
    # friendly pieces block the corners).
    board = _empty()
    board.grid[1, 4] = RED * ADVISOR
    assert set(pieces.advisor_moves(board, 1, 4)) == {(0, 3), (0, 5), (2, 3), (2, 5)}


def test_general_confined_to_palace():
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    assert set(pieces.general_moves(board, 0, 4)) == {(0, 3), (0, 5), (1, 4)}


# ─── Soldier ─────────────────────────────────────────────────────────────────
def test_soldier_forward_only_before_river():
    board = _empty()
    board.grid[3, 4] = RED * SOLDIER      # Red soldier, not yet across
    assert set(pieces.soldier_moves(board, 3, 4)) == {(4, 4)}


def test_soldier_gains_sideways_after_river():
    board = _empty()
    board.grid[5, 4] = RED * SOLDIER      # Red soldier has crossed (row >= 5)
    assert set(pieces.soldier_moves(board, 5, 4)) == {(6, 4), (5, 5), (5, 3)}


def test_black_soldier_moves_downward():
    board = _empty()
    board.grid[6, 4] = BLACK * SOLDIER    # Black moves toward row 0
    assert set(pieces.soldier_moves(board, 6, 4)) == {(5, 4)}


# ─── Dispatch ────────────────────────────────────────────────────────────────
def test_piece_moves_dispatch_and_empty():
    board = Board()
    assert pieces.piece_moves(board, 4, 4) == []             # empty midfield
    assert set(pieces.piece_moves(board, 0, 0)) != set()     # a real chariot
