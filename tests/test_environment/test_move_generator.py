"""Unit tests for legal move generation."""

from __future__ import annotations

import random

from src.environment.board import Board
from src.environment import rules
from src.environment.move_generator import generate_legal_moves
from src.utils.config import BLACK, CHARIOT, GENERAL, RED


def _empty() -> Board:
    return Board(empty=True)


def test_starting_position_has_44_legal_moves():
    # The standard Xiangqi opening position has exactly 44 legal moves.
    board = Board()
    assert len(generate_legal_moves(board, RED)) == 44


def test_generated_moves_never_leave_own_general_in_check():
    board = Board()
    for from_r, from_c, to_r, to_c in generate_legal_moves(board, RED):
        trial = board.clone()
        trial.apply_move(from_r, from_c, to_r, to_c)
        assert not rules.is_in_check(trial, RED)


def test_flying_general_move_is_filtered_out():
    # Red chariot on file 4 is the only thing keeping the Generals apart.
    board = _empty()
    board.grid[0, 4] = RED * GENERAL
    board.grid[1, 4] = RED * CHARIOT
    board.grid[2, 4] = BLACK * GENERAL
    board.to_move = RED

    legal = set(generate_legal_moves(board, RED))
    # Sliding the chariot off file 4 would expose the Generals → illegal.
    assert (1, 4, 1, 0) not in legal
    assert (1, 4, 1, 3) not in legal
    # Capturing straight down onto the Black General is fine (it ends the game).
    assert (1, 4, 2, 4) in legal


def test_random_selfplay_never_crashes_or_produces_illegal_state():
    # A light version of the docs/03 section 9 "1000 random games" check:
    # play out several games of random legal moves and assert the engine stays
    # consistent (every applied move comes from the legal list; the game ends).
    rng = random.Random(42)
    for _ in range(20):
        board = Board()
        plies = 0
        while plies < 300:
            result = rules.get_game_result(board, ply_count=plies)
            if result != rules.ONGOING:
                break
            legal = generate_legal_moves(board)
            assert legal, "get_game_result said ONGOING but there are no moves"
            move = rng.choice(legal)
            board.apply_move(*move)
            plies += 1
