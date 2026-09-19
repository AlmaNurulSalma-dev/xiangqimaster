"""Unit tests for the fixed action-space map and legal mask."""

from __future__ import annotations

from src.environment import action_space
from src.environment.board import Board
from src.utils.config import ACTION_SPACE_SIZE, RED


def test_action_space_has_exactly_2086_moves():
    assert len(action_space.INDEX_TO_MOVE) == ACTION_SPACE_SIZE == 2086


def test_map_is_bijective_and_roundtrips():
    assert len(action_space.MOVE_TO_INDEX) == ACTION_SPACE_SIZE
    for index in (0, 1, 500, 1000, 2085):
        move = action_space.index_to_move(index)
        assert action_space.move_to_index(move) == index


def test_legal_mask_matches_legal_move_count_at_start():
    from src.environment.move_generator import generate_legal_moves

    board = Board()
    mask = action_space.legal_mask(board, RED)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == bool
    assert int(mask.sum()) == len(generate_legal_moves(board, RED)) == 44


def test_every_legal_move_is_encodable():
    # Play a handful of random-ish positions and confirm every generated legal
    # move has an index (i.e. the 2086 set covers all real moves).
    import random

    from src.environment.move_generator import generate_legal_moves

    rng = random.Random(7)
    board = Board()
    for _ in range(60):
        legal = generate_legal_moves(board)
        if not legal:
            break
        for move in legal:
            assert move in action_space.MOVE_TO_INDEX  # would KeyError otherwise
        board.apply_move(*rng.choice(legal))
