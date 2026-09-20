"""Unit tests for the imitation-learning dataset."""

from __future__ import annotations

from src.data.dataset import XiangqiILDataset, mirror_move
from src.data.wxf_parser import Game
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.utils.config import RED


def _sample_game(n_plies: int = 6, outcome: int = RED) -> Game:
    board = Board()
    moves = []
    for _ in range(n_plies):
        legal = generate_legal_moves(board)
        move = legal[0]
        moves.append(move)
        board.apply_move(*move)
    return Game(moves=tuple(moves), outcome=outcome)


def test_dataset_length_and_example_shapes():
    game = _sample_game()
    ds = XiangqiILDataset([game])
    assert len(ds) == 6
    tensor, action, value = ds[0]
    assert tuple(tensor.shape) == (14, 10, 9)
    assert action == action_space.move_to_index(game.moves[0])


def test_value_labels_flip_by_mover():
    game = _sample_game(outcome=RED)
    ds = XiangqiILDataset([game])
    # Ply 0 mover is Red (outcome Red) → +1; ply 1 mover is Black → -1.
    assert ds[0][2] == 1.0
    assert ds[1][2] == -1.0


def test_draw_value_used_for_draws():
    from src.data.wxf_parser import DRAW

    game = _sample_game(outcome=DRAW)
    ds = XiangqiILDataset([game], draw_value=0.0)
    assert ds[0][2] == 0.0


def test_mirror_augmentation_doubles_dataset():
    game = _sample_game()
    plain = XiangqiILDataset([game])
    mirrored = XiangqiILDataset([game], mirror=True)
    assert len(mirrored) == 2 * len(plain)


def test_mirror_move_flips_columns():
    assert mirror_move((0, 0, 1, 0)) == (0, 8, 1, 8)
    assert mirror_move((2, 7, 2, 4)) == (2, 1, 2, 4)
