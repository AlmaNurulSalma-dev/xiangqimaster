"""Unit tests for the imitation-learning dataset."""

from __future__ import annotations

import torch

from src.data.dataset import LazyXiangqiILDataset, XiangqiILDataset, mirror_move
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


# ─── LazyXiangqiILDataset ────────────────────────────────────────────────────


def _same_example(a, b) -> bool:
    ta, aa, va = a
    tb, ab, vb = b
    return bool(torch.equal(ta, tb)) and aa == ab and va == vb


def test_lazy_dataset_matches_eager_no_mirror():
    games = [_sample_game(6, outcome=RED), _sample_game(8, outcome=RED)]
    eager = XiangqiILDataset(games)
    lazy = LazyXiangqiILDataset(games)
    assert len(lazy) == len(eager) == 14
    # Without mirror both lay out game-major, ply-order → identical per index.
    for i in range(len(eager)):
        assert _same_example(lazy[i], eager[i]), f"mismatch at {i}"


def test_lazy_dataset_mirror_doubles_and_matches_multiset():
    games = [_sample_game(6, outcome=RED)]
    eager = XiangqiILDataset(games, mirror=True)
    lazy = LazyXiangqiILDataset(games, mirror=True)
    assert len(lazy) == len(eager) == 12
    # Layout differs (eager interleaves, lazy appends), so compare as a multiset
    # of (action, value, tensor-bytes).
    def multiset(ds):
        out = []
        for i in range(len(ds)):
            t, a, v = ds[i]
            out.append((a, v, t.numpy().tobytes()))
        return sorted(out)
    assert multiset(lazy) == multiset(eager)


def test_lazy_dataset_shapes_and_draw_value():
    from src.data.wxf_parser import DRAW

    game = _sample_game(6, outcome=DRAW)
    lazy = LazyXiangqiILDataset([game], draw_value=0.0)
    tensor, action, value = lazy[0]
    assert tuple(tensor.shape) == (14, 10, 9)
    assert action == action_space.move_to_index(game.moves[0])
    assert value == 0.0


def test_lazy_dataset_is_memory_light():
    # It must NOT pre-materialise tensors (the whole point).
    game = _sample_game(6)
    lazy = LazyXiangqiILDataset([game])
    assert not hasattr(lazy, "_tensors")
