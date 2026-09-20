"""Unit tests for game-level splitting and DataLoader construction."""

from __future__ import annotations

from src.data.data_loader import make_dataloader, split_games
from src.data.dataset import XiangqiILDataset
from src.data.wxf_parser import Game
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.utils.config import RED


def _dummy_games(n: int) -> list[Game]:
    # Distinct one-move games so we can check split membership by identity.
    board = Board()
    first = generate_legal_moves(board)[0]
    return [Game(moves=(first,), outcome=RED) for _ in range(n)]


def test_split_sizes_and_no_overlap():
    games = _dummy_games(10)
    # Make them individually identifiable.
    tagged = [Game(moves=g.moves, outcome=i) for i, g in enumerate(games)]
    train, val, test = split_games(tagged, (0.8, 0.1, 0.1), seed=42)
    assert (len(train), len(val), len(test)) == (8, 1, 1)
    ids = lambda gs: {g.outcome for g in gs}
    assert ids(train) & ids(val) == set()
    assert ids(train) & ids(test) == set()
    assert ids(val) & ids(test) == set()


def test_split_is_deterministic():
    games = [Game(moves=(), outcome=i) for i in range(20)]
    a = split_games(games, seed=7)
    b = split_games(games, seed=7)
    assert [g.outcome for g in a[0]] == [g.outcome for g in b[0]]


def test_split_rejects_bad_fractions():
    import pytest

    with pytest.raises(ValueError):
        split_games(_dummy_games(4), (0.5, 0.4, 0.2))


def test_dataloader_yields_batched_tensors():
    board = Board()
    moves = []
    for _ in range(6):
        m = generate_legal_moves(board)[0]
        moves.append(m)
        board.apply_move(*m)
    ds = XiangqiILDataset([Game(moves=tuple(moves), outcome=RED)])

    loader = make_dataloader(ds, batch_size=4, shuffle=False)
    tensors, actions, values = next(iter(loader))
    assert tuple(tensors.shape) == (4, 14, 10, 9)
    assert actions.shape[0] == 4
    assert values.shape[0] == 4
