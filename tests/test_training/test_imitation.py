"""Smoke/sanity tests for imitation-learning training (docs/07 section 11)."""

from __future__ import annotations

import torch

from src.agents.random_agent import RandomAgent
from src.data.data_loader import make_dataloader
from src.data.dataset import XiangqiILDataset
from src.data.wxf_parser import Game
from src.environment import action_space
from src.environment.board import Board
from src.models.network import PolicyValueNetwork
from src.training.imitation import evaluate, train_imitation
from src.utils.config import RED, BLACK, DEFAULT_SEED


def _random_game(seed: int, plies: int = 8, outcome: int = RED) -> Game:
    board = Board()
    agent = RandomAgent(seed=seed)
    moves = []
    for _ in range(plies):
        action = agent.select_move(board)
        move = action_space.index_to_move(action)
        moves.append(move)
        board.apply_move(*move)
    return Game(moves=tuple(moves), outcome=outcome)


def _small_dataset() -> XiangqiILDataset:
    games = [
        _random_game(seed=1, outcome=RED),
        _random_game(seed=2, outcome=BLACK),
        _random_game(seed=3, outcome=RED),
    ]
    return XiangqiILDataset(games)


def _tiny_net() -> PolicyValueNetwork:
    return PolicyValueNetwork(channels=8, num_blocks=1)


def test_training_run_completes_and_reduces_loss():
    ds = _small_dataset()
    loader = make_dataloader(ds, batch_size=8, shuffle=False)
    net = _tiny_net()
    _, history = train_imitation(
        loader, network=net, epochs=25, learning_rate=1e-2, seed=DEFAULT_SEED
    )
    assert len(history) == 25
    assert history[-1].train_loss < history[0].train_loss  # loss decreases


def test_network_can_memorize_small_set():
    ds = _small_dataset()
    loader = make_dataloader(ds, batch_size=8, shuffle=False)
    net = _tiny_net()

    initial_acc = evaluate(net, loader, torch.device("cpu"))
    net, _ = train_imitation(
        loader, network=net, epochs=40, learning_rate=1e-2, seed=DEFAULT_SEED
    )
    final_acc = evaluate(net, loader, torch.device("cpu"))

    assert final_acc > initial_acc
    assert final_acc >= 0.5  # a tiny net should memorize a handful of positions


def test_evaluate_returns_valid_accuracy_and_history_has_val():
    ds = _small_dataset()
    train_loader = make_dataloader(ds, batch_size=8, shuffle=False)
    val_loader = make_dataloader(ds, batch_size=8, shuffle=False)
    net = _tiny_net()
    _, history = train_imitation(
        train_loader, val_loader, network=net, epochs=3, seed=DEFAULT_SEED
    )
    assert all(0.0 <= h.val_accuracy <= 1.0 for h in history)
