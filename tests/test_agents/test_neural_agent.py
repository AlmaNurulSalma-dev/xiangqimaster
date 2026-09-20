"""Unit tests for the NeuralAgent (masked play from a network)."""

from __future__ import annotations

import numpy as np

from src.agents.neural_agent import NeuralAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.models.network import PolicyValueNetwork
from src.utils.config import ACTION_SPACE_SIZE


def _tiny_net() -> PolicyValueNetwork:
    return PolicyValueNetwork(channels=8, num_blocks=1)


def test_deterministic_move_is_legal():
    board = Board()
    agent = NeuralAgent(_tiny_net(), deterministic=True)
    move = action_space.index_to_move(agent.select_move(board))
    assert move in set(generate_legal_moves(board))


def test_stochastic_move_is_legal():
    board = Board()
    agent = NeuralAgent(_tiny_net(), deterministic=False, seed=0)
    legal = set(generate_legal_moves(board))
    for _ in range(10):
        move = action_space.index_to_move(agent.select_move(board))
        assert move in legal


def test_respects_external_mask():
    # If only one action is allowed by the mask, that action must be chosen.
    board = Board()
    legal = generate_legal_moves(board)
    only = action_space.move_to_index(legal[0])
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[only] = True
    agent = NeuralAgent(_tiny_net(), deterministic=True)
    assert agent.select_move(board, mask) == only
