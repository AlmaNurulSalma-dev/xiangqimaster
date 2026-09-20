"""Unit tests for the RandomAgent."""

from __future__ import annotations

from src.agents.random_agent import RandomAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves


def test_random_agent_returns_a_legal_move():
    board = Board()
    agent = RandomAgent(seed=0)
    legal = set(generate_legal_moves(board))
    for _ in range(20):
        move = action_space.index_to_move(agent.select_move(board))
        assert move in legal


def test_random_agent_is_seeded_reproducible():
    board = Board()
    a = RandomAgent(seed=123)
    b = RandomAgent(seed=123)
    assert [a.select_move(board) for _ in range(10)] == [
        b.select_move(board) for _ in range(10)
    ]
