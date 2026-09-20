"""Unit tests for MCTS core and MCTSAgent (docs/06-AGENTS.md section 8)."""

from __future__ import annotations

from src.agents.mcts import run_mcts, visit_count_policy
from src.agents.mcts_agent import MCTSAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.models.network import PolicyValueNetwork
from src.utils.config import BLACK, CHARIOT, GENERAL, RED


def _tiny_net() -> PolicyValueNetwork:
    return PolicyValueNetwork(channels=8, num_blocks=1)


def _mate_in_one() -> Board:
    board = Board(empty=True)
    board.grid[9, 4] = BLACK * GENERAL
    board.grid[0, 0] = RED * CHARIOT
    board.grid[8, 1] = RED * CHARIOT
    board.grid[0, 3] = RED * GENERAL
    board.to_move = RED
    return board


def test_visit_counts_sum_to_simulations():
    root = run_mcts(Board(), _tiny_net(), n_simulations=20)
    assert root.visit_count == 20
    assert sum(root.children[a].visit_count for a in root.children) == 20


def test_root_children_are_legal_actions():
    board = Board()
    root = run_mcts(board, _tiny_net(), n_simulations=8)
    legal_indices = {action_space.move_to_index(m) for m in generate_legal_moves(board)}
    assert set(visit_count_policy(root)) == legal_indices


def test_select_move_is_legal():
    board = Board()
    agent = MCTSAgent(_tiny_net(), n_simulations=16)
    move = action_space.index_to_move(agent.select_move(board))
    assert move in set(generate_legal_moves(board))


def test_search_does_not_corrupt_the_board():
    board = Board()
    snapshot = board.grid.copy()
    MCTSAgent(_tiny_net(), n_simulations=16).select_move(board)
    assert (board.grid == snapshot).all()
    assert board.to_move == RED


def test_mcts_finds_the_forced_win():
    board = _mate_in_one()
    agent = MCTSAgent(_tiny_net(), n_simulations=120, seed=0)
    action = agent.select_move(board)
    board.apply_move(*action_space.index_to_move(action))
    assert generate_legal_moves(board, BLACK) == []  # Black has no reply
