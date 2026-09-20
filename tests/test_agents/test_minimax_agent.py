"""Unit tests for the Minimax baseline agent (docs/06-AGENTS.md section 8)."""

from __future__ import annotations

from src.agents.minimax_agent import MinimaxAgent
from src.environment import action_space, rules
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.utils.config import BLACK, CHARIOT, GENERAL, RED


def _empty() -> Board:
    return Board(empty=True)


def test_select_move_returns_a_legal_action_at_start():
    agent = MinimaxAgent(depth=2)
    board = Board()
    action = agent.select_move(board)
    move = action_space.index_to_move(action)
    assert move in set(generate_legal_moves(board))


def test_captures_free_material():
    # Red chariot can capture an undefended Black chariot for +9; the material
    # evaluation should make that the chosen move.
    board = _empty()
    board.grid[5, 0] = RED * CHARIOT
    board.grid[5, 5] = BLACK * CHARIOT     # undefended enemy chariot
    board.grid[0, 3] = RED * GENERAL
    board.grid[9, 4] = BLACK * GENERAL
    board.to_move = RED

    agent = MinimaxAgent(depth=2)
    action = agent.select_move(board)
    assert action_space.index_to_move(action) == (5, 0, 5, 5)  # the capture


def _mate_in_one_position() -> Board:
    # Verified genuine checkmate-in-1: Red chariot (0,0) -> (9,0) delivers a
    # back-rank mate. The Red general on file 3 pins the (9,3) escape via the
    # flying-general rule, the row-9 chariot covers (9,5) once the general
    # vacates (9,4), and the chariot on (8,1) covers (8,4).
    board = _empty()
    board.grid[9, 4] = BLACK * GENERAL
    board.grid[0, 0] = RED * CHARIOT
    board.grid[8, 1] = RED * CHARIOT
    board.grid[0, 3] = RED * GENERAL
    board.to_move = RED
    return board


def test_mate_in_one_position_has_a_real_checkmate():
    # Confirm the position genuinely contains a checkmate (in check + no reply),
    # not merely a stalemate.
    board = _mate_in_one_position()
    assert not rules.is_in_check(board, BLACK)  # legal, undecided position
    board.apply_move(0, 0, 9, 0)
    assert rules.is_checkmate(board, BLACK)


def test_finds_forced_win_in_one():
    board = _mate_in_one_position()
    agent = MinimaxAgent(depth=2)
    action = agent.select_move(board)
    board.apply_move(*action_space.index_to_move(action))
    # Black is left with no legal reply → Red has won this move.
    assert generate_legal_moves(board, BLACK) == []


def test_deeper_search_still_legal_on_a_small_position():
    board = _empty()
    board.grid[0, 0] = RED * CHARIOT
    board.grid[9, 8] = BLACK * CHARIOT
    board.grid[0, 4] = RED * GENERAL
    board.grid[9, 3] = BLACK * GENERAL
    board.to_move = RED

    agent = MinimaxAgent(depth=3)
    action = agent.select_move(board)
    assert action_space.index_to_move(action) in set(generate_legal_moves(board))


def test_agent_has_descriptive_name():
    assert MinimaxAgent(depth=4).name == "Minimax(d=4)"
