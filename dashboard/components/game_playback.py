"""Play a game between two agents and record it for step-through playback.

Used by the live-game and game-explorer pages: it returns the moves plus a
board snapshot after each ply (index 0 is the starting position), so a slider
can scrub through the game rendering each position.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import FullMove, generate_legal_moves
from src.utils.config import MAX_PLIES, RED


def record_game(
    red_agent: BaseAgent,
    black_agent: BaseAgent,
    *,
    seed: int | None = None,
    max_plies: int = MAX_PLIES,
) -> tuple[list[FullMove], list[Board]]:
    """Return ``(moves, snapshots)`` where ``snapshots[k]`` is the board after
    ``k`` plies (so ``len(snapshots) == len(moves) + 1``)."""
    board = Board()
    snapshots = [board.clone()]
    moves: list[FullMove] = []

    while len(moves) < max_plies:
        legal = generate_legal_moves(board)
        if not legal:
            break  # terminal: side to move has no reply
        agent = red_agent if board.to_move == RED else black_agent
        mask = action_space.legal_mask(board)
        move = action_space.index_to_move(agent.select_move(board, mask))
        board.apply_move(*move)
        moves.append(move)
        snapshots.append(board.clone())

    return moves, snapshots


__all__ = ["record_game"]
