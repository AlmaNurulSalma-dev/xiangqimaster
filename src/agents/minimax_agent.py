"""Agent 4 — classical Minimax baseline (docs/06-AGENTS.md section 2).

Alpha-beta game-tree search with a handcrafted, material-based evaluation. No
learning. This is the traditional approach to Xiangqi AI and the floor the RL
agents must beat (~1200-1500 Elo at depth 4). It also doubles as a correctness
check on the environment: a working Minimax proves move generation and rules
are sound.

Implementation notes:
* Uses the **negamax** formulation of alpha-beta — a single routine that always
  scores from the side-to-move's perspective and negates for the opponent.
* Moves are ordered captures-first (most valuable capture first) so alpha-beta
  prunes far more of the tree.
* Terminal "no legal move" = a loss for the side to move (checkmate OR
  stalemate, both losses in Xiangqi), scored near +/-MATE with a depth term so
  quicker mates are preferred.
"""

from __future__ import annotations

import math

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import FullMove, generate_legal_moves
from src.utils.config import MINIMAX_DEPTH, PIECE_VALUES, RED

# A score that dwarfs any material total, used for checkmate/stalemate.
MATE_SCORE: float = 1_000_000.0


class MinimaxAgent(BaseAgent):
    """Alpha-beta search with material evaluation."""

    def __init__(self, depth: int = MINIMAX_DEPTH, name: str | None = None) -> None:
        if depth < 1:
            raise ValueError("Minimax depth must be at least 1")
        self.depth = depth
        self.name = name or f"Minimax(d={depth})"

    # ─── Public API ───────────────────────────────────────────────────────
    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        best_move: FullMove | None = None
        best_score = -math.inf
        alpha, beta = -math.inf, math.inf

        for move in self._ordered_moves(board):
            child = board.clone()
            child.apply_move(*move)
            score = -self._negamax(child, self.depth - 1, -beta, -alpha)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        assert best_move is not None, "select_move called on a finished game"
        return action_space.move_to_index(best_move)

    # ─── Search ───────────────────────────────────────────────────────────
    def _negamax(
        self, board: Board, depth: int, alpha: float, beta: float
    ) -> float:
        legal = self._ordered_moves(board)
        if not legal:
            # Side to move has no reply → it loses. Deeper (sooner) mates score
            # slightly worse for the loser, so the winner prefers quicker mates.
            return -(MATE_SCORE + depth)
        if depth == 0:
            return self._evaluate(board)

        best = -math.inf
        for move in legal:
            child = board.clone()
            child.apply_move(*move)
            score = -self._negamax(child, depth - 1, -beta, -alpha)
            if score > best:
                best = score
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # this branch can't improve the result → prune
        return best

    # ─── Evaluation & ordering ────────────────────────────────────────────
    @staticmethod
    def _evaluate(board: Board) -> float:
        """Material balance from the perspective of the side to move."""
        color = board.to_move
        score = 0.0
        grid = board.grid
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                value = int(grid[row, col])
                if value == 0:
                    continue
                piece_value = PIECE_VALUES[abs(value)]
                owner = RED if value > 0 else -RED
                score += piece_value if owner == color else -piece_value
        return score

    @staticmethod
    def _ordered_moves(board: Board) -> list[FullMove]:
        """Legal moves, most valuable captures first (for better pruning)."""
        moves = generate_legal_moves(board)
        moves.sort(
            key=lambda m: PIECE_VALUES.get(board.type_at(m[2], m[3]), 0.0),
            reverse=True,
        )
        return moves


__all__ = ["MinimaxAgent", "MATE_SCORE"]
