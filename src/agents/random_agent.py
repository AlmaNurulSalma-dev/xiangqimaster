"""A trivial agent that plays a uniformly random legal move.

Useful as a sanity opponent, a self-play bootstrap opponent, and a weakest-
possible baseline in the tournament.
"""

from __future__ import annotations

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environment import action_space
from src.environment.board import Board


class RandomAgent(BaseAgent):
    """Selects a legal move uniformly at random."""

    def __init__(self, seed: int | None = None, name: str = "Random") -> None:
        self.rng = np.random.default_rng(seed)
        self.name = name

    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        if legal_mask is None:
            legal_mask = action_space.legal_mask(board)
        legal_indices = np.flatnonzero(legal_mask)
        if legal_indices.size == 0:
            raise ValueError("no legal moves available (game is already over)")
        return int(self.rng.choice(legal_indices))


__all__ = ["RandomAgent"]
