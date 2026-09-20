"""Common interface for all four agents (docs/06-AGENTS.md section 1).

Every agent — Minimax, PPO, IL+RL, MCTS — exposes the same ``select_move`` so
the tournament and dashboard can treat them identically despite very different
internals.

The "state" an agent receives is the current :class:`Board`. Neural agents
encode it to a tensor themselves; the Minimax agent searches it directly. A
precomputed ``legal_mask`` may be passed in for convenience, but agents are
free to derive legality from the board.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.environment.board import Board


class BaseAgent(ABC):
    """Abstract base every agent inherits from."""

    #: Human-readable identifier for logging and result tables.
    name: str = "agent"

    @abstractmethod
    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        """Choose a move for the side to move.

        Args:
            board: the current position (``board.to_move`` is the side to act).
            legal_mask: optional boolean mask of length 2086; if omitted the
                agent computes legality itself.

        Returns:
            The chosen move as an ``action_index`` in ``[0, 2086)``. The
            returned action is always legal.
        """
        raise NotImplementedError


__all__ = ["BaseAgent"]
