"""BaseAgent wrapper around a UCCI engine (ElephantEye / Pikafish).

Lets a fixed-strength engine play in the tournament like any other agent, which
is how absolute Elo anchoring works (docs/08-EVALUATION.md): include an
``EngineAgent`` at a known depth in the round-robin and re-centre the fitted
ratings on its assigned Elo.
"""

from __future__ import annotations

import numpy as np

from src.agents.base_agent import BaseAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.elephanteye import ElephantEyeEngine


class EngineAgent(BaseAgent):
    """Plays the move returned by a UCCI engine subprocess."""

    def __init__(
        self,
        engine: ElephantEyeEngine,
        *,
        depth: int | None = None,
        name: str | None = None,
    ) -> None:
        self.engine = engine
        self.depth = depth
        self.name = name or f"Engine(d={depth or engine.depth})"

    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        move = self.engine.bestmove(board, depth=self.depth)
        return action_space.move_to_index(move)


__all__ = ["EngineAgent"]
