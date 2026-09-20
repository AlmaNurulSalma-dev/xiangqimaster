"""Agent 1 — PPO self-play agent (docs/06-AGENTS.md section 3).

This wraps a trained MaskablePPO model behind the common :class:`BaseAgent`
interface so the tournament and dashboard treat it like any other agent. At
play time it encodes the board, applies the legal-move mask, and asks the model
for a move (greedy by default; stochastic sampling optionally, for exploration).

Training lives in ``src/training/ppo_train.py``; this file is only the
play-time wrapper.
"""

from __future__ import annotations

import numpy as np
from sb3_contrib import MaskablePPO

from src.agents.base_agent import BaseAgent
from src.environment import action_space, encoder
from src.environment.board import Board


class PPOAgent(BaseAgent):
    """Plays moves from a trained MaskablePPO policy, with action masking."""

    def __init__(
        self,
        model: MaskablePPO,
        name: str = "PPO",
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.name = name
        self.deterministic = deterministic

    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        if legal_mask is None:
            legal_mask = action_space.legal_mask(board)
        observation = encoder.encode(board)
        action, _ = self.model.predict(
            observation,
            action_masks=legal_mask,
            deterministic=self.deterministic,
        )
        return int(action)

    @classmethod
    def load(cls, path: str, **kwargs) -> "PPOAgent":
        """Load a PPOAgent from a saved MaskablePPO checkpoint."""
        return cls(MaskablePPO.load(path), **kwargs)


__all__ = ["PPOAgent"]
