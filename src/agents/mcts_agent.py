"""Agent 3 — MCTS + neural network, AlphaZero-style (docs/06-AGENTS.md 5).

Wraps the MCTS core behind the common :class:`BaseAgent` interface. At play
time it runs ``n_simulations`` of tree search guided by the network, then plays
the most-visited move (robust) — or samples proportionally to visit counts when
``deterministic`` is off, for opening variety.

This is the strongest but slowest agent; ``n_simulations`` trades strength for
speed (Experiment 5 studies this).
"""

from __future__ import annotations

import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.mcts import run_mcts, visit_count_policy
from src.environment.board import Board
from src.models.network import PolicyValueNetwork
from src.utils.config import MCTS_C_PUCT, MCTS_SIMULATIONS


class MCTSAgent(BaseAgent):
    """Plays the move MCTS visits most often."""

    def __init__(
        self,
        network: PolicyValueNetwork,
        *,
        n_simulations: int = MCTS_SIMULATIONS,
        c_puct: float = MCTS_C_PUCT,
        deterministic: bool = True,
        name: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.network = network
        self.network.eval()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.deterministic = deterministic
        self.name = name or f"MCTS(n={n_simulations})"
        self._rng = np.random.default_rng(seed)

    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        root = run_mcts(
            board,
            self.network,
            n_simulations=self.n_simulations,
            c_puct=self.c_puct,
        )
        counts = visit_count_policy(root)
        if not counts:
            raise ValueError("MCTS called on a position with no legal moves")

        actions = list(counts)
        visits = np.array([counts[a] for a in actions], dtype=float)
        if self.deterministic:
            return int(actions[int(np.argmax(visits))])
        probs = visits / visits.sum()
        return int(self._rng.choice(actions, p=probs))


__all__ = ["MCTSAgent"]
