"""An agent that plays directly from a Policy-Value network's policy head.

Used to evaluate a raw network's playing strength — most importantly the
imitation-learning network (Agent 2, Phase 1) before any RL refinement, but
also any PolicyValueNetwork checkpoint. It applies the mandatory legal-move
mask before choosing, so it never plays an illegal move (docs/06-AGENTS.md 3.3).

This is *not* MCTS: it plays the network's favourite legal move in one forward
pass. MCTS (Agent 3) will wrap the same network with lookahead search.
"""

from __future__ import annotations

import numpy as np
import torch

from src.agents.base_agent import BaseAgent
from src.environment import action_space, encoder
from src.environment.board import Board
from src.models.network import PolicyValueNetwork


class NeuralAgent(BaseAgent):
    """Masked greedy (or sampled) play from a PolicyValueNetwork policy head."""

    def __init__(
        self,
        network: PolicyValueNetwork,
        name: str = "Neural",
        deterministic: bool = True,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.network = network.to(device)
        self.network.eval()
        self.name = name
        self.deterministic = deterministic
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def select_move(self, board: Board, legal_mask: np.ndarray | None = None) -> int:
        if legal_mask is None:
            legal_mask = action_space.legal_mask(board)

        observation = encoder.encode(board)
        tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        logits, _ = self.network(tensor)
        logits = logits.squeeze(0).cpu().numpy()

        # Mask illegal moves to -inf so they can never be selected.
        masked = np.where(legal_mask, logits, -np.inf)

        if self.deterministic:
            return int(np.argmax(masked))

        # Stochastic: softmax over legal logits, then sample.
        finite = masked[np.isfinite(masked)]
        shifted = masked - np.max(finite)
        probs = np.where(legal_mask, np.exp(shifted), 0.0)
        probs /= probs.sum()
        return int(self._rng.choice(len(probs), p=probs))


__all__ = ["NeuralAgent"]
