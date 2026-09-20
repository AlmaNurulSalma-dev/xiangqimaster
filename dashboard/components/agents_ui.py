"""Agent factories for the dashboard's selectors.

Only agents that need no trained checkpoint are offered by default (Random and
Minimax at a few depths), so the dashboard runs self-contained. Trained agents
(PPO / IL / MCTS) can be added by loading their checkpoints on the relevant
pages.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent

#: Selectable agents that need no checkpoint.
AGENT_NAMES: list[str] = ["Random", "Minimax d1", "Minimax d2", "Minimax d3"]


def make_agent(name: str) -> BaseAgent:
    """Instantiate a checkpoint-free agent by its display name."""
    if name == "Random":
        return RandomAgent()
    if name.startswith("Minimax d"):
        return MinimaxAgent(depth=int(name.rsplit("d", 1)[1]))
    raise ValueError(f"unknown agent: {name!r}")


__all__ = ["AGENT_NAMES", "make_agent"]
