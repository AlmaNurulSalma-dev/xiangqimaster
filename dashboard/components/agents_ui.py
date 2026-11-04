"""Agent factories for the dashboard's selectors.

Checkpoint-free agents (Random, Minimax) are always offered so the dashboard
runs self-contained. Trained agents (IL, MCTS v2, Pikafish-distilled) are added
when their checkpoints exist, and "Pikafish MAX" is added when the Pikafish
engine is present — the superhuman benchmark for the live demo.

The Pikafish engine subprocess is cached as a module-level singleton so repeated
Streamlit reruns reuse one process instead of spawning a new one each click.
"""

from __future__ import annotations

import os

from src.agents.base_agent import BaseAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent

# ─── Paths ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PIKA_EXE = os.path.join(
    _ROOT, "engines", "pikafish", "extracted", "Pikafish-Windows-x86-64-universal.exe"
)
_CKPTS = {
    "IL (human)": os.path.join(_ROOT, "results", "checkpoints", "il_agent2_phase1.pt"),
    "MCTS v2": os.path.join(_ROOT, "results", "checkpoints", "mcts_agent3_v2.best.pt"),
    "Distill (Pikafish)": os.path.join(_ROOT, "results", "checkpoints", "distill_pikafish.pt"),
}

# Pikafish MAX demo strength: deep enough to crush any human, fast enough that a
# full game still generates in seconds (NNUE is very fast at these depths).
_PIKA_DEPTH = 12

_engine_singleton = None  # reuse one Pikafish subprocess across reruns


def _pikafish_engine():
    """Lazily start (and cache) one strong Pikafish engine subprocess."""
    global _engine_singleton
    if _engine_singleton is None:
        from src.environment.elephanteye import ElephantEyeEngine

        threads = max(1, (os.cpu_count() or 2) - 1)
        _engine_singleton = ElephantEyeEngine(
            [_PIKA_EXE], depth=_PIKA_DEPTH, protocol="uci",
            cwd=os.path.dirname(_PIKA_EXE), threads=threads, hash_mb=256,
        ).start()
    return _engine_singleton


def available_agents() -> list[str]:
    """Display names of every agent that can be instantiated right now."""
    names = ["Random", "Minimax d1", "Minimax d2", "Minimax d3"]
    names += [n for n, path in _CKPTS.items() if os.path.exists(path)]
    if os.path.exists(_PIKA_EXE):
        names.append("Pikafish MAX")
    return names


#: Selectable agents (checkpoint-free ones always; others when available).
AGENT_NAMES: list[str] = available_agents()


def make_agent(name: str, *, mcts_sims: int = 100) -> BaseAgent:
    """Instantiate an agent by its display name.

    ``mcts_sims`` sets the MCTS agent's search width; the simulation ablation
    showed strength is nearly flat in this count, so the demo can use fewer for
    a big speed-up without materially weakening the agent.
    """
    if name == "Random":
        return RandomAgent()
    if name.startswith("Minimax d"):
        return MinimaxAgent(depth=int(name.rsplit("d", 1)[1]))
    if name == "Pikafish MAX":
        from src.agents.engine_agent import EngineAgent

        return EngineAgent(_pikafish_engine(), name="Pikafish MAX")
    if name in _CKPTS:
        from src.training.imitation import load_network

        net = load_network(_CKPTS[name], device="cpu")
        if name == "MCTS v2":
            from src.agents.mcts_agent import MCTSAgent

            return MCTSAgent(net, n_simulations=mcts_sims, name="MCTS v2")
        from src.agents.neural_agent import NeuralAgent

        return NeuralAgent(net, name=name, device="cpu")
    raise ValueError(f"unknown agent: {name!r}")


def pikafish_available() -> bool:
    """True when the Pikafish engine binary is present (for the advantage bar)."""
    return os.path.exists(_PIKA_EXE)


def pikafish_engine():
    """The shared Pikafish subprocess (started on first use), reused for eval."""
    return _pikafish_engine()


def value_net_checkpoint() -> str | None:
    """A policy–value checkpoint to drive the advantage bar when Pikafish is
    absent — prefer the human-imitation net, then distillation, then MCTS."""
    for key in ("IL (human)", "Distill (Pikafish)", "MCTS v2"):
        path = _CKPTS[key]
        if os.path.exists(path):
            return path
    return None


__all__ = [
    "AGENT_NAMES",
    "available_agents",
    "make_agent",
    "pikafish_available",
    "pikafish_engine",
    "value_net_checkpoint",
]
