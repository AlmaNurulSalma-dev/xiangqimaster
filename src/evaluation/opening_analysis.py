"""Opening-theory analysis (docs/09-EXPERIMENTS.md section 4, docs/04 section 10).

Tools to answer RQ3 — do RL agents rediscover classical Xiangqi openings? An
opening is the first few plies of a game; we build a probability distribution
over openings for each agent and compare it to a reference (professional)
distribution using KL divergence. A small KL means the agent plays "book"
openings; a large KL means it found its own.

The pure functions (signature, distribution, KL, classification) are separate
from game generation so they are easy to test.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import FullMove
from src.environment.xiangqi_env import XiangqiEnv
from src.utils.config import CANNON, ELEPHANT, HORSE, MAX_PLIES, RED, SOLDIER, WIN_REWARD

#: The central file (0-indexed) — a cannon moving here is the 当头炮 opening.
CENTRAL_FILE: int = 4

Signature = tuple[FullMove, ...]


def opening_signature(moves: Sequence[FullMove], n_plies: int) -> Signature:
    """The first ``n_plies`` moves of a game, as a hashable opening key."""
    return tuple(moves[:n_plies])


def build_distribution(signatures: Sequence[Signature]) -> dict[Signature, float]:
    """Normalize a list of opening signatures into a probability distribution."""
    counts = Counter(signatures)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {sig: n / total for sig, n in counts.items()}


def kl_divergence(
    p: dict[Signature, float],
    q: dict[Signature, float],
    *,
    epsilon: float = 1e-9,
) -> float:
    """KL(p || q) over opening distributions.

    Missing mass in ``q`` is floored to ``epsilon`` so unseen openings do not
    make the divergence infinite. ``KL(p || p) == 0`` exactly.
    """
    total = 0.0
    for sig, p_prob in p.items():
        if p_prob == 0.0:
            continue
        q_prob = q.get(sig)
        if not q_prob:
            q_prob = epsilon
        total += p_prob * math.log(p_prob / q_prob)
    return total


def classify_opening(moves: Sequence[FullMove]) -> str:
    """Classify a game's opening by Red's first move (a few named openings)."""
    if not moves:
        return "empty"
    board = Board()  # standard start; the first move is always Red's
    from_row, from_col, _to_row, to_col = moves[0]
    piece_type = board.type_at(from_row, from_col)
    if piece_type == CANNON:
        return "central_cannon" if to_col == CENTRAL_FILE else "cannon_flank"
    if piece_type == ELEPHANT:
        return "flying_elephant"
    if piece_type == HORSE:
        return "horse_opening"
    if piece_type == SOLDIER:
        return "pawn_opening"
    return "other"


def play_recorded_game(
    red_agent: BaseAgent,
    black_agent: BaseAgent,
    *,
    seed: int | None = None,
    max_plies: int = MAX_PLIES,
) -> tuple[int | None, list[FullMove]]:
    """Play a game and return ``(winner, moves)`` — winner is RED/BLACK/None."""
    env = XiangqiEnv()
    _, info = env.reset(seed=seed)
    moves: list[FullMove] = []
    while True:
        mover = env.get_current_player()
        agent = red_agent if mover == RED else black_agent
        action = agent.select_move(env.board, info["legal_mask"])
        moves.append(action_space.index_to_move(action))
        _, reward, terminated, truncated, info = env.step(action)
        if terminated:
            return (mover if reward == WIN_REWARD else None), moves
        if truncated:
            return None, moves


def collect_agent_openings(
    agent: BaseAgent,
    opponent: BaseAgent,
    *,
    n_games: int,
    n_plies: int,
    seed: int | None = 0,
) -> list[Signature]:
    """Play ``agent`` (as Red) vs ``opponent`` and collect opening signatures."""
    signatures: list[Signature] = []
    for i in range(n_games):
        game_seed = None if seed is None else seed + i
        _, moves = play_recorded_game(agent, opponent, seed=game_seed)
        signatures.append(opening_signature(moves, n_plies))
    return signatures


__all__ = [
    "CENTRAL_FILE",
    "Signature",
    "opening_signature",
    "build_distribution",
    "kl_divergence",
    "classify_opening",
    "play_recorded_game",
    "collect_agent_openings",
]
