"""Tournament orchestration (docs/08-EVALUATION.md section 4).

Plays agents against each other (or, later, against an engine), always
balancing colours to cancel Red's first-move advantage, and turns the results
into match statistics and Elo estimates.

Agent-vs-agent evaluation works today with no external engine; anchoring to an
absolute Elo scale via ElephantEye/Pikafish plugs in later through the same
:func:`play_match` by wrapping the engine as a ``BaseAgent``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.base_agent import BaseAgent
from src.evaluation import elo
from src.evaluation.metrics import DRAW, GameOutcome, LOSS, MatchStats, WIN, summarize
from src.environment.xiangqi_env import XiangqiEnv
from src.utils.config import BLACK, MAX_PLIES, RED, WIN_REWARD


def play_game(
    red_agent: BaseAgent,
    black_agent: BaseAgent,
    *,
    seed: int | None = None,
    max_plies: int = MAX_PLIES,
) -> tuple[int | None, int]:
    """Play one game. Returns ``(winner, plies)`` where winner is RED, BLACK, or
    ``None`` for a draw."""
    env = XiangqiEnv()
    _, info = env.reset(seed=seed)
    while True:
        mover = env.get_current_player()
        agent = red_agent if mover == RED else black_agent
        action = agent.select_move(env.board, info["legal_mask"])
        _, reward, terminated, truncated, info = env.step(action)
        if terminated:
            # A terminal WIN reward means the side that just moved won.
            winner = mover if reward == WIN_REWARD else None
            return winner, env.ply_count
        if truncated:
            return None, env.ply_count


def play_match(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    n_games: int,
    *,
    seed: int | None = None,
) -> list[GameOutcome]:
    """Play ``n_games`` between two agents, alternating which colour A plays.

    Results are recorded from agent A's perspective.
    """
    outcomes: list[GameOutcome] = []
    for i in range(n_games):
        a_plays_red = i % 2 == 0
        a_color = RED if a_plays_red else BLACK
        red, black = (agent_a, agent_b) if a_plays_red else (agent_b, agent_a)
        game_seed = None if seed is None else seed + i
        winner, plies = play_game(red, black, seed=game_seed)

        if winner is None:
            result = DRAW
        elif winner == a_color:
            result = WIN
        else:
            result = LOSS
        outcomes.append(GameOutcome(result=result, plies=plies, a_color=a_color))
    return outcomes


@dataclass(frozen=True)
class EloEstimate:
    """An agent's Elo estimate against a known-rating opponent, with its CI."""

    rating: float
    confidence_interval: float  # 95% half-width, in Elo points
    stats: MatchStats


def estimate_elo(
    agent: BaseAgent,
    opponent: BaseAgent,
    opponent_rating: float,
    n_games: int,
    *,
    seed: int | None = None,
) -> EloEstimate:
    """Estimate ``agent``'s Elo from a match against a known-rating opponent."""
    outcomes = play_match(agent, opponent, n_games, seed=seed)
    stats = summarize(outcomes)
    rating = elo.estimate_rating_from_score(opponent_rating, stats.score)
    ci = elo.elo_confidence_interval(stats.score, n_games)
    return EloEstimate(rating=rating, confidence_interval=ci, stats=stats)


def round_robin(
    agents: dict[str, BaseAgent],
    n_games: int,
    *,
    seed: int | None = None,
) -> dict[tuple[str, str], MatchStats]:
    """Every agent plays every other agent. Returns stats keyed by (A, B) name.

    Stats are from A's perspective for each ordered pair that is actually played
    (each unordered pair is played once, as A vs B).
    """
    names = list(agents)
    results: dict[tuple[str, str], MatchStats] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            pair_seed = None if seed is None else seed + 1000 * (i + 1)
            outcomes = play_match(
                agents[name_a], agents[name_b], n_games, seed=pair_seed
            )
            results[(name_a, name_b)] = summarize(outcomes)
    return results


__all__ = [
    "play_game",
    "play_match",
    "EloEstimate",
    "estimate_elo",
    "round_robin",
]
