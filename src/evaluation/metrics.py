"""Match outcome records and summary metrics (docs/08-EVALUATION.md section 5)."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.config import RED

# A single game's result, always from the perspective of "agent A" in a match.
WIN: str = "win"
LOSS: str = "loss"
DRAW: str = "draw"


@dataclass(frozen=True)
class GameOutcome:
    """One game's result from agent A's perspective."""

    result: str  # WIN / LOSS / DRAW
    plies: int   # number of half-moves played
    a_color: int  # the colour agent A played (RED or BLACK)


@dataclass(frozen=True)
class MatchStats:
    """Aggregate statistics over a match (list of GameOutcomes)."""

    games: int
    wins: int
    losses: int
    draws: int
    score: float             # (wins + 0.5*draws) / games
    win_rate: float
    loss_rate: float
    draw_rate: float
    decisiveness: float      # (wins + losses) / games
    avg_game_length: float   # mean plies
    win_rate_as_red: float
    win_rate_as_black: float


def summarize(outcomes: list[GameOutcome]) -> MatchStats:
    """Aggregate a list of game outcomes into a :class:`MatchStats`."""
    n = len(outcomes)
    if n == 0:
        raise ValueError("cannot summarize an empty match")

    wins = sum(1 for o in outcomes if o.result == WIN)
    losses = sum(1 for o in outcomes if o.result == LOSS)
    draws = sum(1 for o in outcomes if o.result == DRAW)

    as_red = [o for o in outcomes if o.a_color == RED]
    as_black = [o for o in outcomes if o.a_color != RED]

    return MatchStats(
        games=n,
        wins=wins,
        losses=losses,
        draws=draws,
        score=(wins + 0.5 * draws) / n,
        win_rate=wins / n,
        loss_rate=losses / n,
        draw_rate=draws / n,
        decisiveness=(wins + losses) / n,
        avg_game_length=sum(o.plies for o in outcomes) / n,
        win_rate_as_red=_win_rate(as_red),
        win_rate_as_black=_win_rate(as_black),
    )


def _win_rate(outcomes: list[GameOutcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.result == WIN) / len(outcomes)


__all__ = ["WIN", "LOSS", "DRAW", "GameOutcome", "MatchStats", "summarize"]
