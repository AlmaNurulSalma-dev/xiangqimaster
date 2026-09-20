"""Elo rating math (docs/08-EVALUATION.md section 3).

Elo is a *relative* skill scale: the rating difference between two players
predicts their expected score. A 400-point gap means the stronger player is
expected to score about 10x as often.

Functions here are pure math with no game logic, so they are trivial to test:

* :func:`expected_score` — predicted score from a rating difference.
* :func:`update_rating` — the incremental Elo update (for live/streaming ratings).
* :func:`estimate_rating_from_score` — invert the expected-score formula to rate
  an agent from its observed score against a known-rating opponent (Approach A).
* :func:`elo_confidence_interval` — 95% CI half-width for such an estimate.
"""

from __future__ import annotations

import math
from collections import defaultdict

#: Standard Elo scale constant (a 400-point gap ⇒ 10x expected-score ratio).
ELO_SCALE: float = 400.0
#: K-factor for developing players (docs/08 uses 32).
DEFAULT_K: float = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score of A vs B (1 = certain win, 0.5 = even, 0 = certain loss)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def update_rating(
    rating: float, opponent_rating: float, actual_score: float, k: float = DEFAULT_K
) -> float:
    """One incremental Elo update after a game (or a batch treated as one).

    Args:
        rating: the player's current rating.
        opponent_rating: the opponent's rating.
        actual_score: observed score (1 win, 0.5 draw, 0 loss).
        k: K-factor (step size).
    """
    return rating + k * (actual_score - expected_score(rating, opponent_rating))


def estimate_rating_from_score(
    opponent_rating: float, score: float, *, clamp: float = 1e-4
) -> float:
    """Estimate a rating from an observed score vs a known-rating opponent.

    Inverts :func:`expected_score`: rating = opp + 400 * log10(s / (1 - s)).
    The score is clamped away from 0 and 1 (an undefeated/winless record would
    otherwise map to +/- infinity).
    """
    s = min(max(score, clamp), 1.0 - clamp)
    return opponent_rating + ELO_SCALE * math.log10(s / (1.0 - s))


def elo_confidence_interval(
    score: float, n_games: int, *, z: float = 1.96, clamp: float = 1e-4
) -> float:
    """95% confidence half-width (in Elo points) for a score-based estimate.

    Uses the delta method: the standard error of the win rate is propagated
    through the logistic Elo mapping. Roughly ±40-50 Elo at 400 games
    (docs/08-EVALUATION.md section 3.4).
    """
    if n_games <= 0:
        return math.inf
    p = min(max(score, clamp), 1.0 - clamp)
    se_p = math.sqrt(p * (1.0 - p) / n_games)
    se_elo = (ELO_SCALE / math.log(10.0)) * se_p / (p * (1.0 - p))
    return z * se_elo


def fit_ratings(
    pairwise: list[tuple[str, str, float, int]],
    *,
    anchor_mean: float = 1500.0,
    k: float = 16.0,
    iterations: int = 3000,
) -> dict[str, float]:
    """Fit Elo ratings to round-robin results (Approach B, docs/08 section 3.2).

    Args:
        pairwise: list of ``(name_a, name_b, score_a, n_games)`` where
            ``score_a`` is agent A's average score (1 win, 0.5 draw, 0 loss)
            over ``n_games`` against B.
        anchor_mean: the fitted ratings are recentred so their mean equals this.
        k, iterations: step size and iteration count for the fit.

    Returns:
        A mapping of agent name → fitted rating. Because the scale is only
        relative, ratings are recentred on ``anchor_mean``; a fixed-rating
        opponent (e.g. ElephantEye) can be used to anchor absolutely instead.
    """
    games: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for name_a, name_b, score_a, n in pairwise:
        games[name_a].append((name_b, score_a, n))
        games[name_b].append((name_a, 1.0 - score_a, n))

    ratings: dict[str, float] = {name: anchor_mean for name in games}
    for _ in range(iterations):
        updated = dict(ratings)
        for name, results in games.items():
            expected_total = 0.0
            actual_total = 0.0
            total_games = 0
            for opponent, score, n in results:
                expected_total += n * expected_score(ratings[name], ratings[opponent])
                actual_total += n * score
                total_games += n
            if total_games:
                updated[name] = ratings[name] + k * (
                    actual_total - expected_total
                ) / total_games
        ratings = updated

    if ratings:
        shift = anchor_mean - sum(ratings.values()) / len(ratings)
        ratings = {name: r + shift for name, r in ratings.items()}
    return ratings


__all__ = [
    "ELO_SCALE",
    "DEFAULT_K",
    "expected_score",
    "update_rating",
    "estimate_rating_from_score",
    "elo_confidence_interval",
    "fit_ratings",
]
