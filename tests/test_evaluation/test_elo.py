"""Unit tests for Elo math (docs/08-EVALUATION.md section 3)."""

from __future__ import annotations

import math

from src.evaluation import elo


def test_equal_ratings_expect_half():
    assert elo.expected_score(1500, 1500) == 0.5


def test_expected_scores_sum_to_one():
    a, b = 1600, 1400
    assert math.isclose(elo.expected_score(a, b) + elo.expected_score(b, a), 1.0)


def test_higher_rating_favoured():
    assert elo.expected_score(1800, 1400) > 0.9


def test_400_point_gap_is_ten_to_one():
    # A 400-point gap ⇒ expected score ≈ 10/11.
    assert math.isclose(elo.expected_score(1900, 1500), 10 / 11, rel_tol=1e-9)


def test_update_moves_toward_result():
    # Winning against an equal opponent raises the rating.
    new = elo.update_rating(1500, 1500, actual_score=1.0)
    assert new > 1500
    # Losing lowers it.
    assert elo.update_rating(1500, 1500, actual_score=0.0) < 1500


def test_estimate_rating_from_score():
    # Scoring 50% against a 1600 opponent ⇒ ~1600.
    assert math.isclose(elo.estimate_rating_from_score(1600, 0.5), 1600.0)
    # Scoring better ⇒ higher rating; worse ⇒ lower.
    assert elo.estimate_rating_from_score(1600, 0.75) > 1600
    assert elo.estimate_rating_from_score(1600, 0.25) < 1600


def test_confidence_interval_shrinks_with_more_games():
    wide = elo.elo_confidence_interval(0.5, 100)
    narrow = elo.elo_confidence_interval(0.5, 400)
    assert wide > narrow > 0


def test_confidence_interval_around_400_games():
    # docs/08: ~400 games ⇒ roughly ±40-50 Elo at a 50% score.
    ci = elo.elo_confidence_interval(0.5, 400)
    assert 30 < ci < 60


def test_fit_ratings_orders_a_transitive_field():
    # A beats B beats C beats D; fitted ratings must respect that order.
    pairwise = [
        ("A", "B", 0.75, 100),
        ("B", "C", 0.75, 100),
        ("C", "D", 0.75, 100),
        ("A", "C", 0.90, 100),
        ("A", "D", 0.97, 100),
        ("B", "D", 0.90, 100),
    ]
    ratings = elo.fit_ratings(pairwise, anchor_mean=1500.0)
    assert ratings["A"] > ratings["B"] > ratings["C"] > ratings["D"]
    # Recentred on the anchor mean.
    assert abs(sum(ratings.values()) / len(ratings) - 1500.0) < 1e-6
