"""Unit tests for opening-analysis pure functions (docs/09 section 4)."""

from __future__ import annotations

import math

from src.evaluation import opening_analysis as oa


def test_opening_signature_takes_first_n_plies():
    moves = [(0, 0, 1, 0), (9, 0, 8, 0), (2, 7, 2, 4)]
    assert oa.opening_signature(moves, 2) == tuple(moves[:2])


def test_build_distribution_sums_to_one():
    sigs = [((0, 0, 1, 0),), ((0, 0, 1, 0),), ((2, 7, 2, 4),)]
    dist = oa.build_distribution(sigs)
    assert math.isclose(sum(dist.values()), 1.0)
    assert math.isclose(dist[((0, 0, 1, 0),)], 2 / 3)


def test_kl_divergence_zero_for_identical_and_positive_otherwise():
    p = {("a",): 0.5, ("b",): 0.5}
    q = {("a",): 0.9, ("b",): 0.1}
    assert oa.kl_divergence(p, p) == 0.0
    assert oa.kl_divergence(p, q) > 0.0


def test_classify_central_cannon():
    # "C2.5": the cannon on (2,7) moves to the central file (col 4).
    assert oa.classify_opening([(2, 7, 2, 4)]) == "central_cannon"


def test_classify_flying_elephant_and_others():
    assert oa.classify_opening([(0, 2, 2, 4)]) == "flying_elephant"   # elephant
    assert oa.classify_opening([(0, 1, 2, 2)]) == "horse_opening"     # horse
    assert oa.classify_opening([(3, 0, 4, 0)]) == "pawn_opening"      # soldier
    assert oa.classify_opening([(2, 7, 2, 6)]) == "cannon_flank"      # non-central cannon
    assert oa.classify_opening([]) == "empty"
