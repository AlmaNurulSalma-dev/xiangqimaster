"""Unit tests for the global seeding utility."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.seeding import set_seed


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_set_seed_makes_all_rngs_reproducible():
    set_seed(123)
    first = _draw()
    set_seed(123)
    second = _draw()
    assert first == second


def test_different_seeds_differ():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a != b


def test_returns_the_seed():
    assert set_seed(42) == 42
