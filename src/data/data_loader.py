"""Game-level splits and DataLoader construction (docs/04-DATA-PIPELINE.md 7-8).

Splitting is done by GAME, never by position: positions from one game must not
appear in two splits, or information leaks between train and test. The split is
seeded so it is reproducible.
"""

from __future__ import annotations

import random

from torch.utils.data import DataLoader, Dataset

from src.data.wxf_parser import Game
from src.utils.config import DEFAULT_SEED


def split_games(
    games: list[Game],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Game], list[Game], list[Game]]:
    """Split games into (train, val, test) at the GAME level, reproducibly."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    shuffled = list(games)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader (shuffle for training; no shuffle for val/test)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


__all__ = ["split_games", "make_dataloader"]
