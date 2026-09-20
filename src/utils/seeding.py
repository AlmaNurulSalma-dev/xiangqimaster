"""Global seeding for reproducibility (docs/11-CODING-STANDARDS.md section 5).

A single ``set_seed`` seeds Python's ``random``, NumPy, and PyTorch (CPU and,
if present, CUDA) so a run can be reproduced. Every training/experiment script
should call it once at startup and log the seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.config import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Seed all random number generators used in the project.

    Args:
        seed: the seed value.
        deterministic: if True, ask PyTorch/cuDNN for deterministic algorithms
            (slower, but bit-for-bit reproducible on GPU). Document the choice
            in the thesis if enabled.

    Returns:
        The seed used (so callers can log it).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


__all__ = ["set_seed"]
