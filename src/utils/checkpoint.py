"""Checkpoint save/load for training resume (docs/07-TRAINING.md section 6).

Colab disconnects, so training must be resumable. These helpers save the model
weights plus (optionally) the optimizer state, a step counter, and any extra
metadata (seed, git commit, Elo history), and load them back.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint to ``path``.

    Saves the model weights, and (when given) the optimizer state, a step
    counter, and an arbitrary ``extra`` dict for reproducibility metadata.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload: dict[str, Any] = {"model_state": network.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = step
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``network`` (and ``optimizer`` if given).

    Returns the full payload so the caller can read ``step`` / ``extra`` to
    resume training from where it left off.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    network.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


__all__ = ["save_checkpoint", "load_checkpoint"]
