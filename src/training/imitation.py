"""Imitation Learning training — Agent 2, Phase 1 (docs/07-TRAINING.md section 2).

Supervised pre-training of the Policy-Value network to imitate human moves:

* policy loss = cross-entropy between the policy logits and the human's move,
* value loss  = MSE between the predicted value and the game outcome,
* total loss  = policy_loss + value_loss.

The resulting checkpoint is the initialization for Agent 2 Phase 2 (PPO
fine-tuning via ``ppo_train``). Success is measured by top-1 move-prediction
accuracy on a held-out set (40-55% is typical for professional imitation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.network import PolicyValueNetwork
from src.utils.config import (
    IL_LEARNING_RATE,
    IL_EPOCHS,
    NN_CHANNEL_WIDTH,
    NN_L2_WEIGHT_DECAY,
    NN_NUM_RES_BLOCKS,
)


@dataclass
class EpochStats:
    """Metrics recorded after one training epoch."""

    epoch: int
    train_loss: float
    train_policy_loss: float
    train_value_loss: float
    val_accuracy: float | None = None


def _prepare_batch(
    batch: tuple, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors, actions, values = batch
    tensors = tensors.to(device).float()
    actions = actions.to(device).long()
    values = values.to(device).float().unsqueeze(1)  # (batch, 1)
    return tensors, actions, values


def run_epoch(
    network: PolicyValueNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train for one epoch; return (avg_total, avg_policy, avg_value) loss."""
    network.train()
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    total, total_p, total_v, n_batches = 0.0, 0.0, 0.0, 0
    for batch in loader:
        tensors, actions, values = _prepare_batch(batch, device)
        logits, value = network(tensors)
        policy_loss = policy_loss_fn(logits, actions)
        value_loss = value_loss_fn(value, values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        total_p += policy_loss.item()
        total_v += value_loss.item()
        n_batches += 1

    return total / n_batches, total_p / n_batches, total_v / n_batches


@torch.no_grad()
def evaluate(
    network: PolicyValueNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return top-1 move-prediction accuracy over ``loader``."""
    network.eval()
    correct, total = 0, 0
    for batch in loader:
        tensors, actions, _ = _prepare_batch(batch, device)
        logits, _ = network(tensors)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == actions).sum().item())
        total += actions.shape[0]
    return correct / total if total else 0.0


def train_imitation(
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    network: PolicyValueNetwork | None = None,
    epochs: int = IL_EPOCHS,
    learning_rate: float = IL_LEARNING_RATE,
    weight_decay: float = NN_L2_WEIGHT_DECAY,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> tuple[PolicyValueNetwork, list[EpochStats]]:
    """Train (or continue training) the network by imitation learning.

    Returns the trained network and per-epoch stats.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device)
    if network is None:
        network = PolicyValueNetwork()
    network.to(device)

    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[EpochStats] = []
    for epoch in range(epochs):
        train_loss, p_loss, v_loss = run_epoch(
            network, train_loader, optimizer, device
        )
        val_acc = evaluate(network, val_loader, device) if val_loader else None
        history.append(
            EpochStats(
                epoch=epoch,
                train_loss=train_loss,
                train_policy_loss=p_loss,
                train_value_loss=v_loss,
                val_accuracy=val_acc,
            )
        )
    return network, history


def save_checkpoint(network: PolicyValueNetwork, path: str) -> None:
    """Save network weights for use as the Agent 2 Phase 2 initialization."""
    torch.save(network.state_dict(), path)


def load_network(
    path: str,
    *,
    channels: int = NN_CHANNEL_WIDTH,
    num_blocks: int = NN_NUM_RES_BLOCKS,
    device: str | torch.device = "cpu",
) -> PolicyValueNetwork:
    """Load an IL checkpoint (saved by :func:`save_checkpoint`) into a network.

    ``channels``/``num_blocks`` must match those the checkpoint was trained with.
    This is how Agent 2 Phase 2 (PPO fine-tuning) picks up the IL body — see
    ``ppo_train.transfer_il_weights``.
    """
    network = PolicyValueNetwork(channels=channels, num_blocks=num_blocks)
    network.load_state_dict(torch.load(path, map_location=device))
    return network


__all__ = [
    "EpochStats",
    "run_epoch",
    "evaluate",
    "train_imitation",
    "save_checkpoint",
    "load_network",
]
