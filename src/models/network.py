"""The Policy-Value network — the shared "brain" for the RL agents.

AlphaZero-style architecture (docs/05-NEURAL-NETWORK.md):

    input conv (14 → C)  →  N residual blocks  →  ┬─ policy head → logits(2086)
                                                   └─ value head  → scalar[-1,1]

The same network is used by Imitation Learning, PPO (actor + critic), and
MCTS (priors + leaf evaluation).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ConvBlock, ResidualBlock
from src.models.heads import PolicyHead, ValueHead
from src.utils.config import (
    NN_CHANNEL_WIDTH,
    NN_NUM_RES_BLOCKS,
    NUM_CHANNELS,
)


class PolicyValueNetwork(nn.Module):
    """Shared-body, two-head Policy-Value network.

    Args:
        channels: body channel width ``C`` (default from config).
        num_blocks: number of residual blocks ``N`` (default from config).
        in_channels: number of input planes (default 14, the board tensor).

    Forward:
        Input  ``(batch, 14, 10, 9)``
        Output ``(policy_logits, value)`` with shapes ``(batch, 2086)`` and
        ``(batch, 1)``. Policy is raw logits (mask + softmax externally).
    """

    def __init__(
        self,
        channels: int = NN_CHANNEL_WIDTH,
        num_blocks: int = NN_NUM_RES_BLOCKS,
        in_channels: int = NUM_CHANNELS,
    ) -> None:
        super().__init__()
        self.input_conv = ConvBlock(in_channels, channels)
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_blocks))
        )
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.input_conv(x)
        features = self.residual_tower(features)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience inference call in eval mode (no gradients)."""
        self.eval()
        return self.forward(x)


__all__ = ["PolicyValueNetwork"]
