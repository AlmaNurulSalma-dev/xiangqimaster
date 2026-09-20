"""A stable-baselines3 features extractor built from our ResNet body.

MaskablePPO (sb3-contrib) supplies its own policy/value output heads, but it
lets us plug in a custom features extractor. We reuse the AlphaZero-style body
from ``blocks.py`` (input conv + residual tower) so PPO learns on the SAME
convolutional representation the rest of the project uses (docs/05-NEURAL-
NETWORK.md), rather than a generic MLP over the flattened board.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.models.blocks import ConvBlock, ResidualBlock
from src.utils.config import (
    NN_CHANNEL_WIDTH,
    NN_NUM_RES_BLOCKS,
    PPO_FEATURES_DIM,
)


class XiangqiResNetExtractor(BaseFeaturesExtractor):
    """Input conv → residual tower → flatten → linear → feature vector.

    Args:
        observation_space: the ``(14, 10, 9)`` board Box space.
        channels: residual-tower width ``C``.
        num_blocks: number of residual blocks ``N``.
        features_dim: size of the output feature vector handed to SB3's heads.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        channels: int = NN_CHANNEL_WIDTH,
        num_blocks: int = NN_NUM_RES_BLOCKS,
        features_dim: int = PPO_FEATURES_DIM,
    ) -> None:
        super().__init__(observation_space, features_dim)
        in_channels, rows, cols = observation_space.shape
        self.input_conv = ConvBlock(in_channels, channels)
        self.tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_blocks))
        )
        self.fc = nn.Linear(channels * rows * cols, features_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(observations)
        x = self.tower(x)
        x = torch.flatten(x, start_dim=1)
        return self.relu(self.fc(x))


__all__ = ["XiangqiResNetExtractor"]
