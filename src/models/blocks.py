"""Convolutional building blocks for the Policy-Value network.

These are the reusable pieces of the AlphaZero-style body: an input
convolution that lifts the 14-channel board tensor to the working width ``C``,
and a residual block that forms the repeated unit of the residual tower
(docs/05-NEURAL-NETWORK.md section 4).
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv2D(3x3, pad 1) → BatchNorm → ReLU.

    Used as the input convolution (14 → C). Padding 1 keeps the 10x9 board
    spatial size unchanged.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """A pre-activation-free residual block that preserves shape (C x 10 x 9).

    Structure (docs/05-NEURAL-NETWORK.md 4.2):
        Conv3x3 → BN → ReLU → Conv3x3 → BN → (+ skip) → ReLU
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # skip connection, before the final activation
        return self.relu(out)


__all__ = ["ConvBlock", "ResidualBlock"]
