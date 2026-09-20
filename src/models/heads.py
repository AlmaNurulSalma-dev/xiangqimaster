"""Output heads for the Policy-Value network (docs/05-NEURAL-NETWORK.md 4.3-4.4).

The shared body produces a (C x 10 x 9) feature map. Two heads consume it:

* :class:`PolicyHead` → raw logits over the 2086 possible moves. Logits, NOT
  probabilities: illegal-move masking and softmax happen later in the agent so
  illegal moves can be pushed to -inf before the softmax.
* :class:`ValueHead` → a single scalar in [-1, 1] (tanh) estimating the outcome
  from the current player's perspective.
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils.config import (
    ACTION_SPACE_SIZE,
    BOARD_COLS,
    BOARD_ROWS,
    NN_POLICY_HEAD_CHANNELS,
    NN_VALUE_HEAD_HIDDEN,
)

_BOARD_AREA = BOARD_ROWS * BOARD_COLS  # 90


class PolicyHead(nn.Module):
    """Conv1x1 → BN → ReLU → Flatten → Linear → raw move logits."""

    def __init__(
        self,
        in_channels: int,
        head_channels: int = NN_POLICY_HEAD_CHANNELS,
        action_size: int = ACTION_SPACE_SIZE,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(head_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(head_channels * _BOARD_AREA, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        out = torch.flatten(out, start_dim=1)
        return self.fc(out)  # raw logits, shape (batch, action_size)


class ValueHead(nn.Module):
    """Conv1x1 → BN → ReLU → Flatten → Linear → ReLU → Linear → tanh."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = NN_VALUE_HEAD_HIDDEN,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(_BOARD_AREA, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        out = torch.flatten(out, start_dim=1)
        out = self.relu(self.fc1(out))
        out = self.tanh(self.fc2(out))
        return out  # shape (batch, 1), values in [-1, 1]


__all__ = ["PolicyHead", "ValueHead"]
