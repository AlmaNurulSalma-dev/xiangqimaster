"""Unit tests for the Policy-Value network (docs/05-NEURAL-NETWORK.md 10)."""

from __future__ import annotations

import numpy as np
import torch

from src.environment.board import Board
from src.environment import encoder
from src.models.network import PolicyValueNetwork
from src.utils.config import ACTION_SPACE_SIZE, NUM_CHANNELS


def _small_net() -> PolicyValueNetwork:
    # A tiny network keeps the tests fast; shapes and behaviour are identical.
    return PolicyValueNetwork(channels=16, num_blocks=2)


def test_output_shapes():
    net = _small_net().eval()
    x = torch.zeros(4, NUM_CHANNELS, 10, 9)
    policy, value = net(x)
    assert policy.shape == (4, ACTION_SPACE_SIZE)
    assert value.shape == (4, 1)


def test_value_is_within_tanh_range():
    net = _small_net().eval()
    x = torch.randn(8, NUM_CHANNELS, 10, 9)
    _, value = net(x)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_deterministic_in_eval_mode():
    net = _small_net().eval()
    x = torch.randn(2, NUM_CHANNELS, 10, 9)
    p1, v1 = net(x)
    p2, v2 = net(x)
    assert torch.allclose(p1, p2)
    assert torch.allclose(v1, v2)


def test_gradient_flows_to_all_parameters():
    net = _small_net().train()
    x = torch.randn(2, NUM_CHANNELS, 10, 9)
    policy, value = net(x)
    loss = policy.sum() + value.sum()
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"


def test_single_train_step_reduces_loss_on_fixed_batch():
    # Sanity check: the network can memorise a tiny fixed batch.
    torch.manual_seed(0)
    net = _small_net().train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    x = torch.randn(4, NUM_CHANNELS, 10, 9)
    policy_target = torch.randint(0, ACTION_SPACE_SIZE, (4,))
    value_target = torch.rand(4, 1) * 2 - 1  # in [-1, 1]

    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    def compute_loss() -> torch.Tensor:
        policy, value = net(x)
        return policy_loss_fn(policy, policy_target) + value_loss_fn(value, value_target)

    initial = compute_loss().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
    final = compute_loss().item()
    assert final < initial


def test_accepts_encoded_board():
    # End-to-end: a real encoded position flows through the network.
    net = _small_net().eval()
    tensor = encoder.encode(Board())              # (14, 10, 9)
    batch = torch.from_numpy(np.expand_dims(tensor, 0))  # (1, 14, 10, 9)
    policy, value = net(batch)
    assert policy.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1, 1)
