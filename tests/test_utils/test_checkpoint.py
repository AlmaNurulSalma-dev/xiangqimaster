"""Unit tests for checkpoint save/load."""

from __future__ import annotations

import torch

from src.models.network import PolicyValueNetwork
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def _tiny_net() -> PolicyValueNetwork:
    return PolicyValueNetwork(channels=8, num_blocks=1)


def test_save_and_load_restores_weights(tmp_path):
    net = _tiny_net()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, net, step=100, extra={"seed": 42})

    restored = _tiny_net()
    # Sanity: fresh net differs before loading.
    w_a = net.input_conv.conv.weight
    w_b = restored.input_conv.conv.weight
    assert not torch.allclose(w_a, w_b)

    payload = load_checkpoint(path, restored)
    assert torch.allclose(restored.input_conv.conv.weight, w_a)
    assert payload["step"] == 100
    assert payload["extra"]["seed"] == 42


def test_optimizer_state_roundtrips(tmp_path):
    net = _tiny_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # Take one step so the optimizer has state.
    loss = net(torch.zeros(1, 14, 10, 9))[1].sum()
    loss.backward()
    optimizer.step()

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, net, optimizer, step=1)

    net2 = _tiny_net()
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    payload = load_checkpoint(path, net2, opt2)
    assert payload["step"] == 1
    assert opt2.state_dict()["param_groups"][0]["lr"] == 1e-3


def test_creates_missing_directory(tmp_path):
    net = _tiny_net()
    path = str(tmp_path / "nested" / "dir" / "ckpt.pt")
    save_checkpoint(path, net)  # should create the directories
    import os

    assert os.path.exists(path)
