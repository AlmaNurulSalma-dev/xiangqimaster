"""Unit tests for the Gymnasium environment wrapper."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.environment.xiangqi_env import XiangqiEnv
from src.utils.config import (
    ACTION_SPACE_SIZE,
    BLACK,
    NUM_CHANNELS,
    RED,
)


def _first_legal_action(info) -> int:
    return int(np.flatnonzero(info["legal_mask"])[0])


def test_reset_returns_obs_and_info():
    env = XiangqiEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (NUM_CHANNELS, 10, 9)
    assert env.observation_space.contains(obs)
    assert info["legal_mask"].shape == (ACTION_SPACE_SIZE,)
    assert int(info["legal_mask"].sum()) == 44  # opening position
    assert info["current_player"] == RED


def test_step_advances_turn_and_ply():
    env = XiangqiEnv()
    _, info = env.reset(seed=0)
    action = _first_legal_action(info)
    obs, reward, terminated, truncated, info2 = env.step(action)
    assert obs.shape == (NUM_CHANNELS, 10, 9)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert info2["current_player"] == BLACK  # turn passed to Black
    assert info2["ply_count"] == 1


def test_illegal_action_raises():
    env = XiangqiEnv()
    _, info = env.reset(seed=0)
    illegal = int(np.flatnonzero(~info["legal_mask"])[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_clone_is_independent():
    env = XiangqiEnv()
    _, info = env.reset(seed=0)
    twin = env.clone()
    env.step(_first_legal_action(info))
    # Advancing the original must not change the clone.
    assert twin.ply_count == 0
    assert twin.get_current_player() == RED


def test_full_random_game_terminates_with_a_result():
    env = XiangqiEnv()
    obs, info = env.reset(seed=123)
    rng = random.Random(123)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = rng.choice(np.flatnonzero(info["legal_mask"]).tolist())
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps <= 300
    assert "result" in info
    assert info["result"] in {"red_win", "black_win", "draw"}


def test_render_ansi_returns_text():
    env = XiangqiEnv(render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert isinstance(text, str)
    assert "to move" in text
