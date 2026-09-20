"""Unit tests for the single-agent self-play wrapper."""

from __future__ import annotations

import numpy as np

from src.training.self_play_env import SelfPlayEnv
from src.utils.config import (
    ACTION_SPACE_SIZE,
    DRAW_REWARD,
    LOSS_REWARD,
    NUM_CHANNELS,
    WIN_REWARD,
)


def test_reset_returns_agent_perspective_observation():
    env = SelfPlayEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (NUM_CHANNELS, 10, 9)
    # It is the learning agent's turn right after reset.
    assert info["current_player"] == info["agent_color"]


def test_action_masks_are_legal_and_nonempty():
    env = SelfPlayEnv()
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_step_returns_agent_turn_again():
    env = SelfPlayEnv()
    _, _ = env.reset(seed=1)
    action = int(np.flatnonzero(env.action_masks())[0])
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (NUM_CHANNELS, 10, 9)
    assert isinstance(reward, float)
    if not (terminated or truncated):
        # After the opponent replied, it is the agent's turn once more.
        assert info["current_player"] == info["agent_color"]


def test_full_random_rollout_terminates_with_valid_agent_reward():
    env = SelfPlayEnv()
    _, _ = env.reset(seed=7)
    terminated = truncated = False
    reward = 0.0
    steps = 0
    while not (terminated or truncated):
        action = int(np.flatnonzero(env.action_masks())[0])
        _, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        assert steps <= 400
    # The final reward is from the agent's perspective.
    assert reward in {WIN_REWARD, LOSS_REWARD, DRAW_REWARD}
