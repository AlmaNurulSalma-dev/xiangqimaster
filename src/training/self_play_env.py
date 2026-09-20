"""Single-agent self-play wrapper for PPO training.

stable-baselines3 expects a single-agent environment, but Xiangqi is two-player
and alternates turns. This wrapper turns the two-player game into a single-agent
problem by *baking the opponent into the environment*: on each ``step`` the
learning agent makes its move, then the opponent immediately replies, and the
observation returned is again the learning agent's to act on. Reward is sparse
and expressed from the LEARNING AGENT's perspective (+1 win, -1 loss, +0.1
draw), so PPO maximizes the agent's own winning chances.

The opponent is any ``BaseAgent`` (default: a random agent — the simplest
self-play bootstrap, docs/07-TRAINING.md 3.6). It can later be swapped for a
frozen snapshot of the learning policy for true self-play.

Action masking: ``action_masks()`` exposes the legal-move mask for the side the
agent is about to move, which is what MaskablePPO consumes.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.environment import action_space, encoder
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.environment.xiangqi_env import XiangqiEnv
from src.utils.config import (
    ACTION_SPACE_SIZE,
    BLACK,
    BOARD_COLS,
    BOARD_ROWS,
    DRAW_REWARD,
    LOSS_REWARD,
    NUM_CHANNELS,
    RED,
    WIN_REWARD,
)


class SelfPlayEnv(gym.Env):
    """Two-player Xiangqi presented as a single-agent env for PPO."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent: BaseAgent | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.opponent: BaseAgent = opponent or RandomAgent()
        self._env = XiangqiEnv()
        self.agent_color: int = RED

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_CHANNELS, BOARD_ROWS, BOARD_COLS),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    # ─── Gymnasium API ────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._env.reset(seed=seed)
        # Randomize which colour the learning agent plays, for balance.
        self.agent_color = RED if self.np_random.random() < 0.5 else BLACK
        # If the agent is Black, the opponent (Red) makes the first move.
        if self._env.get_current_player() != self.agent_color:
            self._opponent_move()
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # 1) The learning agent moves.
        _, reward, terminated, truncated, _ = self._env.step(action)
        if terminated or truncated:
            agent_reward = self._agent_reward(reward, mover_is_agent=True)
            return self._observation(), agent_reward, terminated, truncated, self._info()

        # 2) The opponent replies.
        opp_reward, terminated, truncated = self._opponent_move()
        if terminated or truncated:
            agent_reward = self._agent_reward(opp_reward, mover_is_agent=False)
            return self._observation(), agent_reward, terminated, truncated, self._info()

        # 3) Back to the agent, game continues (sparse reward → 0 mid-game).
        return self._observation(), 0.0, False, False, self._info()

    def render(self) -> str | None:
        return self._env.render()

    # ─── Masking hook for MaskablePPO ─────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        """Legal-move mask for the learning agent's current turn."""
        return self._env.legal_action_mask()

    # ─── Internals ─────────────────────────────────────────────────────────
    def _opponent_move(self) -> tuple[float, bool, bool]:
        """Let the opponent make one move; return (reward, terminated, truncated)."""
        mask = self._env.legal_action_mask()
        opp_action = self.opponent.select_move(self._env.board, mask)
        _, reward, terminated, truncated, _ = self._env.step(opp_action)
        return reward, terminated, truncated

    @staticmethod
    def _agent_reward(inner_reward: float, mover_is_agent: bool) -> float:
        """Convert the inner env's mover-perspective reward to the agent's.

        The inner env only ever emits a WIN (+1) for the mover or a DRAW (+0.1);
        a loss is expressed as the opponent winning.
        """
        if inner_reward == DRAW_REWARD:
            return DRAW_REWARD
        return WIN_REWARD if mover_is_agent else LOSS_REWARD

    def _observation(self) -> np.ndarray:
        return encoder.encode(self._env.board)

    def _info(self) -> dict[str, Any]:
        return {
            "agent_color": self.agent_color,
            "current_player": self._env.get_current_player(),
            "ply_count": self._env.ply_count,
        }


__all__ = ["SelfPlayEnv"]
