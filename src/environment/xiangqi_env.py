"""Gymnasium environment wrapping the Xiangqi game logic.

This ties Layer 1 together into the standard ``reset()`` / ``step()`` API that
RL libraries (stable-baselines3 via sb3-contrib's MaskablePPO) expect
(docs/03-ENVIRONMENT.md section 5).

Key contracts:

* Observation: the ``14 x 10 x 9`` player-relative tensor from the perspective
  of the side to move (see ``encoder.py``).
* Action: an integer in ``[0, 2086)`` indexing the fixed action space.
* Reward: sparse and from the perspective of the player who just moved — win
  ``+1``, draw ``+0.1``, otherwise ``0``. A loss is not emitted as a step
  reward; the self-play loop assigns it to the loser's stored experiences
  (docs/03-ENVIRONMENT.md section 4.3).
* ``info`` always carries ``legal_mask`` — callers MUST mask the policy so an
  illegal action is never selected.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment import action_space as actions
from src.environment import encoder, rules
from src.environment.board import Board
from src.utils.config import (
    ACTION_SPACE_SIZE,
    BLACK,
    BOARD_COLS,
    BOARD_ROWS,
    DRAW_REWARD,
    MAX_PLIES,
    NUM_CHANNELS,
    RED,
    REPETITION_LIMIT,
    WIN_REWARD,
    opponent,
)


class XiangqiEnv(gym.Env):
    """A single-agent-per-turn Xiangqi environment (self-play compatible)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_CHANNELS, BOARD_ROWS, BOARD_COLS),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.board: Board = Board(empty=True)
        self.ply_count: int = 0
        self._position_counts: dict[tuple, int] = defaultdict(int)

    # ─── Gymnasium API ────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board = Board()  # standard starting position, Red to move
        self.ply_count = 0
        self._position_counts = defaultdict(int)
        self._position_counts[self.board.position_key()] += 1
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        move = actions.index_to_move(int(action))
        if move not in set(self._legal_moves()):
            raise ValueError(
                f"illegal action {action} ({move}); mask the policy with "
                "info['legal_mask'] before sampling"
            )

        mover = self.board.to_move
        self.board.apply_move(*move)
        self.ply_count += 1

        position_key = self.board.position_key()
        self._position_counts[position_key] += 1
        repetition_count = self._position_counts[position_key]

        reward = 0.0
        terminated = False
        truncated = False

        next_player = self.board.to_move  # opponent of `mover`
        if not rules.has_legal_moves(self.board, next_player):
            # Opponent has no reply (checkmate or stalemate) → the mover wins.
            reward = WIN_REWARD
            terminated = True
        elif repetition_count >= REPETITION_LIMIT:
            reward = DRAW_REWARD
            terminated = True
        elif self.ply_count >= MAX_PLIES:
            reward = DRAW_REWARD
            truncated = True

        info = self._info()
        info["mover"] = mover
        if terminated or truncated:
            info["result"] = self._result_label(reward, terminated, mover)
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        text = self.board.to_ascii()
        if self.render_mode == "ansi":
            return text
        print(text)
        return None

    # ─── Extra helpers used by agents / MCTS ──────────────────────────────
    def legal_action_mask(self) -> np.ndarray:
        """Boolean mask (length 2086) of legal actions in the current state."""
        return actions.legal_mask(self.board)

    def get_current_player(self) -> int:
        """Whose turn it is: ``RED`` or ``BLACK``."""
        return self.board.to_move

    def clone(self) -> "XiangqiEnv":
        """Deep, independent copy — required by MCTS (docs/03 section 6)."""
        twin = XiangqiEnv(render_mode=self.render_mode)
        twin.board = self.board.clone()
        twin.ply_count = self.ply_count
        twin._position_counts = defaultdict(int, self._position_counts)
        return twin

    # ─── Internals ─────────────────────────────────────────────────────────
    def _legal_moves(self) -> list[tuple[int, int, int, int]]:
        from src.environment.move_generator import generate_legal_moves

        return generate_legal_moves(self.board)

    def _observation(self) -> np.ndarray:
        return encoder.encode(self.board)

    def _info(self) -> dict[str, Any]:
        return {
            "legal_mask": self.legal_action_mask(),
            "current_player": self.board.to_move,
            "ply_count": self.ply_count,
        }

    @staticmethod
    def _result_label(reward: float, terminated: bool, mover: int) -> str:
        if terminated and reward == WIN_REWARD:
            return "red_win" if mover == RED else "black_win"
        return "draw"


__all__ = ["XiangqiEnv"]
