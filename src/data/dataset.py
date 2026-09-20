"""PyTorch dataset of imitation-learning examples (docs/04-DATA-PIPELINE.md).

Each example is one position from a human game:
* input  = the 14x10x9 board tensor (from the mover's perspective),
* policy label = the action index of the move the human played,
* value label  = the game outcome from that mover's perspective
  (+1 win, -1 loss, ``draw_value`` for a draw).

The encoder and action-space map are the SAME ones the environment uses, so
training data and self-play observations are guaranteed identical.

Optional horizontal-mirror augmentation exploits Xiangqi's left-right symmetry
(docs/04 section 9). Mirroring is done at the *board* level (flip columns) and
the move is mirrored to match, keeping the position legal and the (board, move)
pair consistent.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.wxf_parser import DRAW, Game
from src.environment import action_space, encoder
from src.environment.board import Board
from src.environment.move_generator import FullMove
from src.utils.config import BOARD_COLS


def mirror_move(move: FullMove) -> FullMove:
    """Mirror a move left-right (column c → 8 - c)."""
    from_row, from_col, to_row, to_col = move
    last = BOARD_COLS - 1
    return (from_row, last - from_col, to_row, last - to_col)


def mirror_board(board: Board) -> Board:
    """Return a left-right mirrored copy of a board (same side to move)."""
    twin = board.clone()
    twin.grid = np.fliplr(board.grid).copy()
    return twin


def _value_for(outcome: int, mover: int, draw_value: float) -> float:
    """Outcome from the mover's perspective: +1 win, -1 loss, draw_value draw."""
    if outcome == DRAW:
        return draw_value
    return float(outcome * mover)  # +1 if outcome matches mover, else -1


class XiangqiILDataset(Dataset):
    """In-memory dataset of (tensor, action_index, value) training examples."""

    def __init__(
        self,
        games: list[Game],
        *,
        draw_value: float = 0.0,
        mirror: bool = False,
    ) -> None:
        self._tensors: list[np.ndarray] = []
        self._actions: list[int] = []
        self._values: list[float] = []

        for game in games:
            board = Board()
            for move in game.moves:
                mover = board.to_move
                value = _value_for(game.outcome, mover, draw_value)

                self._tensors.append(encoder.encode(board))
                self._actions.append(action_space.move_to_index(move))
                self._values.append(value)

                if mirror:
                    m_board = mirror_board(board)
                    self._tensors.append(encoder.encode(m_board))
                    self._actions.append(action_space.move_to_index(mirror_move(move)))
                    self._values.append(value)

                board.apply_move(*move)

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        tensor = torch.from_numpy(self._tensors[index])
        return tensor, self._actions[index], self._values[index]


__all__ = ["XiangqiILDataset", "mirror_move", "mirror_board"]
