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

import bisect

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


class LazyXiangqiILDataset(Dataset):
    """Lazy variant of :class:`XiangqiILDataset` for large corpora.

    :class:`XiangqiILDataset` materialises every encoded position in RAM (~5 KB
    each), which does not scale to 100k+ games (millions of positions → tens of
    GB). This variant stores only the games and encodes each requested position
    on access by replaying the game up to that ply (~0.13 ms/item), so memory is
    O(games) instead of O(positions).

    It yields the SAME ``(tensor, action_index, value)`` examples as
    :class:`XiangqiILDataset` — one per ply, plus a left-right mirror per ply when
    ``mirror`` is set — so it is a drop-in replacement in the DataLoader.
    """

    def __init__(
        self,
        games: list[Game],
        *,
        draw_value: float = 0.0,
        mirror: bool = False,
    ) -> None:
        self._games = games
        self._draw_value = draw_value
        self._mirror = mirror
        # Prefix sums of ply counts → map a flat position index to (game, ply)
        # in O(log n) without materialising a per-position index.
        self._cum: list[int] = []
        running = 0
        for game in games:
            running += len(game.moves)
            self._cum.append(running)
        self._n_positions = running  # positions in a single orientation

    def __len__(self) -> int:
        return self._n_positions * (2 if self._mirror else 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        # Indices [0, n) are normal; [n, 2n) are the mirrored copies.
        mirrored = self._mirror and index >= self._n_positions
        base = index - self._n_positions if mirrored else index

        game_idx = bisect.bisect_right(self._cum, base)
        prev = self._cum[game_idx - 1] if game_idx > 0 else 0
        ply = base - prev
        game = self._games[game_idx]

        board = Board()
        for k in range(ply):
            board.apply_move(*game.moves[k])

        move = game.moves[ply]
        value = _value_for(game.outcome, board.to_move, self._draw_value)

        if mirrored:
            board = mirror_board(board)
            move = mirror_move(move)

        tensor = torch.from_numpy(encoder.encode(board))
        return tensor, action_space.move_to_index(move), value


__all__ = [
    "XiangqiILDataset",
    "LazyXiangqiILDataset",
    "mirror_move",
    "mirror_board",
]
