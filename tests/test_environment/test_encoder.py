"""Unit tests for the board→tensor encoder (docs/03-ENVIRONMENT.md section 2)."""

from __future__ import annotations

from src.environment.board import Board
from src.environment import encoder
from src.utils.config import (
    BLACK,
    CHARIOT,
    NUM_CHANNELS,
    RED,
)


def test_shape_and_dtype():
    tensor = encoder.encode(Board())
    assert tensor.shape == (NUM_CHANNELS, 10, 9)
    assert tensor.dtype.name == "float32"


def test_all_pieces_encoded_exactly_once():
    tensor = encoder.encode(Board())
    assert tensor.sum() == 32  # 16 pieces per side, one 1.0 each


def test_red_perspective_own_and_opponent_general():
    board = Board()
    tensor = encoder.encode(board, perspective=RED)
    # Channel 0 = own General; Red's General is at (0, 4).
    assert tensor[0, 0, 4] == 1.0
    # Channel 7 = opponent's General; Black's General is at (9, 4).
    assert tensor[7, 9, 4] == 1.0


def test_player_relative_black_is_rotated_to_bottom():
    board = Board()
    tensor = encoder.encode(board, perspective=BLACK)
    # From Black's perspective the board is rotated 180°, so Black's General at
    # absolute (9, 4) lands in its OWN channel 0 at (0, 4).
    assert tensor[0, 0, 4] == 1.0
    # Red's General at absolute (0, 4) becomes the opponent (channel 7) at
    # rotated (9, 4).
    assert tensor[7, 9, 4] == 1.0


def test_rotation_maps_coordinates_180_degrees():
    board = Board(empty=True)
    board.grid[0, 0] = RED * CHARIOT  # a lone Red chariot in the corner
    # Red's view: own chariot (type CHARIOT=5 → channel 4) at (0, 0).
    red_view = encoder.encode(board, perspective=RED)
    assert red_view[4, 0, 0] == 1.0
    # Black's view: that chariot is the opponent (channel 7+4=11) and rotates
    # 180° from (0,0) to (9,8).
    black_view = encoder.encode(board, perspective=BLACK)
    assert black_view[11, 9, 8] == 1.0
    assert black_view[4, 0, 0] == 0.0


def test_defaults_to_side_to_move():
    board = Board()  # Red to move at the start
    assert (encoder.encode(board) == encoder.encode(board, perspective=RED)).all()
