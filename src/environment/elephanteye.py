"""UCCI engine interface for ElephantEye / Pikafish (docs/08-EVALUATION.md 2.3).

Xiangqi engines speak **UCCI** (Universal Chinese Chess Interface), not UCI.
This module converts our board to a Xiangqi FEN, converts moves to/from UCCI
coordinate strings (``b2e2``), and drives an engine subprocess to get its best
move for a position — giving an absolute, fixed-strength benchmark opponent.

Layer note: this stays Layer-1 pure (imports ``board``/``config`` only). The
``BaseAgent`` wrapper that plugs an engine into the tournament lives in
``src/agents/engine_agent.py``.

UCCI coordinates: files ``a``-``i`` = columns 0-8, ranks ``0``-``9`` = rows 0-9
(rank 0 is Red's back rank at the bottom), so our ``(row, col)`` maps directly.
"""

from __future__ import annotations

import subprocess

from src.environment.board import Board
from src.environment.move_generator import FullMove
from src.utils.config import (
    ADVISOR,
    BOARD_COLS,
    BOARD_ROWS,
    CANNON,
    CHARIOT,
    ELEPHANT,
    GENERAL,
    HORSE,
    RED,
    SOLDIER,
)

# Xiangqi FEN piece letters (uppercase = Red). Horse=N, Elephant=B (chess-style).
_FEN_CHAR = {
    GENERAL: "k", ADVISOR: "a", ELEPHANT: "b", HORSE: "n",
    CHARIOT: "r", CANNON: "c", SOLDIER: "p",
}


def move_to_ucci(move: FullMove) -> str:
    """``(from_row, from_col, to_row, to_col)`` → a UCCI string like ``a0a1``."""
    from_row, from_col, to_row, to_col = move
    return (
        f"{chr(ord('a') + from_col)}{from_row}"
        f"{chr(ord('a') + to_col)}{to_row}"
    )


def ucci_to_move(text: str) -> FullMove:
    """UCCI string like ``a0a1`` → ``(from_row, from_col, to_row, to_col)``."""
    text = text.strip()
    from_col = ord(text[0]) - ord("a")
    from_row = int(text[1])
    to_col = ord(text[2]) - ord("a")
    to_row = int(text[3])
    return (from_row, from_col, to_row, to_col)


def board_to_fen(board: Board) -> str:
    """Serialize a board to a Xiangqi FEN (ranks top→bottom, plus side to move)."""
    ranks: list[str] = []
    for row in range(BOARD_ROWS - 1, -1, -1):
        run = 0
        cells = ""
        for col in range(BOARD_COLS):
            value = board.piece_at(row, col)
            if value == 0:
                run += 1
                continue
            if run:
                cells += str(run)
                run = 0
            char = _FEN_CHAR[abs(value)]
            cells += char.upper() if value > 0 else char
        if run:
            cells += str(run)
        ranks.append(cells)
    side = "w" if board.to_move == RED else "b"
    return "/".join(ranks) + " " + side


class ElephantEyeEngine:
    """Drives a UCCI engine subprocess (ElephantEye, Pikafish, …).

    Pass the engine command (a path, or a list like ``[python, fake_engine.py]``
    for testing). Use as a context manager, or call ``start()`` / ``close()``.
    """

    def __init__(
        self,
        command: str | list[str],
        *,
        depth: int = 6,
        read_limit: int = 10_000,
    ) -> None:
        self.command = [command] if isinstance(command, str) else list(command)
        self.depth = depth
        self.read_limit = read_limit
        self.proc: subprocess.Popen | None = None

    def start(self) -> "ElephantEyeEngine":
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("ucci")
        self._read_until("ucciok")
        return self

    def bestmove(self, board: Board, *, depth: int | None = None) -> FullMove:
        """Ask the engine for its best move in ``board``'s position."""
        self._send(f"position fen {board_to_fen(board)}")
        self._send(f"go depth {depth or self.depth}")
        line = self._read_until("bestmove")
        return ucci_to_move(line.split()[1])

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self._send("quit")
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def __enter__(self) -> "ElephantEyeEngine":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ─── Low-level I/O ────────────────────────────────────────────────────
    def _send(self, command: str) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str) -> str:
        assert self.proc is not None and self.proc.stdout is not None
        for _ in range(self.read_limit):
            line = self.proc.stdout.readline()
            if line == "":  # engine exited unexpectedly
                raise RuntimeError(f"engine ended before '{token}'")
            if line.strip().startswith(token):
                return line.strip()
        raise RuntimeError(f"'{token}' not received within {self.read_limit} lines")


__all__ = [
    "move_to_ucci",
    "ucci_to_move",
    "board_to_fen",
    "ElephantEyeEngine",
]
