"""WXF move-notation parser and game replay/validation (docs/04-DATA-PIPELINE.md).

WXF (World Xiangqi Federation) notation records each move relative to the moving
player. A move is ``<piece><file><op><value>`` — for example ``C2.5`` (the
"central cannon": the cannon on the mover's file 2 traverses to file 5). Files
are numbered 1-9 from each player's OWN RIGHT, so Red and Black mirror each
other. When two identical pieces share a file, a ``+`` (front) or ``-`` (rear)
prefix replaces the file digit, e.g. ``+R+1``.

Operators:
* ``+`` forward (toward the enemy), ``-`` backward, ``.`` or ``=`` sideways.

The final value means:
* sideways (``.``): the destination file (mover-relative);
* forward/backward of a straight-line piece (King, Chariot, Cannon, Pawn): the
  number of steps;
* forward/backward of a diagonal piece (Horse, Advisor, Elephant): the
  destination file (the row change is implied by the shape).

This module converts a WXF token (given the current board) into an absolute
``(from_row, from_col, to_row, to_col)`` move, and replays a whole game with
legality validation (docs/04 section 5) so corrupt records are caught.
"""

from __future__ import annotations

from collections import defaultdict
import re
from dataclasses import dataclass

from src.environment.board import Board
from src.environment.move_generator import FullMove, generate_legal_moves
from src.utils.config import (
    ADVISOR,
    CANNON,
    CHARIOT,
    DRAW_REWARD,  # noqa: F401  (kept for reference; value mapping uses 0 for draw)
    ELEPHANT,
    GENERAL,
    HORSE,
    RED,
    SOLDIER,
)

#: Draw outcome sentinel for a parsed game (distinct from RED=+1 / BLACK=-1).
DRAW: int = 0

# Tolerant letter map (different sources use N/H for horse, B/E for elephant).
_LETTER_TO_TYPE: dict[str, int] = {
    "K": GENERAL, "G": GENERAL,
    "A": ADVISOR,
    "E": ELEPHANT, "B": ELEPHANT,
    "H": HORSE, "N": HORSE,
    "R": CHARIOT,
    "C": CANNON,
    "P": SOLDIER,
}

_STRAIGHT_MOVERS = {GENERAL, CHARIOT, CANNON, SOLDIER}


class WXFParseError(ValueError):
    """Raised when a WXF token cannot be parsed or is illegal in context."""


def _forward_dir(color: int) -> int:
    """Row delta for a forward step: Red +1 (up the board), Black -1."""
    return 1 if color == RED else -1


def _file_to_col(file_num: int, color: int) -> int:
    """Convert a mover-relative file (1-9) to an absolute column (0-8)."""
    if not 1 <= file_num <= 9:
        raise WXFParseError(f"file out of range: {file_num}")
    return 9 - file_num if color == RED else file_num - 1


def parse_wxf_move(board: Board, token: str) -> FullMove:
    """Convert a single WXF token into an absolute move for the side to move."""
    color = board.to_move
    tok = token.strip()
    if len(tok) < 4:
        raise WXFParseError(f"token too short: {token!r}")

    # Split into (piece, file-or-prefix, operator, value).
    if tok[0] in "+-" and tok[1].isalpha():
        prefix, piece_letter, op, value_char = tok[0], tok[1], tok[2], tok[3]
        start_file = None
    else:
        piece_letter, file_char, op, value_char = tok[0], tok[1], tok[2], tok[3]
        prefix = None
        if not file_char.isdigit():
            raise WXFParseError(f"expected a file digit in {token!r}")
        start_file = int(file_char)

    piece_type = _LETTER_TO_TYPE.get(piece_letter.upper())
    if piece_type is None:
        raise WXFParseError(f"unknown piece letter {piece_letter!r} in {token!r}")
    if op not in "+-.=":
        raise WXFParseError(f"unknown operator {op!r} in {token!r}")
    if not value_char.isdigit():
        raise WXFParseError(f"expected a numeric value in {token!r}")
    value = int(value_char)

    from_row, from_col = _resolve_source(
        board, color, piece_type, start_file, prefix, token
    )
    to_row, to_col = _resolve_destination(
        color, piece_type, from_row, from_col, op, value
    )
    return (from_row, from_col, to_row, to_col)


def _resolve_source(
    board: Board,
    color: int,
    piece_type: int,
    start_file: int | None,
    prefix: str | None,
    token: str,
) -> tuple[int, int]:
    candidates = [
        (r, c) for r, c, t in board.pieces_of(color) if t == piece_type
    ]
    if not candidates:
        raise WXFParseError(f"no {piece_type} piece for move {token!r}")

    if start_file is not None:
        col = _file_to_col(start_file, color)
        same_file = [(r, c) for r, c in candidates if c == col]
        if len(same_file) == 1:
            return same_file[0]
        if len(same_file) == 0:
            raise WXFParseError(f"no piece on file for move {token!r}")
        raise WXFParseError(
            f"ambiguous move {token!r}: {len(same_file)} pieces share the file"
        )

    # Front/rear disambiguation prefix (+ = front, - = rear).
    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in candidates:
        by_col[c].append((r, c))
    groups = [g for g in by_col.values() if len(g) >= 2]
    if not groups:
        raise WXFParseError(f"no stacked pieces to disambiguate for {token!r}")
    group = sorted(groups[0], key=lambda rc: rc[0])  # ascending row
    # "Front" is furthest in the mover's forward direction.
    front, rear = (group[-1], group[0]) if color == RED else (group[0], group[-1])
    return front if prefix == "+" else rear


def _resolve_destination(
    color: int,
    piece_type: int,
    from_row: int,
    from_col: int,
    op: str,
    value: int,
) -> tuple[int, int]:
    if op in ".=":  # sideways: value is the destination file
        return from_row, _file_to_col(value, color)

    sign = _forward_dir(color) if op == "+" else -_forward_dir(color)

    if piece_type in _STRAIGHT_MOVERS:  # value is a step count along the file
        return from_row + sign * value, from_col

    # Diagonal movers: value is the destination file; the row change is implied.
    dest_col = _file_to_col(value, color)
    col_delta = abs(dest_col - from_col)
    if piece_type == HORSE:
        row_delta = 2 if col_delta == 1 else 1
    elif piece_type == ADVISOR:
        row_delta = 1
    else:  # ELEPHANT
        row_delta = 2
    return from_row + sign * row_delta, dest_col


@dataclass(frozen=True)
class Game:
    """A validated game: the moves as absolute coordinates plus the result."""

    moves: tuple[FullMove, ...]
    outcome: int  # RED (+1), BLACK (-1), or DRAW (0)


# ─── ICCS / UCCI coordinate notation ────────────────────────────────────────
# Coordinate moves name the from/to squares directly: file a-i = col 0-8, rank
# 0-9 = row 0-9 with rank 0 at Red's side, mapping straight onto our (row, col).
# Two dialects appear in the wild and are both accepted here:
#   * UCCI (engines, XiangqiCore): lowercase, no separator — "h2e2"
#   * ICCS (xqbase / dpxq / WXF-Federation datasets): uppercase, dash — "C2-E2"
# We normalise (drop the dash, lowercase) so a single code path handles both.
_ICCS_RE = re.compile(r"^[a-i][0-9][a-i][0-9]$")


def _normalize_coord(token: str) -> str:
    """Normalise a coordinate token to the canonical ``h2e2`` form."""
    return token.strip().replace("-", "").lower()


def parse_iccs_move(token: str) -> FullMove:
    """Parse a coordinate move (``h2e2`` UCCI or ``C2-E2`` ICCS) into a move."""
    tok = _normalize_coord(token)
    if not _ICCS_RE.match(tok):
        raise WXFParseError(f"not a coordinate move: {token!r}")
    from_col = ord(tok[0]) - ord("a")
    from_row = int(tok[1])
    to_col = ord(tok[2]) - ord("a")
    to_row = int(tok[3])
    return (from_row, from_col, to_row, to_col)


def detect_notation(token: str) -> str:
    """Return ``"iccs"`` if the token is coordinate notation, else ``"wxf"``."""
    return "iccs" if _ICCS_RE.match(_normalize_coord(token)) else "wxf"


def parse_move(board: Board, token: str, notation: str = "auto") -> FullMove:
    """Parse a single move token in WXF, ICCS, or auto-detected notation."""
    if notation == "auto":
        notation = detect_notation(token)
    if notation == "iccs":
        return parse_iccs_move(token)
    if notation == "wxf":
        return parse_wxf_move(board, token)
    raise WXFParseError(f"unknown notation {notation!r}")


def parse_game(
    tokens: list[str],
    outcome: int,
    *,
    notation: str = "auto",
    validate: bool = True,
) -> Game:
    """Replay move tokens from the start position into a validated :class:`Game`.

    ``notation`` is ``"auto"`` (detect per token), ``"iccs"`` (``b2e2``), or
    ``"wxf"`` (``C2.5``). Each move is parsed relative to the current board and,
    if ``validate`` is set, checked against the legal-move generator — an illegal
    move raises :class:`WXFParseError`, so corrupt records are rejected
    (docs/04 section 5).
    """
    board = Board()
    moves: list[FullMove] = []
    for ply, token in enumerate(tokens):
        move = parse_move(board, token, notation)
        if validate and move not in set(generate_legal_moves(board)):
            raise WXFParseError(
                f"illegal move {token!r} (ply {ply}) → {move}"
            )
        board.apply_move(*move)
        moves.append(move)
    return Game(moves=tuple(moves), outcome=outcome)


__all__ = [
    "DRAW",
    "WXFParseError",
    "parse_wxf_move",
    "parse_iccs_move",
    "detect_notation",
    "parse_move",
    "Game",
    "parse_game",
]
