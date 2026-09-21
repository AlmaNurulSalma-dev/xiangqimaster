"""Adapter: Xiangqi PGN files → records for the game loader.

Turns standard Xiangqi PGN text into ``(result, tokens)`` records that
``game_loader.parse_game_records`` validates. A PGN game is a block of
``[Tag "value"]`` pairs followed by movetext, e.g.::

    [Event "..."]
    [Result "1-0"]

    1. C2.5 C8.5 2. H2+3 H8+7 3. R1.2 R9.8 1-0

We read the ``Result`` tag and strip move numbers, comments ``{...}``, NAGs
``$n`` and the trailing result token to recover the bare move list.

Movetext may be Roman WXF (``C2.5``, ``H2+3``) or ICCS coordinates (``h2e2``);
the parser auto-detects per token, or a ``notation`` can be forced. Datasets in
Chinese characters (炮二平五) would still need a different move parser; that is a
documented limitation, not handled here.
"""

from __future__ import annotations

import os
import re

from src.data.game_loader import LoadStats, parse_game_records
from src.data.wxf_parser import Game

_TAG_RE = re.compile(r'\[(\w+)\s+"(.*)"\]')
_COMMENT_RE = re.compile(r"\{[^}]*\}")
_NAG_RE = re.compile(r"\$\d+")
_MOVE_NUMBER_PREFIX = re.compile(r"^\d+\.+")
_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


def split_pgn_games(text: str) -> list[tuple[list[str], list[str]]]:
    """Split PGN text into games as ``(tag_lines, movetext_lines)``."""
    games: list[tuple[list[str], list[str]]] = []
    tags: list[str] = []
    moves: list[str] = []
    seen_moves = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            if seen_moves:  # a tag after movetext → the next game begins
                games.append((tags, moves))
                tags, moves, seen_moves = [], [], False
            tags.append(line)
        elif line:
            moves.append(line)
            seen_moves = True
    if tags or moves:
        games.append((tags, moves))
    return games


def parse_tags(tag_lines: list[str]) -> dict[str, str]:
    """Parse ``[Key "value"]`` lines into a dict."""
    tags: dict[str, str] = {}
    for line in tag_lines:
        match = _TAG_RE.match(line)
        if match:
            tags[match.group(1)] = match.group(2)
    return tags


def extract_move_tokens(movetext: str) -> list[str]:
    """Strip comments, NAGs, move numbers and the result → bare move tokens."""
    text = _COMMENT_RE.sub(" ", movetext)
    text = _NAG_RE.sub(" ", text)
    tokens: list[str] = []
    for raw in text.split():
        token = _MOVE_NUMBER_PREFIX.sub("", raw)  # drop a leading "12." prefix
        if not token or token in _RESULT_TOKENS:
            continue
        tokens.append(token)
    return tokens


def pgn_to_records(text: str) -> list[tuple[str, list[str]]]:
    """Convert PGN text into ``(result, tokens)`` records (skips ``*`` games)."""
    records: list[tuple[str, list[str]]] = []
    for tag_lines, move_lines in split_pgn_games(text):
        tags = parse_tags(tag_lines)
        result = tags.get("Result", "*")
        if result == "*":
            continue  # unfinished game
        tokens = extract_move_tokens(" ".join(move_lines))
        if tokens:
            records.append((result, tokens))
    return records


def load_games_from_pgn(
    source: str,
    *,
    min_plies: int = 10,
    validate: bool = True,
    notation: str = "auto",
) -> tuple[list[Game], LoadStats]:
    """Load games from a PGN file path (or raw PGN text) with validation.

    ``notation`` (``"auto"``/``"iccs"``/``"wxf"``) is passed to the parser.
    """
    text = source
    if os.path.exists(source):
        with open(source, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    records = pgn_to_records(text)
    return parse_game_records(
        records, min_plies=min_plies, validate=validate, notation=notation
    )


__all__ = [
    "split_pgn_games",
    "parse_tags",
    "extract_move_tokens",
    "pgn_to_records",
    "load_games_from_pgn",
]
