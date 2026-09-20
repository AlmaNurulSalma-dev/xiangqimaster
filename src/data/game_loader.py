"""Load and replay-validate game records into :class:`Game` objects.

This is the front of the data pipeline (docs/04-DATA-PIPELINE.md section 11):
read raw records, parse WXF moves, replay-validate every game, apply quality
filters, and drop anything corrupt — logging how many games passed each stage.

A record is ``(result, tokens)`` where ``result`` is a string like ``"red"`` /
``"black"`` / ``"draw"`` (or PGN-style ``"1-0"`` / ``"0-1"`` / ``"1/2-1/2"``)
and ``tokens`` is the list of WXF move strings. A specific dataset file format
(e.g. kaifeiji/xiangqi PGN) only needs a thin adapter that turns its lines into
these records; this loader then validates and filters them.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.wxf_parser import DRAW, Game, WXFParseError, parse_game
from src.utils.config import BLACK, RED

_RESULT_MAP: dict[str, int] = {
    "red": RED, "1-0": RED, "r": RED,
    "black": BLACK, "0-1": BLACK, "b": BLACK,
    "draw": DRAW, "1/2-1/2": DRAW, "1/2": DRAW, "d": DRAW,
}


@dataclass
class LoadStats:
    """How many games passed each stage of loading."""

    total: int = 0
    valid: int = 0
    invalid: int = 0        # failed to parse / illegal move on replay
    too_short: int = 0      # dropped by the minimum-length filter


def parse_result(result: str) -> int:
    """Map a result string to RED / BLACK / DRAW."""
    key = result.strip().lower()
    if key not in _RESULT_MAP:
        raise WXFParseError(f"unknown result {result!r}")
    return _RESULT_MAP[key]


def parse_game_records(
    records: list[tuple[str, list[str]]],
    *,
    min_plies: int = 10,
    validate: bool = True,
) -> tuple[list[Game], LoadStats]:
    """Parse and validate ``(result, tokens)`` records into games + stats."""
    games: list[Game] = []
    stats = LoadStats()
    for result, tokens in records:
        stats.total += 1
        try:
            outcome = parse_result(result)
            game = parse_game(tokens, outcome, validate=validate)
        except WXFParseError:
            stats.invalid += 1
            continue
        if len(game.moves) < min_plies:
            stats.too_short += 1
            continue
        games.append(game)
        stats.valid += 1
    return games, stats


def load_games_from_file(
    path: str,
    *,
    min_plies: int = 10,
    validate: bool = True,
) -> tuple[list[Game], LoadStats]:
    """Load games from a simple text file: one game per line as
    ``<result> tok1 tok2 ...``. Blank lines and ``#`` comments are ignored."""
    records: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            records.append((parts[0], parts[1:]))
    return parse_game_records(records, min_plies=min_plies, validate=validate)


__all__ = ["LoadStats", "parse_result", "parse_game_records", "load_games_from_file"]
