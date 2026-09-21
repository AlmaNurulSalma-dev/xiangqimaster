"""Unit tests for the game-record loader / replay validator."""

from __future__ import annotations

import pytest

from src.data.game_loader import (
    load_games_from_file,
    parse_game_records,
    parse_result,
)
from src.data.wxf_parser import DRAW, WXFParseError
from src.utils.config import BLACK, RED

# A valid 4-ply opening (verified legal from the start position).
VALID = ["C2.5", "C2.5", "H2+3", "H8+7"]


def test_parse_result_mapping():
    assert parse_result("red") == RED
    assert parse_result("0-1") == BLACK
    assert parse_result("1/2-1/2") == DRAW
    with pytest.raises(WXFParseError):
        parse_result("banana")


def test_parse_records_counts_valid_invalid_and_short():
    records = [
        ("red", VALID),          # valid
        ("draw", ["R1+9"]),      # illegal move → invalid
        ("black", ["C2.5"]),     # 1 ply → too short
    ]
    games, stats = parse_game_records(records, min_plies=4)
    assert stats.total == 3
    assert stats.valid == 1
    assert stats.invalid == 1
    assert stats.too_short == 1
    assert len(games) == 1
    assert games[0].outcome == RED
    assert len(games[0].moves) == 4


def test_load_games_from_file(tmp_path):
    path = tmp_path / "games.txt"
    path.write_text(
        "# a tiny dataset\n"
        f"red {' '.join(VALID)}\n"
        "draw R1+9\n"
        "black C2.5\n",
        encoding="utf-8",
    )
    games, stats = load_games_from_file(str(path), min_plies=4)
    assert stats.total == 3 and stats.valid == 1
    assert len(games) == 1


def test_parse_records_passes_iccs_notation_through():
    # dpxq-style coordinate records: both sides open with the central cannon.
    # ICCS equivalents of VALID: central cannons then knights out.
    records = [("red", ["h2e2", "h7e7", "h0g2", "h9g7"])]
    games, stats = parse_game_records(records, min_plies=4, notation="iccs")
    assert stats.valid == 1
    assert games[0].moves[0] == (2, 7, 2, 4)
