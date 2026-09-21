"""Unit tests for the Xiangqi PGN adapter."""

from __future__ import annotations

from src.data import pgn_adapter
from src.utils.config import RED

PGN = """[Event "T1"]
[Red "Alice"]
[Black "Bob"]
[Result "1-0"]

1. C2.5 C8.5 2. H2+3 H8+7 3. R1.2 R9.8 1-0

[Event "T2"]
[Result "0-1"]

1. C2.5 C8.5 0-1
"""


def test_extract_move_tokens_strips_numbers_comments_nags_result():
    text = "1. C2.5 {good} C8.5 $1 2.H2+3 H8+7 1-0"
    assert pgn_adapter.extract_move_tokens(text) == ["C2.5", "C8.5", "H2+3", "H8+7"]


def test_move_numbers_do_not_corrupt_moves():
    # The '.' inside "C2.5" must survive; only leading move numbers are stripped.
    assert pgn_adapter.extract_move_tokens("12. C2.5") == ["C2.5"]


def test_split_and_tags():
    games = pgn_adapter.split_pgn_games(PGN)
    assert len(games) == 2
    tags = pgn_adapter.parse_tags(games[0][0])
    assert tags["Result"] == "1-0"
    assert tags["Red"] == "Alice"


def test_pgn_to_records():
    records = pgn_adapter.pgn_to_records(PGN)
    assert len(records) == 2
    assert records[0][0] == "1-0"
    assert records[0][1][0] == "C2.5"


def test_load_games_from_pgn_validates_and_filters():
    games, stats = pgn_adapter.load_games_from_pgn(PGN, min_plies=4)
    assert stats.total == 2
    assert stats.valid == 1          # game 1 (6 plies) passes
    assert stats.too_short == 1      # game 2 (2 plies) filtered
    assert games[0].outcome == RED
    assert len(games[0].moves) == 6


def test_unfinished_games_are_skipped():
    pgn = '[Result "*"]\n\n1. C2.5 C8.5 *\n'
    assert pgn_adapter.pgn_to_records(pgn) == []


def test_load_games_from_pgn_handles_iccs_movetext():
    # dpxq-style PGN with ICCS coordinate moves (auto-detected).
    pgn = '[Result "1-0"]\n\n1. h2e2 h7e7 2. h0g2 h9g7 1-0\n'
    games, stats = pgn_adapter.load_games_from_pgn(pgn, min_plies=4)
    assert stats.valid == 1
    assert games[0].moves[0] == (2, 7, 2, 4)  # same as WXF "C2.5"


def test_ellipsis_placeholder_is_skipped():
    # XiangqiCore emits "..." for a round with no move on one side.
    tokens = pgn_adapter.extract_move_tokens("1. C2=5  c8=5 2. ...  h8+7")
    assert "..." not in tokens
    assert tokens == ["C2=5", "c8=5", "h8+7"]


# XiangqiCore's DefaultPgnGenerationService writes this exact tag block, with
# "1-0"/"0-1"/"1/2-1/2" Result values and "<n>. <red>  <black>" rounds. English
# movetext uses uppercase Red / lowercase Black and "=" for a traversing move.
XIANGQICORE_PGN_ENGLISH = """[Game "Chinese Chess"]
[Event "Test Cup"]
[Site "Somewhere"]
[Date "2024.01.01"]
[Red "Alice"]
[RedTeam "TeamA"]
[Black "Bob"]
[BlackTeam "TeamB"]
[Result "1-0"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]

1. C2=5  c8=5
2. H2+3  h8+7
3. R1=2  r9=8
"""

XIANGQICORE_PGN_UCCI = """[Game "Chinese Chess"]
[Result "0-1"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]

1. h2e2  h7e7
2. h0g2  h9g7
3. i0h0  i9h9
"""


def test_load_xiangqicore_english_pgn():
    games, stats = pgn_adapter.load_games_from_pgn(XIANGQICORE_PGN_ENGLISH, min_plies=4)
    assert stats.valid == 1
    assert games[0].outcome == RED
    assert games[0].moves[0] == (2, 7, 2, 4)  # C2=5 central cannon
    assert len(games[0].moves) == 6


def test_load_xiangqicore_ucci_pgn():
    games, stats = pgn_adapter.load_games_from_pgn(XIANGQICORE_PGN_UCCI, min_plies=4)
    assert stats.valid == 1
    assert games[0].moves[0] == (2, 7, 2, 4)  # h2e2 == C2.5
    assert len(games[0].moves) == 6
