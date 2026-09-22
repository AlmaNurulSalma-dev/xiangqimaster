"""Tests for the end-to-end data pipeline (docs/04-DATA-PIPELINE.md section 11)."""

from __future__ import annotations

import json

import pytest

from src.data.pipeline import (
    build_dataset,
    find_raw_files,
    load_raw_records,
    load_split_jsonl,
    records_from_file,
    save_split_jsonl,
)
from src.data.wxf_parser import Game

# A valid 6-ply opening in WXF and its exact ICCS equivalent.
WXF_GAME = "C2.5 C2.5 H2+3 H8+7 R1.2 R9.8"
ICCS_GAME = "h2e2 h7e7 h0g2 h9g7 i0h0 i9h9"


def _write_raw(tmp_path):
    (tmp_path / "a.txt").write_text(
        f"red {WXF_GAME}\n"
        f"draw {WXF_GAME}\n"
        "# comment\n"
        "black C2.5\n",  # too short
        encoding="utf-8",
    )
    (tmp_path / "b.pgn").write_text(
        f'[Result "1-0"]\n\n1. {ICCS_GAME} 1-0\n', encoding="utf-8"
    )
    return tmp_path


def test_find_raw_files_filters_and_sorts(tmp_path):
    (tmp_path / "b.pgn").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("x", encoding="utf-8")
    files = [f.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for f in find_raw_files(str(tmp_path))]
    assert files == ["a.txt", "b.pgn"]  # sorted, .md excluded


def test_find_raw_files_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        find_raw_files("does/not/exist")


def test_find_raw_files_descends_into_subfolders(tmp_path):
    # The scraper nests games in a per-player subfolder; the scan must recurse.
    sub = tmp_path / "XuYinchuan_game_records_2024"
    sub.mkdir()
    (sub / "g1.pgn").write_text("x", encoding="utf-8")
    (tmp_path / "top.txt").write_text("x", encoding="utf-8")
    names = sorted(f.replace("\\", "/").rsplit("/", 1)[-1] for f in find_raw_files(str(tmp_path)))
    assert names == ["g1.pgn", "top.txt"]


def test_records_from_file_dispatches_by_extension(tmp_path):
    (tmp_path / "g.txt").write_text(f"red {WXF_GAME}\n", encoding="utf-8")
    (tmp_path / "g.pgn").write_text(
        f'[Result "1-0"]\n\n1. {ICCS_GAME} 1-0\n', encoding="utf-8"
    )
    txt = records_from_file(str(tmp_path / "g.txt"))
    pgn = records_from_file(str(tmp_path / "g.pgn"))
    assert txt[0][0] == "red" and len(txt[0][1]) == 6
    assert pgn[0][0] == "1-0" and pgn[0][1][0] == "h2e2"


def test_load_raw_records_aggregates_all_files(tmp_path):
    _write_raw(tmp_path)
    records, files = load_raw_records(str(tmp_path))
    assert len(records) == 4  # 3 from a.txt + 1 from b.pgn
    assert len(files) == 2


def test_build_dataset_stats_and_filters(tmp_path):
    _write_raw(tmp_path)
    result = build_dataset(str(tmp_path), min_plies=4)
    s = result.stats
    assert s.total_games == 4
    assert s.too_short == 1        # the "black C2.5" one-ply game
    assert s.invalid == 0
    assert s.final_games == 3
    assert s.red_wins == 2 and s.draws == 1 and s.black_wins == 0
    assert s.avg_plies == 6.0
    # 3 games x 6 plies = 18 (position, move) examples, no mirror.
    assert s.total_examples == 18
    assert len(result.train) + len(result.val) + len(result.test) == 18


def test_build_dataset_max_plies_filter(tmp_path):
    (tmp_path / "a.txt").write_text(f"red {WXF_GAME}\n", encoding="utf-8")
    result = build_dataset(str(tmp_path), min_plies=4, max_plies=4)
    assert result.stats.too_long == 1
    assert result.stats.final_games == 0


def test_build_dataset_mirror_doubles_examples(tmp_path):
    (tmp_path / "a.txt").write_text(f"red {WXF_GAME}\n", encoding="utf-8")
    plain = build_dataset(str(tmp_path), min_plies=4)
    mirrored = build_dataset(str(tmp_path), min_plies=4, mirror=True)
    assert mirrored.stats.total_examples == 2 * plain.stats.total_examples


def test_build_dataset_writes_manifest(tmp_path):
    _write_raw(tmp_path)
    out = tmp_path / "splits"
    build_dataset(str(tmp_path), min_plies=4, splits_dir=str(out))
    manifest_path = out / "dataset_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["stats"]["final_games"] == 3
    assert manifest["split_sizes"]["train_games"] + \
        manifest["split_sizes"]["val_games"] + \
        manifest["split_sizes"]["test_games"] == 3


def test_build_dataset_split_is_reproducible(tmp_path):
    _write_raw(tmp_path)
    a = build_dataset(str(tmp_path), min_plies=4, seed=7)
    b = build_dataset(str(tmp_path), min_plies=4, seed=7)
    assert a.manifest["split_sizes"] == b.manifest["split_sizes"]


def test_build_dataset_empty_dir_is_graceful(tmp_path):
    result = build_dataset(str(tmp_path), min_plies=4)
    assert result.stats.final_games == 0
    assert len(result.train) == 0


def test_split_jsonl_round_trip(tmp_path):
    games = [
        Game(moves=((2, 7, 2, 4), (7, 7, 7, 4)), outcome=1),
        Game(moves=((0, 8, 1, 8),), outcome=0),
    ]
    path = tmp_path / "train.jsonl"
    save_split_jsonl(games, str(path))
    restored = load_split_jsonl(str(path))
    assert restored == games  # dataclass eq: moves (as tuples) + outcome match


def test_build_dataset_persists_split_jsonl(tmp_path):
    _write_raw(tmp_path)
    out = tmp_path / "splits"
    result = build_dataset(str(tmp_path), min_plies=4, splits_dir=str(out))
    # One JSONL per split, and they collectively hold every kept game.
    total = 0
    for name in ("train", "val", "test"):
        f = out / f"{name}.jsonl"
        assert f.exists()
        total += len(load_split_jsonl(str(f)))
    assert total == result.stats.final_games == 3


def test_build_dataset_no_materialize_skips_tensors(tmp_path):
    _write_raw(tmp_path)
    result = build_dataset(str(tmp_path), min_plies=4, materialize_tensors=False)
    assert result.train is None and result.val is None and result.test is None
    # Stats + example counts are still computed analytically.
    assert result.stats.final_games == 3
    assert result.stats.total_examples == 18


def test_example_counts_match_materialized_dataset(tmp_path):
    # The analytic example count must equal the real dataset length.
    _write_raw(tmp_path)
    result = build_dataset(str(tmp_path), min_plies=4, materialize_tensors=True)
    materialized = len(result.train) + len(result.val) + len(result.test)
    assert materialized == result.stats.total_examples == 18


def test_spot_check_reports_validity_rate(tmp_path):
    _write_raw(tmp_path)
    out = tmp_path / "splits"
    result = build_dataset(
        str(tmp_path), min_plies=4, validate=False, spot_check=10,
        materialize_tensors=False, splits_dir=str(out),
    )
    sc = result.manifest["spot_check"]
    # _write_raw yields 4 records (3 from a.txt + 1 from b.pgn); all are legal,
    # so none are rejected on replay regardless of the length filter.
    assert sc["sampled"] == 4
    assert sc["invalid"] == 0
    assert sc["legal_rate"] == 1.0
    assert result.manifest["validated"] is False
