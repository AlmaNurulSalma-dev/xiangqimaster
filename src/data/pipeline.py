"""End-to-end data pipeline: raw game files → validated, split IL datasets.

This is the orchestrator for docs/04-DATA-PIPELINE.md section 11. The individual
pieces already exist (PGN adapter, WXF/ICCS parser + replay validation, the IL
dataset, the game-level split); this module scans a raw-data directory, runs
every game through validate → filter, records the statistics the thesis needs
(section 12), splits by game, builds a :class:`XiangqiILDataset` per split, and
writes a reproducible split manifest.

Usage::

    python -m src.data.pipeline --raw data/raw --out data/splits

Supported raw files (place them in ``data/raw/``, e.g. downloaded from dpxq.com
or the WXF Federation):

* ``*.pgn`` — standard Xiangqi PGN (WXF or ICCS movetext), via the PGN adapter.
* ``*.txt`` — one game per line, ``<result> tok1 tok2 ...`` (the simple format
  ``game_loader.load_games_from_file`` reads).

The move notation is auto-detected per token by default (``C2.5`` → WXF,
``h2e2`` → ICCS); pass ``notation="wxf"``/``"iccs"`` to force one.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

from src.data.data_loader import split_games
from src.data.dataset import XiangqiILDataset
from src.data.game_loader import LoadStats, parse_game_records
from src.data.pgn_adapter import pgn_to_records
from src.data.wxf_parser import DRAW, Game
from src.utils.config import BLACK, DEFAULT_SEED, MAX_PLIES, RED

PGN_EXTENSIONS = (".pgn",)
TEXT_EXTENSIONS = (".txt",)
RAW_EXTENSIONS = PGN_EXTENSIONS + TEXT_EXTENSIONS


@dataclass
class DatasetStats:
    """Dataset statistics for the thesis methodology chapter (docs/04 §12)."""

    files: list[str] = field(default_factory=list)
    total_games: int = 0        # records seen across all files
    invalid: int = 0            # failed to parse / illegal move on replay
    too_short: int = 0          # dropped by the minimum-length filter
    too_long: int = 0           # dropped by the maximum-length filter
    final_games: int = 0        # games kept after every filter
    total_examples: int = 0     # (position, move) training pairs (post-mirror)
    red_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    avg_plies: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _records_from_text(text: str) -> list[tuple[str, list[str]]]:
    """Parse the simple one-game-per-line ``<result> tok...`` text format."""
    records: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        records.append((parts[0], parts[1:]))
    return records


def find_raw_files(raw_dir: str) -> list[str]:
    """Return supported raw game files under ``raw_dir``, recursively.

    The scan descends into subdirectories so the per-player subfolders the dpxq
    scraper creates (``<name>_game_records_<date>/*.pgn``) are picked up. Paths
    are sorted for a deterministic (reproducible) downstream split.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw data directory not found: {raw_dir}")
    found: list[str] = []
    for root, _dirs, names in os.walk(raw_dir):
        for name in names:
            if name.lower().endswith(RAW_EXTENSIONS):
                found.append(os.path.join(root, name))
    return sorted(found)


def records_from_file(path: str) -> list[tuple[str, list[str]]]:
    """Read one raw file into ``(result, tokens)`` records by extension."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if path.lower().endswith(PGN_EXTENSIONS):
        return pgn_to_records(text)
    return _records_from_text(text)


def load_raw_records(raw_dir: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Collect ``(result, tokens)`` records from every raw file, in sorted order.

    Returns the records and the list of source files they came from. Iteration is
    deterministic (files sorted by name, records in file order) so the downstream
    seeded split is reproducible.
    """
    files = find_raw_files(raw_dir)
    records: list[tuple[str, list[str]]] = []
    for path in files:
        records.extend(records_from_file(path))
    return records, files


@dataclass
class BuildResult:
    """Outputs of :func:`build_dataset`."""

    train: XiangqiILDataset
    val: XiangqiILDataset
    test: XiangqiILDataset
    stats: DatasetStats
    manifest: dict


def _compute_stats(
    files: list[str],
    load_stats: LoadStats,
    kept: list[Game],
    too_long: int,
    total_examples: int,
) -> DatasetStats:
    red = sum(1 for g in kept if g.outcome == RED)
    black = sum(1 for g in kept if g.outcome == BLACK)
    draws = sum(1 for g in kept if g.outcome == DRAW)
    total_plies = sum(len(g.moves) for g in kept)
    avg = total_plies / len(kept) if kept else 0.0
    return DatasetStats(
        files=[os.path.basename(f) for f in files],
        total_games=load_stats.total,
        invalid=load_stats.invalid,
        too_short=load_stats.too_short,
        too_long=too_long,
        final_games=len(kept),
        total_examples=total_examples,
        red_wins=red,
        black_wins=black,
        draws=draws,
        avg_plies=round(avg, 2),
    )


def build_dataset(
    raw_dir: str,
    *,
    min_plies: int = 10,
    max_plies: int = MAX_PLIES,
    notation: str = "auto",
    validate: bool = True,
    mirror: bool = False,
    draw_value: float = 0.0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = DEFAULT_SEED,
    splits_dir: str | None = None,
) -> BuildResult:
    """Run the full pipeline: scan ``raw_dir`` → validate/filter → split → datasets.

    If ``splits_dir`` is given, a JSON manifest (stats + reproducibility metadata)
    is written to ``<splits_dir>/dataset_manifest.json``.
    """
    records, files = load_raw_records(raw_dir)

    games, load_stats = parse_game_records(
        records, min_plies=min_plies, validate=validate, notation=notation
    )

    # Maximum-length filter (parse_game_records already handles the minimum).
    kept = [g for g in games if len(g.moves) <= max_plies]
    too_long = len(games) - len(kept)

    train_games, val_games, test_games = split_games(
        kept, fractions, seed=seed
    )
    train = XiangqiILDataset(train_games, draw_value=draw_value, mirror=mirror)
    val = XiangqiILDataset(val_games, draw_value=draw_value, mirror=mirror)
    test = XiangqiILDataset(test_games, draw_value=draw_value, mirror=mirror)

    total_examples = len(train) + len(val) + len(test)
    stats = _compute_stats(files, load_stats, kept, too_long, total_examples)

    manifest = {
        "seed": seed,
        "fractions": list(fractions),
        "min_plies": min_plies,
        "max_plies": max_plies,
        "notation": notation,
        "mirror": mirror,
        "draw_value": draw_value,
        "split_sizes": {
            "train_games": len(train_games),
            "val_games": len(val_games),
            "test_games": len(test_games),
            "train_examples": len(train),
            "val_examples": len(val),
            "test_examples": len(test),
        },
        "stats": stats.as_dict(),
    }

    if splits_dir is not None:
        os.makedirs(splits_dir, exist_ok=True)
        manifest_path = os.path.join(splits_dir, "dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return BuildResult(train=train, val=val, test=test, stats=stats, manifest=manifest)


def _format_stats(stats: DatasetStats) -> str:
    lines = [
        "Data pipeline summary",
        "---------------------",
        f"  source files      : {len(stats.files)} ({', '.join(stats.files) or '-'})",
        f"  games seen        : {stats.total_games}",
        f"  invalid (dropped) : {stats.invalid}",
        f"  too short         : {stats.too_short}",
        f"  too long          : {stats.too_long}",
        f"  final games       : {stats.final_games}",
        f"  training examples : {stats.total_examples}",
        f"  outcome R/B/D     : {stats.red_wins}/{stats.black_wins}/{stats.draws}",
        f"  avg game length   : {stats.avg_plies} plies",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build IL datasets from raw games.")
    parser.add_argument("--raw", default="data/raw", help="raw game-file directory")
    parser.add_argument("--out", default="data/splits", help="split manifest output dir")
    parser.add_argument("--min-plies", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=MAX_PLIES)
    parser.add_argument(
        "--notation", choices=["auto", "wxf", "iccs"], default="auto"
    )
    parser.add_argument("--mirror", action="store_true", help="add mirror augmentation")
    parser.add_argument("--no-validate", action="store_true", help="skip replay validation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    result = build_dataset(
        args.raw,
        min_plies=args.min_plies,
        max_plies=args.max_plies,
        notation=args.notation,
        validate=not args.no_validate,
        mirror=args.mirror,
        seed=args.seed,
        splits_dir=args.out,
    )
    print(_format_stats(result.stats))
    print(f"\nManifest written to {os.path.join(args.out, 'dataset_manifest.json')}")


__all__ = [
    "DatasetStats",
    "BuildResult",
    "find_raw_files",
    "records_from_file",
    "load_raw_records",
    "build_dataset",
    "main",
]


if __name__ == "__main__":
    main()
