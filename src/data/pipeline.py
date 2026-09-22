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

PGN_EXTENSIONS = (".pgn", ".pgns")  # some dumps (CGLemon dpxq/WXF) use .pgns
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
    """Outputs of :func:`build_dataset`.

    ``train``/``val``/``test`` are materialised :class:`XiangqiILDataset` objects
    only when ``materialize_tensors`` is set; otherwise they are ``None`` and the
    split is available on disk as JSONL (see :func:`save_split_jsonl`).
    """

    train: XiangqiILDataset | None
    val: XiangqiILDataset | None
    test: XiangqiILDataset | None
    stats: DatasetStats
    manifest: dict


SPLIT_NAMES = ("train", "val", "test")


def save_split_jsonl(games: list[Game], path: str) -> None:
    """Persist a split as JSON Lines: one ``{"o": outcome, "m": [[fr,fc,tr,tc]…]}``
    record per game. This stores the parsed moves so training can rebuild a
    dataset without re-reading and re-parsing the raw files."""
    with open(path, "w", encoding="utf-8") as f:
        for game in games:
            record = {"o": game.outcome, "m": [list(m) for m in game.moves]}
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_split_jsonl(path: str) -> list[Game]:
    """Reconstruct the games persisted by :func:`save_split_jsonl`."""
    games: list[Game] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            moves = tuple(tuple(m) for m in record["m"])
            games.append(Game(moves=moves, outcome=record["o"]))
    return games


def _example_count(games: list[Game], mirror: bool) -> int:
    """Number of (position, move) training pairs a split yields, computed
    analytically (one per ply, doubled under mirror) so it needs no tensors."""
    plies = sum(len(g.moves) for g in games)
    return plies * (2 if mirror else 1)


def spot_check_validity(
    records: list[tuple[str, list[str]]],
    *,
    sample: int,
    notation: str,
    min_plies: int,
    seed: int,
) -> dict:
    """Replay-validate a random sample of records and report the validity rate.

    Full validation of a large pre-cleaned corpus is impractical (~0.5 s/game),
    so we sample instead to get a quality figure for the thesis without the wait.
    """
    import random

    n = min(sample, len(records))
    if n == 0:
        return {"sampled": 0, "invalid": 0, "legal_rate": 1.0}
    chosen = random.Random(seed).sample(records, n)
    # min_plies=0 so the length filter doesn't count against the integrity rate;
    # ``invalid`` then means only "rejected as illegal on replay".
    _games, stats = parse_game_records(
        chosen, min_plies=0, validate=True, notation=notation
    )
    return {
        "sampled": n,
        "invalid": stats.invalid,
        "legal_rate": round((n - stats.invalid) / n, 4),
    }


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
    materialize_tensors: bool = True,
    spot_check: int = 0,
) -> BuildResult:
    """Run the pipeline: scan ``raw_dir`` → (validate)/filter → split → persist.

    ``validate`` replay-checks every move's legality; it is ~0.5 s/game, so for a
    large pre-cleaned corpus prefer ``validate=False`` with ``spot_check=N`` to
    sample a validity rate instead of paying the full cost.

    ``materialize_tensors`` builds an in-memory :class:`XiangqiILDataset` per
    split. That holds every encoded position in RAM (~5 KB each), which does not
    scale to 100k+ games — leave it ``False`` for large corpora and let training
    build a dataset lazily from the persisted split.

    When ``splits_dir`` is given, writes ``train.jsonl``/``val.jsonl``/
    ``test.jsonl`` (the split games) plus ``dataset_manifest.json`` (stats +
    reproducibility metadata).
    """
    records, files = load_raw_records(raw_dir)

    games, load_stats = parse_game_records(
        records, min_plies=min_plies, validate=validate, notation=notation
    )

    # Maximum-length filter (parse_game_records already handles the minimum).
    kept = [g for g in games if len(g.moves) <= max_plies]
    too_long = len(games) - len(kept)

    train_games, val_games, test_games = split_games(kept, fractions, seed=seed)
    splits = dict(zip(SPLIT_NAMES, (train_games, val_games, test_games)))

    # Example counts are analytic (one per ply, doubled if mirrored) so we never
    # need to build tensors just to report or split.
    example_counts = {
        name: _example_count(g, mirror) for name, g in splits.items()
    }
    total_examples = sum(example_counts.values())
    stats = _compute_stats(files, load_stats, kept, too_long, total_examples)

    manifest = {
        "seed": seed,
        "fractions": list(fractions),
        "min_plies": min_plies,
        "max_plies": max_plies,
        "notation": notation,
        "mirror": mirror,
        "draw_value": draw_value,
        "validated": validate,
        "split_sizes": {
            "train_games": len(train_games),
            "val_games": len(val_games),
            "test_games": len(test_games),
            "train_examples": example_counts["train"],
            "val_examples": example_counts["val"],
            "test_examples": example_counts["test"],
        },
        "stats": stats.as_dict(),
    }
    if spot_check and not validate:
        manifest["spot_check"] = spot_check_validity(
            records, sample=spot_check, notation=notation,
            min_plies=min_plies, seed=seed,
        )

    if splits_dir is not None:
        os.makedirs(splits_dir, exist_ok=True)
        for name, split_group in splits.items():
            save_split_jsonl(split_group, os.path.join(splits_dir, f"{name}.jsonl"))
        manifest_path = os.path.join(splits_dir, "dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    if materialize_tensors:
        datasets = {
            name: XiangqiILDataset(g, draw_value=draw_value, mirror=mirror)
            for name, g in splits.items()
        }
    else:
        datasets = {name: None for name in SPLIT_NAMES}

    return BuildResult(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        stats=stats,
        manifest=manifest,
    )


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
    parser.add_argument("--out", default="data/splits", help="split output dir")
    parser.add_argument("--min-plies", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=MAX_PLIES)
    parser.add_argument(
        "--notation", choices=["auto", "wxf", "iccs"], default="auto"
    )
    parser.add_argument("--mirror", action="store_true", help="add mirror augmentation")
    parser.add_argument(
        "--validate", action="store_true",
        help="replay-validate EVERY game (~0.5 s/game; slow on large corpora)",
    )
    parser.add_argument(
        "--spot-check", type=int, default=500,
        help="when not validating, replay-validate this many random games for a "
             "quality rate (0 to disable)",
    )
    parser.add_argument(
        "--materialize", action="store_true",
        help="also build in-memory tensor datasets (only for small corpora)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    result = build_dataset(
        args.raw,
        min_plies=args.min_plies,
        max_plies=args.max_plies,
        notation=args.notation,
        validate=args.validate,
        spot_check=args.spot_check,
        mirror=args.mirror,
        seed=args.seed,
        splits_dir=args.out,
        materialize_tensors=args.materialize,
    )
    print(_format_stats(result.stats))
    sc = result.manifest.get("spot_check")
    if sc:
        print(f"  spot-check        : {sc['sampled']} sampled, {sc['invalid']} "
              f"illegal ({sc['legal_rate'] * 100:.1f}% legal)")
    sizes = result.manifest["split_sizes"]
    print(f"  split (games)     : train {sizes['train_games']} / "
          f"val {sizes['val_games']} / test {sizes['test_games']}")
    print(f"\nSplits + manifest written to {args.out}/")


__all__ = [
    "DatasetStats",
    "BuildResult",
    "SPLIT_NAMES",
    "find_raw_files",
    "records_from_file",
    "load_raw_records",
    "save_split_jsonl",
    "load_split_jsonl",
    "spot_check_validity",
    "build_dataset",
    "main",
]


if __name__ == "__main__":
    main()
