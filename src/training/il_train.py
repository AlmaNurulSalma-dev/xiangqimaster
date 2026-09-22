"""Run imitation learning on the persisted data splits — Agent 2, Phase 1.

Ties the data pipeline to the trainer: load the game-level splits written by
``src.data.pipeline`` (JSONL), wrap them in a memory-light
:class:`LazyXiangqiILDataset`, and train the Policy-Value network to imitate
human moves (``train_imitation``). The resulting checkpoint initialises Agent 2
Phase 2 (PPO fine-tuning).

Usage::

    # full corpus (a GPU / Colab job)
    python -m src.training.il_train --splits data/splits --epochs 30 --device cuda

    # quick local smoke test on a subset
    python -m src.training.il_train --limit-train 2000 --limit-val 500 --epochs 1

Training over the full ~9.3M-position train split is a GPU job; ``--limit-train``
caps the number of GAMES loaded so the flow can be exercised locally.
"""

from __future__ import annotations

import argparse
import os

from src.data.dataset import LazyXiangqiILDataset
from src.data.data_loader import make_dataloader
from src.data.pipeline import load_split_jsonl
from src.data.wxf_parser import Game
from src.training.imitation import save_checkpoint, train_imitation
from src.utils.config import IL_BATCH_SIZE, IL_EPOCHS, IL_LEARNING_RATE


def _load_split(splits_dir: str, name: str, limit: int | None) -> list[Game]:
    path = os.path.join(splits_dir, f"{name}.jsonl")
    games = load_split_jsonl(path)
    if limit is not None:
        games = games[:limit]
    return games


def train_from_splits(
    splits_dir: str = "data/splits",
    *,
    epochs: int = IL_EPOCHS,
    batch_size: int = IL_BATCH_SIZE,
    learning_rate: float = IL_LEARNING_RATE,
    mirror: bool = False,
    draw_value: float = 0.0,
    limit_train: int | None = None,
    limit_val: int | None = None,
    num_workers: int = 0,
    device: str = "cpu",
    seed: int | None = None,
    save_path: str | None = None,
):
    """Load splits, build lazy datasets + loaders, and run imitation training."""
    train_games = _load_split(splits_dir, "train", limit_train)
    val_games = _load_split(splits_dir, "val", limit_val)

    train_ds = LazyXiangqiILDataset(
        train_games, draw_value=draw_value, mirror=mirror
    )
    val_ds = LazyXiangqiILDataset(val_games, draw_value=draw_value)

    print(
        f"train: {len(train_games)} games / {len(train_ds)} examples | "
        f"val: {len(val_games)} games / {len(val_ds)} examples"
    )

    train_loader = make_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = make_dataloader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    network, history = train_imitation(
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )

    for stats in history:
        acc = "n/a" if stats.val_accuracy is None else f"{stats.val_accuracy:.3f}"
        print(
            f"epoch {stats.epoch:>3} | loss {stats.train_loss:.4f} "
            f"(p {stats.train_policy_loss:.4f} / v {stats.train_value_loss:.4f}) "
            f"| val top-1 {acc}"
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_checkpoint(network, save_path)
        print(f"checkpoint saved to {save_path}")

    return network, history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Agent 2 Phase 1 (IL).")
    parser.add_argument("--splits", default="data/splits", help="split dir (JSONL)")
    parser.add_argument("--epochs", type=int, default=IL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=IL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=IL_LEARNING_RATE)
    parser.add_argument("--mirror", action="store_true", help="mirror augmentation")
    parser.add_argument("--limit-train", type=int, default=None, help="cap train games")
    parser.add_argument("--limit-val", type=int, default=None, help="cap val games")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save", default="results/checkpoints/il_agent2_phase1.pt",
        help="checkpoint output path (empty to skip saving)",
    )
    args = parser.parse_args(argv)

    train_from_splits(
        args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mirror=args.mirror,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        save_path=args.save or None,
    )


__all__ = ["train_from_splits", "main"]


if __name__ == "__main__":
    main()
