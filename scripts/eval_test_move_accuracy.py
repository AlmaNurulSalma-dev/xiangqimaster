"""Evaluate move-prediction accuracy of a trained Policy-Value network on the
held-out TEST split, overall and broken down by game phase.

This is the quantitative test-set report analogous to the "board-move accuracy"
tables reported by recent Xiangqi papers. For each position it compares the
network's predicted move against the human expert's actual move and reports:

* ``top1_unmasked`` — argmax over all 2086 actions (the exact metric
  ``imitation.evaluate`` used during training, so it is directly comparable to
  the ~34% validation number),
* ``top1_masked`` / ``top3_masked`` — after masking illegal moves, i.e. how the
  agent actually selects (see ``NeuralAgent``),

each reported overall and per phase (opening / midgame / endgame, split by ply).

Usage::

    <venv-python> scripts/eval_test_move_accuracy.py \
        --splits data/splits --checkpoint results/checkpoints/il_agent2_phase1.pt \
        --limit-games 2000 --device cuda --out results/test_report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# Make ``src`` importable when run as a plain file from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.pipeline import load_split_jsonl
from src.environment import action_space, encoder
from src.environment.board import Board
from src.training.imitation import load_network
from src.utils.config import NN_CHANNEL_WIDTH, NN_NUM_RES_BLOCKS

# Game-phase bins by ply (Xiangqi games average ~82 plies).
PHASES = (("opening", 0, 20), ("midgame", 20, 60), ("endgame", 60, 10 ** 9))


def phase_of(ply: int) -> str:
    for name, lo, hi in PHASES:
        if lo <= ply < hi:
            return name
    return "endgame"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--channels", type=int, default=NN_CHANNEL_WIDTH)
    ap.add_argument("--num-blocks", type=int, default=NN_NUM_RES_BLOCKS)
    ap.add_argument("--limit-games", type=int, default=2000,
                    help="evaluate on the first N test games (0 = all)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/test_report")
    args = ap.parse_args()

    t0 = time.time()
    games = load_split_jsonl(os.path.join(args.splits, f"{args.split}.jsonl"))
    if args.limit_games:
        games = games[: args.limit_games]
    print(f"Loaded {len(games)} {args.split} games", flush=True)

    device = torch.device(args.device)
    net = load_network(args.checkpoint, channels=args.channels,
                       num_blocks=args.num_blocks, device=device)
    net.to(device).eval()

    keys = ["overall", "opening", "midgame", "endgame"]
    stats = {k: {"n": 0, "top1": 0, "top1m": 0, "top3m": 0} for k in keys}

    bt, ba, bm, bp = [], [], [], []  # tensors, actions, masks, phases

    def flush() -> None:
        if not bt:
            return
        x = torch.from_numpy(np.stack(bt)).to(device).float()
        with torch.no_grad():
            logits, _ = net(x)
        logits = logits.cpu().numpy()
        for i in range(len(bt)):
            a, mask, ph, lg = ba[i], bm[i], bp[i], logits[i]
            top1 = int(np.argmax(lg) == a)
            masked = np.where(mask, lg, -1e9)
            order = np.argpartition(masked, -3)[-3:]
            order = order[np.argsort(masked[order])[::-1]]
            top1m = int(order[0] == a)
            top3m = int(a in order[:3])
            for key in ("overall", ph):
                s = stats[key]
                s["n"] += 1
                s["top1"] += top1
                s["top1m"] += top1m
                s["top3m"] += top3m
        bt.clear(); ba.clear(); bm.clear(); bp.clear()

    for gi, game in enumerate(games):
        board = Board()
        for ply, move in enumerate(game.moves):
            bt.append(encoder.encode(board))
            ba.append(action_space.move_to_index(move))
            bm.append(action_space.legal_mask(board))
            bp.append(phase_of(ply))
            if len(bt) >= args.batch_size:
                flush()
            board.apply_move(*move)
        if (gi + 1) % 500 == 0:
            print(f"  {gi + 1}/{len(games)} games "
                  f"({stats['overall']['n']} positions, {time.time() - t0:.0f}s)",
                  flush=True)
    flush()

    rows = []
    for key in keys:
        s = stats[key]
        n = max(s["n"], 1)
        rows.append({
            "phase": key,
            "positions": s["n"],
            "top1_unmasked": round(s["top1"] / n, 4),
            "top1_masked": round(s["top1m"] / n, 4),
            "top3_masked": round(s["top3m"] / n, 4),
        })

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "test_move_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=" * 56)
    print(json.dumps(rows, indent=2))
    print(f"\nWrote {csv_path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
