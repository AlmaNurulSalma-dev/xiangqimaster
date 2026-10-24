"""Generate a distillation dataset by Pikafish self-play (Option B).

Each game is Pikafish vs Pikafish, but the first few plies are random legal
moves so the dataset covers diverse openings (deterministic engine play would
otherwise produce the same game every time). Games are written in the SAME
JSONL format as the human corpus (``{"o": outcome, "m": [[fr,fc,tr,tc]...]}``),
so the existing IL pipeline can train on them unchanged — the only difference
is the *teacher*: a superhuman engine instead of human players.

Writes incrementally (one line per finished game) so a long run is crash-safe
and resumable: re-running with the same ``--out`` continues until ``--games``
total lines exist.

CLI::

    python -m src.data.pikafish_selfplay --games 1000 --depth 8 \
        --opening-random 6 --out data/pikafish/selfplay.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.agents.engine_agent import EngineAgent  # noqa: F401  (kept for parity/use)
from src.environment.board import Board
from src.environment.elephanteye import ElephantEyeEngine
from src.environment.move_generator import generate_legal_moves
from src.data.wxf_parser import DRAW
from src.utils.config import MAX_PLIES, REPETITION_LIMIT, opponent

DEFAULT_EXE = os.path.join(
    "engines", "pikafish", "extracted", "Pikafish-Windows-x86-64-universal.exe"
)


def play_pikafish_game(
    engine: ElephantEyeEngine,
    rng: np.random.Generator,
    *,
    depth: int,
    opening_random_plies: int,
    max_plies: int = MAX_PLIES,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Play one game; return (moves, outcome) with outcome RED/BLACK/DRAW."""
    board = Board()
    moves: list[tuple[int, int, int, int]] = []
    position_counts: dict = {}
    ply = 0
    winner: int | None = None

    while True:
        legal = generate_legal_moves(board)
        if not legal:
            winner = opponent(board.to_move)  # side to move cannot reply → loses
            break

        if ply < opening_random_plies:
            move = tuple(legal[int(rng.integers(len(legal)))])
        else:
            move = engine.bestmove(board, depth=depth)
            if move not in {tuple(m) for m in legal}:  # safety: FEN/engine mismatch
                winner = None  # end as draw; should essentially never happen
                break

        board.apply_move(*move)
        moves.append(tuple(move))
        ply += 1

        key = board.position_key()
        position_counts[key] = position_counts.get(key, 0) + 1
        if position_counts[key] >= REPETITION_LIMIT or ply >= max_plies:
            winner = None  # draw
            break

    outcome = DRAW if winner is None else winner
    return moves, outcome


def generate(
    *,
    n_games: int,
    out_path: str,
    depth: int = 8,
    opening_random_plies: int = 6,
    seed: int = 0,
    exe: str = DEFAULT_EXE,
) -> None:
    """Generate ``n_games`` self-play games, appending each to ``out_path``."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            done = sum(1 for _ in fh)
        print(f"resuming: {done} games already in {out_path}", flush=True)

    rng = np.random.default_rng(seed + done)  # advance RNG past finished games
    exe_abs = os.path.abspath(exe)
    cwd = os.path.dirname(exe_abs)  # so Pikafish finds pikafish.nnue
    engine = ElephantEyeEngine([exe_abs], depth=depth, protocol="uci", cwd=cwd).start()
    print(f"Pikafish started (depth {depth}); target {n_games} games", flush=True)

    wins = {1: 0, -1: 0, 0: 0}
    t_all = time.time()
    try:
        with open(out_path, "a", encoding="utf-8") as fh:
            for g in range(done, n_games):
                t0 = time.time()
                moves, outcome = play_pikafish_game(
                    engine, rng, depth=depth, opening_random_plies=opening_random_plies
                )
                fh.write(json.dumps({"o": outcome, "m": [list(m) for m in moves]}) + "\n")
                fh.flush()
                wins[outcome] = wins.get(outcome, 0) + 1
                if (g + 1) % 5 == 0 or g == done:
                    print(
                        f"game {g+1}/{n_games} | {len(moves)} plies | outcome {outcome} "
                        f"| R/B/D {wins[1]}/{wins[-1]}/{wins[0]} "
                        f"| {(time.time()-t0):.1f}s | total {(time.time()-t_all)/60:.1f} min",
                        flush=True,
                    )
    finally:
        engine.close()
    print(f"done: {n_games} games -> {out_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pikafish self-play dataset generator.")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--opening-random", type=int, default=6,
                   help="random legal plies at the start (diversity)")
    p.add_argument("--out", default="data/pikafish/selfplay.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exe", default=DEFAULT_EXE)
    args = p.parse_args(argv)
    generate(
        n_games=args.games, out_path=args.out, depth=args.depth,
        opening_random_plies=args.opening_random, seed=args.seed, exe=args.exe,
    )


if __name__ == "__main__":
    main()
