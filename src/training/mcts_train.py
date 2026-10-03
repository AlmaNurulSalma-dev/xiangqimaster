"""Agent 3 — AlphaZero-style MCTS self-play training (docs/07-TRAINING.md §4).

The loop alternates two phases, for a number of iterations:

1. **Self-play** — play ``games_per_iter`` games where *both* sides move by
   running MCTS (``src.agents.mcts``) guided by the current network, with
   Dirichlet root noise for exploration. For every position we record
   ``(board_tensor, visit-count policy, side to move)``. When the game ends, each
   recorded position gets a value target ``z`` = the game result from that side's
   perspective (+1 win / -1 loss / ``draw_value`` draw).
2. **Train** — fit the network on the collected samples: a soft cross-entropy
   between the policy head and the MCTS visit distribution, plus MSE between the
   value head and ``z``. A sliding replay ``window`` keeps recent games.

The checkpoint is a plain ``state_dict`` (same format as the IL checkpoint), so
``MCTSAgent(load_network(path))`` plays it.

This is the most compute-heavy agent — each move costs ``n_simulations`` network
evaluations. Keep ``--simulations`` modest (e.g. 100) on a single GPU; batched
leaf evaluation (docs/07 §4.4) is a future optimization. Warm-starting from the
IL checkpoint (``--il-checkpoint``) greatly speeds convergence.

Usage::

    python -m src.training.mcts_train --iterations 20 --games-per-iter 25 \
        --simulations 100 --device cuda \
        --il-checkpoint results/checkpoints/il_agent2_phase1.pt \
        --save results/checkpoints/mcts_agent3.pt
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from torch import nn

from src.agents.mcts import run_mcts, visit_count_policy
from src.environment import action_space, encoder
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.models.network import PolicyValueNetwork
from src.training.imitation import load_network, save_checkpoint
from src.utils.config import (
    ACTION_SPACE_SIZE,
    MAX_PLIES,
    MCTS_C_PUCT,
    NN_CHANNEL_WIDTH,
    NN_L2_WEIGHT_DECAY,
    NN_NUM_RES_BLOCKS,
    REPETITION_LIMIT,
    opponent,
)

Sample = tuple[np.ndarray, np.ndarray, np.float32]  # (state, policy_target, z)


@torch.no_grad()
def play_selfplay_game(
    network: PolicyValueNetwork,
    *,
    n_simulations: int,
    c_puct: float,
    rng: np.random.Generator,
    draw_value: float = 0.0,
    temperature_moves: int = 30,
    max_plies: int = MAX_PLIES,
) -> tuple[list[Sample], int | None, int]:
    """Play one MCTS self-play game; return (samples, winner, plies).

    ``winner`` is ``RED``/``BLACK`` or ``None`` for a draw. Early moves
    (``< temperature_moves``) are sampled in proportion to visit counts for
    exploration; later moves pick the most-visited action.
    """
    board = Board()
    records: list[tuple[np.ndarray, np.ndarray, int]] = []
    position_counts: dict = {}
    ply = 0
    winner: int | None = None

    while True:
        if not generate_legal_moves(board):
            winner = opponent(board.to_move)  # side to move cannot reply → loses
            break

        root = run_mcts(
            board, network, n_simulations=n_simulations, c_puct=c_puct,
            add_noise=True, rng=rng,
        )
        counts = visit_count_policy(root)
        total = sum(counts.values())
        pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for action, visits in counts.items():
            pi[action] = visits / total if total else 0.0
        records.append((encoder.encode(board), pi, board.to_move))

        actions = list(counts)
        visit_arr = np.array([counts[a] for a in actions], dtype=np.float64)
        if ply < temperature_moves and visit_arr.sum() > 0:
            action = int(rng.choice(actions, p=visit_arr / visit_arr.sum()))
        else:
            action = int(actions[int(np.argmax(visit_arr))])

        board.apply_move(*action_space.index_to_move(action))
        ply += 1

        key = board.position_key()
        position_counts[key] = position_counts.get(key, 0) + 1
        if position_counts[key] >= REPETITION_LIMIT or ply >= max_plies:
            winner = None  # draw
            break

    samples: list[Sample] = []
    for state, pi, player in records:
        if winner is None:
            z = draw_value
        else:
            z = 1.0 if winner == player else -1.0
        samples.append((state, pi, np.float32(z)))
    return samples, winner, ply


def _train_on_buffer(
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: list[Sample],
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Fit the network on the replay buffer; return (avg policy, value loss).

    Policy loss is a soft cross-entropy against the MCTS visit distribution
    (the target is a full distribution, not a single label).
    """
    states = torch.from_numpy(np.stack([b[0] for b in buffer])).to(device)
    pis = torch.from_numpy(np.stack([b[1] for b in buffer])).to(device)
    zs = torch.tensor([b[2] for b in buffer], dtype=torch.float32,
                      device=device).unsqueeze(1)

    network.train()
    log_softmax = nn.LogSoftmax(dim=1)
    mse = nn.MSELoss()
    n = len(buffer)
    tot_p = tot_v = 0.0
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, value = network(states[idx])
            policy_loss = -(pis[idx] * log_softmax(logits)).sum(dim=1).mean()
            value_loss = mse(value, zs[idx])
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_p += policy_loss.item()
            tot_v += value_loss.item()
            steps += 1
    return tot_p / max(steps, 1), tot_v / max(steps, 1)


def train_mcts(
    *,
    iterations: int,
    games_per_iter: int,
    n_simulations: int = 100,
    c_puct: float = MCTS_C_PUCT,
    epochs_per_iter: int = 2,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = NN_L2_WEIGHT_DECAY,
    window: int = 20000,
    draw_value: float = 0.0,
    temperature_moves: int = 30,
    max_plies: int = MAX_PLIES,
    channels: int = NN_CHANNEL_WIDTH,
    num_blocks: int = NN_NUM_RES_BLOCKS,
    il_checkpoint: str | None = None,
    save_path: str | None = None,
    checkpoint_every: int = 1,
    resume: bool = False,
    device: str = "cuda",
    seed: int | None = None,
) -> PolicyValueNetwork:
    """Run the AlphaZero-style self-play training loop and return the network.

    Starting point: ``--resume`` (continue from ``save_path``) > ``il_checkpoint``
    (warm start) > random init.
    """
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)
    dev = torch.device(device)

    if resume and save_path and os.path.exists(save_path):
        network = load_network(save_path, channels=channels, num_blocks=num_blocks,
                               device=str(dev))
        print(f"resumed from {save_path}", flush=True)
    elif il_checkpoint:
        network = load_network(il_checkpoint, channels=channels,
                               num_blocks=num_blocks, device=str(dev))
        print(f"warm-start from IL checkpoint {il_checkpoint}", flush=True)
    else:
        network = PolicyValueNetwork(channels=channels, num_blocks=num_blocks)
        print("starting from random init", flush=True)
    network.to(dev)

    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    buffer: deque[Sample] = deque(maxlen=window)

    for it in range(iterations):
        t0 = time.time()
        network.eval()
        n_new = 0
        lengths: list[int] = []
        wins = {1: 0, -1: 0, None: 0}
        for _ in range(games_per_iter):
            samples, winner, plies = play_selfplay_game(
                network, n_simulations=n_simulations, c_puct=c_puct, rng=rng,
                draw_value=draw_value, temperature_moves=temperature_moves,
                max_plies=max_plies,
            )
            buffer.extend(samples)
            n_new += len(samples)
            lengths.append(plies)
            wins[winner] = wins.get(winner, 0) + 1

        p_loss, v_loss = _train_on_buffer(
            network, optimizer, list(buffer),
            epochs=epochs_per_iter, batch_size=batch_size, device=dev,
        )
        if save_path and (it + 1) % checkpoint_every == 0:
            save_checkpoint(network, save_path)

        print(
            f"iter {it:>3}/{iterations} | games {games_per_iter} "
            f"(R/B/D {wins.get(1,0)}/{wins.get(-1,0)}/{wins.get(None,0)}) | "
            f"new {n_new} | buffer {len(buffer)} | avg_len {np.mean(lengths):.0f} | "
            f"p_loss {p_loss:.4f} | v_loss {v_loss:.4f} | {(time.time()-t0)/60:.1f} min",
            flush=True,
        )

    if save_path:
        save_checkpoint(network, save_path)
        print(f"final checkpoint saved to {save_path}", flush=True)
    return network


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Agent 3 (MCTS + NN self-play).")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--games-per-iter", type=int, default=25)
    p.add_argument("--simulations", type=int, default=100,
                   help="MCTS simulations per move (strength vs. speed)")
    p.add_argument("--epochs-per-iter", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=20000, help="replay buffer size")
    p.add_argument("--temperature-moves", type=int, default=30)
    p.add_argument("--max-plies", type=int, default=MAX_PLIES)
    p.add_argument("--il-checkpoint", default=None, help="warm-start IL .pt")
    p.add_argument("--save", default="results/checkpoints/mcts_agent3.pt")
    p.add_argument("--checkpoint-every", type=int, default=1, help="iterations")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    train_mcts(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        n_simulations=args.simulations,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window=args.window,
        temperature_moves=args.temperature_moves,
        max_plies=args.max_plies,
        il_checkpoint=args.il_checkpoint,
        save_path=args.save or None,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        device=args.device,
        seed=args.seed,
    )


__all__ = ["play_selfplay_game", "train_mcts", "main"]


if __name__ == "__main__":
    main()
