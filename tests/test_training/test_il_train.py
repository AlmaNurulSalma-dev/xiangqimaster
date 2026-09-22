"""Test the imitation-learning entrypoint that trains from persisted splits."""

from __future__ import annotations

from src.data.pipeline import save_split_jsonl
from src.data.wxf_parser import Game
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.training.il_train import train_from_splits
from src.utils.config import RED


def _game(n_plies: int = 6, outcome: int = RED) -> Game:
    board = Board()
    moves = []
    for _ in range(n_plies):
        move = generate_legal_moves(board)[0]
        moves.append(move)
        board.apply_move(*move)
    return Game(moves=tuple(moves), outcome=outcome)


def _write_splits(tmp_path):
    games = [_game() for _ in range(4)]
    save_split_jsonl(games, str(tmp_path / "train.jsonl"))
    save_split_jsonl(games[:2], str(tmp_path / "val.jsonl"))


def test_train_from_splits_runs_and_returns_history(tmp_path):
    _write_splits(tmp_path)
    network, history = train_from_splits(
        str(tmp_path), epochs=2, batch_size=8, seed=0, save_path=None
    )
    assert len(history) == 2
    assert history[0].val_accuracy is not None  # val loader was built
    assert network is not None


def test_train_from_splits_respects_game_limit(tmp_path):
    _write_splits(tmp_path)
    # limit_train=1 → only the first game's plies become training examples.
    network, history = train_from_splits(
        str(tmp_path), epochs=1, batch_size=4, limit_train=1, limit_val=1,
        seed=0, save_path=None,
    )
    assert len(history) == 1


def test_train_from_splits_saves_checkpoint(tmp_path):
    _write_splits(tmp_path)
    save_path = tmp_path / "ckpt" / "il.pt"
    train_from_splits(
        str(tmp_path), epochs=1, batch_size=8, seed=0, save_path=str(save_path)
    )
    assert save_path.exists()


def test_load_network_round_trips_weights(tmp_path):
    import torch

    from src.models.network import PolicyValueNetwork
    from src.training.imitation import load_network, save_checkpoint

    original = PolicyValueNetwork(channels=8, num_blocks=1)
    path = tmp_path / "il.pt"
    save_checkpoint(original, str(path))
    loaded = load_network(str(path), channels=8, num_blocks=1)
    assert torch.allclose(
        loaded.input_conv.conv.weight, original.input_conv.conv.weight
    )


def test_il_checkpoint_on_disk_transfers_into_ppo(tmp_path):
    # Full Agent 2 bridge: IL checkpoint on disk → PPO features extractor body.
    import torch

    from src.models.network import PolicyValueNetwork
    from src.training.imitation import load_network, save_checkpoint
    from src.training.ppo_train import build_model

    il = PolicyValueNetwork(channels=8, num_blocks=1)
    ckpt = tmp_path / "il.pt"
    save_checkpoint(il, str(ckpt))

    loaded = load_network(str(ckpt), channels=8, num_blocks=1)
    model = build_model(
        channels=8, num_blocks=1, n_steps=64, batch_size=32, seed=0,
        il_network=loaded,
    )
    assert torch.allclose(
        model.policy.features_extractor.input_conv.conv.weight.detach(),
        il.input_conv.conv.weight.detach(),
    )
