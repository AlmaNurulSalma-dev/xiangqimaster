"""Fast smoke tests for AlphaZero-style MCTS self-play training (Agent 3)."""

from __future__ import annotations

import numpy as np

from src.agents.mcts_agent import MCTSAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.models.network import PolicyValueNetwork
from src.training.imitation import load_network
from src.training.mcts_train import play_selfplay_game, train_mcts
from src.utils.config import ACTION_SPACE_SIZE


def test_play_selfplay_game_shapes():
    net = PolicyValueNetwork(channels=4, num_blocks=1)
    rng = np.random.default_rng(0)
    samples, winner, plies = play_selfplay_game(
        net, n_simulations=3, c_puct=1.5, rng=rng, max_plies=12
    )
    assert plies <= 12 and len(samples) == plies
    state, pi, z = samples[0]
    assert state.shape == (14, 10, 9)
    assert pi.shape == (ACTION_SPACE_SIZE,)
    assert abs(float(pi.sum()) - 1.0) < 1e-5   # policy target is a distribution
    assert winner in (1, -1, None)
    assert float(z) in (-1.0, 0.0, 1.0)


def test_train_mcts_runs_and_checkpoint_is_playable(tmp_path):
    save = tmp_path / "mcts.pt"
    train_mcts(
        iterations=1, games_per_iter=1, n_simulations=3, epochs_per_iter=1,
        batch_size=16, window=5000, max_plies=12, channels=4, num_blocks=1,
        device="cpu", seed=0, save_path=str(save),
    )
    assert save.exists()
    net = load_network(str(save), channels=4, num_blocks=1)
    agent = MCTSAgent(net, n_simulations=3)
    board = Board()
    action = agent.select_move(board)
    assert action_space.index_to_move(action) in set(generate_legal_moves(board))
