"""Smoke tests for PPO training and the PPOAgent wrapper.

These use a tiny network and very few timesteps — they check the whole
MaskablePPO + SelfPlayEnv + ResNet-extractor integration runs end to end, not
that the agent becomes strong (that needs hundreds of thousands of steps).
"""

from __future__ import annotations

from src.agents.ppo_agent import PPOAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.move_generator import generate_legal_moves
from src.training.ppo_train import build_model, train


def test_build_model_and_ppo_agent_plays_legally():
    # No training needed: even an untrained masked policy must return a legal move.
    model = build_model(channels=8, num_blocks=1, n_steps=64, batch_size=32, seed=0)
    agent = PPOAgent(model)
    board = Board()
    move = action_space.index_to_move(agent.select_move(board))
    assert move in set(generate_legal_moves(board))


def test_tiny_training_run_completes():
    model = train(
        total_timesteps=128,
        channels=8,
        num_blocks=1,
        n_steps=64,
        batch_size=32,
        seed=0,
        verbose=0,
    )
    # After a tiny run the model still produces legal moves via the agent.
    agent = PPOAgent(model, deterministic=True)
    board = Board()
    action = agent.select_move(board)
    assert action_space.index_to_move(action) in set(generate_legal_moves(board))
