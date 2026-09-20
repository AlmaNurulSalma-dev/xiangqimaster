"""PPO self-play training for Agent 1 (docs/07-TRAINING.md section 3).

Builds a :class:`SelfPlayEnv`, plugs our ResNet body in as the MaskablePPO
features extractor, and trains. The heavy hyperparameters come from config so
they are visible and adjustable; this module wires them into sb3-contrib's
MaskablePPO, which handles the collect → GAE → clipped-update loop internally.

Usage (from the repo root)::

    python -m src.training.ppo_train --timesteps 100000 --save results/checkpoints/ppo_agent1

Keep ``total_timesteps`` small (a few hundred) for a quick smoke test; real
training needs hundreds of thousands of self-play steps.
"""

from __future__ import annotations

import argparse

from sb3_contrib import MaskablePPO

from src.models.network import PolicyValueNetwork
from src.models.sb3_extractor import XiangqiResNetExtractor
from src.training.self_play_env import SelfPlayEnv
from src.utils.config import (
    NN_CHANNEL_WIDTH,
    NN_NUM_RES_BLOCKS,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_FEATURES_DIM,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_MAX_GRAD_NORM,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_VF_COEF,
)


def transfer_il_weights(
    model: MaskablePPO, il_network: PolicyValueNetwork
) -> None:
    """Copy an IL-trained network's body into the PPO features extractor.

    This is the Agent 2 Phase 2 initialization: PPO fine-tuning starts from the
    imitation-learning representation instead of random weights. Only the shared
    body transfers — the input convolution and residual tower, which have the
    same architecture in both — while SB3's policy/value heads stay fresh.
    """
    policy = model.policy
    seen: set[int] = set()
    for attr in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        extractor = getattr(policy, attr, None)
        if isinstance(extractor, XiangqiResNetExtractor) and id(extractor) not in seen:
            seen.add(id(extractor))
            extractor.input_conv.load_state_dict(il_network.input_conv.state_dict())
            extractor.tower.load_state_dict(il_network.residual_tower.state_dict())


def build_model(
    env: SelfPlayEnv | None = None,
    *,
    channels: int = NN_CHANNEL_WIDTH,
    num_blocks: int = NN_NUM_RES_BLOCKS,
    n_steps: int = PPO_N_STEPS,
    batch_size: int = PPO_BATCH_SIZE,
    seed: int | None = None,
    verbose: int = 0,
    il_network: PolicyValueNetwork | None = None,
) -> MaskablePPO:
    """Construct a MaskablePPO model on a SelfPlayEnv using our ResNet body.

    If ``il_network`` is given, its body is transferred into the extractor
    (Agent 2 Phase 2). Its ``channels``/``num_blocks`` must match this model's.
    """
    if env is None:
        env = SelfPlayEnv()
    policy_kwargs = dict(
        features_extractor_class=XiangqiResNetExtractor,
        features_extractor_kwargs=dict(
            channels=channels,
            num_blocks=num_blocks,
            features_dim=PPO_FEATURES_DIM,
        ),
    )
    model = MaskablePPO(
        policy="MlpPolicy",  # MaskableActorCriticPolicy + our custom extractor
        env=env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=verbose,
    )
    if il_network is not None:
        transfer_il_weights(model, il_network)
    return model


def train(
    total_timesteps: int,
    save_path: str | None = None,
    *,
    channels: int = NN_CHANNEL_WIDTH,
    num_blocks: int = NN_NUM_RES_BLOCKS,
    n_steps: int = PPO_N_STEPS,
    batch_size: int = PPO_BATCH_SIZE,
    seed: int | None = None,
    verbose: int = 1,
) -> MaskablePPO:
    """Train a PPO self-play agent and optionally save it."""
    model = build_model(
        channels=channels,
        num_blocks=num_blocks,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps)
    if save_path is not None:
        model.save(save_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Agent 1 (PPO self-play).")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save", type=str, default="results/checkpoints/ppo_agent1")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train(args.timesteps, args.save, seed=args.seed)


if __name__ == "__main__":
    main()
