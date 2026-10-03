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
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

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
    device: str = "auto",
) -> MaskablePPO:
    """Construct a MaskablePPO model on a SelfPlayEnv using our ResNet body.

    If ``il_network`` is given, its body is transferred into the extractor
    (Agent 2 Phase 2). Its ``channels``/``num_blocks`` must match this model's.
    ``device`` is "auto"/"cuda"/"cpu" (auto uses the GPU when available).
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
        device=device,
    )
    if il_network is not None:
        transfer_il_weights(model, il_network)
    return model


class PeriodicCheckpoint(BaseCallback):
    """Overwrite a single checkpoint file every ``save_freq`` steps.

    SB3 only saves at the end of ``learn()`` by default, so a long PPO run that
    is interrupted (Colab disconnect, crash, power loss) loses everything. This
    saves the model to one fixed path periodically, so a ``--resume`` can pick up
    from the last save. One file (overwritten) keeps Google Drive tidy.
    """

    def __init__(self, save_path: str, save_freq: int, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.save_freq and self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose:
                print(
                    f"[checkpoint] {self.num_timesteps} steps -> {self.save_path}.zip",
                    flush=True,
                )
        return True


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
    il_checkpoint: str | None = None,
    checkpoint_every: int = 0,
    resume: bool = False,
    device: str = "auto",
    log_dir: str | None = None,
) -> MaskablePPO:
    """Train a PPO self-play agent and optionally save it.

    * ``il_checkpoint`` — start from an imitation-learning checkpoint (Agent 2
      Phase 2): its body is transferred into the features extractor (must share
      ``channels``/``num_blocks``). Ignored when resuming.
    * ``checkpoint_every`` — also save every N steps during training (not just at
      the end), so an interrupted run can resume. Recommended on Colab.
    * ``resume`` — if ``save_path`` already exists, continue from it and train
      ``total_timesteps`` MORE steps (otherwise start fresh).
    * ``device`` — "auto"/"cuda"/"cpu" (auto picks the GPU if available).
    """
    env = SelfPlayEnv()
    zip_path = f"{save_path}.zip" if save_path else None
    reset_num_timesteps = True

    if resume and zip_path and os.path.exists(zip_path):
        model = MaskablePPO.load(save_path, env=env, device=device)
        reset_num_timesteps = False
        print(
            f"resuming PPO from {zip_path} at {model.num_timesteps} steps",
            flush=True,
        )
    else:
        il_network = None
        if il_checkpoint is not None:
            from src.training.imitation import load_network

            il_network = load_network(
                il_checkpoint, channels=channels, num_blocks=num_blocks
            )
        model = build_model(
            env=env,
            channels=channels,
            num_blocks=num_blocks,
            n_steps=n_steps,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            il_network=il_network,
            device=device,
        )

    if log_dir is not None:
        # Write progress.csv + TensorBoard events (+ stdout) so training is
        # provable: live graphs via `tensorboard --logdir`, and a CSV to plot
        # publication figures from. Works for both fresh and resumed models.
        from stable_baselines3.common.logger import configure

        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    callback = None
    if checkpoint_every and save_path is not None:
        callback = PeriodicCheckpoint(save_path, checkpoint_every, verbose=verbose)

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=callback,
    )
    if save_path is not None:
        model.save(save_path)
    return model


def load_ppo_agent(
    path: str,
    *,
    channels: int = NN_CHANNEL_WIDTH,
    num_blocks: int = NN_NUM_RES_BLOCKS,
    device: str = "cpu",
    name: str = "PPO",
    deterministic: bool = True,
):
    """Load a trained PPO model as a :class:`PPOAgent`, robustly.

    ``MaskablePPO.load`` can fail on some torch/SB3 combinations with a
    ``PytorchStreamReader`` error while reading the zip's inner stream, even
    though the weights are intact. In that case we fall back to rebuilding the
    model (same ``channels``/``num_blocks``) and loading the policy ``state_dict``
    read directly from the ``.zip`` — which works where the streaming reader
    does not. The architecture must match what the checkpoint was trained with.
    """
    from src.agents.ppo_agent import PPOAgent

    try:
        model = MaskablePPO.load(path, device=device)
        return PPOAgent(model, name=name, deterministic=deterministic)
    except Exception:
        import io
        import zipfile

        import torch

        model = build_model(
            env=SelfPlayEnv(), channels=channels, num_blocks=num_blocks,
            device=device,
        )
        zip_path = path if path.endswith(".zip") else f"{path}.zip"
        policy_sd = torch.load(
            io.BytesIO(zipfile.ZipFile(zip_path).read("policy.pth")),
            map_location=device, weights_only=False,
        )
        model.policy.load_state_dict(policy_sd)
        return PPOAgent(model, name=name, deterministic=deterministic)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO self-play (Agent 1, or Agent 2 Phase 2 with --il-checkpoint)."
    )
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="steps to train this run (added on top when --resume)")
    parser.add_argument("--save", type=str, default="results/checkpoints/ppo_agent1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--il-checkpoint", type=str, default=None,
        help="IL checkpoint to initialise from (Agent 2 Phase 2)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="also save every N steps during training (0 = only at the end). "
             "Use on Colab so an interrupted run can resume, e.g. 10000.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from an existing --save checkpoint instead of starting fresh",
    )
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="write progress.csv + TensorBoard events here (live graphs via "
             "`tensorboard --logdir <dir>`), e.g. results/tb/ppo_agent2",
    )
    args = parser.parse_args()
    train(
        args.timesteps,
        args.save,
        seed=args.seed,
        il_checkpoint=args.il_checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        device=args.device,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
