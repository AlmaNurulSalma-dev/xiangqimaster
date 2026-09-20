"""Training callbacks (docs/07-TRAINING.md, docs/09-EXPERIMENTS.md section 3).

``EloEvalCallback`` periodically evaluates the policy's Elo against a fixed
opponent during PPO training and records (timesteps, Elo). The recorded history
is the raw material for Experiment 2's learning curves (Elo vs training games).
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent


class EloEvalCallback(BaseCallback):
    """Evaluate Elo vs a fixed opponent every ``eval_freq`` steps."""

    def __init__(
        self,
        opponent: BaseAgent | None = None,
        *,
        opponent_rating: float = 1000.0,
        eval_freq: int = 10_000,
        n_games: int = 20,
        seed: int | None = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.opponent = opponent or RandomAgent(seed=seed)
        self.opponent_rating = opponent_rating
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.seed = seed
        self.history: list[tuple[int, float]] = []  # (timesteps, elo)

    def _evaluate_now(self) -> None:
        # Imported lazily to avoid a heavy import at module load.
        from src.agents.ppo_agent import PPOAgent
        from src.evaluation.tournament import estimate_elo

        agent = PPOAgent(self.model, deterministic=True)
        estimate = estimate_elo(
            agent,
            self.opponent,
            self.opponent_rating,
            self.n_games,
            seed=self.seed,
        )
        self.history.append((self.num_timesteps, estimate.rating))
        if self.verbose:
            print(f"[EloEval] t={self.num_timesteps} elo={estimate.rating:.1f}")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_now()
        return True


__all__ = ["EloEvalCallback"]
