"""Experiment logging (docs/07-TRAINING.md sections 7-8).

A thin logger that fans metrics out to Weights & Biases and/or TensorBoard when
they are enabled and installed, and always keeps an in-memory history (so code
and tests never depend on an external service). Backends degrade gracefully:
requesting one that is not installed logs a warning and is simply skipped rather
than crashing a long training run.

Also exposes ``git_commit()`` so runs can record the exact code version.
"""

from __future__ import annotations

import subprocess
import warnings
from typing import Any


def git_commit(short: bool = True) -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else \
        ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


class Logger:
    """Fan metrics out to W&B / TensorBoard (if enabled) + in-memory history."""

    def __init__(
        self,
        *,
        project: str = "xiangqimaster",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        log_dir: str = "runs",
        console: bool = False,
    ) -> None:
        self.console = console
        self.history: list[tuple[int | None, dict[str, float]]] = []
        self._wandb = None
        self._tb = None

        if use_wandb:
            self._wandb = self._init_wandb(project, run_name, config)
        if use_tensorboard:
            self._tb = self._init_tensorboard(log_dir)
        if config:
            self.log_config(config)

    # ─── Backend init (lazy, graceful) ────────────────────────────────────
    @staticmethod
    def _init_wandb(project, run_name, config):
        try:
            import wandb
        except ImportError:
            warnings.warn("wandb not installed - W&B logging disabled")
            return None
        return wandb.init(project=project, name=run_name, config=config or {})

    @staticmethod
    def _init_tensorboard(log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            warnings.warn("tensorboard not available - TensorBoard logging disabled")
            return None
        return SummaryWriter(log_dir=log_dir)

    # ─── Logging ──────────────────────────────────────────────────────────
    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record a set of scalar metrics at an optional step."""
        self.history.append((step, dict(metrics)))
        if self.console:
            joined = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(f"[log] step={step} {joined}")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._tb is not None:
            for key, value in metrics.items():
                self._tb.add_scalar(key, value, global_step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Record run configuration (and the git commit for reproducibility)."""
        payload = dict(config)
        payload.setdefault("git_commit", git_commit())
        if self._wandb is not None:
            self._wandb.config.update(payload, allow_val_change=True)
        if self.console:
            print(f"[config] {payload}")

    def finish(self) -> None:
        """Flush and close any open backends."""
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
            self._tb = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()


__all__ = ["Logger", "git_commit"]
