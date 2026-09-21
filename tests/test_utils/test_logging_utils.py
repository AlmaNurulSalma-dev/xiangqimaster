"""Unit tests for the logging utility."""

from __future__ import annotations

from src.utils.logging_utils import Logger, git_commit


def test_history_records_metrics():
    logger = Logger()  # all backends off
    logger.log({"loss": 1.0, "elo": 1500.0}, step=0)
    logger.log({"loss": 0.5, "elo": 1600.0}, step=1)
    assert len(logger.history) == 2
    assert logger.history[0] == (0, {"loss": 1.0, "elo": 1500.0})
    assert logger.history[1][0] == 1


def test_finish_is_safe_with_no_backends():
    logger = Logger()
    logger.log({"x": 1.0})
    logger.finish()  # must not raise


def test_context_manager():
    with Logger() as logger:
        logger.log({"x": 1.0}, step=0)
    assert logger.history == [(0, {"x": 1.0})]


def test_tensorboard_backend_does_not_crash(tmp_path):
    # Enabling a backend (available or not) must not break logging; writes go to
    # a temp dir so nothing pollutes the repo.
    logger = Logger(use_tensorboard=True, log_dir=str(tmp_path / "tb"))
    logger.log({"x": 1.0}, step=0)
    logger.finish()
    assert logger.history == [(0, {"x": 1.0})]


def test_git_commit_returns_str_or_none():
    commit = git_commit()
    assert commit is None or isinstance(commit, str)
