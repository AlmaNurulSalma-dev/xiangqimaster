"""Test the EngineAgent against a fake UCCI engine subprocess.

Uses a minimal fake engine (no real binary needed) to exercise the full
handshake → position → go → bestmove → move-index path.
"""

from __future__ import annotations

import sys
import textwrap

from src.agents.engine_agent import EngineAgent
from src.environment import action_space
from src.environment.board import Board
from src.environment.elephanteye import ElephantEyeEngine

_FAKE_ENGINE = textwrap.dedent(
    """
    import sys
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "ucci":
            print("id name FakeEngine")
            print("ucciok", flush=True)
        elif cmd.startswith("go"):
            print("info depth 1")
            print("bestmove a0a1", flush=True)
        elif cmd == "quit":
            break
    """
)


def test_engine_agent_returns_the_engines_move(tmp_path):
    fake = tmp_path / "fake_engine.py"
    fake.write_text(_FAKE_ENGINE, encoding="utf-8")

    engine = ElephantEyeEngine([sys.executable, str(fake)])
    with engine:
        agent = EngineAgent(engine, depth=1)
        action = agent.select_move(Board())

    # The fake always answers "a0a1" → move (0,0,1,0).
    assert action == action_space.move_to_index((0, 0, 1, 0))


def test_engine_bestmove_parses_coordinates(tmp_path):
    fake = tmp_path / "fake_engine.py"
    fake.write_text(_FAKE_ENGINE, encoding="utf-8")
    with ElephantEyeEngine([sys.executable, str(fake)]) as engine:
        assert engine.bestmove(Board()) == (0, 0, 1, 0)
