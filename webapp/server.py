"""Tiny stdlib web server for the XiangqiMaster demo prototype.

A chessgpt.ai-style single-page front-end (``index.html``) talks to two JSON
endpoints here. No third-party web framework — just ``http.server`` — so it runs
on the same Python that already has the project + PyTorch installed.

Endpoints
---------
GET  /                 → the single-page app (index.html)
GET  /api/agents       → {"agents": [...display names...]}
POST /api/play         → body {"red": name, "black": name} →
                         the whole game as frames (pieces + advantage per ply)

The heavy work (playing the agents, scoring every position with Pikafish or the
value head) happens once per request; the browser then animates the returned
frames with speed control, sound, and a live advantage chart.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from dashboard.components.advantage import evaluate_snapshots  # noqa: E402
from dashboard.components.agents_ui import AGENT_NAMES, make_agent  # noqa: E402
from dashboard.components.game_playback import record_game  # noqa: E402
from src.environment.board import Board  # noqa: E402
from src.utils.config import RED  # noqa: E402
from src.utils.render import piece_label  # noqa: E402

# One game at a time: the Pikafish engine singleton used for scoring is a single
# subprocess, so serialise play requests.
_PLAY_LOCK = threading.Lock()
_MAX_PLIES = 140   # keep demo games from dragging on
_MCTS_SIMS = 48    # fewer sims = much faster, strength barely changes (ablation)


def _serialize(board) -> list[dict]:
    """A board → a flat list of pieces for the front-end to draw."""
    pieces = []
    for color in (RED, -RED):
        for row, col, piece_type in board.pieces_of(color):
            pieces.append({
                "r": row, "c": col,
                "t": piece_label(piece_type, color),
                "red": color == RED,
            })
    return pieces


def play_game(red: str, black: str) -> dict:
    """Play ``red`` vs ``black`` and return every position + its advantage."""
    moves, snapshots = record_game(
        make_agent(red, mcts_sims=_MCTS_SIMS),
        make_agent(black, mcts_sims=_MCTS_SIMS),
        max_plies=_MAX_PLIES,
    )
    evals = evaluate_snapshots(snapshots)

    frames = []
    prev_count = None
    for k, board in enumerate(snapshots):
        pieces = _serialize(board)
        move = None
        capture = False
        if k > 0:
            fr, fc, tr, tc = moves[k - 1]
            move = {"fr": fr, "fc": fc, "tr": tr, "tc": tc}
            capture = prev_count is not None and len(pieces) < prev_count
        ev = evals[k]
        frames.append({
            "pieces": pieces,
            "move": move,
            "capture": capture,
            "redCp": ev["red_cp"],
            "fill": ev["fill"],
            "label": ev["label"],
            "mate": ev["is_mate"],
        })
        prev_count = len(pieces)

    return {
        "red": red,
        "black": black,
        "plies": len(moves),
        "source": evals[0]["source"] if evals else "n/a",
        "frames": frames,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                with open(os.path.join(_HERE, "index.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(404, b"index.html not found", "text/plain")
        elif self.path == "/api/agents":
            start = {
                "pieces": _serialize(Board()), "move": None, "capture": False,
                "redCp": 0, "fill": 0.5, "label": "Starting position", "mate": False,
            }
            self._json({"agents": AGENT_NAMES, "start": start})
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/play":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            red = req.get("red") or AGENT_NAMES[-1]
            black = req.get("black") or AGENT_NAMES[0]
            with _PLAY_LOCK:
                result = play_game(red, black)
            self._json(result)
        except Exception as exc:  # report the error to the UI instead of 500-crashing
            self._json({"error": str(exc)}, code=400)

    def log_message(self, *args) -> None:  # quieter console
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8502"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"XiangqiMaster demo on http://127.0.0.1:{port}  (agents: {len(AGENT_NAMES)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
