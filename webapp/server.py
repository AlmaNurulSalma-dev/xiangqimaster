"""Tiny stdlib web server for the XiangqiMaster demo prototype.

A chessgpt.ai-style single-page front-end (``index.html``) talks to two JSON
endpoints here. No third-party web framework — just ``http.server`` — so it runs
on the same Python that already has the project + PyTorch installed.

Endpoints
---------
GET  /                 → the single-page app (index.html)
GET  /api/agents       → {"agents": [...display names...], "characters": {...}}
POST /api/play         → body {"red": name, "black": name} →
                         the whole game as frames (pieces + advantage per ply)
POST /api/move         → human-vs-AI, one move at a time (see _handle_move)

The heavy work (playing the agents, scoring every position with Pikafish or the
value head) happens once per request; the browser then animates the returned
frames with speed control, sound, and a live advantage chart.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from dashboard.components.advantage import evaluate_snapshots  # noqa: E402
from dashboard.components.agents_ui import AGENT_NAMES, make_agent  # noqa: E402
from dashboard.components.game_playback import record_game  # noqa: E402
from src.environment import action_space  # noqa: E402
from src.environment.board import Board  # noqa: E402
from src.environment.move_generator import generate_legal_moves  # noqa: E402
from src.utils.config import RED  # noqa: E402
from src.utils.render import piece_label  # noqa: E402

# One game at a time: the Pikafish engine singleton used for scoring is a single
# subprocess, so serialise play requests.
_PLAY_LOCK = threading.Lock()
_MAX_PLIES = 140   # keep demo games from dragging on
_MCTS_SIMS = 48    # fewer sims = much faster, strength barely changes (ablation)

# Human-vs-AI sessions, keyed by a generated session id. Each entry holds the
# live Board, the instantiated AI agent (kept alive across moves so an engine
# subprocess or loaded network isn't recreated on every request), which colour
# the human plays, and the piece count after the last move (for capture
# detection, mirroring _serialize's use in play_game).
_SESSIONS: dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()

# Character metadata for the opponent-select screen. Elo values are the
# project's own measured results (see research/13_advisor_consultation.md,
# Section VIII); only agents actually available on this machine are exposed
# to the frontend (filtered against AGENT_NAMES below).
_CHARACTERS = {
    "Random": {"persona": "The Rookie", "elo": "~1300"},
    "Minimax d1": {"persona": "The Calculator", "elo": "~1460"},
    "Minimax d2": {"persona": "The Calculator", "elo": "~1460"},
    "Minimax d3": {"persona": "The Calculator", "elo": "~1460"},
    "IL (human)": {"persona": "The Student", "elo": "~1600"},
    "MCTS v2": {"persona": "The Strategist", "elo": "~1690"},
    "Distill (Pikafish)": {"persona": "The Apprentice", "elo": "~1500-1570"},
    "Pikafish MAX": {"persona": "The Grandmaster", "elo": "2000+"},
}


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


class HumanMoveError(ValueError):
    """An illegal or malformed human move, or an unknown session.

    Raised instead of letting a bad request crash the server; ``do_POST``
    catches it and reports a 400 with the message, per the project's
    convention of surfacing errors to the UI rather than 500-crashing.
    """


def _frame_for(board, move_tuple, prev_count, ev) -> tuple[dict, int]:
    """Build one playback frame (see ``play_game``'s ``frames`` entries) for a
    single board, given the move that produced it and the previous piece
    count (for capture detection). Returns ``(frame, new_piece_count)``."""
    pieces = _serialize(board)
    move = None
    capture = False
    if move_tuple is not None:
        fr, fc, tr, tc = move_tuple
        move = {"fr": fr, "fc": fc, "tr": tr, "tc": tc}
        capture = prev_count is not None and len(pieces) < prev_count
    frame = {
        "pieces": pieces,
        "move": move,
        "capture": capture,
        "redCp": ev["red_cp"],
        "fill": ev["fill"],
        "label": ev["label"],
        "mate": ev["is_mate"],
    }
    return frame, len(pieces)


def _play_ai_ply(session_id: str) -> tuple[dict | None, bool]:
    """Let a session's AI agent play one move. Returns ``(frame, game_over)``;
    ``frame`` is ``None`` only if the game was already over."""
    sess = _SESSIONS[session_id]
    board = sess["board"]
    if not generate_legal_moves(board):
        return None, True
    mask = action_space.legal_mask(board)
    move = action_space.index_to_move(sess["agent"].select_move(board, mask))
    board.apply_move(*move)
    game_over = not generate_legal_moves(board)
    ev = evaluate_snapshots([board])[0]
    frame, prev_count = _frame_for(board, move, sess["prev_count"], ev)
    sess["prev_count"] = prev_count
    return frame, game_over


def start_human_game(human_side: str, ai_agent_name: str) -> dict:
    """Start a new human-vs-AI session. If the AI plays Red, it moves first,
    since Red always moves first in Xiangqi."""
    if human_side not in ("red", "black"):
        raise HumanMoveError("humanSide must be 'red' or 'black'")
    if ai_agent_name not in AGENT_NAMES:
        raise HumanMoveError(f"unknown agent: {ai_agent_name!r}")

    human_color = RED if human_side == "red" else -RED
    ai_color = -human_color
    agent = make_agent(ai_agent_name, mcts_sims=_MCTS_SIMS)

    board = Board()
    session_id = uuid.uuid4().hex
    ev0 = evaluate_snapshots([board])[0]
    frame, prev_count = _frame_for(board, None, None, ev0)
    frames = [frame]

    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = {
            "board": board,
            "agent": agent,
            "agent_name": ai_agent_name,
            "human_color": human_color,
            "ai_color": ai_color,
            "prev_count": prev_count,
        }

    game_over = False
    if ai_color == RED:
        ai_frame, game_over = _play_ai_ply(session_id)
        if ai_frame is not None:
            frames.append(ai_frame)

    return {
        "sessionId": session_id,
        "humanColor": human_side,
        "frames": frames,
        "gameOver": game_over,
    }


def human_move(session_id: str, fr: int, fc: int, tr: int, tc: int) -> dict:
    """Apply one human move, validating legality, then let the session's AI
    reply if the game isn't over. Raises ``HumanMoveError`` on anything that
    should be reported to the UI as a 400 rather than crash the server."""
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(session_id)
    if sess is None:
        raise HumanMoveError("unknown or expired session")

    board = sess["board"]
    if board.to_move != sess["human_color"]:
        raise HumanMoveError("it is not the human's turn")

    try:
        action = action_space.move_to_index((fr, fc, tr, tc))
    except KeyError:
        raise HumanMoveError("not a move this action space can encode") from None
    if not action_space.legal_mask(board)[action]:
        raise HumanMoveError("illegal move")

    board.apply_move(fr, fc, tr, tc)
    game_over = not generate_legal_moves(board)
    ev = evaluate_snapshots([board])[0]
    frame, prev_count = _frame_for(board, (fr, fc, tr, tc), sess["prev_count"], ev)
    sess["prev_count"] = prev_count
    frames = [frame]

    if not game_over:
        ai_frame, game_over = _play_ai_ply(session_id)
        if ai_frame is not None:
            frames.append(ai_frame)

    if game_over:
        with _SESSIONS_LOCK:
            _SESSIONS.pop(session_id, None)

    return {"sessionId": session_id, "frames": frames, "gameOver": game_over}


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
            characters = {name: _CHARACTERS[name] for name in AGENT_NAMES if name in _CHARACTERS}
            self._json({"agents": AGENT_NAMES, "characters": characters, "start": start})
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/play":
            self._handle_play()
        elif self.path == "/api/move":
            self._handle_move()
        else:
            self._send(404, b"not found", "text/plain")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _handle_play(self) -> None:
        try:
            req = self._read_json()
            red = req.get("red") or AGENT_NAMES[-1]
            black = req.get("black") or AGENT_NAMES[0]
            with _PLAY_LOCK:
                result = play_game(red, black)
            self._json(result)
        except Exception as exc:  # report the error to the UI instead of 500-crashing
            self._json({"error": str(exc)}, code=400)

    def _handle_move(self) -> None:
        """One human-vs-AI step: either start a session (``action: "start"``)
        or play the human's move and get the AI's reply (``action: "move"``).
        A ``HumanMoveError`` (illegal move, bad session, bad request) is
        reported as a 400 with a clear message rather than a 500 crash;
        anything else is still caught, matching ``_handle_play``'s contract.
        """
        try:
            req = self._read_json()
            action = req.get("action")
            with _PLAY_LOCK:
                if action == "start":
                    result = start_human_game(
                        req.get("humanSide", "red"), req.get("aiAgent", AGENT_NAMES[-1]),
                    )
                elif action == "move":
                    result = human_move(
                        req["sessionId"], int(req["fr"]), int(req["fc"]),
                        int(req["tr"]), int(req["tc"]),
                    )
                else:
                    raise HumanMoveError("action must be 'start' or 'move'")
            self._json(result)
        except HumanMoveError as exc:
            self._json({"error": str(exc)}, code=400)
        except (KeyError, TypeError, ValueError) as exc:
            self._json({"error": f"bad request: {exc}"}, code=400)
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
