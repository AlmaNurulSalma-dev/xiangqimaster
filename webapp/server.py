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
from src.environment import action_space, rules  # noqa: E402
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
# the human plays, and the PieceTracker carrying piece identity across the
# session (so the browser can animate moves and captures).
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


class PieceTracker:
    """Stable identity for every piece on the board.

    A board snapshot only says *what sits where*, never *which piece moved*.
    Without identity the browser has to tear down and rebuild all 32 pieces on
    every ply, which makes smooth movement and capture animation impossible.
    This assigns each piece an id once at the start of a game and carries that
    id along as moves are applied, so the front end can diff frame to frame.
    """

    __slots__ = ("ids",)

    def __init__(self, board) -> None:
        self.ids: dict[tuple[int, int], str] = {}
        counters: dict[tuple[str, int], int] = {}
        for color in (RED, -RED):
            side = "r" if color == RED else "b"
            for row, col, piece_type in board.pieces_of(color):
                key = (side, piece_type)
                n = counters.get(key, 0)
                counters[key] = n + 1
                self.ids[(row, col)] = f"{side}{piece_type}_{n}"

    def apply(self, move) -> str | None:
        """Carry the mover's id to its destination.

        Returns the id of the captured piece, or ``None`` for a quiet move —
        which is also how the front end knows a capture happened and which
        element to animate out.
        """
        from_row, from_col, to_row, to_col = move
        captured = self.ids.pop((to_row, to_col), None)
        moved = self.ids.pop((from_row, from_col), None)
        if moved is not None:
            self.ids[(to_row, to_col)] = moved
        return captured

    def id_at(self, row: int, col: int) -> str:
        # The fallback should never fire in normal play; it keeps a desynced
        # tracker from crashing the demo mid-presentation.
        return self.ids.get((row, col), f"x{row}_{col}")


def _serialize(board, tracker: PieceTracker) -> list[dict]:
    """A board → a flat list of pieces (with stable ids) for the front end."""
    pieces = []
    for color in (RED, -RED):
        for row, col, piece_type in board.pieces_of(color):
            pieces.append({
                "id": tracker.id_at(row, col),
                "r": row, "c": col,
                "t": piece_label(piece_type, color),
                "red": color == RED,
            })
    return pieces


def _status(board) -> dict:
    """Check / game-over state of ``board``, from the side to move's view.

    The side to move losing when it has no legal reply covers BOTH checkmate
    and stalemate, since stalemate is a loss in Xiangqi (docs/01-GAME-RULES.md
    section 6.3) — the two are distinguished only for the wording shown to the
    viewer. ``checkPos`` is where the threatened General stands, so the board
    can point at it.
    """
    over = not generate_legal_moves(board)
    in_check = rules.is_in_check(board, board.to_move)
    status: dict = {"check": in_check and not over, "over": over}
    if in_check or over:
        position = board.find_general(board.to_move)
        if position is not None:
            status["checkPos"] = {"r": position[0], "c": position[1]}
    if over:
        status["result"] = "black" if board.to_move == RED else "red"
        status["reason"] = "checkmate" if in_check else "stalemate"
    return status


def play_game(red: str, black: str) -> dict:
    """Play ``red`` vs ``black`` and return every position + its advantage."""
    moves, snapshots = record_game(
        make_agent(red, mcts_sims=_MCTS_SIMS),
        make_agent(black, mcts_sims=_MCTS_SIMS),
        max_plies=_MAX_PLIES,
    )
    evals = evaluate_snapshots(snapshots)

    tracker = PieceTracker(snapshots[0])
    frames = [
        _frame_for(board, moves[k - 1] if k > 0 else None, tracker, evals[k])
        for k, board in enumerate(snapshots)
    ]
    # record_game stops at the ply cap as well as at a terminal position, so a
    # game that simply ran out of moves is a draw rather than a win.
    if frames and not frames[-1]["over"] and len(moves) >= _MAX_PLIES:
        frames[-1].update(over=True, result="draw", reason="move limit")

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


def _frame_for(board, move_tuple, tracker: PieceTracker, ev) -> dict:
    """Build one playback frame for a single board.

    ``board`` is the position *after* ``move_tuple`` was played, so the tracker
    is advanced by that move first — which is what yields ``capturedId``.
    """
    captured = tracker.apply(move_tuple) if move_tuple is not None else None
    frame = {
        "pieces": _serialize(board, tracker),
        "move": None,
        "capture": captured is not None,
        "capturedId": captured,
        "redCp": ev["red_cp"],
        "fill": ev["fill"],
        "label": ev["label"],
        "mate": ev["is_mate"],
    }
    if move_tuple is not None:
        fr, fc, tr, tc = move_tuple
        frame["move"] = {"fr": fr, "fc": fc, "tr": tr, "tc": tc}
    frame.update(_status(board))
    return frame


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
    ev = evaluate_snapshots([board])[0]
    frame = _frame_for(board, move, sess["tracker"], ev)
    return frame, frame["over"]


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
    tracker = PieceTracker(board)
    session_id = uuid.uuid4().hex
    ev0 = evaluate_snapshots([board])[0]
    frames = [_frame_for(board, None, tracker, ev0)]

    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = {
            "board": board,
            "agent": agent,
            "agent_name": ai_agent_name,
            "human_color": human_color,
            "ai_color": ai_color,
            "tracker": tracker,
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
    ev = evaluate_snapshots([board])[0]
    frame = _frame_for(board, (fr, fc, tr, tc), sess["tracker"], ev)
    game_over = frame["over"]
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
            board = Board()
            start = _frame_for(board, None, PieceTracker(board), {
                "red_cp": 0, "fill": 0.5,
                "label": "Starting position", "is_mate": False,
            })
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
