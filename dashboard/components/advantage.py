"""Advantage ("who's winning") evaluation for the live-demo eval bar.

This powers the chess.com-style meter that lets a non-Xiangqi audience follow a
game: for every position it produces a single Red-positive score, a 0–1 fill for
the bar, and a plain-language label (balanced / slight / big advantage).

Preferred source is **Pikafish** (a centipawn score, thresholds 100/800), the
strongest open engine; when the engine binary is absent it falls back to a
trained **policy–value network's value head**, scaled to the same thresholds.
Scores are always expressed from **Red's** perspective, regardless of whose turn
it is, so the bar reads consistently across the whole game.
"""

from __future__ import annotations

from src.utils.config import RED

# Centipawn thresholds (Pikafish scale): below SLIGHT = balanced; SLIGHT..BIG =
# slight advantage; at/above BIG = big advantage. Matches the demo blueprint.
_SLIGHT_CP = 100
_BIG_CP = 800
# Shallow search depth per position: NNUE is fast here, so a full game scores in
# a few seconds while still giving a meaningful evaluation.
_EVAL_DEPTH = 6
# Centipawns that map to a fully one-sided bar.
_FULL_CP = 1000.0


def _label(red_cp: int, is_mate: bool) -> str:
    """Plain-language verdict from a Red-positive centipawn score."""
    if is_mate:
        return "Red has a forced mate" if red_cp > 0 else "Black has a forced mate"
    magnitude = abs(red_cp)
    if magnitude < _SLIGHT_CP:
        return "Balanced"
    side = "Red" if red_cp > 0 else "Black"
    return f"{side}: {'slight' if magnitude < _BIG_CP else 'big'} advantage"


def _fill(red_cp: float) -> float:
    """Red's share of the bar in ``[0, 1]`` (0.5 = balanced)."""
    clamped = max(-1.0, min(1.0, red_cp / _FULL_CP))
    return 0.5 + 0.5 * clamped


def _entry(red_cp: int, is_mate: bool, source: str) -> dict:
    return {
        "red_cp": int(red_cp),
        "is_mate": is_mate,
        "fill": _fill(red_cp),
        "label": _label(red_cp, is_mate),
        "source": source,
    }


def _terminal_entry(board, source: str) -> dict | None:
    """If ``board`` is game-over (the side to move has no legal reply — checkmate,
    or stalemate which is also a loss in Xiangqi), return a finished-game entry;
    otherwise ``None``. Handled explicitly because an engine's score sign for an
    already-mated position is ambiguous."""
    from src.environment.move_generator import generate_legal_moves

    if generate_legal_moves(board):
        return None
    winner_is_red = board.to_move != RED
    red_cp = 30000 if winner_is_red else -30000
    return {
        "red_cp": red_cp,
        "is_mate": True,
        "fill": _fill(red_cp),
        "label": f"{'Red' if winner_is_red else 'Black'} wins — checkmate",
        "source": source,
    }


def evaluate_snapshots(snapshots, *, prefer_engine: bool = True) -> list[dict]:
    """Score every board in ``snapshots`` for the advantage bar.

    Returns one dict per snapshot: ``red_cp`` (Red-positive centipawns),
    ``is_mate``, ``fill`` (0–1), ``label``, and ``source``.
    """
    from dashboard.components.agents_ui import (
        pikafish_available,
        pikafish_engine,
        value_net_checkpoint,
    )

    if prefer_engine and pikafish_available():
        engine = pikafish_engine()
        out: list[dict] = []
        for board in snapshots:
            terminal = _terminal_entry(board, "Pikafish")
            if terminal is not None:
                out.append(terminal)
                continue
            try:
                cp, is_mate = engine.evaluate(board, depth=_EVAL_DEPTH)
            except Exception:  # keep the demo alive if one eval hiccups
                cp, is_mate = 0, False
            red_cp = cp if board.to_move == RED else -cp
            out.append(_entry(red_cp, is_mate, "Pikafish"))
        return out

    # Fallback: a trained network's value head (expected result for side-to-move,
    # in [-1, 1]); scale to centipawns so the same thresholds/labels apply.
    path = value_net_checkpoint()
    if path is None:
        return [_entry(0, False, "n/a") for _ in snapshots]

    import torch

    from src.environment import encoder
    from src.training.imitation import load_network

    net = load_network(path, device="cpu")
    net.eval()
    out = []
    with torch.no_grad():
        for board in snapshots:
            terminal = _terminal_entry(board, "Value head")
            if terminal is not None:
                out.append(terminal)
                continue
            obs = encoder.encode(board)
            tensor = torch.from_numpy(obs).unsqueeze(0)
            _, value = net(tensor)
            v = float(value.squeeze().item())  # side-to-move perspective
            red_v = v if board.to_move == RED else -v
            out.append(_entry(int(red_v * _FULL_CP), False, "Value head"))
    return out


def bar_html(adv: dict) -> str:
    """An HTML advantage bar for ``st.markdown(..., unsafe_allow_html=True)``."""
    red_pct = max(0.0, min(100.0, adv["fill"] * 100.0))
    black_pct = 100.0 - red_pct
    cp = adv["red_cp"]
    if adv["is_mate"]:
        score_txt = "mate"
    else:
        score_txt = f"{'+' if cp >= 0 else '−'}{abs(cp)} cp"
    return f"""
<div style="font-family:inherit;margin:0.25rem 0 0.75rem 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.85rem;
              font-weight:600;margin-bottom:4px;">
    <span style="color:#c0392b;">● Red</span>
    <span>{adv['label']} &nbsp;·&nbsp; {score_txt}
          <span style="opacity:0.55;font-weight:400;">({adv['source']})</span></span>
    <span style="color:#2c3e50;">Black ●</span>
  </div>
  <div style="position:relative;height:22px;border-radius:11px;overflow:hidden;
              display:flex;box-shadow:inset 0 0 0 1px rgba(0,0,0,0.15);">
    <div style="width:{red_pct:.1f}%;background:#c0392b;"></div>
    <div style="width:{black_pct:.1f}%;background:#2c3e50;"></div>
    <div style="position:absolute;left:50%;top:0;bottom:0;width:2px;
                background:rgba(255,255,255,0.85);transform:translateX(-1px);"></div>
  </div>
</div>
"""


__all__ = ["evaluate_snapshots", "bar_html"]
