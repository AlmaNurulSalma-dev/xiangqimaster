"""Global constants for XiangqiMaster.

Every fixed number in the project lives here — board dimensions, the action
space size, player and piece codes, palace/river boundaries, and piece values.
Logic modules import these names instead of hardcoding "magic numbers"
(see docs/11-CODING-STANDARDS.md section 4).

Tunable hyperparameters (learning rate, number of games, MCTS simulations)
do NOT belong here — those go in per-experiment YAML files under configs/.
"""

from __future__ import annotations

# ─── Board geometry (docs/01-GAME-RULES.md section 2) ──────────────────────
BOARD_ROWS: int = 10  # ranks; row 0 = Red's back rank (bottom), row 9 = Black's
BOARD_COLS: int = 9   # files; col 0 = leftmost, col 8 = rightmost
NUM_POINTS: int = BOARD_ROWS * BOARD_COLS  # 90 playable intersections

# ─── Tensor / action space (docs/03-ENVIRONMENT.md) ───────────────────────
NUM_CHANNELS: int = 14         # board_tensor shape is (14, 10, 9)
ACTION_SPACE_SIZE: int = 2086  # count of all geometrically possible moves

# ─── Players ──────────────────────────────────────────────────────────────
# Encoded as +1 / -1 so a piece's sign also encodes its owner on the board.
RED: int = 1
BLACK: int = -1


def opponent(color: int) -> int:
    """Return the other player."""
    return -color


# ─── Piece type codes ─────────────────────────────────────────────────────
# Order matches the neural-network channel order (docs/03-ENVIRONMENT.md 2.1).
# On the board, a Red piece is stored as +code and a Black piece as -code;
# an empty point is 0.
GENERAL: int = 1
ADVISOR: int = 2
ELEPHANT: int = 3
HORSE: int = 4
CHARIOT: int = 5
CANNON: int = 6
SOLDIER: int = 7

PIECE_TYPES: tuple[int, ...] = (
    GENERAL,
    ADVISOR,
    ELEPHANT,
    HORSE,
    CHARIOT,
    CANNON,
    SOLDIER,
)
EMPTY: int = 0

# Single-letter symbols for human-readable rendering / debugging.
# Uppercase = Red, lowercase = Black.
PIECE_LETTERS: dict[int, str] = {
    GENERAL: "G",
    ADVISOR: "A",
    ELEPHANT: "E",
    HORSE: "H",
    CHARIOT: "R",  # R for Rook/Chariot (avoids clashing with Cannon's C)
    CANNON: "C",
    SOLDIER: "S",
}

# ─── Palace & river boundaries (0-indexed, inclusive) ─────────────────────
PALACE_COLS: tuple[int, int] = (3, 5)       # both palaces span columns 3–5
RED_PALACE_ROWS: tuple[int, int] = (0, 2)   # Red general/advisors live here
BLACK_PALACE_ROWS: tuple[int, int] = (7, 9)  # Black general/advisors live here

# The river runs between row 4 and row 5.
# Red's half is rows 0–4; Black's half is rows 5–9.
RED_SIDE_ROWS: tuple[int, int] = (0, 4)
BLACK_SIDE_ROWS: tuple[int, int] = (5, 9)

# ─── Piece values (docs/01-GAME-RULES.md section 8) ───────────────────────
# Used by the Minimax baseline and optional dense reward shaping.
# The General is game-ending; a large finite value stands in for "infinite".
PIECE_VALUES: dict[int, float] = {
    GENERAL: 10000.0,
    CHARIOT: 9.0,
    CANNON: 4.5,
    HORSE: 4.0,
    ADVISOR: 2.0,
    ELEPHANT: 2.0,
    SOLDIER: 1.0,  # a soldier that has crossed the river is worth more;
    #                that context-dependent bonus is applied in evaluation.
}

# ─── Episode limits (docs/01-GAME-RULES.md section 6.5) ────────────────────
MAX_PLIES: int = 300              # move limit → draw
REPETITION_LIMIT: int = 3        # same position 3 times → draw

# ─── Rewards, from the mover's perspective (docs/03-ENVIRONMENT.md 4.1) ─────
WIN_REWARD: float = 1.0
LOSS_REWARD: float = -1.0
DRAW_REWARD: float = 0.1         # small positive: draws are rare in Xiangqi

# ─── Neural network architecture (docs/05-NEURAL-NETWORK.md section 9) ─────
NN_CHANNEL_WIDTH: int = 64        # C: body channel width (64/128/256)
NN_NUM_RES_BLOCKS: int = 10      # N: residual blocks (6 fast … 19 strong)
NN_POLICY_HEAD_CHANNELS: int = 2  # 1x1 conv channels in the policy head
NN_VALUE_HEAD_HIDDEN: int = 256  # hidden units in the value head (128–512)
NN_L2_WEIGHT_DECAY: float = 1e-4  # weight decay used by the optimizer

# ─── Minimax baseline (docs/06-AGENTS.md section 2) ────────────────────────
MINIMAX_DEPTH: int = 4            # search depth in plies (deeper = stronger, slower)

# ─── MCTS (Agent 3, docs/06-AGENTS.md section 5.6) ─────────────────────────
MCTS_SIMULATIONS: int = 400        # simulations per move (Experiment 5 studies this)
MCTS_C_PUCT: float = 1.5           # exploration constant in the PUCT formula
MCTS_DIRICHLET_ALPHA: float = 0.3  # root exploration noise (training self-play)
MCTS_DIRICHLET_EPSILON: float = 0.25

# ─── Imitation learning (docs/07-TRAINING.md section 2.3) ──────────────────
IL_LEARNING_RATE: float = 1e-3
IL_EPOCHS: int = 30
IL_BATCH_SIZE: int = 512

# ─── PPO self-play (docs/07-TRAINING.md section 3.4) ───────────────────────
PPO_LEARNING_RATE: float = 3e-4
PPO_N_STEPS: int = 2048          # steps collected per policy update
PPO_BATCH_SIZE: int = 512
PPO_N_EPOCHS: int = 4            # passes over the collected data per update
PPO_CLIP_RANGE: float = 0.2     # PPO clipping epsilon
PPO_GAMMA: float = 0.99         # discount factor
PPO_GAE_LAMBDA: float = 0.95    # GAE smoothing
PPO_ENT_COEF: float = 0.01      # entropy bonus (exploration)
PPO_VF_COEF: float = 0.5        # value loss coefficient
PPO_MAX_GRAD_NORM: float = 0.5  # gradient clipping
PPO_FEATURES_DIM: int = 256     # ResNet extractor output size fed to SB3 heads

# ─── Reproducibility ──────────────────────────────────────────────────────
DEFAULT_SEED: int = 42
