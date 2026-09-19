# 03 — Game Environment Specification

> **Read this when:** building or modifying the Xiangqi Gymnasium environment — the board state representation, action space, reward function, and the reset/step API. This is the foundation every agent depends on.

---

## 1. Purpose of the Environment

The environment is the simulator the agents interact with. It must conform to the **Gymnasium interface** so that standard RL libraries (stable-baselines3) can use it. It wraps the pure game logic (board, pieces, rules) into a standard `reset()` / `step()` API.

The environment answers four questions:
1. **What is the state?** — the board encoded as a tensor
2. **What are the actions?** — the set of possible moves, and which are legal now
3. **What is the reward?** — the signal that tells the agent how well it's doing
4. **When does the episode end?** — terminal conditions

---

## 2. State Representation

### 2.1 The State Tensor

The board state is encoded as a **14 × 10 × 9 tensor** (channels × rows × columns).

- 14 channels: 7 piece types × 2 players
- 10 rows, 9 columns: the board grid

**Channel assignment (fixed — never change ordering):**

| Channel | Contents |
|---|---|
| 0 | Current player's Generals (1 where present, 0 elsewhere) |
| 1 | Current player's Advisors |
| 2 | Current player's Elephants |
| 3 | Current player's Horses |
| 4 | Current player's Chariots |
| 5 | Current player's Cannons |
| 6 | Current player's Soldiers |
| 7 | Opponent's Generals |
| 8 | Opponent's Advisors |
| 9 | Opponent's Elephants |
| 10 | Opponent's Horses |
| 11 | Opponent's Chariots |
| 12 | Opponent's Cannons |
| 13 | Opponent's Soldiers |

Each channel is a 10×9 binary plane: 1.0 where a piece of that type/color exists, 0.0 elsewhere.

### 2.2 Player-Relative Encoding (Critical Design Decision)

The state is always encoded **from the perspective of the player to move**. Channels 0–6 are always "my pieces" and 7–13 are always "opponent's pieces" — regardless of whether it's Red or Black to move.

**Why:** This lets a single neural network play both sides without needing to know its color. It always sees "me vs. opponent." When it's Black's turn, the board is flipped so Black's pieces occupy channels 0–6.

**Implementation note:** When it's Black to move, rotate/flip the board 180° so the current player always "moves upward." This canonical orientation must be applied consistently in both the environment and the encoder. Document the exact flip operation and test it carefully.

### 2.3 Optional Additional Channels (Advanced)

For richer state, optionally add channels (making it e.g. 15 or 17 channels). Only do this if experiments show it helps. Candidates:
- A "whose turn" plane (all 1s or all 0s) — often unnecessary given player-relative encoding
- Move-count plane (normalized) — helps with draw awareness
- Repetition-count plane — helps avoid perpetual check

**Default: stick with 14 channels.** Note any addition in config and thesis.

---

## 3. Action Space

### 3.1 Action Representation

The action space is **discrete with 2,086 possible actions**. Each action index maps to a unique (from_position → to_position) move.

**Why 2,086:** This is the total count of all *geometrically possible* moves any piece could make on the board, pre-enumerated. Not all are legal in a given position — most positions have ~30–40 legal moves.

### 3.2 Action Index Mapping

Build a fixed, bidirectional lookup table at initialization:
- `action_index → (from_row, from_col, to_row, to_col)`
- `(from_row, from_col, to_row, to_col) → action_index`

This table is computed once and reused. It must be identical across training and evaluation (save it or regenerate deterministically).

### 3.3 Legal Action Masking (MANDATORY)

At every step, the environment must provide a **legal action mask**: a boolean array of length 2,086, where `True` marks a legal move and `False` marks an illegal one.

**This is non-negotiable.** The agent must NEVER select an illegal action. Masking is applied by setting the logits of illegal actions to negative infinity before the softmax/sampling step.

The mask is produced by:
1. `move_generator.py` generates all legal moves for the current position
2. Each legal move is converted to its action index
3. Those indices are set `True` in the mask, all others `False`

The environment exposes this mask via a method like `env.legal_action_mask()` or includes it in the `info` dict returned by `step()`.

---

## 4. Reward Function

### 4.1 Default: Sparse Terminal Reward

The primary reward is sparse — given only at game end, from the perspective of the player who just moved:

| Outcome | Reward |
|---|---|
| Win (checkmate opponent, or opponent stalemated) | +1.0 |
| Loss (get checkmated, or get stalemated) | −1.0 |
| Draw (repetition or move limit) | +0.1 |

Non-terminal steps return 0.0.

**Why draw = +0.1 (small positive):** In Xiangqi, decisive games are common, but a small positive draw reward discourages the agent from throwing games while still valuing wins far more.

### 4.2 Optional: Dense Reward Shaping (Experimental)

For faster early learning, optionally add a small material-advantage reward at each step:

```
dense_reward = (my_material − opponent_material) × 0.01
```

using the piece values from 01-GAME-RULES.md section 8.

**Trade-off:** Dense reward speeds learning but can cause the agent to over-value capturing pieces rather than winning. Make this a config toggle (`use_dense_reward: bool`) and compare both in experiments if time allows.

### 4.3 Reward Perspective

Rewards are always from the perspective of the player who made the move that led to the state. In self-play, this must be handled carefully — the reward for player A's winning move is +1, and the same terminal state gives −1 to player B. The self-play loop (07-TRAINING.md) assigns outcomes to each player's stored experiences correctly.

---

## 5. The Gymnasium API

The environment must implement the standard Gymnasium interface:

### 5.1 `reset(seed=None) → (observation, info)`
- Resets to the standard Xiangqi starting position
- Returns the initial state tensor (14×10×9) and an info dict (containing at least the legal action mask)
- Must accept a seed for reproducibility

### 5.2 `step(action) → (observation, reward, terminated, truncated, info)`
- Applies the given action (an integer index)
- If the action is illegal (should never happen with masking), raise an error or return a large penalty — document the chosen behavior
- Returns:
  - `observation`: new state tensor (from the NEXT player's perspective)
  - `reward`: reward for the player who just moved
  - `terminated`: True if the game ended by rules (checkmate/stalemate/draw)
  - `truncated`: True if ended by move limit
  - `info`: dict with legal action mask, current player, move count, etc.

### 5.3 `render()`
- Produces a human-readable board (ASCII and/or graphical) — used by the dashboard and for debugging

### 5.4 Additional Required Methods
- `legal_action_mask() → np.ndarray[bool]` — the current legal move mask
- `get_current_player() → {RED, BLACK}` — whose turn
- `clone()` / `copy()` — deep copy of state (REQUIRED for MCTS, which simulates future positions)

---

## 6. Special Requirement: State Cloning for MCTS

MCTS (Agent 3) needs to simulate many hypothetical future positions without affecting the real game. The environment MUST support fast, correct deep copying:
- `env.clone()` returns an independent copy whose moves don't affect the original
- This must be efficient — MCTS calls it thousands of times per move
- Test that mutating a clone never mutates the original

---

## 7. Episode Termination Summary

| Condition | terminated | truncated | reward to mover |
|---|---|---|---|
| Checkmate delivered | True | False | +1.0 |
| Got checkmated | True | False | −1.0 |
| Opponent stalemated (no moves) | True | False | +1.0 |
| Self stalemated (no moves) | True | False | −1.0 |
| Threefold repetition | True | False | +0.1 |
| Move limit (300 plies) | False | True | +0.1 |

---

## 8. Existing Environment: gym-xiangqi

An open-source environment (`gym-xiangqi`) exists and may be used as a starting point. Evaluate it against this spec:
- Does it use the 14-channel player-relative encoding? If not, wrap or adapt it.
- Does it provide a legal action mask? If not, add one.
- Does it support `clone()` for MCTS? If not, add it.
- Does its action space match (or can it be mapped to) the 2,086-move convention?

**Decision:** Either use gym-xiangqi with a wrapper that adds the missing pieces, OR implement from scratch following this spec. Document the choice. Using it as a validated reference for move legality (cross-checking your own move generator) is valuable regardless.

---

## 9. Testing the Environment

The environment is the foundation — bugs here corrupt everything above. Test rigorously (see 11-CODING-STANDARDS.md for testing conventions):
- [ ] Starting position is correct
- [ ] `reset()` returns correct tensor shape (14, 10, 9)
- [ ] Legal action mask only contains legal moves (cross-check vs move_generator)
- [ ] Every action in the mask, when applied, produces a valid board
- [ ] Terminal conditions trigger correctly (test known checkmate positions)
- [ ] Reward signs are correct from the mover's perspective
- [ ] `clone()` produces independent copies
- [ ] Player-relative encoding correctly flips for Black to move
- [ ] Playing 1,000 random legal games never crashes or produces illegal states

---

*Cross-references: 01-GAME-RULES.md (the rules being encoded), 04-DATA-PIPELINE.md (encoder shared with data), 05-NEURAL-NETWORK.md (consumes the state tensor), 06-AGENTS.md (uses the mask).*
