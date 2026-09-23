# XiangqiMaster — Complete Project Guide

> A detailed, self-contained explanation of the whole project: what it is, why it
> exists, how every part works, what is done, and what remains. Written to be read
> top-to-bottom by someone who wants to fully understand the codebase. For the
> short version, see [`README.md`](README.md); for the formal specs, see
> [`docs/`](docs/).

---

## Table of Contents

1. [What this project is (in plain words)](#1-what-this-project-is)
2. [The research questions it answers](#2-the-research-questions)
3. [The four agents — what each one is and what it tests](#3-the-four-agents)
4. [Why we benchmark against ElephantEye](#4-why-elephanteye)
5. [How strength is measured: Elo](#5-how-strength-is-measured-elo)
6. [End-to-end flow (data → training → evaluation)](#6-end-to-end-flow)
7. [System architecture (the 4 layers)](#7-system-architecture)
8. [Project structure — every folder and file explained](#8-project-structure)
9. [Key design decisions and why](#9-key-design-decisions)
10. [What is done vs. what remains](#10-status)
11. [The plan going forward](#11-the-plan)
12. [Requirements / what you need to run it](#12-requirements)

---

## 1. What this project is

**XiangqiMaster** is a **bachelor's thesis research project** (dual degree, UII ×
NXU) that studies **how different deep-reinforcement-learning (RL) methods learn to
play Xiangqi (Chinese Chess), and which method works best.**

Xiangqi is one of the most-played board games in the world, but — unlike Go, Chess,
and Shogi (which got AlphaGo/AlphaZero) — **nobody has published a systematic
comparison of modern deep-RL methods on Xiangqi under one fair, standardized
framework.** This project fills that gap.

Concretely, the project **builds four game-playing agents**, each using a different
approach, **trains them, and measures each one's playing strength on a common
scale (Elo)** so they can be compared apples-to-apples. It also investigates
whether an agent that first *learns from human games* becomes stronger/faster than
one that learns *purely from playing itself*, and whether a self-taught agent
rediscovers classical Xiangqi opening theory.

**In one sentence:** *design, train, and compare four RL approaches to Xiangqi, and
measure them scientifically.*

---

## 2. The research questions

The whole project is organized around answering these (they map to the experiments
in `experiments/`):

- **RQ1 — Does imitation learning help?** Does pre-training on ~141k human games
  (then refining with RL) beat learning purely from self-play, in final strength
  and in convergence speed? *(Experiments 1 & 2)*
- **RQ2 — Which RL method is strongest for Xiangqi?** PPO self-play vs. MCTS+NN
  (AlphaZero-style) vs. the classical Minimax baseline. *(Experiment 1)*
- **RQ3 — Does self-play rediscover opening theory?** Do the learned agents
  independently arrive at known openings like the Central Cannon (当头炮)?
  *(Experiment 3)*
- **RQ4 — Secondary questions:** Does a curriculum (progressively stronger
  opponents) beat plain self-play (Experiment 4)? How does MCTS strength trade off
  against its simulation count / inference time (Experiment 5)?

---

## 3. The four agents

All four agents implement the **same interface** (`BaseAgent.select_move(board,
legal_mask) -> action_index`), so the tournament and dashboard treat them
identically. What differs is *how each one chooses a move*.

Think of it as four contestants using four different philosophies. Comparing them
is the point of the thesis.

### Agent 4 — Minimax (the classical baseline; **no learning**)
- **File:** `src/agents/minimax_agent.py`
- **What it does:** Classic game-tree search. It looks ahead a fixed number of
  half-moves (`MINIMAX_DEPTH = 4`), assuming both sides play their best, and picks
  the move that leads to the best position. "Best" is judged by a **handcrafted
  material score** (sum of piece values from `config.PIECE_VALUES`).
- **How, in code:** `_negamax()` is the **negamax** form of alpha-beta search — one
  recursive function that always scores from the side-to-move's view and negates
  for the opponent. `_ordered_moves()` sorts captures first (most valuable capture
  first) so **alpha-beta pruning** (the `if alpha >= beta: break`) can skip large
  parts of the tree. Reaching a position with **no legal move = a loss** (in
  Xiangqi both checkmate *and* stalemate are losses), scored near `±MATE_SCORE`
  with a depth term so quicker mates are preferred.
- **What it tests / why it exists:** It is the **reference floor** ("the
  traditional approach used for 30+ years"). An Elo number is meaningless without a
  baseline; Minimax answers *"how much better are the learning agents than classic
  search?"* It also **doubles as a correctness test of the environment** — if
  Minimax plays legal, sensible chess, then move generation and rules are sound.

### Agent 1 — PPO self-play (**learns from scratch**)
- **Files:** `src/agents/ppo_agent.py` (play-time wrapper), trained by
  `src/training/ppo_train.py` on `src/training/self_play_env.py`.
- **What it does:** Starts knowing nothing and **learns by playing millions of
  games against an opponent** (initially a random agent; later a copy of itself),
  using **PPO** (Proximal Policy Optimization), a standard policy-gradient RL
  algorithm. It gradually adjusts the neural network so moves that led to wins
  become more likely.
- **How, in code:** Xiangqi is two-player, but PPO expects a *single-agent*
  environment, so `SelfPlayEnv` **bakes the opponent into the environment**: on each
  `step()` the learning agent moves, then the opponent immediately replies, and the
  observation handed back is again the learner's turn. Reward is **sparse and from
  the learner's perspective** (+1 win, −1 loss, +0.1 draw). We use **MaskablePPO**
  (from `sb3-contrib`) so **illegal moves are masked out** (`action_masks()`),
  which is essential in a 2086-move space.
- **What it tests / why it exists:** The "pure RL, no human knowledge" data point
  — the AlphaZero philosophy. Central to **RQ1** (compare against the
  imitation-primed agent) and **RQ2**.

### Agent 2 — Imitation Learning + PPO (**learns from humans, then self-play**)
- **Files:** Phase 1 (imitation) `src/training/imitation.py` +
  `src/training/il_train.py`; the network is played by
  `src/agents/neural_agent.py`. Phase 2 (RL fine-tune) reuses `ppo_train.py`.
- **What it does — two phases:**
  - **Phase 1 (Imitation Learning):** Show the network **141k real human games**
    (the CGLemon dataset). For every position, train it to (a) predict the move the
    human actually played (policy, cross-entropy loss) and (b) predict who won the
    game (value, MSE loss). This is **supervised learning** — copy the experts.
  - **Phase 2 (PPO fine-tune):** Take that human-trained network as the starting
    point and refine it with the same self-play PPO as Agent 1.
- **How, in code:** The imitation dataset (`LazyXiangqiILDataset`) yields
  `(board_tensor, human_move_index, game_outcome)` triples. The bridge into PPO is
  `ppo_train.transfer_il_weights()`: it copies the **shared body** (input conv +
  residual tower) of the imitation network into the PPO features extractor, so RL
  starts from the human-learned representation instead of random weights.
- **What it tests / why it exists:** This is the **headline question (RQ1)** — the
  same idea that made AlphaGo work. Does starting from human knowledge make the
  agent stronger and faster to train than pure self-play? *This has never been
  formally studied for Xiangqi.*

### Agent 3 — MCTS + Neural Network (**AlphaZero-style; strongest, slowest**)
- **Files:** `src/agents/mcts.py` (search core), `src/agents/mcts_agent.py`
  (wrapper).
- **What it does:** For each move, it runs **N simulations** (`MCTS_SIMULATIONS =
  400`) of look-ahead search guided by the neural network, then plays the
  **most-visited** move. It combines a neural net's intuition with explicit search.
- **How, in code:** each simulation has four phases (`_simulate()`):
  1. **Selection** — descend the tree by the **PUCT** score `−Q(child) + c_puct ·
     P(child) · √ΣN / (1 + N(child))` (`_select_child`). The `−Q` is because a
     child's stored value is from the *opponent's* perspective.
  2. **Expansion** — at an unexpanded leaf, ask the network for **priors** over the
     legal moves and create child nodes (`_evaluate`).
  3. **Evaluation** — use the network's **value head** for the leaf (no random
     rollout). A leaf with no legal move is terminal → value `−1` (side to move
     lost).
  4. **Backpropagation** — walk back up, **negating the value at each step**
     (negamax) and incrementing visit counts (`_backpropagate`).
- **What it tests / why it exists:** The other major modern paradigm (AlphaZero).
  Central to **RQ2**, and **Experiment 5** studies how its strength scales with the
  simulation count (strength vs. speed).

**Also present:** `RandomAgent` (`random_agent.py`) — plays a random legal move; the
simplest possible opponent and a sanity baseline. `EngineAgent`
(`engine_agent.py`) — wraps the external ElephantEye engine as a `BaseAgent` so it
can play in the same tournament (see next section).

---

## 4. Why ElephantEye

The agents can play *each other* (a round-robin), but that only tells us who is
best **relative to the group** — if all four are weak, one is still "best of the
weak." To make an **absolute** claim of strength, and to get a reproducible number,
we benchmark against a fixed, strong, external opponent: **ElephantEye**.

There are three concrete reasons, and they are the real logic:

- **A — It's the fixed "measuring stick" for absolute strength.** ElephantEye is
  one of the strongest **open-source** Xiangqi engines and can be set to different
  difficulty levels (search depths). Beating it — or how close you get, and at what
  depth — is a meaningful, absolute statement of an agent's strength.
- **B — It anchors the Elo scale.** Elo is *relative*; it only means something
  against a reference. By playing our agents against ElephantEye at known settings,
  we pin them onto a real Elo scale ("Agent 2 ≈ 1800 Elo") instead of just "Agent 2
  beat Agent 1."
- **C — Reproducibility (the academic reason).** It is free, open-source, and
  deterministic, so a thesis examiner can download the same engine and reproduce
  our numbers. Benchmarking against a human or a closed app would not be
  reproducible.

**How it's wired:** `src/environment/elephanteye.py` speaks the engine's protocol —
note it's **UCCI (Universal *Chinese* Chess Interface)**, *not* the UCI used by
international-chess engines. It converts our board to a Xiangqi **FEN** string
(`board_to_fen`), sends it to the engine subprocess, and reads back the best move
(`bestmove`). `src/agents/engine_agent.py` wraps that as a `BaseAgent`, so the
engine drops straight into the same tournament as our four agents. **Pikafish** is a
stronger, modern alternative that plugs in the same way.

---

## 5. How strength is measured: Elo

**File:** `src/evaluation/elo.py`.

Elo is the chess-style rating scale: the *difference* between two ratings predicts
their expected score. A 400-point gap ⇒ the stronger player is expected to score
~10× as often. Key functions:

- `expected_score(a, b)` — the logistic formula: expected score of A vs B.
- `estimate_rating_from_score(opp_rating, score)` — **Approach A**: invert the
  formula to rate an agent from its score against a *known-rating* opponent (e.g.
  ElephantEye). Score is clamped away from 0/1 so an undefeated record doesn't map
  to infinity.
- `elo_confidence_interval(score, n_games)` — the 95% confidence half-width (via
  the delta method). Roughly ±40–50 Elo at 400 games — which is *why* the thesis
  plans ~400 games per matchup.
- `fit_ratings(pairwise, ...)` — **Approach B**: given all round-robin results,
  iteratively fit a consistent set of Elo ratings for every agent at once, recentred
  on an anchor mean (or on ElephantEye's known rating for absolute anchoring).

---

## 6. End-to-end flow

```
                                RAW HUMAN GAMES
                        (CGLemon .pgn/.pgns, ICCS notation)
                                      │
             ┌────────────────────────┴─ src/data/pipeline.py ─────────────┐
             │  parse (WXF/ICCS) → replay-validate → length-filter →        │
             │  game-level split (train/val/test) → save JSONL + manifest   │
             └────────────────────────┬─────────────────────────────────────┘
                                      │  data/splits/{train,val,test}.jsonl
                                      ▼
        ┌──────────────────── Agent 2 Phase 1: IMITATION LEARNING ───────────┐
        │  src/training/il_train.py → LazyXiangqiILDataset →                  │
        │  supervised policy+value training → il_agent2_phase1.pt             │
        └────────────────────────────────┬───────────────────────────────────┘
                                          │ transfer body
                                          ▼
   ┌── Agent 1: PPO self-play ──┐   ┌── Agent 2 Phase 2: PPO fine-tune ──┐
   │  ppo_train.py (random init) │   │  ppo_train.py --il-checkpoint ...   │
   └─────────────┬───────────────┘   └──────────────┬──────────────────────┘
                 │            ┌── Agent 3: MCTS+NN ──┐   ┌ Agent 4: Minimax ┐
                 │            │  self-play (planned) │   │ (no training)     │
                 └──────┬─────┴───────────┬──────────┴───┬──────────────────┘
                        ▼                 ▼              ▼
              ┌──────────────── EVALUATION (src/evaluation/) ────────────────┐
              │ round-robin + games vs ElephantEye → Elo + head-to-head       │
              └───────────────────────────┬──────────────────────────────────┘
                                          ▼
              experiments/exp1..5  →  CSV tables & curves  →  dashboard/ & thesis
```

---

## 7. System architecture

The code is organized into **four dependency layers** (see
[`docs/02-ARCHITECTURE.md`](docs/02-ARCHITECTURE.md)). The rule: a layer may import
from layers **below** it, never above. This keeps things testable and prevents
circular dependencies.

```
Layer 4  Applications      experiments/ , dashboard/
Layer 3  Agents & Training src/agents/ , src/training/ , src/evaluation/
Layer 2  Learning          src/models/ (neural network) , src/data/ (pipeline)
Layer 1  Core (foundation) src/environment/ (rules, board, encoder, action space)
Layer 0  Shared            src/utils/ (config, seeding, checkpoint, logging, render)
```

Everything rests on **Layer 1**, the game engine. If that is correct, everything
above it can trust it.

---

## 8. Project structure

Below, every folder and its files are explained — what they contain, what each
does, and the key functions/logic inside.

### `src/environment/` — Layer 1: the game engine (the foundation)
Implements Xiangqi itself: the board, the rules, and the interfaces the AI needs.

- **`board.py`** — the `Board` class. Holds a **NumPy 10×9 grid** where each cell is
  a signed integer: a Red piece is `+code`, a Black piece is `−code`, empty is `0`
  (so the *sign encodes ownership*). Key methods: `apply_move()` (move a piece,
  return captured value, flip side to move), `clone()` (deep copy — needed by search
  agents), `find_general()`, `pieces_of(color)`, `position_key()` (a hashable
  snapshot used for repetition detection), `to_ascii()` (debug print). Coordinates:
  `(row, col)`, 0-indexed, **row 0 = Red's back rank at the bottom**.
- **`pieces.py`** — pseudo-legal move generation **per piece type**:
  `general_moves`, `advisor_moves`, `elephant_moves`, `horse_moves`,
  `chariot_moves`, `cannon_moves`, `soldier_moves`, dispatched by `piece_moves()`.
  Encodes each piece's real Xiangqi rules — e.g. the horse's "blocked leg", the
  cannon's jump-to-capture, the elephant not crossing the river, soldiers gaining
  sideways movement after crossing. "Pseudo-legal" = geometry only; it doesn't yet
  check whether the move leaves your own General in check.
- **`move_generator.py`** — `generate_legal_moves(board, color)`: takes the
  pseudo-legal moves and **filters out any move that leaves your own General in
  check** (including the flying-general rule) by trying each move on a clone and
  asking the rules module. This is the single source of truth for "what moves are
  legal here."
- **`rules.py`** — the "referee." `is_in_check()`, `is_checkmate()`,
  `is_stalemate()`, `has_legal_moves()`, `get_game_result()`, and `generals_face()`
  (the **flying-general** rule: two Generals may not face each other on an open
  file). **Xiangqi specifics enforced here:** stalemate is a **LOSS** (not a draw
  like chess), and facing generals is illegal.
- **`action_space.py`** — the **fixed 2086-move encoding**. A neural net's policy
  head outputs one number per possible move, so every move needs a stable integer
  index. `_build_moves()` enumerates all geometrically-possible moves (all
  straight-line moves, 8 horse jumps, the fixed advisor/elephant diagonals) in the
  standard "cchess-zero" order; `INDEX_TO_MOVE` / `MOVE_TO_INDEX` are the two
  lookup tables (built once at import, asserted to equal 2086). `legal_mask(board)`
  returns a boolean array of length 2086 marking which indices are legal now —
  **the mandatory action mask** every policy must apply before choosing.
- **`encoder.py`** — `encode(board)` turns a position into the **14×10×9 float
  tensor** the network reads. **Key design decision (player-relative encoding):**
  channels 0–6 are *the current player's* seven piece types, channels 7–13 are the
  opponent's; and when Black is to move the board is **rotated 180°** so the mover
  always "sits at the bottom and moves upward." This lets **one network play both
  colours**.
- **`xiangqi_env.py`** — `XiangqiEnv`, the **Gymnasium** wrapper (`reset()` /
  `step()`) that RL libraries expect. `step(action)` validates the action against
  the legal mask, applies the move, updates repetition/ply counters, and returns
  `(observation, reward, terminated, truncated, info)`. Reward is **sparse, from the
  mover's perspective** (+1 win, +0.1 draw, else 0); the `info` dict always carries
  `legal_mask`. Detects game end (opponent has no reply → mover wins; repetition or
  300-ply cap → draw).
- **`elephanteye.py`** — the **UCCI engine interface** (see §4): FEN serialization,
  UCCI move ↔ our-move conversion, and the `ElephantEyeEngine` subprocess driver.

### `src/models/` — Layer 2: the neural network (the shared "brain")
The **same network** is used by imitation learning, PPO (as actor+critic), and MCTS
(priors + leaf value).

- **`blocks.py`** — reusable building blocks: `ConvBlock` (Conv3×3 → BatchNorm →
  ReLU; the input conv that lifts 14 channels to width C, padding keeps the 10×9
  size) and `ResidualBlock` (Conv→BN→ReLU→Conv→BN + **skip connection** → ReLU; the
  repeated unit of the tower).
- **`heads.py`** — the two outputs. `PolicyHead`: Conv1×1 → BN → ReLU → flatten →
  Linear → **raw logits over 2086 moves** (logits, not probabilities — masking &
  softmax happen later so illegal moves can be set to −∞ first). `ValueHead`:
  Conv1×1 → BN → ReLU → flatten → Linear → ReLU → Linear → **tanh → scalar in
  [−1, 1]** estimating the outcome from the mover's view.
- **`network.py`** — `PolicyValueNetwork` = input conv → **N residual blocks** →
  two heads. Defaults `C = 64` channels, `N = 10` blocks (~1.15M parameters).
  `forward(x)` returns `(policy_logits, value)`.
- **`sb3_extractor.py`** — `XiangqiResNetExtractor`: the same body (conv + residual
  tower) packaged as a **stable-baselines3 features extractor**, so MaskablePPO
  learns on the *same* convolutional representation (not a generic MLP). This is the
  piece the imitation-learning body is transferred *into* for Agent 2 Phase 2.

### `src/data/` — Layer 2: the data pipeline
Turns raw human game files into training-ready data.

- **`wxf_parser.py`** — parses move notation into absolute moves and replays games.
  `parse_wxf_move()` handles **Roman WXF** relative notation (e.g. `C2.5`, front/rear
  disambiguation, mover-relative file numbering). `parse_iccs_move()` handles
  **coordinate** notation in both UCCI (`h2e2`) and xqbase-ICCS (`C3-C4`, uppercase +
  dash) dialects. `detect_notation()` auto-picks; `parse_game()` replays a token list
  from the start position and (optionally) **replay-validates** every move against
  the legal-move generator, raising on any illegal move.
- **`pgn_adapter.py`** — converts standard PGN text into `(result, tokens)` records:
  `split_pgn_games`, `parse_tags`, `extract_move_tokens` (strips move numbers,
  comments, NAGs, result tokens, and the `...` placeholder), `pgn_to_records`,
  `load_games_from_pgn`.
- **`game_loader.py`** — `parse_game_records()` (validate + length-filter a batch of
  records into `Game` objects, tracking a `LoadStats` count of valid/invalid/short)
  and `parse_result()` (maps `"1-0"` / `"red"` / `"1/2-1/2"` → RED/BLACK/DRAW).
- **`dataset.py`** — the PyTorch datasets. `XiangqiILDataset` (**eager**: encodes
  every position into RAM up front — only viable for small data) and
  **`LazyXiangqiILDataset`** (encodes each position **on access** by replaying the
  game to that ply; O(games) memory, uses a prefix-sum `bisect` index). Both yield
  `(tensor, action_index, value)` and optionally add a **left-right mirror**
  augmentation (Xiangqi is left-right symmetric). `_value_for()` gives the outcome
  from that ply's mover's perspective.
- **`data_loader.py`** — `split_games()` (**game-level** train/val/test split — never
  split by position, or data leaks between sets; seeded for reproducibility) and
  `make_dataloader()` (wraps a dataset in a torch `DataLoader`).
- **`pipeline.py`** — the **orchestrator** that ties it all together. `build_dataset()`
  scans `data/raw/` recursively (`find_raw_files`), parses every game
  (`load_raw_records`), applies length filters, splits by game, computes statistics
  (`DatasetStats`: outcome distribution, avg length, per-filter counts) and — for a
  large corpus — persists each split as JSONL (`save_split_jsonl` /
  `load_split_jsonl`) **without materializing tensors**. Because full replay-
  validation of 141k games would take ~22h, it defaults to **not** validating and
  instead **spot-checks** a random sample for a legality rate (`spot_check_validity`).
  CLI: `python -m src.data.pipeline`.

### `src/agents/` — Layer 3: the players
All subclass `BaseAgent`. Covered individually in [§3](#3-the-four-agents):
`base_agent.py` (abstract interface), `minimax_agent.py`, `ppo_agent.py`,
`neural_agent.py` (plays a network's policy head greedily/sampled — used to
evaluate the imitation network), `mcts.py` + `mcts_agent.py`, `random_agent.py`,
`engine_agent.py`.

### `src/training/` — Layer 3: the learning procedures
- **`self_play_env.py`** — `SelfPlayEnv`: presents two-player Xiangqi as a
  single-agent env for PPO by baking the opponent in (see Agent 1).
- **`ppo_train.py`** — builds and trains **MaskablePPO**. `build_model()` wires the
  ResNet extractor and PPO hyperparameters; `transfer_il_weights()` copies an
  imitation network's body into the extractor (the **Agent 2 Phase 2 bridge**);
  `train(..., il_checkpoint=...)` runs it, optionally starting from an IL checkpoint.
- **`imitation.py`** — supervised training of the network on human games.
  `run_epoch()` (policy cross-entropy + value MSE), `evaluate()` (top-1
  move-prediction accuracy on validation), `train_imitation()` (the loop, with an
  `on_epoch_end` checkpoint callback and `start_epoch` for resuming),
  `save_checkpoint()` / `load_network()`.
- **`il_train.py`** — the **entry point** for Agent 2 Phase 1: loads the JSONL
  splits → `LazyXiangqiILDataset` → DataLoaders → `train_imitation` → checkpoint.
  `train_from_splits()` writes a checkpoint **every epoch** and supports `--resume`
  (important because Colab disconnects). CLI: `python -m src.training.il_train`.
- **`callbacks.py`** — `EloEvalCallback`: an SB3 callback that periodically measures
  the training agent's Elo vs a reference during PPO, for learning-curve data.

### `src/evaluation/` — Layer 3: measuring strength
- **`elo.py`** — the Elo math (see [§5](#5-how-strength-is-measured-elo)).
- **`metrics.py`** — `GameOutcome` (one game's WIN/LOSS/DRAW from A's view) and
  `MatchStats` (aggregate: score, win/draw/loss rates, decisiveness, avg length,
  win-rate as Red vs Black); `summarize()` computes them.
- **`tournament.py`** — runs games. `play_game()` (one game to completion),
  `play_match()` (N games, **alternating colours** to cancel Red's first-move
  advantage), `estimate_elo()` (rate an agent vs a known-rating opponent),
  `round_robin()` (everyone plays everyone). This is where an engine wrapped as a
  `BaseAgent` plugs in for absolute Elo.
- **`opening_analysis.py`** — RQ3 tools. `opening_signature()` (first N plies as a
  key), `build_distribution()` (→ probabilities), `kl_divergence()` (compare an
  agent's opening distribution to the professional one — small KL = plays "book"),
  `classify_opening()` (name Red's first move: central cannon, flying elephant, …),
  and helpers to play and collect games.

### `src/utils/` — Layer 0: shared helpers
- **`config.py`** — **every fixed constant** (board size, action-space size, piece
  codes/values, palace/river boundaries, rewards, and all hyperparameter defaults).
  Logic modules import from here instead of hardcoding numbers.
- **`seeding.py`** — `set_seed()` for reproducible runs.
- **`checkpoint.py`** — generic `save_checkpoint` / `load_checkpoint` (weights +
  optimizer + step + metadata) for resumable training.
- **`logging_utils.py`** — a `Logger` that logs to **Weights & Biases** and/or
  **TensorBoard** if installed, and **degrades gracefully** (in-memory) if not; plus
  `git_commit()` to stamp runs with the code version.
- **`render.py`** — `board_to_plotly()`: draws a board as an interactive Plotly
  figure for the dashboard.

### `experiments/` — Layer 4: the five thesis experiments
Each is a script that runs one experiment and writes CSV/figures. `exp1_main_comparison.py`
(round-robin → fitted Elo + head-to-head tables), `exp2_training_efficiency.py`
(learning curves: Elo vs games, IL+RL vs pure RL), `exp3_opening_theory.py`
(opening distributions + KL vs professional play), `exp4_curriculum.py` (curriculum
vs plain self-play), `exp5_mcts_ablation.py` (MCTS strength vs simulation count).
*(These are coded and unit-tested but not yet run on trained agents.)*

### `dashboard/` — Layer 4: the Streamlit web app
`app.py` (entry) plus five pages (`01_live_game`, `02_elo_comparison`,
`03_learning_curves`, `04_opening_analysis`, `05_game_explorer`) and reusable
`components/` (results I/O, game playback, agent pickers). It visualizes results and
lets you watch agents play. Boots locally today; public deployment is planned.

### `docs/` — the formal specifications
13 markdown documents (game rules, architecture, environment, data pipeline, neural
network, agents, training, evaluation, experiments, dashboard, coding standards,
glossary) + a master index. These are the authoritative design specs; this guide
summarizes and connects them.

### `notebooks/` — `xiangqi_colab_training.ipynb`
The Colab notebook: GPU check → clone/mount → install → download data → build
splits → IL → PPO Phase 2 → PPO Agent 1, checkpointing to Google Drive so a
disconnect never loses progress.

### `tests/` — the test suite (221 tests, all passing)
Mirrors `src/` (`test_environment`, `test_models`, `test_agents`, `test_data`,
`test_training`, `test_evaluation`, `test_experiments`, `test_dashboard`,
`test_utils`). Run `python -m pytest -q` from the repo root. These are what let us
trust the code before any (expensive) training.

### Root files
`README.md` (short entry point), this `PROJECT_GUIDE.md`, `requirements.txt` /
`environment.yml` (dependencies), `LICENSE` (MIT), `configs/` (per-experiment YAML,
currently empty), `data/` (gitignored raw/splits; only the tiny
`dataset_manifest.json` is committed), `results/` (checkpoints/outputs, gitignored).

---

## 9. Key design decisions

Each major decision and the reason behind it:

- **Custom environment instead of `gym-xiangqi`.** The existing package is built on
  the *legacy* Gym API and didn't fit our action-masking and dependency needs.
  Writing our own (`src/environment/`) gives full control and a clean Gymnasium
  interface — and doubles as a learning artifact for the thesis.
- **Player-relative 14×10×9 encoding (rotate for Black).** So **one** network plays
  both colours ("me at the bottom vs. opponent at the top"), halving what must be
  learned. *(encoder.py)*
- **Fixed 2086-move action space + mandatory legal mask.** A policy head needs a
  fixed output size; masking illegal moves to −∞ before softmax means an agent can
  **never** play an illegal move — the #2 risk flagged in the proposal, solved by
  `MaskablePPO` + `legal_mask`.
- **One shared Policy-Value network for all learning agents.** IL, PPO, and MCTS all
  use the same architecture, so weights transfer between phases (the IL→PPO bridge)
  and results are comparable.
- **Stalemate = loss; flying-general illegal.** These are real Xiangqi rules that
  differ from chess; getting them wrong would silently corrupt everything. They're
  enforced in `rules.py` and covered by tests.
- **Lazy dataset + skip full validation for the big corpus.** 141k games = ~11.6M
  positions. Materializing all tensors would need tens of GB, and full replay-
  validation would take ~22h. So we persist splits as JSONL, encode on-the-fly, and
  spot-check legality instead. *(pipeline.py, dataset.py)*
- **Checkpoint every epoch + resume.** Colab disconnects mid-run; per-epoch
  checkpoints to Google Drive make training survivable. *(il_train.py)*
- **Colour-balanced matches + confidence intervals.** Red moves first (an
  advantage), so `play_match` alternates colours; Elo comes with a 95% CI so claims
  are statistically honest. *(tournament.py, elo.py)*
- **ElephantEye via UCCI, wrapped as a BaseAgent.** Absolute, reproducible Elo
  anchor that plugs into the same tournament code (see [§4](#4-why-elephanteye)).
- **Everything imports constants from `config.py`.** No "magic numbers" scattered in
  logic — one place to read and change the fixed values.

---

## 10. Status

The single most important thing to understand: **all the CODE is built and tested,
but no agent has actually been TRAINED yet.** That line — "implemented" vs
"trained" — is where the project currently sits.

### ✅ Done (code & foundation — 221 tests passing)
- Full game engine (board, pieces, move generation, rules) — Layer 1.
- Board→tensor encoder + 2086-move action space + legal masking.
- Gymnasium environment + single-agent self-play wrapper.
- Policy-Value network (ResNet body + policy/value heads) + SB3 extractor.
- All four agents (Minimax, PPO, IL/Neural, MCTS) + Random + Engine wrapper.
- Evaluation stack: Elo math, match/metrics, tournament/round-robin, opening
  analysis.
- **Data pipeline + a real 141k-game dataset built and split** (train 113,084 /
  val 14,135 / test 14,137 games; ~11.6M training positions; spot-checked 100%
  legal). Manifest committed.
- Lazy dataset + IL training entry point + IL→PPO transfer bridge +
  per-epoch checkpoint/resume.
- All 5 experiment scripts (coded + unit-tested).
- Streamlit dashboard (boots locally).
- Colab training notebook; repo public with README and this guide.

### ⏳ Not done (the execution phase — needs a GPU)
- **Training runs:** no agent is trained yet — IL (Agent 2 P1), PPO fine-tune
  (Agent 2 P2), PPO self-play (Agent 1), MCTS (Agent 3).
- **ElephantEye binary:** the *interface* exists; the engine still needs to be
  compiled and pointed at for absolute Elo.
- **MCTS training script:** the search + agent exist; a dedicated self-play training
  entry point (like `il_train`/`ppo_train`) is not finalized.
- **Experiment results:** scripts exist but haven't been run on trained agents, so
  there are **no Elo tables / learning curves / opening analyses yet**.
- **Deliverables that depend on the above:** trained model weights (#1),
  experimental results (#2), deployed dashboard (#3).
- **Thesis document (#4):** the *proposal* is done; the 5-chapter thesis is not
  written.

### Mapping to the proposal's 6-phase timeline
- **Phase 1 (Foundation)** → ✅ complete (env + dataset + Minimax).
- **Phases 2–3 (train the agents)** → code ready, **training not yet run ← you are
  here**.
- **Phase 4 (experiments + dashboard)** → dashboard ✅; experiments await trained
  agents.
- **Phases 5–6 (writing + revision)** → not started (only the proposal exists).

---

## 11. The plan

The critical path from here (most of it is *running* things on a GPU, not writing
code):

1. **Train Agent 2 — Imitation Learning (Phase 1)** on Colab (~6–8 h on a T4),
   checkpointing to Drive → `il_agent2_phase1.pt`.
2. **Train Agent 2 — PPO fine-tune (Phase 2)** from that checkpoint.
3. **Train Agent 1 — PPO self-play** from scratch (~15–20 h; split across sessions).
4. **Train Agent 3 — MCTS+NN** (most compute-heavy; may be de-scoped per the
   fallback plan if too slow).
5. **Compile ElephantEye** and wire `EngineAgent` for absolute Elo.
6. **Run Experiments 1–5** → Elo tables, learning curves, opening analysis.
7. **Write the thesis** (5 chapters) using those results; deploy the dashboard.

**Fallback plan (from the proposal), if time/compute runs short:** drop Agent 3 →
drop Agent 2 → fine-tune an existing net → minimum viable thesis (one PPO agent +
Minimax baseline + Elo vs ElephantEye). The non-negotiable core is *one RL agent +
one baseline + standardized Elo evaluation*.

---

## 12. Requirements

- **Python 3.10+** (project target; local dev may use 3.14, but training libs prefer
  3.10). Install via `conda env create -f environment.yml` **or**
  `pip install -r requirements.txt`.
- **Core libraries:** PyTorch, gymnasium, stable-baselines3 + sb3-contrib
  (MaskablePPO), NumPy, pandas. Optional: wandb / TensorBoard (logging degrades
  gracefully without them), Streamlit + Plotly (dashboard).
- **A GPU** for training — practically, **Google Colab** (the CPU is fine for the
  code, tests, and building the dataset, but not for training over millions of
  positions).
- **The dataset:** raw game files go in `data/raw/` (the CGLemon dpxq/WXF `.pgns`
  files), then `python -m src.data.pipeline` builds the splits.
- **(Optional) ElephantEye/Pikafish** compiled binary for absolute-Elo evaluation.

Run the tests any time with `python -m pytest -q` from the repo root.

---

*This guide reflects the codebase as implemented. For authoritative per-topic detail
see [`docs/`](docs/); for the current build state, see the project memory / commit
history.*
