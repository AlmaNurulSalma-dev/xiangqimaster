# XiangqiMaster — Complete Project Guide

> A detailed, self-contained learning resource for the whole project: what it is,
> why it exists, the game it plays, the AI concepts behind it, how **every** part
> of the code works, worked examples that trace real data through the system, what
> is done, and what remains.
>
> **This is the long, teach-me-everything document.** For a 1-minute overview see
> [`README.md`](README.md); for the formal specs see [`docs/`](docs/).
>
> **How to read it:**
> - *New to the whole thing?* Read §1 → §2 → §3 (Xiangqi) → §4 (AI concepts) top to
>   bottom, then skim the rest.
> - *Know the game, want the code?* Jump to §7 (architecture) and §8 (every file).
> - *Preparing for the thesis defense?* §4, §9 (worked examples), §12
>   (experiments), and §15 (design decisions + rationale) are the meat.

---

## Table of Contents

1. [What this project is (plain words)](#1-what-this-project-is)
2. [The research questions](#2-the-research-questions)
3. [A Xiangqi primer — the game itself](#3-a-xiangqi-primer)
4. [The AI concepts, explained from zero](#4-the-ai-concepts-explained-from-zero)
5. [The four agents — what each is and what it tests](#5-the-four-agents)
6. [Why we benchmark against ElephantEye + how Elo works](#6-elephanteye-and-elo)
7. [System architecture (the layers)](#7-system-architecture)
8. [Project structure — every folder and file](#8-project-structure)
9. [Worked examples (trace real data through the system)](#9-worked-examples)
10. [The data pipeline in depth](#10-the-data-pipeline-in-depth)
11. [The neural network in depth](#11-the-neural-network-in-depth)
12. [Training in depth](#12-training-in-depth)
13. [Evaluation in depth](#13-evaluation-in-depth)
14. [The five experiments](#14-the-five-experiments)
15. [Key design decisions and why](#15-key-design-decisions)
16. [Testing & reproducibility](#16-testing--reproducibility)
17. [Status: done vs. remaining](#17-status)
18. [The plan going forward](#18-the-plan)
19. [Requirements & command cheat-sheet](#19-requirements--commands)
20. [Glossary (plain language)](#20-glossary)
21. [FAQ](#21-faq)
22. ["Where do I look for X?" index](#22-where-do-i-look-for-x)

---

## 1. What this project is

**XiangqiMaster** is a **bachelor's-thesis research project** (dual degree, UII ×
NXU) that studies **how different deep-reinforcement-learning (RL) methods learn to
play Xiangqi (Chinese Chess), and which method works best.**

Xiangqi is one of the most-played board games on Earth (~600 million players), yet —
unlike Go, Chess, and Shogi, which got AlphaGo/AlphaZero — **nobody has published a
systematic, apples-to-apples comparison of modern deep-RL methods on Xiangqi.** This
project fills that gap.

Concretely it **builds four game-playing agents**, each using a different approach,
**trains them, and measures each one's playing strength on a common scale (Elo)** so
they can be compared fairly. It also asks two deeper questions: does *learning from
human games first* make an agent stronger/faster than *learning purely from
self-play*? And does a self-taught agent *rediscover* classical Xiangqi opening
theory?

**In one sentence:** *design, train, and scientifically compare four RL approaches to
Xiangqi.*

Why it's a legitimate research contribution:
- **No prior systematic comparison** of PPO / imitation-learning / MCTS+NN /
  Minimax on Xiangqi under one standardized Elo framework.
- **Imitation-learning transfer is unvalidated for Xiangqi** — AlphaGo showed it
  helps in Go, but Xiangqi has unique mechanics (the Cannon's jump-capture, the
  River, the Palace) that could behave differently.
- **Emergent opening theory** — whether self-play rediscovers openings like the
  Central Cannon (当头炮) is genuinely open and interesting.

---

## 2. The research questions

Everything is organized to answer these (they map to `experiments/`):

| # | Question | Experiment(s) |
|---|----------|---------------|
| **RQ1** | Does imitation learning (pre-train on humans) beat pure self-play, in final strength **and** convergence speed? | Exp 1, **Exp 2** |
| **RQ2** | Which RL method is strongest for Xiangqi (PPO vs MCTS+NN vs Minimax)? | Exp 1 |
| **RQ3** | Does self-play rediscover classical opening theory? | Exp 3 |
| **RQ4** | Does a curriculum of stronger opponents beat plain self-play? How does MCTS strength trade off vs. its simulation count? | Exp 4, Exp 5 |

---

## 3. A Xiangqi primer

You cannot fully understand the code without understanding the game, because the
environment (`src/environment/`) *is* the rules. Here is exactly what the code
implements.

### 3.1 The board
- **10 rows × 9 columns = 90 intersections** (pieces sit on line intersections, not
  in squares). In code: `BOARD_ROWS = 10`, `BOARD_COLS = 9`.
- Coordinates are `(row, col)`, 0-indexed. **Row 0 is Red's back rank at the
  bottom; row 9 is Black's back rank at the top.** Column 0 is leftmost.
- **The River** runs between row 4 and row 5. Red's half is rows 0–4; Black's is
  rows 5–9.
- **The Palace** is a 3×3 box (columns 3–5) on each side (Red rows 0–2, Black rows
  7–9). The General and Advisors can never leave it.

### 3.2 The pieces and their exact movement rules
Each side has 16 pieces of 7 types. In code, a piece is a **signed integer**: Red =
`+type`, Black = `−type`, empty = `0`. The type codes (`config.py`) and values
(used by the Minimax evaluation) are:

| Piece | Code | Value | How it moves (as implemented in `pieces.py`) |
|-------|------|-------|-----------------------------------------------|
| **General (将/帅)** | 1 | 10000 | One orthogonal step, **confined to the Palace**. Plus the "flying general" rule (below). |
| **Advisor (士/仕)** | 2 | 2 | One **diagonal** step, confined to the Palace. |
| **Elephant (象/相)** | 3 | 2 | Exactly **two points diagonally**; **cannot cross the River**; blocked if the midpoint ("eye", 塞象眼) is occupied. |
| **Horse (马)** | 4 | 4 | L-shape (like a knight), **but blocked at the "leg"** (蹩马腿): the orthogonal point it steps through must be empty. |
| **Chariot (车)** | 5 | 9 | Slides orthogonally any distance until it hits a piece; captures an enemy, blocked by a friend. (Like a chess rook — the strongest piece.) |
| **Cannon (炮)** | 6 | 4.5 | Moves like a Chariot over **empty** points, but **captures only by jumping exactly one "screen" piece** and landing on an enemy beyond it. |
| **Soldier (兵/卒)** | 7 | 1 | One point **forward** only; **after crossing the River** it may also move sideways; **never backward**, never diagonal. |

The `general_moves`, `advisor_moves`, … functions each return the *pseudo-legal*
destinations for one piece (geometry only). "Pseudo-legal" = doesn't yet check
whether the move leaves your own General in check; that filtering is done later.

### 3.3 Special rules the code enforces (and where)
- **Check** (`rules.is_in_check`): your General is under attack if any enemy piece
  can reach its point.
- **Flying General / facing generals** (`rules.generals_face`): the two Generals may
  **not** face each other down an open file with nothing between them. This is
  treated as check, so any move creating it is illegal.
- **Checkmate = loss, Stalemate = loss** (`rules.is_checkmate`, `is_stalemate`):
  **This is the biggest difference from chess.** In chess, stalemate (no legal move
  but not in check) is a draw; **in Xiangqi it is a loss.** So the referee only needs
  to ask "does the side to move have any legal move?" — if not, that side loses,
  whether it's checkmate or stalemate.
- **Draws:** threefold repetition (`REPETITION_LIMIT = 3`) or the move cap
  (`MAX_PLIES = 300`). *(Note: real tournament "perpetual check/chase" rules are
  more nuanced; this project uses the simpler repetition + ply-cap draw model.)*
- **Legal move generation** (`move_generator.generate_legal_moves`): take all
  pseudo-legal moves, and for each, play it on a *clone* and discard it if it leaves
  your own General in check. What survives is the legal move list — the single
  source of truth.

### 3.4 The starting position (what `Board.reset()` builds)
Standard opening setup, Red to move:
- Row 0 (Red back rank): Chariot, Horse, Elephant, Advisor, **General**, Advisor,
  Elephant, Horse, Chariot. Row 9 mirrors it for Black.
- Cannons: row 2, columns 1 and 7 (Red); row 7 for Black.
- Soldiers: row 3, columns 0, 2, 4, 6, 8 (Red); row 6 for Black.

From this position there are **exactly 44 legal moves for Red** — a number the tests
use as a correctness anchor.

---

## 4. The AI concepts, explained from zero

This section explains the ideas the four agents rely on. No prior RL knowledge
assumed.

### 4.1 The game as a "Markov Decision Process" (MDP)
RL frames a game as: an **agent** observes a **state**, picks an **action**, and gets
a **reward** and a **new state**. In our project:
- **State** = the board position (encoded as a tensor — see §11).
- **Action** = one of 2086 possible moves (an integer index).
- **Reward** = sparse and only at the end: **+1** for winning, **−1** for losing,
  **+0.1** for a draw; **0** on every non-terminal move. (Draws get a small positive
  value because draws are relatively rare in Xiangqi and better than losing.)
- **Episode** = one full game.
- **Goal** = learn a **policy** (a rule for choosing actions) that maximizes expected
  reward — i.e., wins more.

A **ply** = one half-move (one side moving once). "300 plies" = 150 full moves each.

### 4.2 The neural network: a policy + a value in one
Instead of a lookup table (impossible — there are astronomically many positions), we
use a **neural network** that *generalizes* from positions it has seen to ones it
hasn't. Ours is a **Policy-Value network** with two outputs:
- **Policy head** → a score for each of the 2086 moves ("which moves look good?").
- **Value head** → one number in [−1, 1] ("am I winning from here?").

The same network is the shared "brain" used by imitation learning, PPO, and MCTS.
Details in §11.

### 4.3 Imitation Learning (IL) = "copy the experts" (supervised learning)
Show the network **millions of positions from real human games** and train it to:
- **predict the move the human played** (a classification problem → **cross-entropy
  loss** on the policy head), and
- **predict who eventually won** (a regression problem → **mean-squared-error loss**
  on the value head).

This is ordinary supervised learning — no game-playing during training, just
"here's a position, here's the right answer." It gives the network a strong starting
point (it already plays like a decent human) before any RL. **This is the crux of
RQ1.**

### 4.4 PPO (Proximal Policy Optimization) = "learn by playing"
PPO is a **policy-gradient RL algorithm**. The agent plays games, and for each move
it estimates an **advantage** (how much better that move turned out than expected).
Moves with positive advantage are made more likely; negative, less likely. Key ideas,
in plain terms:
- **Advantage / GAE (Generalized Advantage Estimation):** a smoothed estimate of "was
  this move better than the value head predicted?" (`PPO_GAE_LAMBDA = 0.95`,
  `PPO_GAMMA = 0.99` — the discount that makes near-term rewards matter more).
- **Clipped surrogate objective:** PPO's signature trick — it limits how far the
  policy can change in one update (`PPO_CLIP_RANGE = 0.2`), which keeps training
  stable (no wild swings).
- **Entropy bonus** (`PPO_ENT_COEF = 0.01`): rewards keeping some randomness, so the
  agent keeps exploring instead of prematurely committing to one plan.
- **Value loss** (`PPO_VF_COEF = 0.5`): trains the value head alongside the policy.

We use **MaskablePPO** (from `sb3-contrib`) so illegal moves are masked out before the
agent chooses — essential with 2086 candidate actions.

### 4.5 MCTS (Monte Carlo Tree Search) + network = "think ahead, guided by intuition"
Instead of trusting the policy in one shot, MCTS **builds a search tree** by running
many **simulations** (`MCTS_SIMULATIONS = 400`). Each simulation:
1. **Selection** — walk down the tree, at each node choosing the child with the best
   **PUCT** score, which balances *"this move looked good so far"* (exploitation)
   against *"we haven't tried this much"* (exploration):

   `PUCT(child) = −Q(child) + c_puct · P(child) · √(ΣN) / (1 + N(child))`

   where `Q` = the child's average value (negated because it's from the opponent's
   view), `P` = the network's prior for that move, `N` = visit counts, and `c_puct =
   1.5` tunes the exploration strength.
2. **Expansion** — at a new leaf, ask the network for priors over the legal moves and
   create child nodes.
3. **Evaluation** — use the network's **value head** to score the leaf (AlphaZero
   replaced slow random rollouts with this).
4. **Backpropagation** — send that value back up the path, **negating it at each
   level** (what's good for me is bad for my opponent).

After all simulations, play the **most-visited** move (more robust than the raw
value). More simulations = stronger but slower — exactly what Experiment 5 measures.

### 4.6 Minimax = the classical, no-learning approach
Search the game tree assuming both sides play optimally, scoring leaf positions with
a **handcrafted material count**. It doesn't learn; it's the baseline the learners
must beat. Details in §5.

---

## 5. The four agents

All four implement the same interface — `BaseAgent.select_move(board, legal_mask) ->
action_index` (`src/agents/base_agent.py`) — so the tournament and dashboard treat
them identically. What differs is *how each chooses a move*.

### Agent 4 — Minimax (classical baseline; **no learning**)
- **File:** `src/agents/minimax_agent.py`.
- **Mechanism:** alpha-beta game-tree search to depth `MINIMAX_DEPTH = 4`, in the
  **negamax** formulation (`_negamax`) — one recursive routine that always scores
  from the side-to-move's view and negates for the opponent. Leaf positions are
  scored by **material balance** (`_evaluate`: sum of `PIECE_VALUES`, mine minus
  opponent's). Moves are **ordered captures-first** (`_ordered_moves`) so alpha-beta
  pruning (`if alpha >= beta: break`) skips large parts of the tree. A position with
  no legal move = a loss, scored near `±MATE_SCORE` with a depth term so quicker mates
  are preferred.
- **What it tests / why it exists:** the **reference floor** — "how much better are
  the learning agents than 30-year-old classical search?" It also **doubles as an
  environment correctness test**: if Minimax plays legal, sensible Xiangqi, move
  generation and rules are sound.

### Agent 1 — PPO self-play (**learns from scratch**)
- **Files:** `src/agents/ppo_agent.py` (play-time wrapper), trained by
  `src/training/ppo_train.py` on `src/training/self_play_env.py`.
- **Mechanism:** starts from random weights and improves by playing many games via
  PPO (§4.4). `SelfPlayEnv` turns two-player Xiangqi into the single-agent env SB3
  expects by **baking the opponent in**: on each `step()` the learner moves, the
  opponent immediately replies, and the returned observation is again the learner's
  turn. Reward is **sparse, from the learner's perspective**. Opponent starts as a
  `RandomAgent` (simplest bootstrap) and can be swapped for a frozen self-copy.
- **What it tests:** the "pure RL, no human knowledge" data point (AlphaZero
  philosophy). Central to **RQ1** and **RQ2**.

### Agent 2 — Imitation Learning + PPO (**learn from humans, then self-play**)
- **Files:** Phase 1 (imitation) `src/training/imitation.py` +
  `src/training/il_train.py`; the trained network plays via
  `src/agents/neural_agent.py`. Phase 2 (RL) reuses `ppo_train.py`.
- **Mechanism — two phases:**
  - **Phase 1 (Imitation):** train the network on 141k human games to predict human
    moves (cross-entropy) and outcomes (MSE). See §4.3.
  - **Phase 2 (PPO fine-tune):** transfer that network's **shared body** into the PPO
    features extractor (`ppo_train.transfer_il_weights`) and refine with self-play.
    (SB3's own policy/value heads start fresh; only the convolutional body transfers.)
- **What it tests:** the **headline question (RQ1)** — does human pre-training make
  the agent stronger/faster than pure self-play? Never formally studied for Xiangqi.

### Agent 3 — MCTS + Neural Network (**AlphaZero-style; strongest, slowest**)
- **Files:** `src/agents/mcts.py` (search core), `src/agents/mcts_agent.py` (wrapper).
- **Mechanism:** for each move, run `MCTS_SIMULATIONS = 400` simulations of
  network-guided PUCT search (§4.5) and play the most-visited move. Uses `Board.clone()`
  to explore without touching the real position.
- **What it tests:** the other modern paradigm (AlphaZero). Central to **RQ2**;
  **Experiment 5** studies strength vs. simulation count.

**Also present:** `RandomAgent` (random legal move — the simplest opponent/sanity
baseline) and `EngineAgent` (wraps ElephantEye as a `BaseAgent`).

---

## 6. ElephantEye and Elo

### 6.1 Why benchmark against ElephantEye
The agents can play *each other* (round-robin), but that only ranks them **relative**
to the group. To make an **absolute** claim, and a **reproducible** one, we benchmark
against a fixed strong external opponent — **ElephantEye**:
- **A — absolute measuring stick:** one of the strongest **open-source** Xiangqi
  engines, adjustable by search depth; beating it (and at what depth) is a real,
  absolute statement of strength.
- **B — anchors the Elo scale:** Elo is relative; playing a known-rating opponent
  pins our agents onto a real scale ("≈1800 Elo") rather than "beat Agent 1."
- **C — reproducibility (the academic reason):** free, open-source, deterministic —
  an examiner can download it and reproduce the numbers. A human or closed app could
  not be reproduced.

**How it's wired:** `src/environment/elephanteye.py` speaks **UCCI (Universal
*Chinese* Chess Interface)** — *not* the UCI used by international-chess engines. It
serializes the board to a Xiangqi **FEN** (`board_to_fen`), sends it to the engine
subprocess, and reads the best move back (`bestmove`). `src/agents/engine_agent.py`
wraps that as a `BaseAgent`, so the engine drops straight into the same tournament.
**Pikafish** (stronger, modern) plugs in identically.

### 6.2 How Elo works (`src/evaluation/elo.py`)
Elo turns "win/lose/draw" records into a single skill number. A 400-point gap ⇒ the
stronger player is expected to score ~10× as often.
- `expected_score(a, b) = 1 / (1 + 10^((b−a)/400))` — predicted score of A vs B.
- **Approach A** (`estimate_rating_from_score`): invert the formula to rate an agent
  from its score vs a known-rating opponent — `rating = opp + 400·log10(s/(1−s))`.
- **Approach B** (`fit_ratings`): given all round-robin results, iteratively fit a
  consistent rating for every agent at once, recentred on an anchor.
- `elo_confidence_interval(score, n_games)`: 95% CI half-width (delta method) —
  roughly ±40–50 Elo at 400 games, which is *why* matches use ~400 games.

---

## 7. System architecture

Code is organized into **dependency layers** ([`docs/02-ARCHITECTURE.md`](docs/02-ARCHITECTURE.md)).
The rule: a layer may import from layers **below**, never above. This prevents
circular dependencies and keeps each layer independently testable.

```
Layer 4  Applications      experiments/ , dashboard/
Layer 3  Agents & Training src/agents/ , src/training/ , src/evaluation/
Layer 2  Learning          src/models/ (network) , src/data/ (pipeline)
Layer 1  Core (foundation) src/environment/ (rules, board, encoder, action space)
Layer 0  Shared            src/utils/ (config, seeding, checkpoint, logging, render)
```

Everything rests on **Layer 1**, the game engine. If it's correct, everything above
can trust it — which is why the environment is the most heavily tested layer.

A subtle but important detail: the **encoder** lives in `environment/` (Layer 1), not
`data/`, even though it "feels" like data code. Reason: the environment must emit
observations, and the dependency rule forbids `environment` importing `data`. The
data pipeline imports the *same* encoder, guaranteeing training data and self-play
observations are byte-for-byte identical.

---

## 8. Project structure

Every folder and file, what it contains, and the key functions/logic.

### `src/environment/` — Layer 1: the game engine
- **`board.py`** — `Board`: a **NumPy 10×9 grid** of signed ints (sign = owner).
  Methods: `reset()` (standard start), `apply_move()` (move a piece, capture the
  destination, flip side to move, return the captured code — *no legality check
  here*), `clone()` (independent deep copy — essential for search/legality tests),
  `find_general()`, `pieces_of(color)`, `position_key()` (hashable snapshot for
  repetition detection: `grid.tobytes()` + side to move), `to_ascii()`.
- **`pieces.py`** — one function per piece type (`general_moves`, `advisor_moves`,
  `elephant_moves`, `horse_moves`, `chariot_moves`, `cannon_moves`, `soldier_moves`),
  dispatched by `piece_moves()`. Encodes the exact rules from §3.2 (horse leg-block,
  cannon screen, elephant eye + no river crossing, palace confinement, soldier
  forward/sideways). Returns **pseudo-legal** destinations.
- **`move_generator.py`** — `generate_legal_moves(board, color)`: filters pseudo-legal
  moves, discarding any that leave your own General in check (tries each on a clone).
  The single source of truth for legality.
- **`rules.py`** — the referee: `is_in_check`, `is_checkmate`, `is_stalemate`,
  `has_legal_moves`, `get_game_result`, `generals_face`. Enforces stalemate-is-a-loss
  and the flying-general rule.
- **`action_space.py`** — the **fixed 2086-move map**. `_build_moves()` enumerates
  every geometrically possible move (all straight-line moves per point + 8 horse
  jumps + the 16 fixed advisor diagonals + 32 fixed elephant diagonals) in the
  "cchess-zero" order; `INDEX_TO_MOVE`/`MOVE_TO_INDEX` are the lookup tables (built
  once, asserted == 2086). `legal_mask(board)` → boolean array of length 2086 marking
  legal indices — the mandatory mask.
- **`encoder.py`** — `encode(board)` → the **14×10×9 tensor** (see §11.1), with the
  **player-relative** convention (rotate 180° for Black so the mover always "moves
  upward"; `_orient`).
- **`xiangqi_env.py`** — `XiangqiEnv`, the **Gymnasium** env. `step(action)` validates
  against the legal mask, applies the move, updates repetition/ply counters, and
  returns `(obs, reward, terminated, truncated, info)`; `info` always carries
  `legal_mask`. `clone()` deep-copies (for MCTS).
- **`elephanteye.py`** — the **UCCI** engine interface: `board_to_fen`, `move_to_ucci`
  / `ucci_to_move`, and the `ElephantEyeEngine` subprocess driver (`start`,
  `bestmove`, `close`; usable as a context manager).

### `src/models/` — Layer 2: the neural network
- **`blocks.py`** — `ConvBlock` (Conv3×3→BN→ReLU; the 14→C input conv, padding keeps
  10×9) and `ResidualBlock` (Conv→BN→ReLU→Conv→BN + **skip** → ReLU).
- **`heads.py`** — `PolicyHead` (Conv1×1→BN→ReLU→flatten→Linear → **2086 raw
  logits**) and `ValueHead` (Conv1×1→BN→ReLU→flatten→Linear→ReLU→Linear→**tanh →
  scalar in [−1,1]**).
- **`network.py`** — `PolicyValueNetwork`: input conv → **N residual blocks** → two
  heads. Defaults `C=64`, `N=10` (~1.15M params). `forward(x) → (logits, value)`.
- **`sb3_extractor.py`** — `XiangqiResNetExtractor`: the same body packaged as a
  stable-baselines3 features extractor, so PPO learns on the same representation. This
  is what the IL body is transferred *into* for Agent 2 Phase 2.

### `src/data/` — Layer 2: the data pipeline
- **`wxf_parser.py`** — parses move notation to absolute moves and replays games.
  `parse_wxf_move` (Roman WXF `C2.5`, with front/rear disambiguation), `parse_iccs_move`
  (coordinate notation, both UCCI `h2e2` and xqbase-ICCS `C3-C4`), `detect_notation`
  (auto-pick), `parse_game` (replay + optional legality validation, raising on illegal
  moves). `Game` = a validated (moves, outcome) record.
- **`pgn_adapter.py`** — standard PGN text → `(result, tokens)` records
  (`split_pgn_games`, `parse_tags`, `extract_move_tokens` — strips move numbers,
  comments, NAGs, results, and the `...` placeholder — `pgn_to_records`,
  `load_games_from_pgn`).
- **`game_loader.py`** — `parse_game_records` (validate + length-filter a batch into
  `Game`s, tracking `LoadStats`), `parse_result` (`"1-0"`/`"red"`/`"1/2-1/2"` →
  RED/BLACK/DRAW).
- **`dataset.py`** — `XiangqiILDataset` (eager: all tensors in RAM; small data only)
  and **`LazyXiangqiILDataset`** (encodes each position on access by replaying to that
  ply; O(games) memory via a prefix-sum `bisect` index). Both yield `(tensor,
  action_index, value)` and optionally add a **left-right mirror** (Xiangqi symmetry;
  `mirror_move`, `mirror_board`). `_value_for` = outcome from that ply's mover's view.
- **`data_loader.py`** — `split_games` (**game-level** train/val/test split, seeded —
  never split by position or data leaks) and `make_dataloader` (torch DataLoader).
- **`pipeline.py`** — the **orchestrator**. `build_dataset()`: scan `data/raw/`
  recursively (`find_raw_files`), parse everything (`load_raw_records`), length-filter,
  split by game, compute `DatasetStats`, and persist splits as **JSONL**
  (`save_split_jsonl`/`load_split_jsonl`) without materializing tensors. Because full
  validation of 141k games ≈ 22h, it defaults to *not* validating and instead
  **spot-checks** a random sample (`spot_check_validity`). CLI: `python -m
  src.data.pipeline`.

### `src/agents/` — Layer 3: the players
`base_agent.py` (ABC), `minimax_agent.py`, `ppo_agent.py`, `neural_agent.py` (plays a
network's policy head greedily/sampled — used to evaluate the *raw* IL network before
RL), `mcts.py` + `mcts_agent.py`, `random_agent.py`, `engine_agent.py`. All covered in §5.

### `src/training/` — Layer 3: the learning procedures
- **`self_play_env.py`** — `SelfPlayEnv` (two-player → single-agent for PPO; §5).
- **`ppo_train.py`** — `build_model` (wires the ResNet extractor + PPO
  hyperparameters), `transfer_il_weights` (the **IL→PPO body bridge**), `train(...,
  il_checkpoint=...)`, CLI `main`.
- **`imitation.py`** — `run_epoch` (policy CE + value MSE), `evaluate` (top-1 move
  accuracy on val), `train_imitation` (loop with an `on_epoch_end` checkpoint callback
  + `start_epoch` for resume), `save_checkpoint`/`load_network`.
- **`il_train.py`** — the **entry point** for Agent 2 Phase 1: JSONL splits →
  `LazyXiangqiILDataset` → DataLoaders → `train_imitation` → checkpoint. Writes a
  checkpoint **every epoch** and supports `--resume` (Colab disconnects). CLI `main`.
- **`callbacks.py`** — `EloEvalCallback`: measures the training policy's Elo vs a fixed
  opponent every `eval_freq` steps and records `(timesteps, elo)` — the raw material
  for Experiment 2's learning curves.

### `src/evaluation/` — Layer 3: measuring strength
- **`elo.py`** — Elo math (§6.2).
- **`metrics.py`** — `GameOutcome` (one game from A's view) and `MatchStats`
  (aggregate: score, win/draw/loss rates, decisiveness, avg length, win-rate as
  Red/Black); `summarize()`.
- **`tournament.py`** — `play_game` (one game), `play_match` (N games, **alternating
  colours** to cancel Red's first-move edge), `estimate_elo` (rate vs a known
  opponent), `round_robin` (everyone vs everyone).
- **`opening_analysis.py`** — RQ3 tools: `opening_signature`, `build_distribution`,
  `kl_divergence` (compare an agent's opening distribution to the professional one —
  small KL = plays "book"), `classify_opening` (name Red's first move: central cannon,
  flying elephant, …), plus game collection helpers.

### `src/utils/` — Layer 0: shared helpers
`config.py` (**all fixed constants and hyperparameter defaults** — no magic numbers in
logic), `seeding.py` (`set_seed`), `checkpoint.py` (generic save/load with optimizer +
step + metadata), `logging_utils.py` (`Logger` for W&B/TensorBoard with graceful
fallback + `git_commit`), `render.py` (`board_to_plotly` for the dashboard).

### `experiments/` — Layer 4: the five thesis experiments
`exp1_main_comparison.py`, `exp2_training_efficiency.py`, `exp3_opening_theory.py`,
`exp4_curriculum.py`, `exp5_mcts_ablation.py`. Each runs one experiment and writes a
CSV. Detailed in §14. (Coded + unit-tested; not yet run on trained agents.)

### `dashboard/` — Layer 4: the Streamlit web app
`app.py` + five pages (`01_live_game`, `02_elo_comparison`, `03_learning_curves`,
`04_opening_analysis`, `05_game_explorer`) + reusable `components/` (results I/O, game
playback, agent pickers). Visualizes results and lets you watch agents play. Boots
locally; public deployment planned.

### `docs/` — the formal specs
13 markdown documents (game rules, architecture, environment, data pipeline, neural
network, agents, training, evaluation, experiments, dashboard, coding standards,
glossary) + a master index. Authoritative design specs; this guide connects them.

### `notebooks/xiangqi_colab_training.ipynb`
The Colab workflow: GPU check → clone/mount → install → download data → build splits →
IL → PPO Phase 2 → PPO Agent 1, checkpointing to Google Drive.

### `tests/` — 221 tests, all passing
Mirrors `src/` (`test_environment`, `test_models`, `test_agents`, `test_data`,
`test_training`, `test_evaluation`, `test_experiments`, `test_dashboard`,
`test_utils`). `python -m pytest -q` from the repo root. These are what let us trust
the code before spending money on GPU training.

### Root files
`README.md`, this `PROJECT_GUIDE.md`, `requirements.txt` / `environment.yml`,
`LICENSE` (MIT), `configs/` (per-experiment YAML), `data/` (gitignored raw/splits; only
`dataset_manifest.json` committed), `results/` (checkpoints/outputs, gitignored).

---

## 9. Worked examples

These trace real data through the actual code, step by step.

### 9.1 Encoding the opening position into a tensor
`encoder.encode(Board())` with Red to move produces a `(14, 10, 9)` float array:
- For Red's General at `(0, 4)`: color == perspective (Red), piece_type = 1, so
  `base = 0`, `channel = 0`. `_orient` leaves Red coordinates unchanged → set
  `tensor[0, 0, 4] = 1.0`.
- For Black's General at `(9, 4)`: opponent, type 1 → `base = 7`, `channel = 7`.
  (Perspective is Red, so no rotation.) → `tensor[7, 9, 4] = 1.0`.
- If it were **Black** to move, the board would be rotated 180° first, so Black's
  pieces occupy channels 0–6 and sit at the bottom — the network always sees "me
  below, opponent above."

### 9.2 One move, end to end
1. `env.reset()` → standard board, `info["legal_mask"]` has 44 `True` entries.
2. An agent's `select_move(board, mask)` returns an action index, say the Central
   Cannon `C2.5` → `(2, 7, 2, 4)` → `action_space.move_to_index(...)`.
3. `env.step(action)`: `index_to_move` decodes it, checks it's in the legal set
   (raises if you forgot to mask), `board.apply_move(2,7,2,4)` moves the cannon and
   flips to Black, ply counter and repetition counter update.
4. It then checks: does Black have a legal reply? Yes → `reward = 0`, game continues.
   (If Black had none → `reward = +1` for Red, `terminated = True`.)

### 9.3 One imitation-learning training step (`run_epoch`)
1. A DataLoader batch = 512 triples `(tensor[512,14,10,9], action[512], value[512])`
   from `LazyXiangqiILDataset` (each encoded on the fly by replaying its game to the
   right ply).
2. `logits, value_pred = network(tensors)`.
3. **Policy loss** = `CrossEntropy(logits, action)` — push the network toward the
   human's move. **Value loss** = `MSE(value_pred, value)` — push it toward the true
   outcome. `loss = policy_loss + value_loss`.
4. `loss.backward()`; `optimizer.step()` (AdamW, lr 1e-3, weight decay 1e-4).
5. After the epoch, `evaluate()` reports **top-1 accuracy** on validation (fraction of
   positions where the network's top move == the human's). ~40–55% is expected and
   sufficient — the RL phase does the rest.

### 9.4 One MCTS simulation (`_simulate`)
Starting at the root (already expanded with priors):
1. **Select**: `_select_child` picks the highest-PUCT child; `board.apply_move(...)`
   descends; repeat until reaching a node with no children.
2. **Expand/Evaluate**: `generate_legal_moves` at the leaf. If none → terminal, value
   = −1. Else `_evaluate` runs the network → priors + a value; create children.
3. **Backpropagate**: `_backpropagate` walks the path in reverse, adding the value and
   **negating it each step**, incrementing visit counts.
After 400 of these, `visit_count_policy(root)` gives visit counts; the agent plays the
most-visited action.

### 9.5 One Elo estimate
Agent scores 0.65 over 400 games vs an opponent rated 1500:
- `estimate_rating_from_score(1500, 0.65)` = `1500 + 400·log10(0.65/0.35)` ≈ **1500 +
  107 = 1607 Elo**.
- `elo_confidence_interval(0.65, 400)` ≈ **±33 Elo** → report "1607 ± 33".

---

## 10. The data pipeline in depth

### 10.1 The dataset
Source: **CGLemon/chinese-chess-PGN** (the source `docs/04` anticipated) — two
collections, both standard PGN in **ICCS** coordinate notation with illegal moves
pre-filtered, hosted on Google Drive:
- **World Xiangqi Federation:** 41,743 games.
- **DPXQ (东萍) archive:** 99,813 games.

### 10.2 Move notation (the same move, three ways)
The Red Central Cannon opening — cannon from column 7 to the central column 4, on
row 2 — appears as:
- **WXF (Roman):** `C2.5` (cannon on Red's file 2 traverses to file 5). Handled by
  `parse_wxf_move`. Human-relative file numbering (Red counts files from the right),
  with front/rear (`+`/`−`) disambiguation for stacked pieces.
- **UCCI:** `h2e2` (lowercase file letters a–i, no separator). Handled by
  `parse_iccs_move`.
- **xqbase ICCS:** `H2-E2` (uppercase + dash) — the dataset's format. Same parser
  (normalized to lowercase, dash stripped).

All three decode to the absolute move `(2, 7, 2, 4)` and are verified equivalent.

### 10.3 What the pipeline does, and the real numbers
`python -m src.data.pipeline --raw data/raw --out data/splits --notation iccs
--spot-check 500` produced:
- **141,556 games seen** → after filters, **141,356 kept** (26 too short < 10 plies,
  174 too long > 300 plies, **0 illegal**).
- **11,624,921 training positions** (one per ply).
- Outcome distribution **Red / Black / Draw = 53,764 / 39,860 / 47,732** (Red's
  first-move advantage is visible), **avg game length 82.24 plies**.
- **Spot-check: 500/500 legal (100%)** — the dataset is clean, which is why full
  validation is safely skipped.
- **Split:** train **113,084** / val **14,135** / test **14,137** games (80/10/10, at
  the **game** level).

### 10.4 Output files
`data/splits/{train,val,test}.jsonl` — one game per line as `{"o": outcome, "m":
[[fr,fc,tr,tc], …]}` (moves, not raw tokens, so training needs no re-parsing). Plus
`dataset_manifest.json` (the stats above + seed/fractions — the reproducibility
record, and the only data file committed to git). The big JSONL files are gitignored
(reproducible from raw + seed).

### 10.5 Why game-level split (not position-level)
If two positions from the *same* game landed in both train and test, the model could
"memorize" that game and inflate its test accuracy — **data leakage**. Splitting whole
games into disjoint sets prevents this. The split is **seeded** so it's identical
every run.

---

## 11. The neural network in depth

### 11.1 The input: 14 planes of 10×9
`encoder.encode` builds 14 binary planes (each 10×9), **from the mover's
perspective**:
- **Channels 0–6:** the current player's 7 piece types (General, Advisor, Elephant,
  Horse, Chariot, Cannon, Soldier) — a `1.0` wherever such a piece sits.
- **Channels 7–13:** the opponent's 7 piece types.
- When Black is to move, the whole board is rotated 180° so "my pieces" are always at
  the bottom moving up. No separate "side to move" plane is needed — it's baked into
  the my-vs-opponent split.

### 11.2 The body: an AlphaZero-style ResNet
`input conv (14→C) → N residual blocks (each: Conv→BN→ReLU→Conv→BN + skip → ReLU)`.
The **skip connections** (ResidualBlock) let gradients flow through a deep stack
without vanishing — the standard trick that makes deep nets trainable. Padding keeps
the 10×9 board size throughout. Defaults `C=64` width, `N=10` blocks.

### 11.3 The two heads
- **PolicyHead** → 2086 **raw logits** (not probabilities). Logits are kept raw so the
  agent can set illegal moves to −∞ *before* softmax — guaranteeing legality.
- **ValueHead** → one **tanh** scalar in [−1, 1] estimating the game outcome from the
  mover's view (+1 winning, −1 losing).

### 11.4 How it's trained / used by each consumer
- **Imitation:** supervised — CE on the policy vs human moves, MSE on the value vs
  outcomes.
- **PPO:** the body becomes the SB3 features extractor; PPO trains policy + value via
  the clipped surrogate + value loss + entropy (§4.4).
- **MCTS:** the policy head gives priors `P`, the value head evaluates leaves.

One shared architecture across all three is what makes the **IL→PPO transfer** and the
whole comparison coherent.

---

## 12. Training in depth

### 12.1 Imitation Learning (Agent 2 Phase 1)
- Entry: `python -m src.training.il_train --splits data/splits --epochs 30 --device
  cuda --save results/checkpoints/il_agent2_phase1.pt` (add `--resume` to continue).
- Hyperparameters (`config.py`): lr `1e-3`, `30` epochs, batch `512`, AdamW, weight
  decay `1e-4`.
- Uses `LazyXiangqiILDataset` (can't fit 11.6M tensors in RAM). Data-loading cost ≈
  20 min/epoch of encoding, negligible next to GPU compute. Checkpoints **every
  epoch** to survive Colab disconnects. Success metric: val top-1 accuracy ~40–55%.
  Estimated ~6–8 h for 30 epochs on a Colab T4.

### 12.2 PPO self-play (Agent 1, and Agent 2 Phase 2)
- Agent 1 (random init): `python -m src.training.ppo_train --timesteps 500000 --save
  results/checkpoints/ppo_agent1`.
- Agent 2 Phase 2 (IL-initialized): add `--il-checkpoint results/checkpoints/
  il_agent2_phase1.pt` — `transfer_il_weights` copies the IL body into the extractor.
- Hyperparameters (`config.py`): lr `3e-4`, `n_steps 2048`, batch `512`, `4` epochs
  per update, clip `0.2`, γ `0.99`, GAE-λ `0.95`, entropy `0.01`, value coef `0.5`,
  grad clip `0.5`, features dim `256`. Estimated ~15–20 h for 500k steps.

### 12.3 MCTS self-play (Agent 3)
- The search core and agent exist; a dedicated self-play *training loop* (like
  `il_train`/`ppo_train`) is **not yet finalized** — the most compute-heavy piece and
  the first fallback candidate if time runs short.

### 12.4 Checkpointing philosophy
Colab disconnects, so training must be resumable. IL checkpoints every epoch to Google
Drive; `--resume` reloads the latest weights. PPO uses SB3's `.zip` save/load. This is
why the notebook writes to Drive, not Colab's ephemeral disk.

---

## 13. Evaluation in depth

### 13.1 Playing a match (`tournament.play_match`)
N games between two agents, **alternating which colour agent A plays** each game, to
cancel Red's first-move advantage. Each game → a `GameOutcome` (WIN/LOSS/DRAW from A's
view + plies + A's colour). `summarize()` aggregates to `MatchStats` (score, win/draw/
loss rates, decisiveness, avg length, win-rate as Red vs as Black).

### 13.2 Two ways to get Elo
- **vs a known opponent** (`estimate_elo`): play a match vs e.g. ElephantEye, invert
  the score to a rating + CI. Absolute.
- **round-robin + fit** (`round_robin` → `elo.fit_ratings`): everyone plays everyone;
  fit consistent ratings for all agents at once. Relative unless anchored by including
  a known-rating engine.

### 13.3 Statistical honesty
Every rating comes with a 95% confidence interval; ~400 games/matchup targets ±40–50
Elo, so the thesis can state differences that are actually significant.

---

## 14. The five experiments

Each is a script in `experiments/` that runs the analysis and writes a CSV.

- **Exp 1 — Main comparison** (`exp1_main_comparison.py`, `run_exp1`): colour-balanced
  **round-robin** among all agents → `fit_ratings` → **`exp1_elo.csv`** (fitted Elo +
  CI + score + games) and **`exp1_head_to_head.csv`** (win-rate matrix). Answers **RQ2**
  (and RQ1 once IL and pure-RL agents are both in). Engine anchoring plugs in by adding
  ElephantEye to the `agents` dict.
- **Exp 2 — Training efficiency** (`exp2_training_efficiency.py`, `run_exp2`): **the
  thesis's central experiment.** Trains two PPO agents — `scratch` (random init) and
  `il_init` (IL body transferred) — while an `EloEvalCallback` records Elo vs a fixed
  opponent over training. → **`exp2_learning_curves.csv`** (variant, timesteps, elo).
  Answers **RQ1**: does IL start higher and reach a given Elo in fewer steps?
- **Exp 3 — Opening theory** (`exp3_opening_theory.py`, `run_exp3`): for each agent,
  play games, collect opening signatures (first `n_plies`), build a distribution, and
  compute **KL divergence** vs a professional reference (small KL = plays "book"). Also
  classifies Red's first move (central cannon, flying elephant, …). →
  **`exp3_openings.csv`**. Answers **RQ3**.
- **Exp 4 — Curriculum vs self-play** (`exp4_curriculum.py`, `run_exp4`): train PPO
  against **progressively stronger opponents** (Minimax at increasing depth, or
  ElephantEye at increasing depth), advancing a stage once a win-rate threshold (0.6)
  is met. → **`exp4_curriculum.csv`** (stage, opponent, timesteps, win_rate). Compare
  final Elo/efficiency to plain self-play. Answers **RQ4**.
- **Exp 5 — MCTS ablation** (`exp5_mcts_ablation.py`, `run_exp5`): run an MCTS agent at
  several simulation counts (e.g. 100/400/800/1600) vs a fixed opponent, recording
  **score/Elo and average thinking time per move** (a `_TimedAgent` wrapper). →
  **`exp5_mcts_ablation.csv`**. Reveals the strength-vs-speed sweet spot. Answers the
  MCTS part of **RQ4**.

---

## 15. Key design decisions

Each major decision and the reason:

- **Custom environment instead of `gym-xiangqi`.** The existing package is on the
  *legacy* Gym API and didn't fit our action-masking/dependency needs; a clean
  Gymnasium env we control also serves as a thesis artifact.
- **Player-relative 14×10×9 encoding (rotate for Black).** One network plays both
  colours ("me at bottom vs. opponent at top"), halving what must be learned.
- **Fixed 2086-move action space + mandatory legal mask.** A policy head needs a fixed
  output size; masking illegal moves to −∞ before softmax means an agent can **never**
  play illegal — solving the proposal's #2 risk via `MaskablePPO` + `legal_mask`.
- **One shared Policy-Value network for all learning agents.** Enables the IL→PPO
  transfer and makes the comparison coherent.
- **Stalemate = loss, flying-general illegal.** Real Xiangqi rules that differ from
  chess; getting them wrong would silently corrupt everything. Enforced in `rules.py`,
  covered by tests.
- **Negamax + captures-first ordering (Minimax).** One clean recursive routine; move
  ordering makes alpha-beta pruning far more effective.
- **AlphaZero-style value evaluation (no random rollouts) in MCTS.** The value head
  replaces slow simulations — faster and stronger.
- **Lazy dataset + skip full validation for the big corpus.** 141k games ≈ 11.6M
  positions; materializing tensors needs tens of GB and full validation ≈ 22h. So:
  persist JSONL, encode on the fly, spot-check legality.
- **Game-level split, seeded.** Prevents data leakage; reproducible.
- **Checkpoint every epoch + resume.** Colab disconnects; per-epoch Drive checkpoints
  make training survivable.
- **Colour-balanced matches + confidence intervals.** Red moves first (an edge), so
  matches alternate colours; every Elo comes with a CI for honest claims.
- **ElephantEye via UCCI, wrapped as a BaseAgent.** Reproducible absolute-Elo anchor
  that reuses the same tournament code.
- **All constants in `config.py`.** No magic numbers scattered through logic.

---

## 16. Testing & reproducibility

- **221 automated tests** (`tests/`, mirroring `src/`) cover: rule correctness (e.g.
  the opening has exactly 44 legal moves; stalemate is a loss; horse-leg/cannon-screen
  behavior), the action-space round-trip (2086, no duplicates), encoder shape,
  parser/pipeline (WXF/ICCS, replay validation, split reproducibility), datasets
  (lazy == eager), network shapes, each agent plays only legal moves, Elo math,
  tournament, and every experiment on tiny synthetic data. Run: `python -m pytest -q`.
- **Reproducibility:** `set_seed()` seeds Python/NumPy/torch; the split is seeded and
  recorded in `dataset_manifest.json`; `logging_utils.git_commit()` can stamp runs
  with the code version. Re-running the pipeline on the same raw files + seed yields
  the identical split.
- **Why so many tests before training?** GPU training is slow and costs money; a bug
  in the environment or encoder would waste hours and corrupt results. The tests let
  us trust Layer 1–2 completely before spending on Layer 3–4.

---

## 17. Status

**The key distinction: all the CODE is built and tested (221 tests green), but no
agent has been TRAINED yet.**

### ✅ Done (code & foundation)
- Full game engine (board, pieces, move generation, rules) — Layer 1.
- Encoder + 2086-move action space + legal masking; Gymnasium env + self-play wrapper.
- Policy-Value network (ResNet body + heads) + SB3 extractor.
- All four agents (Minimax, PPO, IL/Neural, MCTS) + Random + Engine wrapper.
- Evaluation stack: Elo, metrics, tournament/round-robin, opening analysis.
- **Data pipeline + a real 141k-game dataset built and split** (train 113,084 / val
  14,135 / test 14,137; ~11.6M positions; spot-checked 100% legal). Manifest committed.
- Lazy dataset + IL training entry point + IL→PPO transfer + per-epoch checkpoint/resume.
- All 5 experiment scripts (coded + unit-tested).
- Streamlit dashboard (boots locally); Colab notebook; public repo with README + this guide.

### ⏳ Not done (the execution phase — needs a GPU)
- **Training runs:** IL (Agent 2 P1), PPO fine-tune (Agent 2 P2), PPO self-play (Agent
  1), MCTS (Agent 3) — none trained yet.
- **ElephantEye binary:** interface exists; engine must be compiled for absolute Elo.
- **MCTS training script:** search + agent exist; a dedicated self-play training entry
  point is not finalized.
- **Experiment results:** scripts exist but haven't run on trained agents — no Elo
  tables / curves / opening analyses yet.
- **Thesis document:** proposal done; the 5-chapter thesis is not written.

### Timeline mapping (proposal's 6 phases)
Phase 1 (Foundation) ✅ complete · **Phases 2–3 (train agents) ← you are here** (code
ready, not run) · Phase 4 (experiments + dashboard) → dashboard ✅, experiments await
agents · Phases 5–6 (writing + revision) → not started.

---

## 18. The plan

Critical path (mostly *running* things, not coding):
1. **Train Agent 2 IL (Phase 1)** on Colab (~6–8 h), checkpoint to Drive.
2. **Train Agent 2 PPO fine-tune (Phase 2)** from that checkpoint.
3. **Train Agent 1 PPO** from scratch (~15–20 h, split across sessions).
4. **Train Agent 3 MCTS** (heaviest; de-scope per fallback if too slow).
5. **Compile ElephantEye**, wire `EngineAgent` for absolute Elo.
6. **Run Experiments 1–5** → Elo tables, learning curves, opening analysis.
7. **Write the thesis** using those results; deploy the dashboard.

**Fallback plan** (from the proposal), if compute/time runs short: drop Agent 3 → drop
Agent 2 → fine-tune an existing net → minimum viable thesis (one PPO agent + Minimax +
Elo vs ElephantEye). The non-negotiable core: *one RL agent + one baseline +
standardized Elo evaluation.*

---

## 19. Requirements & commands

**Requirements**
- **Python 3.10+** (`conda env create -f environment.yml` or `pip install -r
  requirements.txt`).
- Core: PyTorch, gymnasium, stable-baselines3 + sb3-contrib, NumPy, pandas. Optional:
  wandb/TensorBoard (logging degrades gracefully), Streamlit + Plotly (dashboard).
- **A GPU** for training (practically, Google Colab). CPU is fine for code, tests, and
  building the dataset.
- Raw game files in `data/raw/`; ElephantEye/Pikafish binary optional (absolute Elo).

**Command cheat-sheet**
```bash
# Run the whole test suite
python -m pytest -q

# Build dataset splits from raw games in data/raw/
python -m src.data.pipeline --raw data/raw --out data/splits --notation iccs --spot-check 500

# Agent 2 Phase 1 — imitation learning (GPU); --resume to continue after a disconnect
python -m src.training.il_train --splits data/splits --epochs 30 --device cuda \
    --save results/checkpoints/il_agent2_phase1.pt

# Agent 2 Phase 2 — PPO fine-tune from the IL checkpoint
python -m src.training.ppo_train --timesteps 500000 \
    --il-checkpoint results/checkpoints/il_agent2_phase1.pt --save results/checkpoints/ppo_agent2

# Agent 1 — PPO self-play from scratch
python -m src.training.ppo_train --timesteps 500000 --save results/checkpoints/ppo_agent1

# Launch the dashboard
streamlit run dashboard/app.py

# Quick local IL smoke test (cap games, 1 epoch)
python -m src.training.il_train --limit-train 2000 --limit-val 500 --epochs 1
```

---

## 20. Glossary

- **Action / action index** — a move, encoded as an integer in [0, 2086).
- **Action masking** — forcing illegal moves to be unselectable (logits → −∞).
- **AdamW** — the optimizer used for imitation learning (Adam + weight decay).
- **AlphaZero** — DeepMind's algorithm: one network + MCTS, trained purely by
  self-play; the model for Agent 3.
- **Advantage** — how much better an action turned out than the value head expected;
  what PPO's updates are proportional to.
- **Alpha-beta pruning** — skipping game-tree branches that can't change the result;
  makes Minimax feasible.
- **BatchNorm** — a layer that normalizes activations to stabilize/speed up training.
- **cchess-zero label scheme** — the standard ordering of the 2086 moves we follow, for
  compatibility with other Xiangqi AlphaZero code.
- **Checkpoint** — saved model weights, so training can resume / the model can be
  reused.
- **CI (confidence interval)** — the ± range on an Elo estimate (95%).
- **Convolution (Conv2D)** — a neural layer that scans local patterns over the board;
  the backbone of the network.
- **Cross-entropy** — the loss for "predict the right class" (here, the human's move).
- **Elo** — a relative skill rating; 400 points ≈ 10× expected-score ratio.
- **Episode** — one full game.
- **FEN** — a compact text string describing a board position (used to talk to the
  engine).
- **GAE (Generalized Advantage Estimation)** — a smoothed advantage estimate used by
  PPO.
- **Gymnasium** — the standard RL environment interface (`reset`/`step`) our env
  implements.
- **ICCS / UCCI / WXF** — three Xiangqi move notations (see §10.2); UCCI is also the
  engine protocol.
- **Imitation Learning (IL)** — supervised pre-training from human games.
- **KL divergence** — a measure of how different two probability distributions are;
  used to compare opening styles.
- **Logits** — raw, unnormalized network scores (before softmax).
- **MaskablePPO** — PPO variant (sb3-contrib) that supports action masking.
- **MCTS** — Monte Carlo Tree Search; look-ahead search guided by the network.
- **MDP (Markov Decision Process)** — the state/action/reward framing of the game.
- **Minimax / negamax** — classical adversarial search; negamax is its one-function
  form.
- **PGN** — a standard text format for recording games.
- **Ply** — one half-move (one side moving once).
- **Policy** — the rule for choosing moves (here, the policy head's distribution).
- **Policy-Value network** — one network with a move-scoring head and a
  position-scoring head.
- **PPO (Proximal Policy Optimization)** — the RL algorithm for Agents 1 & 2 Phase 2.
- **PUCT** — the formula MCTS uses to balance exploration vs. exploitation.
- **Pseudo-legal move** — a geometrically valid move, before checking self-check.
- **Replay validation** — replaying a recorded game move-by-move to confirm every move
  is legal.
- **Residual block / skip connection** — the ResNet unit that makes deep nets
  trainable.
- **Reward** — the +1/−1/+0.1 signal an agent maximizes.
- **Round-robin** — a tournament where everyone plays everyone.
- **Self-play** — an agent improving by playing against itself (or a copy).
- **Softmax** — turns logits into a probability distribution.
- **Tanh** — squashes the value head's output into [−1, 1].
- **Tensor** — a multi-dimensional array (here the 14×10×9 board representation).
- **Top-1 accuracy** — fraction of positions where the network's #1 move matches the
  human's; the IL success metric.
- **Value** — the network's estimate of the outcome from the current position.

---

## 21. FAQ

**Q: Why not just use one strong engine instead of training four agents?**
A: The research question isn't "make the strongest Xiangqi bot," it's "**compare how
different learning methods perform on Xiangqi**." The four agents *are* the experiment.

**Q: Why is stalemate a loss here? Isn't that a bug?**
A: No — it's a real Xiangqi rule (unlike chess). It's deliberately enforced in
`rules.py` and tested.

**Q: 2086 moves — where does that number come from?**
A: It's every *geometrically possible* move on the board (all straight-line moves, all
horse jumps, and the fixed advisor/elephant diagonals), following the standard
"cchess-zero" enumeration. Most aren't legal in a given position — that's what the
legal mask handles.

**Q: Why does the network see the board "rotated" for Black?**
A: So one network can play both sides. It always sees "me at the bottom moving up," so
it never has to learn two mirror-image strategies.

**Q: Why can't I train on my laptop?**
A: The code and dataset build fine on CPU, but training touches ~11.6M positions per
epoch (IL) or hundreds of thousands of self-play steps (PPO) — that needs a GPU
(Colab). CPU would take days per epoch.

**Q: Is the 141k-game dataset in the repo?**
A: No — the raw files and the big JSONL splits are gitignored (they're large and
reproducible). Only the tiny `dataset_manifest.json` (the stats record) is committed.

**Q: What happens if MCTS is too slow to train in time?**
A: The fallback plan drops Agent 3 (and Experiments 4–5) — the three-agent comparison
is still a complete, publishable contribution (§18).

**Q: How do I know the environment is correct if nothing's trained?**
A: 221 tests, plus the Minimax agent playing legal, sensible games, plus the dataset's
141,356 human games all replaying legally (0 illegal in the spot-check) — three
independent confirmations that the rules and move generation are right.

---

## 22. Where do I look for X?

| I want to… | Look at |
|------------|---------|
| Understand the rules of Xiangqi | §3, `src/environment/pieces.py`, `rules.py`, `docs/01-GAME-RULES.md` |
| See how a position becomes a tensor | §11.1, `src/environment/encoder.py` |
| Understand the move numbering (2086) | §8 (action_space), `src/environment/action_space.py` |
| Change a hyperparameter | `src/utils/config.py` |
| Build the dataset | §10, `src/data/pipeline.py` |
| Train imitation learning | §12.1, `src/training/il_train.py` |
| Train / fine-tune PPO | §12.2, `src/training/ppo_train.py` |
| Understand MCTS | §4.5, §9.4, `src/agents/mcts.py` |
| Measure strength / Elo | §6.2, §13, `src/evaluation/` |
| Run an experiment | §14, `experiments/` |
| Talk to the ElephantEye engine | §6.1, `src/environment/elephanteye.py` |
| See results visually | `dashboard/`, `streamlit run dashboard/app.py` |
| Trust the code | `tests/`, `python -m pytest -q` |

---

*This guide reflects the codebase as implemented. For authoritative per-topic detail
see [`docs/`](docs/); for the current build state, see the commit history.*
