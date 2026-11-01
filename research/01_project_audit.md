# 01 — Comprehensive Project Audit: XiangqiMaster

> **Purpose.** This document is the evidence base for the Lab 8 research
> proposal. It records what the `xiangqimaster` codebase *actually* contains —
> verified by reading the source, the result tables, and the git history — so
> every claim in the proposal is traceable to a file, function, or data file.
> It is **not** aspirational: where the design docs describe modules that do not
> exist in the code, that divergence is noted.

**Audit date:** 2026-11-01 · **Auditor:** research-supervisor pass over the repo
at `C:\Users\kinet\OneDrive\Documents\xiangqimaster` (branch `master`, clean
working tree, HEAD `e46c8c1`).

---

## 1. Project Overview

XiangqiMaster (象棋大师) is a **final thesis project** that conducts a
**comparative study of deep reinforcement-learning paradigms for Xiangqi
(Chinese Chess)**. It is not a game-playing product; it is a research codebase
whose purpose is to *train several categories of game-playing agent on the same
environment and measure their relative playing strength on a common Elo scale.*

- **Author:** Alma (student ID `L25020007`), dual-degree UII × NXU
  (`README.md:6`).
- **Central thesis framing** (`README.md:3`): "A Comparative Study of Deep
  Reinforcement Learning Approaches for Xiangqi — From Imitation Learning to
  Self-Play Mastery."
- **Four agent paradigms** are implemented, trained, and benchmarked: pure
  self-play PPO, Imitation-Learning-then-PPO, AlphaZero-style MCTS+network, and
  a classical Minimax baseline, plus a Random control.
- **Maturity:** Advanced. All planned build steps are implemented and
  unit-tested; four agents are trained; five experiments have been run and have
  produced **real Elo tables** (Section 15). The project has already produced
  several non-trivial empirical findings (Section 14).

The project is therefore in the unusual and favourable position of having
**completed experimental infrastructure and preliminary results** that a
research proposal can cite as feasibility evidence rather than promise.

---

## 2. Complete Directory Structure Overview

176 tracked files. The tree below is condensed to the research-relevant
structure (verified via `git ls-files`):

```
xiangqimaster/
├── docs/                    13 design/spec documents (01–12 + READMEs)
├── src/
│   ├── environment/         LAYER 1 — pure game logic (no ML deps)
│   │   ├── board.py            board state, apply_move, clone, position_key
│   │   ├── pieces.py           per-piece move deltas
│   │   ├── move_generator.py   generate_legal_moves (check-filtered)
│   │   ├── rules.py            check / checkmate / stalemate / draw
│   │   ├── action_space.py     fixed 2086-move map + legal_mask (cchess-zero)
│   │   ├── encoder.py          board → 14×10×9 tensor, 180° rotate for Black
│   │   ├── xiangqi_env.py      Gymnasium reset/step, clone() for MCTS
│   │   └── elephanteye.py      UCCI/UCI driver for ElephantEye / Pikafish
│   ├── data/                LAYER 2 — data pipeline
│   │   ├── wxf_parser.py       WXF + ICCS notation → validated move lists
│   │   ├── dataset.py          XiangqiILDataset + LazyXiangqiILDataset
│   │   ├── data_loader.py      reproducible game-level split + DataLoader
│   │   ├── pgn_adapter.py      PGN movetext → records
│   │   ├── game_loader.py      records → replay-validated Games
│   │   ├── pipeline.py         raw games → train/val/test JSONL + manifest
│   │   └── pikafish_selfplay.py engine self-play → distillation JSONL
│   ├── models/              LAYER 2 — neural network
│   │   ├── network.py          PolicyValueNetwork (shared body, two heads)
│   │   ├── blocks.py           ConvBlock, ResidualBlock
│   │   ├── heads.py            PolicyHead (→2086 logits), ValueHead (→tanh)
│   │   └── sb3_extractor.py    ResNet body as a MaskablePPO features extractor
│   ├── agents/              LAYER 3 — the agents (all BaseAgent)
│   │   ├── base_agent.py       abstract select_move(board, mask) + name
│   │   ├── minimax_agent.py    Agent 4 — alpha-beta negamax
│   │   ├── ppo_agent.py        Agent 1 — MaskablePPO policy
│   │   ├── neural_agent.py     Agent 2 Phase 1 — raw policy-head play
│   │   ├── mcts.py             PUCT MCTS core (Node, run_mcts)
│   │   ├── mcts_agent.py       Agent 3 — MCTS + network
│   │   ├── random_agent.py     Random control
│   │   └── engine_agent.py     EngineAgent wrapper for ElephantEye/Pikafish
│   ├── training/            LAYER 2/3 — training loops
│   │   ├── imitation.py        supervised CE(policy)+MSE(value) IL trainer
│   │   ├── il_train.py         IL from disk splits (+ --channels/--num-blocks)
│   │   ├── self_play_env.py    single-agent self-play wrapper (opponent baked)
│   │   ├── ppo_train.py        MaskablePPO training (+ --opponent, --resume)
│   │   ├── mcts_train.py       AlphaZero-style self-play trainer (Agent 3)
│   │   └── callbacks.py        periodic/best checkpoint callbacks
│   ├── evaluation/          LAYER 4 — measurement
│   │   ├── elo.py              expected_score, estimate, CI, fit_ratings
│   │   ├── tournament.py       play_game, colour-balanced match, round_robin
│   │   ├── metrics.py          GameOutcome, MatchStats
│   │   └── opening_analysis.py opening move-distribution analysis
│   └── utils/               config, seeding, checkpoint, logging_utils, render
├── experiments/             exp1..exp5 scripts (one per research question)
├── dashboard/               Streamlit app + 5 pages + components
├── tests/                   48 test files across all layers
├── results/                 Elo CSVs, head-to-head matrices, figures, findings
│   ├── exp1/ exp1_agent3/ exp1_agent3_v2/ exp_distill/ exp_bignet/ exp5_mcts_v2/
│   ├── figures/             11 PNG figures (training curves, Elo bars)
│   └── AGENT3_MCTS_FINDING.md
├── notebooks/               Colab training notebooks (disconnect-safe)
├── data/splits/dataset_manifest.json   reproducibility record (stats)
├── PROJECT_GUIDE.md         plain-English end-to-end walkthrough
└── README.md / environment.yml / requirements.txt
```

**Divergence note (verified).** The idealized layout in `docs/02-ARCHITECTURE.md`
lists several files that do **not** exist as named (`mcts_core.py`,
`self_play.py`, `replay_buffer.py`, `curriculum.py`, `data/encoder.py`). In the
real code the MCTS core is `agents/mcts.py`, the encoder lives in
`environment/encoder.py` (so Layer-1 modules can share it), and the replay
buffer is internal to `training/mcts_train.py`. The proposal must describe the
**code as it is**, not the doc's idealized map.

---

## 3. Technology Stack (verified)

| Layer | Technology | Evidence |
|---|---|---|
| Language | Python (target 3.10 per `environment.yml`; run locally on 3.12 CUDA venv / 3.14) | `requirements.txt`, memory |
| Deep learning | PyTorch 2.14 (CUDA cu126 on RTX 4060 laptop GPU) | `models/network.py`, memory |
| RL | `stable-baselines3` + `sb3-contrib` 2.9 (**MaskablePPO**) | `ppo_train.py`, `sb3_extractor.py` |
| Env API | Gymnasium 1.3 (`gym.Env`, `spaces.Box/Discrete`) | `xiangqi_env.py:25,47` |
| Numerics | NumPy (board grid, masks, Dirichlet noise) | throughout |
| Engine benchmark | ElephantEye / **Pikafish** via UCCI/UCI subprocess | `environment/elephanteye.py`, `engines/pikafish/` |
| Dashboard | Streamlit + Plotly | `dashboard/app.py`, `utils/render.py` |
| Testing | pytest (48 test files, ~175+ tests) | `tests/`, memory |
| Experiment logging | optional W&B / TensorBoard + CSV (graceful fallback) | `utils/logging_utils.py`, `ppo_train.py --log-dir` |

No web/app server, no database, no authentication — this is a **research/ML
codebase**, not an interactive web application. "Persistence" is file-based:
model checkpoints (`.pt`/`.zip`), JSONL data splits, CSV result tables, PNG
figures.

---

## 4. System Architecture

A strict **five-layer, downward-only dependency** architecture
(`docs/02-ARCHITECTURE.md:198`), enforced by convention and verified by the
import structure:

1. **Layer 1 — Environment** (`src/environment/`): pure Xiangqi logic. Imports
   only `utils/`. Deterministic, fully unit-tested, reusable. This is the
   foundation that makes every agent comparison *fair* — all agents act on the
   identical rule engine and action space.
2. **Layer 2 — Learning components** (`models/`, `data/`): the shared
   PolicyValueNetwork and the data pipeline.
3. **Layer 3 — Agents** (`agents/`): the four paradigms behind one interface.
4. **Layer 4 — Experiments & Evaluation** (`evaluation/`, `experiments/`): Elo,
   tournaments, and the five experiment scripts.
5. **Layer 5 — Presentation** (`dashboard/`): Streamlit visualisation.

**Most important architectural decision:** the **uniform agent interface**.
`BaseAgent.select_move(board, legal_mask) -> action_index` + `name`
(`agents/base_agent.py`, confirmed used by every agent) lets the tournament and
dashboard treat a 20-line Minimax and a GPU neural net identically. This is what
makes the comparative study methodologically clean — the *only* thing that
differs between measured agents is the decision function, not the environment,
the action encoding, or the scoring.

**Second key decision:** a **fixed 2086-action space** shared by the network
policy head, PPO, MCTS priors, and the IL targets (`environment/action_space.py`,
`config.ACTION_SPACE_SIZE = 2086`). It follows the "cchess-zero" label scheme
(`action_space.py:9`) so indices are compatible with other AlphaZero-style
Xiangqi codebases. Legality is handled separately by `legal_mask`, keeping the
action space constant while positions vary.

---

## 5. Core Modules and Their Responsibilities

- **`environment/board.py`** — the `Board`: a signed NumPy 10×9 grid (Red `+code`,
  Black `−code`, empty `0`), `to_move`, `apply_move`, `clone`, `position_key`
  (for repetition detection), `type_at`.
- **`environment/move_generator.py`** — `generate_legal_moves(board)` returns
  fully legal moves (pseudo-legal filtered for own-king exposure and the
  flying-general rule).
- **`environment/rules.py`** — `has_legal_moves`, check/checkmate/stalemate and
  draw detection. **Stalemate counts as a loss** (Xiangqi rule), a detail that
  propagates into every agent and the reward.
- **`environment/action_space.py`** — builds the bidirectional move↔index map
  once at import (`move_to_index`, `index_to_move`, `legal_mask`).
- **`environment/encoder.py`** — `encode(board)` → `14×10×9` float tensor from
  the **side-to-move's perspective** (Black's view is rotated 180°), so the
  network always "sees" the board as the mover.
- **`environment/xiangqi_env.py`** — Gymnasium `XiangqiEnv`: `reset`/`step`
  with sparse reward (`+1` win, `+0.1` draw, loss assigned by the self-play
  loop), `MAX_PLIES=300` truncation, 3-fold repetition draw, and `clone()` for
  MCTS (verified `xiangqi_env.py:81–143`).
- **`models/network.py`** — `PolicyValueNetwork`: input conv → N residual
  blocks → policy head (2086 logits) + value head (scalar in [−1,1]). Default
  C=64 / N=10 ≈ **1.15M params**; size is now configurable (`--channels`,
  `--num-blocks`).
- **`evaluation/elo.py`** — relative Elo via iterative round-robin fit
  (`fit_ratings`, anchor 1500) plus score-based estimate and a delta-method 95%
  CI (`elo_confidence_interval`).
- **`evaluation/tournament.py`** — `play_game`, colour-balanced `play_match`,
  `round_robin`; agent-vs-agent needs no external engine.

---

## 6. Xiangqi Game Mechanics Implementation (verified)

- **Board geometry:** 10 rows × 9 columns = 90 points; palaces at columns 3–5,
  rows 0–2 (Red) / 7–9 (Black); river between rows 4 and 5 (`config.py:14–77`).
- **Pieces & values:** General 10000, Chariot 9, Cannon 4.5, Horse 4, Advisor 2,
  Elephant 2, Soldier 1 (`config.PIECE_VALUES`). Values drive the Minimax
  evaluation and optional reward shaping.
- **Move legality:** per-piece deltas in `pieces.py` → pseudo-legal in
  `move_generator.py` → filtered for self-check and the flying-general
  face-off. The **opening position yields exactly 44 legal moves**, the
  project's validation anchor (memory; `tests/test_move_generator.py`).
- **Terminal conditions:** no legal reply ⇒ the side to move has lost
  (checkmate *or* stalemate); 3-fold repetition ⇒ draw; 300-ply cap ⇒ draw
  (`xiangqi_env.py:104–113`).
- **Perspective handling:** the encoder rotates the board for Black so policy
  and value are always learned from the mover's viewpoint — essential for a
  single shared network to play both colours.

---

## 7. Algorithm Analysis

### 7.1 Minimax (Agent 4) — `agents/minimax_agent.py` (verified)
- **Negamax** formulation of alpha-beta (one routine scoring from the
  side-to-move's view, negating for the opponent).
- **Evaluation:** material balance only (`_evaluate`, `minimax_agent.py:87`).
- **Move ordering:** most-valuable captures first (`_ordered_moves`) to sharpen
  pruning.
- **Terminal handling:** no-legal-move = `−(MATE_SCORE + depth)` so quicker
  mates are preferred. Default depth 4 (`config.MINIMAX_DEPTH`); experiments use
  depth 2 as the "club-player floor."

### 7.2 PPO (Agents 1 & 2 Phase 2) — `training/ppo_train.py`, `self_play_env.py`
- **MaskablePPO** (sb3-contrib) with the ResNet body as a features extractor
  (`models/sb3_extractor.py`). Action masking prevents wasting probability on
  illegal moves.
- **Self-play wrapper** (`self_play_env.py`) bakes a frozen or co-evolving
  opponent into a single-agent env; reward is from the training agent's
  perspective; agent colour is randomised per episode.
- **`--opponent {random, self, minimax[:D]}`** selects the self-play opponent
  (commit `d4da962`). Disconnect-safe: `--checkpoint-every`, `--resume`,
  `--device`, `--log-dir` (CSV + TensorBoard).

### 7.3 MCTS + Network (Agent 3) — `agents/mcts.py`, `training/mcts_train.py`
- **PUCT selection** `score = −Q(child) + c_puct·P(child)·√ΣN /(1+N(child))`
  (`mcts.py:80`), network-prior expansion, **value-head leaf evaluation** (no
  rollouts; terminal leaf = −1), negamax backprop, move chosen by **visit
  count** (`mcts.py` verified end-to-end).
- **Training:** AlphaZero-style self-play with Dirichlet root noise → records
  (state, visit-count policy, outcome) → fits the net (soft-CE to visits + value
  MSE) over a sliding replay window (`mcts_train.py`). Supports IL warm-start,
  and (commit `569e8b2`) **persistent replay buffer + best-checkpoint saving**
  on resume — a fix that turned out to matter for strength (Section 14).

### 7.4 Imitation Learning (Agent 2 Phase 1) — `training/imitation.py`, `il_train.py`
- Supervised **cross-entropy (policy) + MSE (value)** over human games, AdamW,
  per-epoch top-1 accuracy, intra-epoch checkpointing for Colab resilience.
- `LazyXiangqiILDataset` replays each game to a ply on access (O(games) RAM),
  enabling training over 9.3M positions without materialising tensors.

---

## 8. AI Implementation Analysis

The project implements **four genuinely different decision mechanisms** over one
shared network and environment — this is its core intellectual content:

| Agent | Human data? | Neural net? | Search at play? | Learns? | Role |
|---|---|---|---|---|---|
| 1 PPO (scratch) | No | Yes | No | Yes | baseline RL |
| 2 IL→PPO | Yes (Phase 1) | Yes | No | Yes | **tests RQ1 (does IL help?)** |
| 3 MCTS+NN | optional warm-start | Yes | **Yes (PUCT)** | Yes | **tests RQ2 (strongest algo?)** |
| 4 Minimax | No | No | Yes (α-β) | No | classical baseline |

All four are verified in source and all four have been **trained/instantiated and
evaluated** (Section 15). The shared PolicyValueNetwork is the single learned
artifact reused by IL, PPO (actor+critic), and MCTS (priors + leaf value).

---

## 9. Data & Data-Management Analysis

- **Source:** `CGLemon/chinese-chess-PGN` — standard PGN in xqbase ICCS
  notation (two collections: WXF-Federation 41,743 games + dpxq 99,813 games).
- **Pipeline** (`data/pipeline.py`): scans `data/raw/` recursively → parse
  (WXF/ICCS) → replay-validate (spot-checked) → length filter → reproducible
  **game-level** split → per-split JSONL + `dataset_manifest.json`.
- **Verified manifest** (`data/splits/dataset_manifest.json`): 141,556 games
  seen → **141,356 kept** (26 too short, 174 too long, 0 invalid);
  **11,624,921 positions**; outcome mix R/B/D = 53,764 / 39,860 / 47,732; avg
  82.24 plies; spot-check 500/500 legal. Split: **113,084 / 14,135 / 14,137**
  games (train 9,296,810 positions).
- **Distillation path:** `data/pikafish_selfplay.py` generates Pikafish-vs-
  Pikafish games (random opening plies for diversity) in the *same* JSONL
  format, so the IL pipeline trains on an engine teacher unchanged.
- Game data is gitignored (large); the **manifest is committed** as the
  reproducibility record (the thesis's dataset-statistics table).

---

## 10. Testing & Quality Assurance Analysis

- **48 test files** under `tests/`, mirroring every source package
  (`test_environment/`, `test_data/`, `test_models/`, `test_agents/`,
  `test_training/`, `test_evaluation/`, `test_experiments/`, `test_utils/`,
  `test_dashboard/`). Memory records ~175+ passing tests at the last full run.
- **TDD discipline:** the move generator, rules, action-space round-trip,
  encoder, Elo math, and tournament are all unit-tested; the environment is
  validated by the "44 legal opening moves" anchor and by a working Minimax.
- **Reproducibility infrastructure:** `utils/seeding.set_seed`, committed
  `dataset_manifest.json`, per-run seeds/configs, and CSV/figure outputs
  regenerable from saved data + scripts.
- **Known gaps:** `tests/test_dashboard/` needs Plotly installed to run; several
  heavy results were produced by scratchpad scripts (e.g. `run_exp1_agent3*.py`)
  rather than committed experiment modules — a reproducibility seam to tighten.

---

## 11. Existing Features (implemented & verified)

1. Complete, tested Xiangqi rule engine + Gymnasium environment.
2. Fixed 2086-action space with legal masking and a bidirectional move map.
3. 14×10×9 player-relative board encoder.
4. Shared, size-configurable PolicyValueNetwork (policy + value heads).
5. Four agents + Random, behind one interface.
6. Three training regimes: IL (supervised), PPO self-play, AlphaZero MCTS.
7. End-to-end data pipeline over 141k real human games (built & split).
8. Pikafish/ElephantEye UCCI/UCI engine interface + engine self-play
   distillation generator.
9. Elo evaluation (relative fit + CIs) and round-robin tournament harness.
10. Five experiment scripts and a Streamlit dashboard with live play.
11. Disconnect-safe, resumable training with checkpointing and logging.

---

## 12. Technical Strengths

- **Methodological cleanliness:** one environment, one action space, one
  network, one Elo scale → a genuinely controlled comparison across paradigms.
- **Real scale:** trained on 9.3M positions from 113k human games; agents
  actually trained on a consumer GPU; experiments actually run.
- **Reproducibility scaffolding:** seeds, manifest, CSV/figure provenance,
  resumable training.
- **Honest, verified results** with confidence intervals — including *negative*
  and *surprising* results that a weaker project would have hidden.
- **Strong test coverage** of the correctness-critical Layer 1.

---

## 13. Technical Limitations (verified, not invented)

- **Model-capacity ceiling:** the 1.15M-param net plateaus at ~34% IL top-1
  move accuracy; four independent interventions failed to break a ~1690 Elo
  ceiling (Section 14).
- **Compute-bound evaluation:** Elo tables use 20–50 games/pair ⇒ CIs ±40–96,
  so several mid-table differences are within noise.
- **Relative-only Elo** in most tables (no absolute anchor except the one
  distillation run that included Pikafish at 2062).
- **Reproducibility seam:** key evaluation runs live in scratchpad scripts, not
  committed experiment modules; `data/raw/` is empty by default (user must
  download games).
- **Training-stability fragility:** fragmented stop/resume once *regressed* the
  MCTS agent (1703→1582) until a replay-buffer-persistence fix restored it
  (1691) — strength proved sensitive to training-engineering details.
- **Docs/code drift:** `docs/02` describes modules that do not exist as named.

---

## 14. Verified Empirical Findings (the research gold)

These are **already-observed results**, verified against the committed CSVs and
`results/AGENT3_MCTS_FINDING.md`. They are the strongest raw material for a
defensible research proposal:

- **F1 — Search over imitation beats everything measured (RQ2).** AlphaZero-style
  MCTS over the IL-trained net is the top agent: **MCTS 1703** vs IL_only 1557 >
  Minimax 1541 > IL+PPO 1411 > Random 1287 (`results/exp1_agent3/exp1_elo.csv`).
- **F2 — PPO fine-tuning *eroded* the imitation policy (surprising RQ1).** In the
  7-agent round-robin, **IL_only 1689 was the single strongest non-search agent;
  no PPO variant beat it**, and self-play fine-tuning *lowered* the IL agent
  (IL+PPO-self 1444 < IL+PPO-rand 1526 < IL_only 1689)
  (`results/exp1/exp1_elo.csv`).
- **F3 — Capacity/compute is the binding constraint, confirmed four ways.**
  (a) more MCTS simulations did not help (`exp5_mcts_ablation.csv`); (b) IL top-1
  plateaued at ~34%; (c) Pikafish distillation did **not** beat human-IL
  (Distill 1501 ≈ IL 1489, with Pikafish anchor at 2062,
  `results/exp_distill/`); (d) a 4× bigger net gave no real strength gain
  (Distill_big 1573 ≈ IL 1561 ≈ Distill_small 1530, `results/exp_bignet/`).
- **F4 — Training-stability is a first-class strength factor (methodological).**
  Replay-buffer continuity across stop/resume directly affected MCTS strength;
  persisting the buffer + saving a best checkpoint recovered a 121-Elo
  regression (`results/AGENT3_MCTS_FINDING.md`).

---

## 15. Verified Evidence from Source Code & Results

| Claim | Evidence |
|---|---|
| Four agents + Random behind one interface | `agents/*.py`, `base_agent.py` |
| MCTS = PUCT + value-head leaf eval + visit-count move | `agents/mcts.py:80,167,178` |
| Minimax = α-β negamax, material eval, captures-first | `agents/minimax_agent.py:63–111` |
| Shared net = conv + N res blocks + 2086-policy + tanh-value | `models/network.py:40–59` |
| 2086 action space, cchess-zero scheme | `environment/action_space.py:1–48`, `config.py:20` |
| Env: sparse reward, stalemate=loss, 300-ply/3-rep draw, clone() | `environment/xiangqi_env.py:99–143` |
| Dataset: 141,356 games / 11.6M positions, split 113k/14k/14k | `data/splits/dataset_manifest.json` |
| Elo: relative fit + delta-method CI | `evaluation/elo.py:59–122` |
| Engine benchmark via UCCI/UCI (Pikafish) | `environment/elephanteye.py:1–40` |
| Real Elo tables (5 experiments) | `results/exp1*/`, `exp_distill/`, `exp_bignet/`, `exp5_mcts_v2/` |
| Findings write-up | `results/AGENT3_MCTS_FINDING.md` |
| 15 feature commits, clean tree | `git log`, `git status` |

---

## 16. Overall Project Maturity Assessment

**Maturity: high / near-complete research project.** This is not a skeleton to
be built into a study — it is a working comparative-RL framework that has
*already run its experiments and produced defensible, confidence-interval-backed
results, including surprising negative findings.* The engineering (environment,
action space, network, training loops, evaluation, dashboard) is complete and
tested; the remaining open work is scientific framing, additional
statistical power (more games/tighter CIs), and optional absolute-Elo anchoring.

**Implication for Lab 8.** The research proposal should be framed as a *forward
specification of the study this codebase embodies*, presenting the completed
infrastructure and the four verified findings (Section 14) as **preliminary
results and feasibility evidence**. The strongest, most honest and most
defensible research narratives available are:

1. A **controlled comparative study** of four learning paradigms under a *fixed
   compute budget*, with the headline that **search + imitation priors
   outperform policy-gradient fine-tuning** (F1, F2).
2. An **empirical bottleneck analysis** showing that for Xiangqi at consumer-GPU
   scale, **model capacity and data/compute — not search depth or RL algorithm —
   bound playing strength** (F3), a multi-pronged, falsifiable finding.
3. A **reproducibility / training-stability** contribution on self-play RL
   engineering (F4), which is the most *Software-Engineering-flavoured* angle.

Candidate research directions built on these are developed and ranked in
`research/02_literature_review.md` / the Phase-2 recommendation. **No research
direction is committed until the user approves one.**
```