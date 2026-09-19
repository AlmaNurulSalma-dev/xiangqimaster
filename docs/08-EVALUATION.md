# 08 — Evaluation & Benchmarking Specification

> **Read this when:** building the Elo rating system, integrating the ElephantEye benchmark engine, running tournaments, or computing evaluation metrics. This defines how agent strength is measured objectively.

---

## 1. Purpose

Evaluation answers the central question: **how strong is each agent, and how do they compare?** All comparisons in the thesis rest on this framework, so it must be rigorous and statistically sound.

The primary tool is **Elo rating**, computed by playing many games against the **ElephantEye** benchmark engine at multiple strength levels.

---

## 2. The Benchmark: ElephantEye

### 2.1 What It Is
ElephantEye (象眼, repo `xqbase/eleeye`, LGPL-2.1, C++) is a well-known
open-source Xiangqi engine. It uses classical search + evaluation and provides
a stable, reproducible opponent of known strength.

> **Note:** ElephantEye ships as C++ source with **no prebuilt binary** on
> GitHub — you must compile it (works on Windows and Linux/Colab). For a
> **stronger, actively-maintained** benchmark, also consider **Pikafish**
> (a Stockfish-based Xiangqi engine) or **Fairy-Stockfish**. All three speak
> UCCI, so the same interface module works for any of them.

### 2.2 Why Use It
- **Fixed reference:** unlike self-play Elo (which is relative and can drift), ElephantEye gives an absolute, stable benchmark
- **Adjustable strength:** its search depth can be configured to create opponents of different Elo levels
- **Reproducible:** anyone can re-run the same benchmark

### 2.3 Integration via UCCI Protocol
ElephantEye communicates through the **UCCI (Universal Chinese Chess Interface)**
protocol — the Xiangqi counterpart of UCI (do NOT assume plain UCI; Xiangqi
engines use UCCI). The `elephanteye.py` module:
- Launches the engine as a subprocess
- Sends the current position (in the engine's expected FEN-like format)
- Sends a "go" command with a depth or time limit
- Reads back the engine's chosen move
- Translates between the engine's move format and the internal action representation

### 2.4 Strength Levels
Configure ElephantEye at several fixed search depths to create a ladder of opponents:

| Level | Approx. Depth | Approx. Elo |
|---|---|---|
| E1 | Depth 1 | ~800 |
| E2 | Depth 3 | ~1200 |
| E3 | Depth 5 | ~1600 |
| E4 | Depth 7 | ~2000 |
| E5 | Depth 9 | ~2400 |

(These Elo mappings are approximate; calibrate them if possible. Document the exact settings used.)

---

## 3. Elo Rating System

### 3.1 What Elo Measures
Elo is a relative skill rating. The difference between two players' ratings predicts the expected score of a match. A 400-point gap means the stronger player is expected to score ~10× as often.

### 3.2 Computing an Agent's Elo
Two approaches — use both and cross-check:

**Approach A — Anchored to ElephantEye:**
- ElephantEye levels have assigned (fixed) Elo values
- Play the agent against each level, record win/draw/loss
- From the win rate against a known-Elo opponent, derive the agent's Elo using the Elo expected-score formula
- Average the estimates across levels (weighting by number of games)

**Approach B — Full tournament + rating fit:**
- Run a round-robin among all agents AND ElephantEye levels
- Fit Elo ratings to all results simultaneously (e.g., via maximum likelihood / logistic regression, like BayesElo)
- Anchor the scale to the fixed ElephantEye Elo values

### 3.3 The Elo Expected-Score Formula
The expected score of player A against player B depends only on the rating difference. From an observed win rate against a known-rating opponent, invert this relationship to estimate the unknown rating. (Standard Elo math — implement in `elo.py`.)

### 3.4 Confidence Intervals
Elo estimates from finite games have uncertainty. Report **95% confidence intervals**. The width depends on the number of games:
- ~400 games → roughly ±40–50 Elo
- More games → tighter intervals
- Always report the interval, never a bare number.

### 3.5 Minimum Games
Play at least **400 games per agent-vs-opponent matchup** (alternating colors equally — half as Red, half as Black — to remove first-move bias). More is better for the final reported numbers.

---

## 4. Tournament Manager

The `tournament.py` module orchestrates evaluation games. It must support:

### 4.1 Agent vs Engine
Play a specified agent against a specified ElephantEye level for K games, alternating colors, recording each result. Returns win/draw/loss counts and derived statistics.

### 4.2 Agent vs Agent
Play two agents against each other (used for the round-robin and for MCTS "new vs best" promotion checks in training).

### 4.3 Round-Robin
Every agent plays every other agent and every ElephantEye level. Produces the full results matrix for Elo fitting.

### 4.4 Color Balancing
Always split games evenly between the agent playing Red and Black. Xiangqi has a first-move advantage for Red; ignoring color balance biases results.

### 4.5 Determinism vs Variety
- For reproducibility, seed the games
- But agents should have SOME variety (otherwise every game is identical). Use low-temperature sampling or seeded starting-position variety so games differ while remaining reproducible. Document the approach.

---

## 5. Metrics

Beyond Elo, record these for richer analysis (`metrics.py`):

| Metric | Definition | Why |
|---|---|---|
| Win rate | wins / total games | Primary head-to-head measure |
| Draw rate | draws / total games | Xiangqi draw tendency |
| Loss rate | losses / total games | |
| Elo rating | derived from win rates | Primary strength measure |
| Average game length | mean plies per game | Playing style indicator |
| Win rate as Red | wins when playing Red | First-move advantage check |
| Win rate as Black | wins when playing Black | Second-player strength |
| Decisiveness | (wins+losses) / total | How often games are decisive |

---

## 6. Evaluation During Training vs Final Evaluation

### 6.1 During Training (Fast, Frequent)
- Run a small number of games (e.g., 50–100) against a mid-level ElephantEye (E2 or E3) every N training games/iterations
- Purpose: track the Elo learning curve (feeds Experiment 2)
- Use faster/shallower settings to keep it quick

### 6.2 Final Evaluation (Thorough, Once)
- After training completes, run the full protocol: 400+ games per agent per ElephantEye level, all colors balanced
- Purpose: the definitive Elo numbers reported in the thesis (Experiment 1)
- Run overnight; this is expensive but done only once per agent

---

## 7. Statistical Rigor (For Publishable Results)

- Report Elo with 95% confidence intervals
- State the number of games behind every reported number
- Use enough games that differences between agents are statistically distinguishable
- When claiming "Agent X > Agent Y," verify their confidence intervals don't heavily overlap, or run a significance test on the win rate
- Balance colors; report per-color results to expose any asymmetry

---

## 8. Reproducibility of Evaluation

- Fix ElephantEye version and settings; document them exactly
- Seed the tournament
- Save all raw game records (so results can be re-derived and audited)
- Save the results matrix to `results/tables/`

---

## 9. Expected Results Shape (For Planning)

A well-run evaluation should produce something like:

| Agent | Elo (95% CI) | Win% vs E3 | Win% vs E4 |
|---|---|---|---|
| Minimax baseline | ~1400 (±50) | ... | ... |
| PPO (Agent 1) | ~1700 (±45) | ... | ... |
| IL+RL (Agent 2) | ~1900 (±45) | ... | ... |
| MCTS+NN (Agent 3) | ~2100 (±50) | ... | ... |

(Numbers illustrative — actual results are the research finding.)

---

## 10. Testing the Evaluation Framework

- [ ] ElephantEye subprocess launches, accepts positions, returns legal moves
- [ ] Move translation between engine format and internal format is correct (round-trip test)
- [ ] Elo formula gives known correct values on textbook examples
- [ ] Confidence interval width shrinks as game count grows
- [ ] Color balancing splits games 50/50
- [ ] A weak agent scores near 0% vs a strong engine level, a strong agent scores near 100% vs a weak level (sanity check)
- [ ] Full tournament produces a complete, symmetric results matrix

---

*Cross-references: 06-AGENTS.md (agents being evaluated), 07-TRAINING.md (in-training evaluation), 09-EXPERIMENTS.md (experiments consuming these metrics), 03-ENVIRONMENT.md (game play).*
