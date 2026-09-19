# 09 — Experiments Specification

> **Read this when:** setting up, running, or analyzing any of the five experiments. Each experiment answers a research question and produces specific result tables/figures for the thesis. This document defines the protocol, inputs, and expected outputs for each.

---

## 1. Experiment Overview

| Exp | Name | Answers | Key Output |
|---|---|---|---|
| 1 | Main Performance Comparison | RQ2 | Elo comparison table across all agents |
| 2 | Training Efficiency | RQ1 | Learning curves (Elo vs training games) |
| 3 | Opening Theory Analysis | RQ3 | Opening distribution comparison |
| 4 | Curriculum vs Self-Play | RQ4 | Curriculum vs standard Elo comparison |
| 5 | MCTS Simulation Ablation | (supporting) | Performance vs simulation-count trade-off |

Each experiment script lives in `experiments/exp<N>_*.py` and writes results to `results/tables/` and `results/figures/`.

---

## 2. Experiment 1 — Main Performance Comparison

### 2.1 Research Question
RQ2: Which RL algorithm achieves the highest Elo rating?

### 2.2 Hypothesis
MCTS+NN (Agent 3) > IL+RL (Agent 2) > PPO (Agent 1) > Minimax (Agent 4).

### 2.3 Protocol
1. Take all four fully-trained agents
2. Each agent plays 400+ games against each ElephantEye level (E1–E5), colors balanced
3. Compute each agent's Elo with 95% confidence interval (08-EVALUATION.md)
4. Also run a round-robin: each agent vs each other agent (400+ games)

### 2.4 Output
- **Table:** Agent × ElephantEye-level win rates + derived Elo (with CI)
- **Table:** Agent vs agent head-to-head win-rate matrix
- **Figure:** bar chart of final Elo per agent with error bars

### 2.5 Analysis Points
- Rank the agents by Elo
- State which differences are statistically significant (non-overlapping CIs)
- Discuss WHY the ranking came out as it did (tie back to method properties)

---

## 3. Experiment 2 — Training Efficiency

### 3.1 Research Question
RQ1: Does imitation learning pre-training accelerate RL convergence? **This is the thesis's central experiment.**

### 3.2 Hypothesis
Agent 2 (IL+RL) starts at a higher Elo and reaches strong play in fewer self-play games than Agent 1 (pure PPO from scratch).

### 3.3 Protocol
1. During training of BOTH Agent 1 and Agent 2 Phase 2, evaluate Elo every N self-play games (e.g., every 25k games) against a fixed ElephantEye level
2. Record (training_games, Elo) pairs throughout training
3. Plot both learning curves on the same axes

### 3.4 Output
- **Figure:** two learning curves (Elo vs training games) — Agent 1 vs Agent 2 Phase 2
- **Table:** games-to-reach-milestones (e.g., games needed to reach Elo 1500, 1700, 1800) for each
- **Metric:** area under the learning curve, final Elo, convergence speed

### 3.5 Analysis Points
- Does Agent 2 start higher? (It should — it begins post-IL)
- Does Agent 2 reach any given Elo faster? (The efficiency claim)
- Does Agent 2 have a higher ceiling, or does Agent 1 eventually catch up?
- **This directly answers RQ1 for Xiangqi — the novel contribution.**

---

## 4. Experiment 3 — Opening Theory Analysis

### 4.1 Research Question
RQ3: Do RL agents rediscover, reject, or extend classical Xiangqi opening theory?

### 4.2 Hypothesis
Stronger agents (especially MCTS+NN and IL+RL) converge toward classical openings; pure self-play (Agent 1) may develop non-standard openings.

### 4.3 Protocol
1. For each trained agent, generate ~1,000 games (agent plays both sides, low temperature)
2. Extract the first ~10 plies of each game
3. Build the agent's opening move-frequency distribution
4. Compare against the professional reference distribution from the opening database (04-DATA-PIPELINE.md section 10)
5. Compute the divergence (e.g., KL divergence) between each agent's opening distribution and the professional distribution

### 4.4 Named Openings to Track
- 当头炮 (Central Cannon) — the most common professional opening
- 顺炮 (Same-Direction Cannons)
- 列炮 / 逆炮 (Opposite Cannons)
- 飞相局 (Flying Elephant Opening)
- 仕角炮 (Advisor's Corner Cannon)

### 4.5 Output
- **Figure:** heatmap of first-move frequency per agent vs professionals
- **Table:** KL divergence from professional openings, per agent
- **Table:** frequency of each named classical opening, per agent
- **Qualitative:** highlight any novel openings a self-play agent invented

### 4.6 Analysis Points
- Which agents play most like professionals in the opening?
- Did the IL agent retain human openings after RL refinement, or drift?
- Did the pure self-play agent independently rediscover 当头炮, or find something new?
- This is an intellectually rich, presentation-friendly finding.

---

## 5. Experiment 4 — Curriculum vs Standard Self-Play

### 5.1 Research Question
RQ4: Does curriculum learning (progressively stronger opponents) beat standard self-play?

### 5.2 Hypothesis
Curriculum training reaches a given Elo faster and/or achieves a higher final Elo than standard self-play, for the same compute budget.

### 5.3 Protocol
1. Train a curriculum variant (07-TRAINING.md section 5): PPO against ElephantEye at increasing depths
2. Compare against standard self-play Agent 1 (already trained in Experiment 2)
3. Match the compute budget (same number of training games) for a fair comparison
4. Evaluate both with the standard Elo protocol

### 5.4 Output
- **Figure:** learning curves — curriculum vs standard self-play
- **Table:** final Elo and training efficiency for both
- **Metric:** games-to-milestone comparison

### 5.5 Analysis Points
- Does curriculum help, hurt, or make no difference?
- Discuss the trade-off: curriculum needs a graded opponent (ElephantEye) available; self-play doesn't

---

## 6. Experiment 5 — MCTS Simulation Ablation

### 6.1 Research Question
Supporting question: how does MCTS strength scale with the number of simulations per move, and what's the best speed/strength trade-off?

### 6.2 Hypothesis
More simulations → higher Elo, with diminishing returns; inference time grows roughly linearly with simulations.

### 6.3 Protocol
1. Take the trained MCTS+NN agent (Agent 3)
2. Evaluate it at different simulation counts: 100, 200, 400, 800, 1600 sims/move
3. For each, measure Elo (vs ElephantEye) AND average inference time per move
4. (Only run if Agent 3 was successfully trained; skip if MCTS was dropped)

### 6.4 Output
- **Figure:** Elo vs simulation count (strength curve)
- **Figure:** inference time vs simulation count
- **Table:** Elo and time per simulation setting
- **Recommendation:** the "sweet spot" simulation count for practical play

### 6.5 Analysis Points
- Where do diminishing returns set in?
- What setting balances strength and speed for the dashboard's live play?

---

## 7. Ablation Studies (Supporting, Optional)

If time allows, additional ablations strengthen the thesis:
- **Network size:** does a bigger network (more channels/blocks) help enough to justify the cost?
- **Dense vs sparse reward:** does material-shaping reward speed learning or distort strategy?
- **Data quality filter:** does filtering IL data by Elo ≥ 1800 improve the IL agent vs using all games?

Each follows the same pattern: change one variable, hold others fixed, compare Elo.

---

## 8. Result Reproducibility & Storage

For every experiment:
- Save raw game records and the results matrix to `results/tables/`
- Save all figures to `results/figures/` (with the generating script and config)
- Log the config, seeds, and git commit for each run
- Every table/figure in the thesis must be regenerable from saved data + script

---

## 9. Mapping Experiments to Thesis Chapter 4

| Thesis Section | Experiment |
|---|---|
| 4.1 Main Performance Comparison | Experiment 1 |
| 4.2 Training Efficiency | Experiment 2 |
| 4.3 Opening Theory Analysis | Experiment 3 |
| 4.4 Curriculum vs Self-Play | Experiment 4 |
| 4.5 MCTS Simulation Ablation | Experiment 5 |
| 4.6 Discussion | synthesis of all |

---

## 10. Minimum Experiments (If Scope Shrinks)

If time is short, the non-negotiable experiments are:
- **Experiment 1** (main comparison — even with just PPO vs Minimax)
- **Experiment 2** (training efficiency — the core RQ1 finding)

Experiments 3, 4, 5 are additive. A thesis with only Experiments 1 and 2 (comparing Agent 1, Agent 2, Agent 4) is still complete and publishable.

---

*Cross-references: 06-AGENTS.md (agents), 07-TRAINING.md (curriculum training), 08-EVALUATION.md (Elo & metrics), 04-DATA-PIPELINE.md (opening database), 10-DASHBOARD.md (visualizing results).*
