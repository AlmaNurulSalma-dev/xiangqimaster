# 07 — Training Procedures Specification

> **Read this when:** writing or running any training script — imitation learning, PPO self-play, MCTS training, or curriculum training. Covers training loops, hyperparameters, checkpointing, and the Colab workflow.

---

## 1. Overview of Training Procedures

There are four distinct training procedures, one per learning approach:

| Procedure | For Agent | Type | Compute |
|---|---|---|---|
| Imitation Learning | Agent 2 Phase 1 | Supervised | Low (~8 hrs) |
| PPO Self-Play | Agent 1, Agent 2 Phase 2 | RL | Medium (~15–20 hrs) |
| MCTS Self-Play | Agent 3 | RL | Very High (~100+ hrs) |
| Curriculum Training | Experiment 4 variant | RL | Medium |

Agent 4 (Minimax) requires no training.

---

## 2. Imitation Learning Training (Agent 2, Phase 1)

### 2.1 Goal
Train the Policy-Value network to imitate professional human moves via supervised learning.

### 2.2 Procedure
1. Load processed training examples from the data pipeline (04-DATA-PIPELINE.md): (board_tensor, human_move, outcome)
2. For each batch:
   - Forward pass → (policy_logits, value)
   - Policy loss: cross-entropy between policy_logits and the human move
   - Value loss: MSE between predicted value and actual outcome
   - Total loss: policy_loss + value_loss
   - Backpropagate, update weights
3. After each epoch, evaluate on the validation set (move-prediction accuracy)
4. Use early stopping if validation accuracy plateaus
5. Save the final network as the initialization for Agent 2 Phase 2

### 2.3 Hyperparameters
| Parameter | Default |
|---|---|
| Optimizer | Adam or AdamW |
| Learning rate | 1e-3 (with decay) |
| Batch size | 512 |
| Epochs | 30–50 (early stop) |
| L2 weight decay | 1e-4 |
| LR schedule | Step decay or cosine annealing |

### 2.4 Success Metric
Top-1 move-prediction accuracy on the validation set. Professional-level imitation typically reaches 40–55% top-1 accuracy (predicting the exact human move) — this is normal and sufficient; the value comes from the RL refinement afterward.

### 2.5 Time Estimate
~6–8 hours on Colab T4 for 30–50 epochs over the filtered dataset.

### 2.6 How to Run (implementation)
The data pipeline (`src.data.pipeline`) writes game-level splits to
`data/splits/{train,val,test}.jsonl`. `src.training.il_train` loads them into a
memory-light `LazyXiangqiILDataset` (encodes each position on access — the full
train split is ~9.3M positions, far too many to hold in RAM) and runs the
supervised trainer, then saves the checkpoint that seeds Agent 2 Phase 2.

```bash
# Full corpus — a GPU / Colab job (CPU is impractical at ~9.3M positions/epoch)
python -m src.training.il_train --splits data/splits --epochs 30 \
    --device cuda --save results/checkpoints/il_agent2_phase1.pt

# Quick local smoke test on a subset (caps GAMES loaded, not positions)
python -m src.training.il_train --limit-train 2000 --limit-val 500 --epochs 1
```

Useful flags: `--mirror` (left-right augmentation), `--batch-size`,
`--num-workers` (parallel data loading; use >0 on Linux/Colab), `--seed`.

> Rebuild the splits first if `data/splits/` is empty:
> `python -m src.data.pipeline --raw data/raw --out data/splits --notation iccs`.

---

## 3. PPO Self-Play Training (Agent 1 & Agent 2 Phase 2)

### 3.1 Goal
Improve the policy through self-play using PPO. Agent 1 starts from random weights; Agent 2 Phase 2 starts from the IL checkpoint.

### 3.2 The Self-Play + Update Loop
```
Repeat for many iterations:
  1. COLLECT: play a batch of self-play games with the current policy
     - each move: mask illegal moves, sample from policy
     - record (state, action, reward, value estimate, log-probability)
  2. COMPUTE ADVANTAGES: use GAE (Generalized Advantage Estimation)
     - advantage = how much better an action was than the value baseline predicted
  3. UPDATE: run several PPO epochs over the collected data
     - clipped surrogate policy loss + value loss + entropy bonus
  4. EVALUATE periodically: measure Elo vs ElephantEye
  5. CHECKPOINT: save weights
```

### 3.3 PPO Loss Components
- **Clipped policy loss:** encourages good actions but clips the update ratio to [1−ε, 1+ε] to prevent destabilizing jumps
- **Value loss:** MSE between predicted value and actual returns
- **Entropy bonus:** rewards keeping the policy uncertain (encourages exploration, prevents premature convergence)
- Total: `policy_loss + c1 × value_loss − c2 × entropy`

### 3.4 Hyperparameters
| Parameter | Default | Notes |
|---|---|---|
| Learning rate | 3e-4 | |
| Batch size | 512 | |
| PPO epochs per update | 4 | how many passes over collected data |
| Clip ε | 0.2 | PPO clipping range |
| Discount γ | 0.99 | future reward discount |
| GAE λ | 0.95 | advantage smoothing |
| Entropy coefficient | 0.01 | exploration; can decay over time |
| Value loss coefficient | 0.5 | |
| Games per iteration | 32–64 | self-play games before each update |
| Gradient clip norm | 0.5 | stability |
| Total training games | 500,000 | for meaningful strength |

### 3.5 Use MaskablePPO
Use `sb3-contrib`'s MaskablePPO so illegal moves are masked during both action selection and probability computation.

### 3.6 Self-Play Opponent Strategy
- **Simplest:** the agent plays against a copy of its own current policy
- **Better:** maintain a pool of past checkpoints and play against a random past version (prevents strategy collapse / cycling)
- **Best (for Experiment 4):** curriculum — play against progressively stronger fixed opponents (see section 5)

### 3.7 Time Estimate
- ~100k games ≈ 3–4 hours (Colab T4)
- 500k games ≈ 15–20 hours → split across multiple overnight sessions with checkpointing

---

## 4. MCTS Self-Play Training (Agent 3)

### 4.1 Goal
Train the network via AlphaZero-style MCTS self-play. This is the most compute-heavy procedure.

### 4.2 The Loop
```
Repeat for many iterations:
  1. SELF-PLAY: play games where each move is chosen by MCTS
     (N simulations per move, using the current network)
     - record (position, MCTS visit-count distribution, outcome) for each move
     - add Dirichlet noise at the root during training for exploration
  2. TRAIN: sample from the replay buffer, train network to predict:
     - the MCTS visit distribution (policy target — cross-entropy)
     - the game outcome (value target — MSE)
  3. EVALUATE: new network vs previous best (e.g., 100–200 games)
     - if new wins > 55%, promote it to "best"
  4. CHECKPOINT every few iterations
```

### 4.3 Hyperparameters
| Parameter | Default |
|---|---|
| Simulations per move | 400 (100–200 for faster iteration) |
| Games per iteration | 20–50 |
| Replay buffer size | last ~500k positions |
| Batch size | 512 |
| Learning rate | 1e-3 → decay |
| c_puct | 1.5 |
| Dirichlet noise α | 0.3 |
| Dirichlet noise weight ε | 0.25 |
| Temperature (moves 1–30) | 1.0 |
| Temperature (moves 31+) | ~0 (greedy) |
| Evaluation win threshold | 55% to promote |

### 4.4 Performance Optimization (Important)
MCTS is slow — optimize aggressively:
- **Batch leaf evaluations:** collect multiple leaves and evaluate them in one network forward pass
- **Reduce simulations for early iterations:** start with 100 sims, increase later
- **Parallelize self-play:** run multiple self-play games concurrently (multiprocessing)
- **Cache network evaluations** for repeated positions where possible

### 4.5 Time Estimate & Strategy
- Each iteration: ~30–60 min on Colab T4
- Needs 200+ iterations → ~100–200 GPU hours total
- **Strategy:** run continuously in background sessions, checkpoint every 5–10 iterations, resume across sessions. Run experiments on OTHER agents while this trains. If it can't reach meaningful strength in the available time, drop it (this is the planned fallback).

---

## 5. Curriculum Training (Experiment 4 Variant)

### 5.1 Goal
Test whether training against progressively stronger fixed opponents (a curriculum) beats standard self-play.

### 5.2 Procedure
Instead of self-play against itself, the agent trains against ElephantEye at increasing depths:
```
Stage 1: train vs ElephantEye Depth 1 until win rate > 60%
Stage 2: train vs ElephantEye Depth 3 until win rate > 55%
Stage 3: train vs ElephantEye Depth 5 until win rate > 50%
Stage 4: train vs ElephantEye Depth 7
```
Advance to the next stage when a win-rate threshold is met. Otherwise identical PPO setup to Agent 1.

### 5.3 Comparison
Compare the curriculum agent's final Elo and training efficiency against standard self-play Agent 1. This answers RQ4.

---

## 6. Checkpointing (Critical — Colab Disconnects)

Colab sessions disconnect unpredictably. Robust checkpointing is mandatory.

### 6.1 What to Save
- Network weights
- Optimizer state (so training resumes correctly)
- Current iteration/game count
- Random number generator states (for reproducibility)
- Elo history / training metrics
- Replay buffer (for MCTS — or a way to rebuild it)

### 6.2 When to Save
- PPO: every 10k games (or every iteration)
- MCTS: every 5–10 iterations
- Imitation: every epoch
- Always save the "best so far" separately from the "latest"

### 6.3 Where to Save
- Mount Google Drive in Colab and save checkpoints there (survives disconnection)
- Also push key checkpoints to a persistent store (GitHub LFS or Drive)

### 6.4 Resume Logic
Every training script must support resuming from the latest checkpoint automatically — check for an existing checkpoint on startup and continue from it.

---

## 7. Experiment Tracking

Use **Weights & Biases (wandb)** to log during every training run:
- Loss curves (policy loss, value loss, total)
- Elo rating over training games/iterations
- Win/draw/loss rates in self-play
- Entropy (for PPO — watch it doesn't collapse to zero)
- Learning rate
- Games/iterations completed

TensorBoard as a backup. These logs directly become the figures in Experiment 2 (learning curves).

---

## 8. Reproducibility Requirements

- Set and log ALL random seeds (Python, NumPy, PyTorch, environment) — see 11-CODING-STANDARDS.md
- Log the full config (all hyperparameters) with each run
- Save the git commit hash with each run
- Deterministic data splits (04-DATA-PIPELINE.md)

---

## 9. Colab Workflow

1. Mount Google Drive (persistent storage for checkpoints)
2. Clone the repo / pull latest code
3. Install dependencies
4. Check for existing checkpoint → resume if found
5. Run training with periodic checkpointing to Drive
6. When session nears timeout or disconnects → resume in a new session from the last checkpoint
7. Pull final results/checkpoints down for analysis and the dashboard

---

## 10. Training Schedule (6-Month Plan)

| When | Train |
|---|---|
| Month 2 | Agent 1 (PPO from scratch), 500k games over multiple sessions |
| Month 3 early | Agent 2 Phase 1 (Imitation Learning), ~8 hrs |
| Month 3 mid | Agent 2 Phase 2 (PPO fine-tune from IL) |
| Month 3–4 | Agent 3 (MCTS) — runs continuously in background |
| Month 4 | Curriculum variant for Experiment 4 |

---

## 11. Testing Training Code

- [ ] A tiny training run (few hundred games) completes without error
- [ ] Loss decreases on a small fixed dataset (sanity check)
- [ ] Checkpoint save + resume produces identical continued training
- [ ] Entropy does not collapse to zero immediately (PPO)
- [ ] Elo evaluation during training produces sensible increasing trend
- [ ] Seeds produce reproducible runs

---

*Cross-references: 05-NEURAL-NETWORK.md (network & losses), 06-AGENTS.md (agent definitions), 08-EVALUATION.md (Elo evaluation during training), 09-EXPERIMENTS.md (curriculum for Exp 4), 11-CODING-STANDARDS.md (seeding, config).*
