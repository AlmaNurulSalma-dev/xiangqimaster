# 06 — The Four Agents Specification

> **Read this when:** implementing any of the four agents' decision-making logic. Each agent is a different approach to choosing moves. This document specifies what each agent is, how it selects moves, and how it's structured. (Training procedures are in 07-TRAINING.md.)

---

## 1. Common Agent Interface

All four agents inherit from a common abstract base (`base_agent.py`). Every agent must implement:

- **`select_move(state, legal_mask) → action_index`** — given the current board state and legal move mask, return the chosen move. This is the core method used during evaluation and play.
- **`name`** — a string identifier for logging and result tables.

This uniform interface lets the tournament and dashboard treat all agents identically. The Minimax agent and the neural agents all expose the same `select_move` method despite very different internals.

---

## 2. Agent 4 — Minimax Baseline (Build This First)

**Why first:** It requires no training and gives an immediate baseline to evaluate the RL agents against. It also validates the environment (a working Minimax proves the move generation and rules are correct).

### 2.1 What It Is
A classical game-tree search using **alpha-beta pruning** with a **handcrafted evaluation function**. No learning. This represents the traditional approach to Xiangqi AI used for decades.

### 2.2 How It Selects Moves
1. Search the game tree to a fixed depth D (default D = 4 plies; deeper = stronger but slower)
2. At each node, alternate between maximizing (own turn) and minimizing (opponent's turn) the evaluation score
3. Use alpha-beta pruning to skip branches that cannot affect the result
4. Return the move leading to the best guaranteed score

### 2.3 Evaluation Function
A weighted sum of features (from the current player's perspective):
- **Material:** sum of own piece values − sum of opponent piece values (piece values from 01-GAME-RULES.md section 8) — this is the dominant term
- **Mobility (optional):** number of legal moves available (more = better)
- **Position (optional):** small bonuses for pieces in strong positions (e.g., Chariots on open files, Soldiers advanced across the river)
- **King safety (optional):** penalty when the General is exposed

Start with material-only; add other terms only if the baseline is too weak.

### 2.4 Simplest Path
The ElephantEye engine at a low fixed depth IS effectively a strong Minimax agent. You may use ElephantEye at a shallow depth as the "Minimax baseline" rather than writing your own — but writing a simple one is educational and gives full control. Document which you chose.

### 2.5 Expected Strength
Depth-4 material-only Minimax ≈ 1200–1500 Elo (weak-to-moderate club player). This is the floor the RL agents should beat.

---

## 3. Agent 1 — Pure Self-Play PPO

### 3.1 What It Is
An agent that learns **entirely from scratch through self-play**, using **PPO (Proximal Policy Optimization)**. It starts knowing nothing (random weights) and improves by playing millions of games against itself. No human data.

### 3.2 The Network
Uses the Policy-Value network from 05-NEURAL-NETWORK.md:
- The **policy head** is the "actor" — it proposes moves
- The **value head** is the "critic" — it estimates how good the position is

### 3.3 How It Selects Moves
- **During training:** the network outputs policy logits → apply legal mask (set illegal moves to −∞) → softmax → sample a move from the resulting distribution (stochastic, for exploration). Early in each game, higher temperature encourages exploration; later, lower temperature for sharper play.
- **During evaluation:** apply mask → softmax → pick the highest-probability legal move (greedy/deterministic).

### 3.4 Why PPO
PPO is a policy-gradient method that:
- Directly optimizes the policy toward moves that led to good outcomes
- Uses a "clipped" objective to prevent destabilizing large policy updates
- Is more stable than vanilla policy gradient and simpler to tune than many alternatives
- Handles large discrete action spaces well (with masking)

### 3.5 Key Requirement: MaskablePPO
Because of illegal moves, use **MaskablePPO** (from `sb3-contrib`) rather than vanilla PPO. It natively supports action masking so the agent never wastes probability mass on illegal moves. This is essential — vanilla PPO would be extremely inefficient here.

### 3.6 Reward
Sparse terminal reward (+1 win / −1 loss / +0.1 draw) from 03-ENVIRONMENT.md. Optionally dense material reward as a config toggle for faster early learning.

### 3.7 Expected Strength
With enough self-play (hundreds of thousands of games), expect ~1600–1800 Elo. Pure self-play from scratch is sample-inefficient, so this is the slowest-improving of the neural agents early on.

---

## 4. Agent 2 — Imitation Learning + RL Refinement

### 4.1 What It Is
The AlphaGo-style two-phase approach — **THIS IS THE THESIS'S KEY AGENT** because it directly tests RQ1 (does human pre-training help?).

**Phase 1 — Imitation Learning (IL):** Train the network by supervised learning to imitate human professional moves from the game dataset (kaifeiji/xiangqi; see 04-DATA-PIPELINE.md).

**Phase 2 — RL Refinement:** Take the IL-trained network as the starting point, then continue improving it via PPO self-play (identical to Agent 1, but starting from a good initialization instead of random).

### 4.2 Phase 1 Details (Imitation Learning)
- Input: board positions from human games
- Target: the move the human played (policy) + the game outcome (value)
- Loss: cross-entropy (policy) + MSE (value)
- This is pure supervised learning — no self-play, no environment interaction
- Result: a network that plays like an average professional (~1600–1800 Elo before any RL)

### 4.3 Phase 2 Details (RL Refinement)
- Identical PPO self-play to Agent 1
- BUT the network weights start from the IL-trained checkpoint
- Hypothesis: this reaches higher Elo faster than Agent 1's from-scratch training

### 4.4 The Core Comparison (RQ1)
Agent 1 vs. Agent 2 is the heart of the thesis:
- Agent 1: random init → self-play
- Agent 2: human init → self-play
- Compare: convergence speed (Elo vs training games) and final Elo ceiling

**Expected finding:** Agent 2 starts much higher and may reach a higher ceiling — but the thesis's value is in MEASURING this precisely for Xiangqi, which has never been done.

### 4.5 Expected Strength
Agent 2 typically reaches ~1800–2000 Elo, higher than Agent 1, and gets there in fewer self-play games.

---

## 5. Agent 3 — MCTS + Neural Network (AlphaZero-style)

> **This is the most powerful but most compute-intensive agent. It is the primary fallback candidate to drop if compute/time runs short.**

### 5.1 What It Is
Combines **Monte Carlo Tree Search (MCTS)** with the Policy-Value network. Instead of directly playing the network's favorite move, it uses the network to guide a lookahead search, then plays the move the search found best. This is the AlphaZero approach.

### 5.2 Why It's Stronger
The raw network is fast but shallow (one forward pass). MCTS uses the network to explore many possible futures before committing — trading compute for much stronger play. The value head replaces the slow random rollouts of classical MCTS.

### 5.3 How MCTS Selects a Move
For each move decision, run N simulations (default N = 400). Each simulation has four phases:

**1. Selection:** Starting from the root (current position), walk down the tree choosing children by the **PUCT formula**, which balances:
- Exploitation: the average value Q(s,a) of that action so far
- Exploration: the network's prior probability P(s,a) for that action, scaled down by how often it's already been visited
- The formula: `PUCT = Q(s,a) + c_puct × P(s,a) × √(ΣN(s,b)) / (1 + N(s,a))`
  - `c_puct` is the exploration constant (default ~1.0–4.0, tune it)

**2. Expansion:** When reaching a leaf (unexpanded node), run the network on that position to get:
- Prior probabilities P for all legal moves (masked)
- A value estimate V for the position
Add the node's children with these priors.

**3. Evaluation:** Use the network's value V for the leaf (no random rollout needed — this is the AlphaZero improvement).

**4. Backpropagation:** Propagate V back up the path to the root, updating visit counts N(s,a) and average values Q(s,a) for every node on the path.

After N simulations, the move is chosen by **visit count** (the most-visited child), not by raw value — visit count is a more robust signal.

### 5.4 Training (Self-Play with MCTS)
See 07-TRAINING.md for the full loop. In brief:
1. Play self-play games where each move is chosen by MCTS
2. Record (position, MCTS visit-count distribution, eventual outcome) for every move
3. Train the network to predict the MCTS distribution (policy) and the outcome (value)
4. The improved network makes better MCTS searches → better data → repeat

### 5.5 Critical Requirements
- The environment must support `clone()` (03-ENVIRONMENT.md section 6) — MCTS simulates thousands of hypothetical positions
- The network must be small/fast enough that thousands of evaluations per move are feasible
- Batching leaf evaluations across simulations speeds things up significantly

### 5.6 Key Hyperparameters
| Parameter | Default | Effect |
|---|---|---|
| Simulations per move (N) | 400 | More = stronger but slower (Experiment 5 studies this) |
| c_puct | 1.5 | Exploration vs exploitation balance |
| Temperature (early game) | 1.0 | Exploration in opening moves |
| Temperature (after move 30) | ~0 | Greedy play in midgame/endgame |
| Dirichlet noise at root | ε=0.25, α=0.3 | Extra exploration during training self-play |

### 5.7 Expected Strength
The strongest agent — potentially ~2000–2200+ Elo — but requires the most GPU time by far (see 07-TRAINING.md time estimates).

---

## 6. Agent Comparison Summary

| | Agent 1 PPO | Agent 2 IL+RL | Agent 3 MCTS+NN | Agent 4 Minimax |
|---|---|---|---|---|
| Uses human data? | No | Yes (Phase 1) | No | No |
| Uses neural network? | Yes | Yes | Yes | No |
| Uses tree search at play time? | No | No | Yes | Yes (alpha-beta) |
| Learns? | Yes | Yes | Yes | No |
| Training cost | Medium | Medium | Very High | None |
| Expected Elo | ~1600–1800 | ~1800–2000 | ~2000–2200 | ~1200–1500 |
| Primary research role | Baseline RL | Tests RQ1 (IL benefit) | Tests RQ2 (best algo) | Classical baseline |
| Drop priority if scope shrinks | Keep (core) | Keep (core) | Drop first | Keep (core) |

---

## 7. Which Agents Answer Which Research Questions

- **RQ1 (does IL help?):** Agent 1 vs Agent 2
- **RQ2 (best RL algorithm?):** Agent 1 vs Agent 2 vs Agent 3
- **RQ3 (opening rediscovery?):** analyze all neural agents' opening moves
- **RQ4 (curriculum vs self-play?):** a curriculum-trained variant vs standard Agent 1 (see 07-TRAINING and 09-EXPERIMENTS)

---

## 8. Testing the Agents

- [ ] Every agent's `select_move` returns only LEGAL moves (test on 10,000 positions)
- [ ] Minimax finds forced mates when they exist (test known mate-in-1 and mate-in-2 positions)
- [ ] PPO agent's move sampling respects the legal mask
- [ ] MCTS `clone()` never corrupts the real game state
- [ ] MCTS visit counts concentrate on reasonable moves as simulations increase
- [ ] All agents can play a full game to completion against each other without errors

---

*Cross-references: 05-NEURAL-NETWORK.md (the shared network), 07-TRAINING.md (how each is trained), 08-EVALUATION.md (how they're measured), 03-ENVIRONMENT.md (clone for MCTS).*
