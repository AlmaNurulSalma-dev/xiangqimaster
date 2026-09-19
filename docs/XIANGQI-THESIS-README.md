# ♟️ XiangqiMaster · 象棋大师 · Sang Master Xiangqi
## Comparative Study of Deep Reinforcement Learning Approaches for Xiangqi (Chinese Chess): From Imitation Learning to Self-Play Mastery

> **Project name across languages:**
> **English:** XiangqiMaster · **中文:** 象棋大师 (Xiàngqí Dàshī) · **Bahasa Indonesia:** XiangqiMaster (Sang Master Xiangqi)
> The name plays on 大师 (Dàshī), the honorific for a master-level Xiangqi player — the system aims to reach master strength.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-AlphaZero%20Style-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)

**Final Thesis Project | Dual-Degree Program**
Universitas Islam Indonesia (UII) × Nanjing Xiaozhuang University (NXU)
Author: Alma (L25020007) | Informatics | 2026–2027

</div>

---

## 📖 Abstract

Xiangqi (Chinese Chess) is one of the world's oldest and most complex strategy games, played by an estimated 600 million people globally — yet it remains significantly underrepresented in deep reinforcement learning research compared to International Chess and Go. While AlphaZero demonstrated superhuman performance in Chess, Go, and Shogi, its application to Xiangqi has not been formally studied, and no comprehensive comparison of modern deep RL approaches exists for this game.

This thesis conducts a systematic comparative study of four deep reinforcement learning approaches applied to Xiangqi:

1. **Pure Self-Play RL** (PPO-based, learning from scratch)
2. **Imitation Learning → RL Refinement** (pre-train on human games, refine with self-play)
3. **MCTS + Neural Network** (AlphaZero-style policy/value network with Monte Carlo Tree Search)
4. **Classical Minimax Baseline** (Alpha-Beta pruning, rule-based — the traditional approach)

The central research question: *Does pre-training on human game records (imitation learning) accelerate convergence and improve final performance compared to learning purely from self-play — and which RL architecture achieves the best Elo rating against established Xiangqi engines?*

---

## 🎯 Research Questions

| ID | Question |
|---|---|
| **RQ1** | Does imitation learning pre-training on professional Xiangqi games accelerate RL convergence compared to training from scratch? |
| **RQ2** | Which RL algorithm (PPO, MCTS+NN) achieves the highest Elo rating against the ElephantEye benchmark engine? |
| **RQ3** | Do RL-trained agents rediscover, reject, or extend classical Xiangqi opening theory? |
| **RQ4** | How does curriculum learning (progressively harder opponents) compare to standard self-play in terms of training efficiency and final performance? |

---

## 🆕 Novel Contributions

### C1: First Systematic RL Comparison on Xiangqi
No existing paper comprehensively compares PPO, MCTS+NN, imitation learning, and a classical Minimax baseline specifically on Xiangqi with standardized Elo evaluation. This fills a clear gap.

### C2: Imitation Learning as RL Initialization for Xiangqi
AlphaGo used human game records as initialization before self-play. This has been validated for International Chess and Go — but **never formally studied for Xiangqi**. This work tests whether the same benefit transfers.

### C3: Opening Theory Analysis via RL Agent Behavior
By analyzing the moves RL agents converge to in the early game, this work examines whether agents independently rediscover classical Xiangqi openings (e.g., 当头炮 Cannon Opening, 顺炮 Same-Direction Cannons) or develop novel strategies — a finding with implications for both AI and Xiangqi theory.

### C4: Curriculum Learning vs. Standard Self-Play for Board Games
Tests whether a structured training curriculum (weak → medium → strong opponents) produces better results than random self-play — a comparison that has not been done specifically for Xiangqi.

---

## 🧠 Background: Why Xiangqi?

### The Game
Xiangqi is a two-player zero-sum perfect information game played on a 9×10 board. It shares ancestry with International Chess but has distinct rules:
- **River boundary** divides the board — pieces have different movement rules on each side
- **Palace** — King (General) and Advisors are confined to a 3×3 palace
- **Cannon** — moves like a Rook but must jump exactly one piece to capture
- **Elephant** — cannot cross the river
- **Average game length:** ~95 moves (longer than Chess ~40, shorter than Go ~200)
- **Branching factor:** ~40 (similar to Chess ~35, much lower than Go ~250)
- **State space:** ~10^{40} (comparable to Chess ~10^{44})

### Why It's Underresearched
| Game | AlphaZero Paper | RL Papers (approx.) | Players Worldwide |
|---|---|---|---|
| Chess | ✅ 2017 | 500+ | 600M |
| Go | ✅ 2016 | 400+ | 60M |
| Shogi | ✅ 2017 | 100+ | 20M |
| **Xiangqi** | ❌ Never | ~20 | **600M** |

Despite having as many players as Chess, Xiangqi has a fraction of the RL research — creating a genuine academic gap.

### Why It's Hard for AI
1. **Cannon movement** — can only capture by jumping over exactly one piece — highly non-standard and hard to generalize from Chess models
2. **River asymmetry** — Elephants cannot cross the river, creating asymmetric board dynamics
3. **Fewer draws** — Xiangqi has more decisive games than Chess, making it a cleaner win/loss environment for RL

---

## 📊 Data Sources

### Primary: kaifeiji/xiangqi — Xiangqi Game Collection

| Property | Detail |
|---|---|
| **Source** | GitHub `kaifeiji/xiangqi` (verified 2026-09) |
| **Size** | ~140,000 games (99,813 PGN + 41,743 WXF) |
| **Format** | PGN and WXF (Xiangqi standard notation) |
| **Quality** | Games from dpxq / WXF sources; some include player Elo & results |
| **Access** | Free download from GitHub |
| **Use in thesis** | Imitation learning pre-training dataset |

**Download:**
```bash
# Verified primary source
git clone https://github.com/kaifeiji/xiangqi
# Contains ~99.8k PGN + ~41.7k WXF games
```

> ⚠️ **Correction (verified 2026-09):** The originally-planned
> `donkeyid/xiangqi-dataset` (claimed ~800k games) does **NOT exist** (GitHub
> 404). `kaifeiji/xiangqi` (~140k games) is the best verified public
> alternative and is more than enough for imitation-learning pre-training.
> Additional fallback sources if more data is needed: Kaggle "onlinexiangqi"
> (~10k blitz games), the WXF Federation game archive, and the DpXq / 01xq
> online databases (scrapeable). Always check each repo's LICENSE and cite it.

### Benchmark Engine: ElephantEye (+ Pikafish)
- ElephantEye (`xqbase/eleeye`, LGPL-2.1, C++): well-known open-source Xiangqi engine
- Ships as **source only — no prebuilt binary**; must be compiled (Windows/Linux/Colab)
- Speaks the **UCCI** protocol (Universal *Chinese Chess* Interface) — the Xiangqi
  counterpart of UCI, **not** plain UCI
- Multiple difficulty levels (configurable search depth = different Elo levels)
- Free, open-source: https://github.com/xqbase/eleeye
- **Recommended stronger alternative:** **Pikafish** (Stockfish-based, actively
  maintained) or **Fairy-Stockfish** — both also speak UCCI, so the same
  interface module works for any of them

### Elo Rating System
- Standard Elo rating used in competitive Xiangqi
- Professional players: Elo ~2400–2700
- Amateur strong players: ~1800–2200
- Average club player: ~1200–1500
- Your goal: agent surpasses ~1800 Elo (strong amateur level)

### Data Pipeline Summary
```
kaifeiji/xiangqi raw PGN + WXF files (~140k games)
        ↓
Parser: PGN/WXF → board state tensors + move labels
        ↓
Filter: Keep games with player Elo > 1800 (quality filter)
        ↓
Split: 80% train / 10% val / 10% test (for imitation learning)
        ↓
Convert to PyTorch Dataset format
        ↓
Use for Phase 1: Imitation Learning pre-training
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION DASHBOARD                         │
│              Streamlit — Live Game Viewer                       │
│   [Board Visualizer] [Elo Chart] [Move Analysis] [Comparison]  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    AGENT COMPARISON LAYER                       │
│                                                                  │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Agent 1    │ │   Agent 2    │ │ Agent 3  │ │  Agent 4   │  │
│  │  Pure RL    │ │  IL + RL     │ │ MCTS+NN  │ │  Minimax   │  │
│  │  (PPO)      │ │  (PPO fine-  │ │ (AlphaZ- │ │ (baseline) │  │
│  │             │ │   tuned)     │ │  style)  │ │            │  │
│  └─────────────┘ └──────────────┘ └──────────┘ └────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    NEURAL NETWORK LAYER                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Policy-Value Network (shared)               │   │
│  │                                                          │   │
│  │  Input: Board State Tensor (14×10×9, channels×rows×cols)│   │
│  │         ↓                                                │   │
│  │  ResNet Backbone (19 residual blocks)                   │   │
│  │         ↓                    ↓                          │   │
│  │  Policy Head            Value Head                      │   │
│  │  (move probabilities)   (win probability)               │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    TRAINING LAYER                               │
│                                                                  │
│  Phase 1: Imitation Learning                                    │
│  Human games → supervised learning → base policy network        │
│                                                                  │
│  Phase 2: RL Refinement                                         │
│  Self-play games → PPO/MCTS → improved policy                  │
│                                                                  │
│  Phase 3: Curriculum Training (comparison experiment)          │
│  Weak opponents → Medium → Strong → ElephantEye levels         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    ENVIRONMENT LAYER                            │
│              Xiangqi Game Engine (gym-xiangqi)                  │
│   Board state · Legal move generation · Win detection           │
│         ElephantEye UCCI interface · Self-play manager          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 The Four Agents — Detailed

### Agent 1: Pure Self-Play RL (PPO)
**What it is:** Learns entirely from scratch through self-play. No human game data used.

**Algorithm: PPO (Proximal Policy Optimization)**
- Policy-gradient method — directly optimizes the probability of taking good actions
- "Proximal" = clips updates to prevent catastrophically large policy changes
- More stable than vanilla policy gradient, more sample-efficient than DQN for complex action spaces

**Training process:**
```
Initialize random policy network
        ↓
Agent plays against itself (current policy vs old policy)
        ↓
Collect (state, action, reward) tuples
        ↓
Compute advantage: A(s,a) = Q(s,a) - V(s)  ← how much better was this action than expected?
        ↓
Update policy using clipped surrogate objective:
L_CLIP = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]
        ↓
Repeat for millions of games
```

**Reward structure:**
```python
# Sparse reward (end of game only)
if game_won:    reward = +1.0
if game_lost:   reward = -1.0
if game_drawn:  reward = +0.1  # small positive — draws are rare in Xiangqi

# Optional: Dense reward shaping (material advantage)
material_reward = (your_piece_values - opponent_piece_values) × 0.01
# Piece values: General=∞, Chariot=9, Cannon=4.5, Horse=4, Elephant=2, Advisor=2, Soldier=1
```

**Baseline comparison value:** Shows how well an agent can learn with zero prior knowledge.

---

### Agent 2: Imitation Learning → RL Refinement (IL+RL)
**What it is:** Mimics AlphaGo's approach — first learn from human experts, then improve via self-play.

**Phase 1: Imitation Learning (Supervised)**
```
Input: Board state (from a human game record)
Label: Next move played by professional player
Loss: Cross-entropy loss between predicted move distribution and actual move

Result: A network that plays like an average professional player
Expected performance after IL: ~1600–1800 Elo
```

**Phase 2: RL Fine-tuning**
```
Take IL-pretrained network as starting point
        ↓
Run self-play using PPO (same as Agent 1)
        ↓
The better starting point = faster convergence hypothesis

Expected result: Reaches higher Elo faster than Agent 1
```

**Why this is your key research question:**
> Does IL initialization actually help, or does RL eventually converge to the same point regardless? For Chess/Go — yes, it helps significantly. For Xiangqi — unknown. Your paper answers this.

---

### Agent 3: MCTS + Neural Network (AlphaZero-style)
**What it is:** The most powerful approach — combines neural network evaluation with tree search.

**Monte Carlo Tree Search (MCTS):**
```
At each move, run N simulations:

1. SELECTION: Walk down the tree using UCB formula:
   UCB = Q(s,a) + c × P(s,a) × √(N(s)) / (1 + N(s,a))
   Q = average value of this action
   P = prior probability from neural network
   N = visit counts
   c = exploration constant

2. EXPANSION: At leaf node, expand with neural network:
   Policy head → prior probabilities for all legal moves
   Value head → expected outcome from this position

3. SIMULATION: Use value head (no random rollout needed)

4. BACKPROPAGATION: Update Q values up the tree

After N simulations → pick move with highest visit count
```

**The neural network (Policy-Value Network):**
```
Input: 14×10×9 tensor (channels × rows × columns)
  - 7 channels for your pieces (one per piece type: General, Advisor,
    Elephant, Horse, Chariot, Cannon, Soldier)
  - 7 channels for opponent pieces (same 7 piece types)
  - Total: 14×10×9 = 1,260 input features
  - Player-relative encoding: channels 0–6 are always "my pieces"
    (a "whose turn" plane is optional — see 03-ENVIRONMENT.md §2.3)

ResNet backbone (19 residual blocks):
  Each block: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → Skip connection

Policy head:
  → Fully connected → softmax → probability over all possible moves (~2,086 total)

Value head:
  → Fully connected → tanh → scalar in [-1, 1]
  → -1 = certain loss, +1 = certain win
```

**Training (self-play loop):**
```
1. Play game using MCTS (1,600 simulations per move)
2. Store (board_state, mcts_policy, game_outcome) for each move
3. Train network to predict: mcts_policy and game_outcome
4. Update network weights
5. Evaluate new network vs old — keep if better
6. Repeat
```

**Expected performance:** Highest of all agents, but slowest to train. Requires most GPU time.

---

### Agent 4: Classical Minimax (Baseline)
**What it is:** Traditional alpha-beta pruning search with handcrafted evaluation function. No learning.

**How it works:**
```
Search game tree to depth D (e.g., D=8 half-moves)
At each node: maximize your score, minimize opponent's score
Alpha-beta pruning: skip branches that can't affect the result

Evaluation function (handcrafted):
score = Σ(your piece values) - Σ(opponent piece values)
      + position bonus (pieces in center/attack positions worth more)
      + mobility bonus (more legal moves = better)
      + king safety penalty
```

**Why include this:** Shows how much deep RL improves over the traditional approach that Xiangqi programmers have used for 30+ years. This is your "old world vs new world" comparison.

**Implementation:** ElephantEye at depth 4–6 (equivalent to strong amateur play)

---

## 🧪 Experiments Design

### Experiment 1: Main Performance Comparison
**Question:** Which agent achieves highest Elo?

**Protocol:**
- Each agent plays 400 games against each ElephantEye strength level
- ElephantEye levels: Depth 1 (~800 Elo), 3 (~1200), 5 (~1600), 7 (~2000), 9 (~2400)
- Win rate at each level → convert to Elo using standard formula
- Report: Elo rating ± 95% confidence interval

**Expected result table:**
| Agent | Elo (estimated) | Win rate vs D5 | Win rate vs D7 |
|---|---|---|---|
| Minimax baseline | ~1400 | — | — |
| Pure RL (PPO) | ~1600–1800 | — | — |
| IL + RL | ~1800–2000 | — | — |
| MCTS + NN | ~2000–2200 | — | — |

### Experiment 2: Training Efficiency Comparison
**Question:** Does IL initialization help converge faster?

**Protocol:**
- Track Elo rating every 10,000 self-play games during training
- Plot learning curves: Elo vs training games
- Compare: Agent 1 (pure RL) vs Agent 2 (IL+RL) convergence speed

**Expected result:** Agent 2 starts higher and reaches peak Elo faster, with potentially higher ceiling.

**This is your key finding for RQ1.**

### Experiment 3: Opening Theory Analysis
**Question:** Do agents rediscover classical openings?

**Protocol:**
- Collect first 10 moves from 1,000 games played by each trained agent
- Compare to frequency distribution of openings in the professional game database
- Classical openings to check:
  - 当头炮 (Central Cannon Opening) — most common in professional play
  - 顺炮 (Same-Direction Cannons)
  - 列炮 (Ranked Cannons)
  - 飞相局 (Flying Elephant Opening)
- Measure: KL divergence between agent opening distribution and professional database

**Expected result:** MCTS+NN agent rediscovers classical openings most closely. Pure RL agent may develop novel non-classical strategies.

### Experiment 4: Curriculum Learning vs Standard Self-Play
**Question:** Does structured curriculum training beat random self-play?

**Protocol:**
- Curriculum agent: Train against ElephantEye Depth 1 → 3 → 5 → 7 (progressive)
- Standard agent: Train against random self-play throughout
- Compare final Elo rating and training efficiency
- Compare to Experiment 1 results

### Experiment 5: Ablation — MCTS Simulation Count
**Question:** How many MCTS simulations are needed for good performance?

**Protocol:**
- Test MCTS+NN with: 100, 400, 800, 1600 simulations per move
- Measure Elo rating vs inference time trade-off
- Find the "sweet spot" for practical deployment

---

## 💻 Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  LANGUAGE         Python 3.10+                                  │
├─────────────────────────────────────────────────────────────────┤
│  DEEP LEARNING    PyTorch 2.0+                                  │
│                   torchvision (ResNet backbone)                 │
├─────────────────────────────────────────────────────────────────┤
│  RL LIBRARY       stable-baselines3 (PPO implementation)        │
│                   gymnasium (environment interface)             │
├─────────────────────────────────────────────────────────────────┤
│  GAME ENGINE      gym-xiangqi (Xiangqi environment)             │
│                   python-xiangqi (board logic)                  │
│                   ElephantEye / Pikafish (benchmark, UCCI)      │
├─────────────────────────────────────────────────────────────────┤
│  DATA             kaifeiji/xiangqi dataset (~140k games)        │
│                   WXF parser (custom, ~200 lines)               │
├─────────────────────────────────────────────────────────────────┤
│  EXPERIMENT       Weights & Biases (wandb) — training tracker   │
│  TRACKING         tensorboard — loss/reward curves              │
├─────────────────────────────────────────────────────────────────┤
│  VISUALIZATION    matplotlib / plotly — Elo curves, win rates   │
│                   Streamlit — live game dashboard               │
│                   python-chess style board renderer             │
├─────────────────────────────────────────────────────────────────┤
│  TRAINING         Google Colab Pro (T4/A100 GPU, ~$10/month)    │
│                   Local CPU for environment/data processing     │
├─────────────────────────────────────────────────────────────────┤
│  VERSIONING       GitHub                                        │
│  TOTAL COST       ~$60 (Colab Pro only, all else free)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
xiangqimaster/
│
├── 📁 data/
│   ├── raw/                        # Raw PGN/WXF game files (kaifeiji/xiangqi)
│   ├── processed/                  # Parsed board state tensors
│   ├── splits/                     # train / val / test splits
│   └── openings/                   # Classical opening database
│
├── 📁 src/
│   ├── 📁 environment/
│   │   ├── xiangqi_env.py          # Custom Gymnasium environment wrapper
│   │   ├── board.py                # Board state representation
│   │   ├── pieces.py               # Piece movement rules
│   │   ├── move_generator.py       # Legal move generation
│   │   ├── win_detector.py         # Win/draw/loss detection
│   │   └── elephanteye_interface.py # UCCI protocol for ElephantEye/Pikafish
│   │
│   ├── 📁 data/
│   │   ├── wxf_parser.py           # WXF format → board tensor converter
│   │   ├── dataset.py              # PyTorch Dataset for imitation learning
│   │   ├── preprocessor.py         # Board state → 14×10×9 tensor
│   │   └── data_loader.py          # DataLoader with augmentation
│   │
│   ├── 📁 models/
│   │   ├── policy_value_net.py     # ResNet policy-value network
│   │   ├── resnet_backbone.py      # 19-block ResNet implementation
│   │   ├── policy_head.py          # Move probability output head
│   │   └── value_head.py           # Win probability output head
│   │
│   ├── 📁 agents/
│   │   ├── ppo_agent.py            # Agent 1: Pure PPO self-play
│   │   ├── il_rl_agent.py          # Agent 2: Imitation Learning + RL
│   │   ├── mcts_agent.py           # Agent 3: MCTS + Neural Network
│   │   ├── minimax_agent.py        # Agent 4: Alpha-beta baseline
│   │   └── mcts.py                 # MCTS core implementation (UCB, backprop)
│   │
│   ├── 📁 training/
│   │   ├── imitation_trainer.py    # Phase 1: Supervised IL training
│   │   ├── rl_trainer.py           # Phase 2: PPO self-play training
│   │   ├── mcts_trainer.py         # AlphaZero-style MCTS training loop
│   │   ├── curriculum_trainer.py   # Experiment 4: Curriculum training
│   │   └── self_play_manager.py    # Parallel self-play game generation
│   │
│   ├── 📁 evaluation/
│   │   ├── elo_calculator.py       # Elo rating computation
│   │   ├── tournament.py           # Agent vs agent / vs ElephantEye
│   │   ├── opening_analyzer.py     # Experiment 3: Opening analysis
│   │   └── metrics.py              # Win rate, draw rate, game length
│   │
│   └── 📁 utils/
│       ├── config.py               # Global hyperparameters
│       ├── logger.py               # Wandb + tensorboard logging
│       ├── checkpoint.py           # Save/load model checkpoints
│       └── visualizer.py           # Board state renderer
│
├── 📁 dashboard/
│   ├── app.py                      # Streamlit app entry point
│   └── 📁 pages/
│       ├── 01_live_game.py         # Watch agent play in real time
│       ├── 02_elo_comparison.py    # Elo rating chart comparison
│       ├── 03_learning_curves.py   # Training progress visualization
│       ├── 04_opening_analysis.py  # Opening move heatmaps
│       └── 05_move_explorer.py     # Step through any game move by move
│
├── 📁 experiments/
│   ├── exp1_main_comparison.py     # Experiment 1: Main Elo comparison
│   ├── exp2_training_efficiency.py # Experiment 2: Learning curves
│   ├── exp3_opening_theory.py      # Experiment 3: Opening analysis
│   ├── exp4_curriculum.py          # Experiment 4: Curriculum vs self-play
│   └── exp5_mcts_ablation.py       # Experiment 5: MCTS simulation count
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_board_state_visualization.ipynb
│   ├── 03_training_progress.ipynb
│   └── 04_results_analysis.ipynb
│
├── 📁 results/
│   ├── tables/                     # CSV result tables
│   ├── figures/                    # Generated plots
│   └── checkpoints/                # Saved model weights (gitignored)
│
├── 📁 docs/
│   ├── xiangqi_rules.md            # Complete Xiangqi rules reference
│   ├── rl_background.md            # RL theory background
│   ├── data_format.md              # WXF format specification
│   └── thesis_outline.md           # Chapter-by-chapter outline
│
├── .env.example
├── .gitignore
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/yourusername/xiangqimaster.git
cd xiangqimaster

# Create conda environment
conda env create -f environment.yml
conda activate xiangqimaster

# Or with pip
pip install -r requirements.txt
```

### 2. Download Data
```bash
# Download the game dataset (verified source)
git clone https://github.com/kaifeiji/xiangqi data/raw/

# Build the ElephantEye engine from source (no prebuilt binary is published).
# ElephantEye is C++; compile it, then place the binary in src/environment/.
git clone https://github.com/xqbase/eleeye
cd eleeye && make            # produces the eleeye binary (see repo build docs)
mv eleeye/eleeye ../src/environment/ && cd ..

# Alternative (recommended, stronger + actively maintained): Pikafish.
# See https://github.com/official-pikafish/Pikafish for prebuilt binaries.
```

### 3. Preprocess Data
```bash
# Parse WXF files → board state tensors
python src/data/wxf_parser.py --input data/raw/ --output data/processed/

# Create train/val/test splits
python src/data/preprocessor.py --min-elo 1800
```

### 4. Train Agents

```bash
# Agent 1: Pure RL (PPO)
python src/training/rl_trainer.py \
    --agent ppo \
    --games 500000 \
    --save-every 10000

# Agent 2: Imitation Learning first
python src/training/imitation_trainer.py \
    --data data/processed/train/ \
    --epochs 50 \
    --save models/il_pretrained.pth

# Then RL fine-tuning
python src/training/rl_trainer.py \
    --agent ppo \
    --pretrained models/il_pretrained.pth \
    --games 500000

# Agent 3: MCTS + Neural Network
python src/training/mcts_trainer.py \
    --simulations 800 \
    --games 200000

# Agent 4: Minimax (no training needed)
# Uses ElephantEye directly
```

### 5. Run Evaluation Tournament
```bash
# Evaluate all agents against ElephantEye
python experiments/exp1_main_comparison.py \
    --games-per-matchup 400 \
    --elephanteye-depths 1 3 5 7 9
```

### 6. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📅 Development Timeline

| Phase | Month | Activities | Milestone |
|---|---|---|---|
| **1 — Foundation** | Month 1 | Study RL theory (PPO, MCTS, IL), set up Xiangqi environment, parse the game dataset, verify ElephantEye/Pikafish integration | ✅ Working Xiangqi environment + clean dataset |
| **2 — Baseline Agents** | Month 2 | Implement PPO agent (Agent 1), train pure RL from scratch, implement minimax baseline (Agent 4), first Elo evaluation | ✅ Two agents running, first results |
| **3 — IL + MCTS** | Month 3 | Implement imitation learning (Agent 2), implement MCTS+NN (Agent 3), run Experiments 1 & 2 | ✅ All four agents trained |
| **4 — Advanced Experiments** | Month 4 | Experiment 3 (opening analysis), Experiment 4 (curriculum), Experiment 5 (MCTS ablation), dashboard build | ✅ All 5 experiments complete |
| **5 — Writing** | Month 5 | Thesis document (all chapters), paper draft, figure generation, final results tables | ✅ Thesis submitted, paper drafted |
| **6 — Buffer** | Month 6 | Revisions, presentation prep, NXU + UII submission | ✅ Final submission |

---

## 📈 Evaluation Metrics

### Primary Metric: Elo Rating
```
Elo formula:
Expected score: E_A = 1 / (1 + 10^((R_B - R_A)/400))
New rating:     R_A' = R_A + K × (S_A - E_A)

K = 32 (standard for developing players)
S_A = actual score (1=win, 0.5=draw, 0=loss)

Elo confidence interval (95%):
CI = ± 1.96 × √(N × p × (1-p)) / N × (800 / ln(10))
where p = win rate, N = number of games
```

### Secondary Metrics
| Metric | Description | Formula |
|---|---|---|
| Win Rate | % of games won against specific opponent | wins / total_games |
| Draw Rate | % of games drawn | draws / total_games |
| Game Length | Average moves per game | Σ(moves) / games |
| Training Efficiency | Elo gain per 10k training games | ΔElo / Δgames |
| Opening Divergence | KL divergence from professional openings | Σ p(x) log(p(x)/q(x)) |

---

## 🎓 Thesis Chapter Outline

```
Chapter 1: Introduction
├── 1.1 Background — Xiangqi and AI
├── 1.2 Research Problem and Motivation
├── 1.3 Research Questions (RQ1–RQ4)
├── 1.4 Novel Contributions (C1–C4)
├── 1.5 Thesis Structure

Chapter 2: Literature Review
├── 2.1 Reinforcement Learning — Key Algorithms
│   ├── 2.1.1 Q-Learning and DQN
│   ├── 2.1.2 Policy Gradient Methods and PPO
│   └── 2.1.3 Monte Carlo Tree Search
├── 2.2 Deep RL for Board Games
│   ├── 2.2.1 AlphaGo and AlphaGo Zero
│   ├── 2.2.2 AlphaZero — Chess, Go, Shogi
│   ├── 2.2.3 MuZero
│   └── 2.2.4 Existing Xiangqi AI Systems
├── 2.3 Imitation Learning in Games
├── 2.4 Curriculum Learning
└── 2.5 Research Gap and Positioning

Chapter 3: Methodology
├── 3.1 Xiangqi Game Formalization
│   ├── 3.1.1 State Space Representation
│   ├── 3.1.2 Action Space
│   └── 3.1.3 Reward Function Design
├── 3.2 Data Collection and Preprocessing
├── 3.3 Neural Network Architecture
├── 3.4 Agent 1: PPO Self-Play
├── 3.5 Agent 2: Imitation Learning + RL
├── 3.6 Agent 3: MCTS + Neural Network
├── 3.7 Agent 4: Minimax Baseline
└── 3.8 Evaluation Protocol

Chapter 4: Experiments and Results
├── 4.1 Experiment 1: Main Performance Comparison
├── 4.2 Experiment 2: Training Efficiency
├── 4.3 Experiment 3: Opening Theory Analysis
├── 4.4 Experiment 4: Curriculum vs Self-Play
├── 4.5 Experiment 5: MCTS Simulation Ablation
└── 4.6 Discussion and Analysis

Chapter 5: Conclusion
├── 5.1 Summary of Contributions
├── 5.2 Answers to Research Questions
├── 5.3 Limitations
└── 5.4 Future Work
```

---

## 📚 Key References

### Must-Read Papers (Read Before Starting)

1. **Silver et al. (2016)** — *Mastering the game of Go with deep neural networks and tree search* — Nature. **[AlphaGo original paper]**

2. **Silver et al. (2017)** — *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* — arXiv. **[AlphaZero paper]**

3. **Schulman et al. (2017)** — *Proximal Policy Optimization Algorithms* — arXiv. **[PPO algorithm you use]**

4. **Mnih et al. (2015)** — *Human-level control through deep reinforcement learning* — Nature. **[DQN paper]**

5. **Pomerleau (1991)** — *Efficient Training of Artificial Neural Networks for Autonomous Navigation* — Neural Computation. **[Foundational imitation learning]**

6. **Ye et al. (2021)** — *Mastering Atari Games with Limited Data* — NeurIPS. **[EfficientZero — relevant MuZero variant]**

7. **Suphx (Li et al., 2020)** — *Suphx: Mastering Mahjong with Deep Reinforcement Learning* — arXiv. **[Most relevant related work — Mahjong RL]**

### Xiangqi-Specific References

8. **Liu & Tsuruoka (2017)** — *Modifications of the AlphaGo Zero approach for games with incomplete information* — AAAI workshop.

9. **WXF Notation Specification** — World Xiangqi Federation (WXF) game-record format; plus the `kaifeiji/xiangqi` dataset repository documentation.

### Target Publication Venues

| Venue | Type | Fit | Deadline cycle |
|---|---|---|---|
| *IEEE Transactions on Games* | Journal Q2 | ⭐⭐⭐⭐⭐ | Rolling |
| *IEEE Conference on Games (CoG)* | Conference | ⭐⭐⭐⭐⭐ | Annual (Aug) |
| *AAAI* | Top conference | ⭐⭐⭐⭐ | Annual (Sep) |
| *IJCAI* | Top conference | ⭐⭐⭐⭐ | Annual (Jan) |
| *ECAI* | Conference | ⭐⭐⭐⭐ | Annual |

---

## 💰 Project Cost Breakdown

| Item | Cost |
|---|---|
| Google Colab Pro (5 months) | ~$50 |
| Game dataset (kaifeiji/xiangqi) | **Free** |
| ElephantEye engine | **Free** |
| All Python libraries | **Free** |
| GitHub | **Free** |
| Streamlit Cloud deployment | **Free** |
| **Total** | **~$50** |

---

## ⚠️ Known Challenges & Solutions

| Challenge | Solution |
|---|---|
| MCTS training is very slow | Use Google Colab A100, parallelize self-play with multiprocessing |
| ~140k games takes time to parse | Run preprocessing once, cache as .pkl tensors |
| Elo evaluation takes many games | Run tournaments overnight, evaluate every 50k training steps |
| MCTS may be too slow for real-time play in dashboard | Pre-compute moves with fixed simulation budget (400 sims), cache results |
| WXF format parsing edge cases | Use existing python-xiangqi library as validator |

---

## 🔄 Fallback Strategy

```
Full thesis (4 agents + 5 experiments)
    ↓ MCTS too slow to train?
Drop Agent 3 (MCTS) → 3 agents + 4 experiments → still publishable
    ↓ IL data pipeline issues?
Drop Agent 2 (IL+RL) → 2 agents (PPO vs Minimax) + Experiment 1 only
→ Still a valid comparison paper
    ↓ Absolute minimum
Pure PPO agent vs ElephantEye benchmarks + Elo analysis
→ "Training a PPO Agent for Xiangqi" — lighter contribution but defensible
```

---

## 👤 Author

**Alma (诺艾玛)**
Student ID: L25020007
Dual-Degree Program — Informatics
Universitas Islam Indonesia (UII) × Nanjing Xiaozhuang University (NXU)

---

## 📄 License

MIT License — open source. See `LICENSE` for details.

---

<div align="center">
<b>♟️ XiangqiMaster</b><br>
<i>Bringing modern deep reinforcement learning to the world's most underrepresented major board game.</i>
</div>
