# ♟️ XiangqiMaster — Master Index & Navigation Guide

> **THIS IS THE ENTRY POINT.** Any AI agent or developer starting work on this project should read this file FIRST. It explains what the project is, how the documentation is organized, and — most importantly — **which document to read for which task**.

---

## 🏷️ Project Name (Trilingual)

| Language | Name | Pronunciation / Notes |
|---|---|---|
| **English** | **XiangqiMaster** | The primary project & repository name |
| **中文 (Chinese)** | **象棋大师** | Xiàngqí Dàshī — "Xiangqi Grandmaster"; 大师 (Dàshī) is the honored title for a master-level Xiangqi player |
| **Bahasa Indonesia** | **XiangqiMaster** *(Sang Master Xiangqi)* | Uses the English name; Indonesian descriptive tagline "Sang Master Xiangqi" = "The Xiangqi Master" |

**Full academic title:**
*"XiangqiMaster (象棋大师): A Comparative Study of Deep Reinforcement Learning Approaches for Xiangqi (Chinese Chess) — From Imitation Learning to Self-Play Mastery"*

**Why this name:** "Master / 大师 (Dàshī)" is the honorific for a master-level Xiangqi player. The name subtly claims the system aims to play at master strength, is instantly recognizable to Chinese academic audiences (carrying real cultural weight in the Xiangqi world), reads clearly in Indonesian and US English, and keeps the full word "Xiangqi" for clarity. Use **XiangqiMaster** as the code/repo name everywhere; use **象棋大师** in Chinese-facing contexts (NXU thesis cover, presentations for Chinese lecturers).

---

## 🎯 What This Project Is (One Paragraph)

XiangqiMaster is a final thesis research project that designs, trains, and compares four deep reinforcement learning approaches for playing Xiangqi (Chinese Chess): (1) Pure Self-Play PPO, (2) Imitation Learning followed by RL refinement, (3) MCTS combined with a neural network in the AlphaZero style, and (4) a classical Minimax baseline. Each trained agent is evaluated using standardized Elo rating against the ElephantEye benchmark engine. The central research question is whether pre-training on professional human game records accelerates learning and improves final playing strength compared to learning purely from self-play. The final output is a set of trained agents, a full experimental comparison, an interactive evaluation dashboard, and an academic thesis targeting publication.

---

## 📚 Documentation Map — Which File Do I Read?

This project's documentation is split into 13 focused files. **Do not guess — use this table to route to the correct document.**

| If the task involves... | Read this document |
|---|---|
| Getting oriented, understanding the whole project | **README.md** (this file) |
| Xiangqi rules, how pieces move, notation, win conditions | **01-GAME-RULES.md** |
| How the codebase is organized, how modules connect, data flow | **02-ARCHITECTURE.md** |
| The game environment: board state, actions, rewards, step/reset | **03-ENVIRONMENT.md** |
| Loading data, parsing WXF/PGN files, the game dataset, preprocessing | **04-DATA-PIPELINE.md** |
| The neural network: layers, tensor shapes, policy/value heads | **05-NEURAL-NETWORK.md** |
| Any of the four agents: PPO, IL+RL, MCTS, Minimax logic | **06-AGENTS.md** |
| Training loops, hyperparameters, checkpointing, Colab workflow | **07-TRAINING.md** |
| Elo rating, ElephantEye benchmark, tournaments, evaluation metrics | **08-EVALUATION.md** |
| Running the five experiments and producing result tables | **09-EXPERIMENTS.md** |
| Building the Streamlit dashboard and its visualizations | **10-DASHBOARD.md** |
| Naming conventions, file structure, docstrings, testing rules | **11-CODING-STANDARDS.md** |
| The meaning of any RL, Xiangqi, or technical term | **12-GLOSSARY.md** |

---

## 🧭 Recommended Reading Order (For a New Contributor)

If reading the project cold, follow this order:

1. **README.md** (this file) — the big picture
2. **12-GLOSSARY.md** — so terms make sense throughout
3. **01-GAME-RULES.md** — understand the game being modeled
4. **02-ARCHITECTURE.md** — understand how the code is organized
5. **03-ENVIRONMENT.md** — the foundation everything is built on
6. **04-DATA-PIPELINE.md** — where training data comes from
7. **05-NEURAL-NETWORK.md** — the brain of the agents
8. **06-AGENTS.md** — the four approaches being compared
9. **07-TRAINING.md** — how agents are trained
10. **08-EVALUATION.md** — how agents are measured
11. **09-EXPERIMENTS.md** — the research questions in action
12. **10-DASHBOARD.md** — how results are presented
13. **11-CODING-STANDARDS.md** — reference while writing code

---

## 🔨 Build Order (For Implementation)

When actually building the project, implement in this dependency order:

```
STEP 1: Environment foundation
   → Read 01-GAME-RULES, 03-ENVIRONMENT, 11-CODING-STANDARDS
   → Build: board representation, move generation, win detection

STEP 2: Data pipeline
   → Read 04-DATA-PIPELINE
   → Build: WXF parser, dataset loaders, preprocessing

STEP 3: Neural network
   → Read 05-NEURAL-NETWORK
   → Build: Policy-Value network

STEP 4: Baseline agent (Minimax)
   → Read 06-AGENTS (Minimax section)
   → Build: Agent 4 + ElephantEye integration (08-EVALUATION)

STEP 5: First RL agent (PPO)
   → Read 06-AGENTS (PPO section), 07-TRAINING
   → Build: Agent 1

STEP 6: Evaluation framework
   → Read 08-EVALUATION
   → Build: Elo calculator, tournament manager

STEP 7: Imitation Learning agent
   → Read 06-AGENTS (IL+RL section), 04-DATA-PIPELINE
   → Build: Agent 2

STEP 8: MCTS agent (hardest, optional)
   → Read 06-AGENTS (MCTS section)
   → Build: Agent 3

STEP 9: Experiments
   → Read 09-EXPERIMENTS
   → Run all experiments, collect results

STEP 10: Dashboard
   → Read 10-DASHBOARD
   → Build: Streamlit app
```

---

## 📐 Core Project Facts (Quick Reference)

| Fact | Value |
|---|---|
| Game | Xiangqi (Chinese Chess), 9×10 board |
| Board state tensor shape | 14 × 10 × 9 (channels × rows × columns) |
| Action space size | 2,086 possible moves |
| Number of agents | 4 (PPO, IL+RL, MCTS+NN, Minimax) |
| Number of experiments | 5 |
| Primary dataset | kaifeiji/xiangqi — ~140,000 games (99.8k PGN + 41.7k WXF) |
| Benchmark engine | ElephantEye / Pikafish (via UCCI protocol) |
| Primary metric | Elo rating |
| Language | Python 3.10+ |
| DL framework | PyTorch 2.0+ |
| RL library | stable-baselines3 (MaskablePPO) |
| Environment interface | Gymnasium |
| Dashboard | Streamlit |
| Training compute | Google Colab Pro (T4/A100) |
| Timeline | 6 months |
| Total budget | ~$50–60 |

---

## 🎓 The Four Research Questions

Every part of this project ultimately serves to answer these:

- **RQ1:** Does imitation learning pre-training accelerate RL convergence for Xiangqi?
- **RQ2:** Which RL algorithm (PPO, MCTS+NN) achieves the highest Elo?
- **RQ3:** Do RL agents rediscover classical Xiangqi opening theory?
- **RQ4:** Does curriculum learning beat standard self-play?

---

## ⚠️ Critical Constraints (Never Violate These)

1. **Action masking is mandatory** — agents must NEVER select illegal moves. Every policy output must be masked by legal moves before sampling. (See 03-ENVIRONMENT and 06-AGENTS.)
2. **Checkpoint frequently** — Colab disconnects. Save model weights every N iterations. (See 07-TRAINING.)
3. **All randomness must be seeded** — for reproducibility of research results. (See 11-CODING-STANDARDS.)
4. **Never train on test data** — keep the train/val/test split strict. (See 04-DATA-PIPELINE.)
5. **Elo evaluation needs enough games** — minimum 400 games per matchup for statistical significance. (See 08-EVALUATION.)

---

## 🗂️ Modular Fallback (If Scope Must Shrink)

The project degrades gracefully. If time or compute runs short, drop components in this order:
1. Drop Agent 3 (MCTS) → keep PPO, IL+RL, Minimax
2. Drop Agent 2 (IL+RL) → keep PPO, Minimax
3. Minimum viable: PPO vs Minimax + Elo evaluation

The non-negotiable core is: **one PPO agent + one Minimax baseline + Elo evaluation against ElephantEye.**

---

*Author: Alma (L25020007) | UII × NXU Dual-Degree Informatics | 2026–2027*
