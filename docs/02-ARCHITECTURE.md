# 02 — System Architecture & Code Organization

> **Read this when:** you need to understand how the whole system fits together, where a new module belongs, how data flows between components, or before creating any new file. This is the map of the codebase.

---

## 1. High-Level System Overview

The system has five layers, each depending only on the layer(s) below it. Data and control flow upward; dependencies point downward.

```
┌──────────────────────────────────────────────────┐
│  LAYER 5: PRESENTATION                            │
│  Streamlit dashboard — visualizes results         │
│  (depends on: results files, trained agents)      │
└──────────────────────────────────────────────────┘
                      ▲
┌──────────────────────────────────────────────────┐
│  LAYER 4: EXPERIMENTS & EVALUATION                │
│  Runs the 5 experiments, computes Elo, tournaments │
│  (depends on: agents, environment)                │
└──────────────────────────────────────────────────┘
                      ▲
┌──────────────────────────────────────────────────┐
│  LAYER 3: AGENTS                                  │
│  PPO, IL+RL, MCTS+NN, Minimax                     │
│  (depends on: neural network, environment)        │
└──────────────────────────────────────────────────┘
                      ▲
┌──────────────────────────────────────────────────┐
│  LAYER 2: LEARNING COMPONENTS                     │
│  Neural network, training loops, data pipeline    │
│  (depends on: environment)                        │
└──────────────────────────────────────────────────┘
                      ▲
┌──────────────────────────────────────────────────┐
│  LAYER 1: FOUNDATION                              │
│  Xiangqi environment — board, rules, moves        │
│  (depends on: nothing — pure game logic)          │
└──────────────────────────────────────────────────┘
```

**The golden rule:** Layer 1 (environment) must NEVER import from any higher layer. It is pure, self-contained game logic. This keeps it testable and reusable.

---

## 2. Complete Folder Structure

```
xiangqimaster/
│
├── docs/                          # ALL documentation (these 13 files)
│
├── data/
│   ├── raw/                       # Downloaded WXF game files (gitignored, large)
│   ├── processed/                 # Parsed board-state tensors (gitignored)
│   ├── splits/                    # train/val/test index files
│   └── openings/                  # Classical opening reference database
│
├── src/
│   ├── environment/               # LAYER 1 — pure game logic
│   │   ├── board.py               # Board state, piece placement
│   │   ├── pieces.py              # Piece definitions & movement rules
│   │   ├── move_generator.py      # Legal move generation
│   │   ├── rules.py               # Check, checkmate, draw detection
│   │   ├── xiangqi_env.py         # Gymnasium environment wrapper
│   │   └── elephanteye.py         # ElephantEye/Pikafish UCCI interface
│   │
│   ├── data/                      # LAYER 2 — data pipeline
│   │   ├── wxf_parser.py          # WXF notation → move sequence
│   │   ├── encoder.py             # Board → 14×10×9 tensor
│   │   ├── dataset.py             # PyTorch Dataset for imitation learning
│   │   └── data_loader.py         # DataLoader construction
│   │
│   ├── models/                    # LAYER 2 — neural network
│   │   ├── network.py             # Full Policy-Value network
│   │   ├── blocks.py              # Residual block, conv block
│   │   └── heads.py               # Policy head, value head
│   │
│   ├── agents/                    # LAYER 3 — the four agents
│   │   ├── base_agent.py          # Abstract agent interface (all agents inherit)
│   │   ├── ppo_agent.py           # Agent 1 & 2 (PPO-based)
│   │   ├── mcts_agent.py          # Agent 3 (MCTS + NN)
│   │   ├── mcts_core.py           # MCTS tree search algorithm
│   │   └── minimax_agent.py       # Agent 4 (baseline)
│   │
│   ├── training/                  # LAYER 2/3 — training procedures
│   │   ├── imitation.py           # Supervised IL training
│   │   ├── ppo_train.py           # PPO self-play loop
│   │   ├── mcts_train.py          # AlphaZero-style training loop
│   │   ├── curriculum.py          # Curriculum training variant
│   │   ├── self_play.py           # Self-play game generator
│   │   └── replay_buffer.py       # Experience storage
│   │
│   ├── evaluation/                # LAYER 4 — measurement
│   │   ├── elo.py                 # Elo rating calculation
│   │   ├── tournament.py          # Round-robin & vs-engine matches
│   │   ├── metrics.py             # Win/draw rates, game length
│   │   └── opening_analysis.py    # Opening move distribution analysis
│   │
│   └── utils/                     # Cross-cutting helpers
│       ├── config.py              # Central config / hyperparameters
│       ├── logging_utils.py       # wandb + tensorboard setup
│       ├── checkpoint.py          # Save/load model weights
│       ├── seeding.py             # Reproducibility (random seeds)
│       └── render.py              # Board visualization
│
├── experiments/                   # LAYER 4 — experiment scripts
│   ├── exp1_main_comparison.py
│   ├── exp2_training_efficiency.py
│   ├── exp3_opening_theory.py
│   ├── exp4_curriculum.py
│   └── exp5_mcts_ablation.py
│
├── dashboard/                     # LAYER 5 — presentation
│   ├── app.py                     # Streamlit entry point
│   ├── pages/                     # Individual dashboard pages
│   └── components/                # Reusable UI widgets
│
├── tests/                         # Unit & integration tests
│   ├── test_environment/          # Rules & move generation tests
│   ├── test_data/                 # Parser & encoder tests
│   ├── test_models/               # Network shape tests
│   └── test_agents/               # Agent behavior tests
│
├── notebooks/                     # Exploration & analysis notebooks
├── results/                       # Output tables, figures, checkpoints
├── configs/                       # YAML config files per experiment
│
├── .env.example
├── .gitignore
├── requirements.txt
├── environment.yml
└── README.md                      # Points to docs/README.md
```

---

## 3. Data Flow — Training a PPO Agent (Example)

This traces how data moves through the system during PPO self-play training:

```
1. xiangqi_env.reset()
   → environment/board.py creates starting position
   → environment/xiangqi_env.py returns initial state

2. encoder.encode(board)
   → data/encoder.py converts board → 14×10×9 tensor

3. ppo_agent.act(state_tensor)
   → agents/ppo_agent.py calls models/network.py
   → network returns (policy_logits, value)
   → move_generator provides legal mask
   → agent applies mask, samples legal action

4. xiangqi_env.step(action)
   → environment applies move, checks rules.py for game end
   → returns (next_state, reward, done)

5. replay_buffer.add(state, action, reward, ...)
   → training/replay_buffer.py stores experience

6. [game ends] → ppo_train.py computes advantages, updates network

7. checkpoint.save(network) every N iterations
   → utils/checkpoint.py writes weights to results/checkpoints/

8. [periodically] tournament.evaluate(agent, elephanteye)
   → evaluation/tournament.py runs games
   → evaluation/elo.py computes rating
   → logged via utils/logging_utils.py to wandb
```

---

## 4. Data Flow — Imitation Learning (Example)

```
1. wxf_parser.parse(game_file)
   → data/wxf_parser.py reads WXF → list of moves

2. For each move: reconstruct board state, encode
   → data/encoder.py → (board_tensor, move_label, outcome)

3. dataset.py wraps these as a PyTorch Dataset

4. data_loader.py batches them

5. imitation.py trains network via cross-entropy loss
   → predict pro's move from board state

6. checkpoint.save(network) → hands off to ppo_train.py for fine-tuning
```

---

## 5. Module Dependency Rules

To keep the codebase clean and prevent circular imports, follow these rules strictly:

| Module | May import from | May NOT import from |
|---|---|---|
| `environment/` | `utils/` only | agents, models, training, evaluation |
| `data/` | `environment/`, `utils/` | agents, models, training |
| `models/` | `utils/` only | agents, environment, data |
| `agents/` | `models/`, `environment/`, `utils/` | training, evaluation, experiments |
| `training/` | `agents/`, `models/`, `environment/`, `data/`, `utils/` | evaluation, experiments |
| `evaluation/` | `agents/`, `environment/`, `utils/` | training, experiments |
| `experiments/` | everything in `src/` | dashboard |
| `dashboard/` | reads `results/` files, may import `agents/` for live play | training, experiments |

**If a needed import would violate these rules, that signals a design problem** — the shared logic probably belongs in `utils/` or `environment/`.

---

## 6. Configuration Strategy

All tunable values live in ONE place: `src/utils/config.py` plus per-experiment YAML files in `configs/`.

- `config.py` holds defaults and constants (board size, action space size, piece values, network dimensions).
- `configs/*.yaml` holds experiment-specific overrides (learning rate, number of games, MCTS simulations).
- **Never hardcode a magic number in a module.** If a value might change, it goes in config.

See 11-CODING-STANDARDS.md for the full configuration convention.

---

## 7. Where Does New Code Go? (Decision Guide)

| I'm writing code that... | It goes in... |
|---|---|
| Defines how a piece moves | `environment/pieces.py` |
| Generates legal moves | `environment/move_generator.py` |
| Detects check/checkmate/draw | `environment/rules.py` |
| Converts a board to a tensor | `data/encoder.py` |
| Reads WXF game files | `data/wxf_parser.py` |
| Defines a neural network layer | `models/blocks.py` or `models/heads.py` |
| Implements an agent's move selection | `agents/<agent>_agent.py` |
| Runs a training loop | `training/<method>_train.py` |
| Computes a metric or Elo | `evaluation/` |
| Runs a full experiment | `experiments/exp<N>_*.py` |
| Displays something in the dashboard | `dashboard/pages/` |
| Is a shared helper used everywhere | `utils/` |

---

*Cross-references: 03-ENVIRONMENT.md (Layer 1 detail), 11-CODING-STANDARDS.md (conventions), README.md (build order).*
