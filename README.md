# ♟️ XiangqiMaster · 象棋大师 · Sang Master Xiangqi

A Comparative Study of Deep Reinforcement Learning Approaches for Xiangqi
(Chinese Chess) — From Imitation Learning to Self-Play Mastery.

**Final thesis project** · Dual-degree UII × NXU · Author: Alma (L25020007)

> 📖 **New here? Read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)** — a complete,
> plain-English walkthrough of the whole project: what it is, the four agents,
> why we benchmark against ElephantEye, the full folder/file/function breakdown,
> every design decision, and what's done vs. still to do.

---

## 📚 Documentation

**All documentation lives in [`docs/`](docs/).** Start with **[`docs/README.md`](docs/README.md)** —
it is the master index and routes you to the right document for any task.

Quick links:
- [Project overview & thesis README](docs/XIANGQI-THESIS-README.md)
- [Game rules](docs/01-GAME-RULES.md) · [Architecture](docs/02-ARCHITECTURE.md) ·
  [Environment](docs/03-ENVIRONMENT.md) · [Data pipeline](docs/04-DATA-PIPELINE.md)
- [Neural network](docs/05-NEURAL-NETWORK.md) · [Agents](docs/06-AGENTS.md) ·
  [Training](docs/07-TRAINING.md) · [Evaluation](docs/08-EVALUATION.md)
- [Experiments](docs/09-EXPERIMENTS.md) · [Dashboard](docs/10-DASHBOARD.md) ·
  [Coding standards](docs/11-CODING-STANDARDS.md) · [Glossary](docs/12-GLOSSARY.md)

## 🚀 Setup

```bash
# Option A — conda
conda env create -f environment.yml
conda activate xiangqimaster

# Option B — pip
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env   # then fill in values
```

See [`docs/XIANGQI-THESIS-README.md`](docs/XIANGQI-THESIS-README.md) for the full
Quick Start (data download, engine build, training, evaluation, dashboard).

## 🏋️ Training

Build the dataset splits from raw games, then train:

```bash
# Build train/val/test splits from raw game files in data/raw/ (WXF or ICCS PGN)
python -m src.data.pipeline --raw data/raw --out data/splits --notation iccs --spot-check 500

# Agent 2 Phase 1 — imitation learning (GPU recommended)
python -m src.training.il_train --splits data/splits --epochs 30 --device cuda \
    --save results/checkpoints/il_agent2_phase1.pt        # --resume to continue

# Agent 2 Phase 2 — PPO fine-tune from the IL checkpoint
python -m src.training.ppo_train --timesteps 500000 \
    --il-checkpoint results/checkpoints/il_agent2_phase1.pt --save results/checkpoints/ppo_agent2

# Agent 1 — PPO self-play from scratch
python -m src.training.ppo_train --timesteps 500000 --save results/checkpoints/ppo_agent1
```

**On Colab:** open [`notebooks/xiangqi_colab_training.ipynb`](notebooks/xiangqi_colab_training.ipynb)
and run top-to-bottom — it clones the repo, installs deps, downloads the data,
builds splits, and trains with checkpoints saved to Google Drive (survives
disconnects). See [`docs/07-TRAINING.md`](docs/07-TRAINING.md) for details.

## 🗂️ Repository layout

```
docs/         Documentation (13 files — read docs/README.md first)
src/          Source code (environment, data, models, agents, training, evaluation, utils)
experiments/  The five experiment scripts
dashboard/    Streamlit dashboard
tests/        Unit & integration tests
configs/      Per-experiment YAML configs
data/         Datasets (gitignored) · results/  Outputs (checkpoints gitignored)
notebooks/    Exploration & analysis
```

See [`docs/02-ARCHITECTURE.md`](docs/02-ARCHITECTURE.md) for the full module map
and dependency rules.

## 📄 License

MIT — see [`LICENSE`](LICENSE).
