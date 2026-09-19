# ♟️ XiangqiMaster · 象棋大师 · Sang Master Xiangqi

A Comparative Study of Deep Reinforcement Learning Approaches for Xiangqi
(Chinese Chess) — From Imitation Learning to Self-Play Mastery.

**Final thesis project** · Dual-degree UII × NXU · Author: Alma (L25020007)

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
