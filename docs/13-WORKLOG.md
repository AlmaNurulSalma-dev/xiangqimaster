# 13 — Work Log: How XiangqiMaster Was Built & Trained

> A chronological record of the project's development and training process,
> reconstructed from the **git history**, the saved **training logs**
> (`results/logs/*.log`), and the running project notes. It is the reference for
> *what was run, when, what it produced, and what went wrong* — useful for the
> thesis methodology chapter and for reproducing any run.
>
> **Note on sources.** The project's live terminal scrollback was not kept
> (terminals discard old lines), and the shell command history contained only
> trivial commands — most training/experiment commands were launched as detached
> background processes. This log therefore reconstructs the process from durable
> artifacts: commit timestamps, the `.log`/`.pid` files each run wrote, and the
> result CSVs. Dates/times are local.

---

## 1. Environment & Hardware

| Item | Detail |
|---|---|
| Machine | Laptop with **NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)** |
| Default Python | 3.14 (system) — **CPU-only** PyTorch; used for tests/dev |
| Training Python | **3.12 venv** at `C:\Users\kinet\xiangqi-venv` — **CUDA** PyTorch (cu126) |
| Run command | `C:\Users\kinet\xiangqi-venv\Scripts\python.exe -m <module> …` |
| Key deps | PyTorch 2.14 (CUDA), Gymnasium 1.3, sb3-contrib / stable-baselines3 2.9 (MaskablePPO), NumPy, Streamlit+Plotly, pytest |
| Engine benchmark | Pikafish (UCI) at `engines/pikafish/extracted/` |
| Dataset | `CGLemon/chinese-chess-PGN` — 141,356 legal games → **11.6M positions**; split 113k/14k/14k (train 9.3M positions) |

**GPU-setup gotcha (resolved):** installing sb3 pulled a CPU PyTorch and
`torchvision` pinned an old CUDA build. Fix: uninstall `torchvision` (unused),
then reinstall the exact CUDA wheel:
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps`.
Verified `torch.cuda.is_available() == True`.

---

## 2. Development Timeline (code milestones, from git)

| Date | Milestone |
|---|---|
| **19 Sep** | Project scaffolding + docs; environment layer (`config`, `Board`, `pieces`, `rules`, move generation, action space, encoder, Gymnasium env). |
| **20 Sep** | The "big bang" day — Policy-Value network; Minimax (Agent 4); PPO self-play (Agent 1); Elo/metrics/tournament; WXF parser + IL dataset; **Imitation Learning + NeuralAgent (Agent 2 P1)**; **MCTS+network (Agent 3)**; all five experiments; Agent 2 Phase 2 transfer; Streamlit dashboard. |
| **21–22 Sep** | UCCI engine interface (ElephantEye/Pikafish); experiment logging; ICCS notation; end-to-end data pipeline; scaled to 100k+ games; lazy IL dataset + train-from-splits; IL→PPO bridge; Colab notebook + checkpointing. |
| **23 Sep** | README training section + `PROJECT_GUIDE`. |
| **1 Oct** | Intra-epoch checkpointing (IL) + periodic checkpointing/resume/device (PPO) — survival features for Colab/laptop interruptions. |
| **3 Oct** | PPO TensorBoard+CSV logging; robust checkpoint loaders; `--opponent {random,self,minimax}`; robust `--resume`. |
| **4 Oct** | AlphaZero-style **MCTS self-play trainer** (Agent 3) committed. |
| **24 Oct** | Pikafish **UCI** support + **self-play distillation** generator; persist replay buffer + best-checkpoint on MCTS resume. |
| **25–26 Oct** | Configurable network size (`--channels`/`--num-blocks`) for IL; dashboard adds trained agents + Pikafish; bigger-network distillation eval. |
| **1–2 Nov** | Lab 8 capstone proposal (this `research/` work). |

---

## 3. Training Runs (chronological, with commands & outcomes)

All commands use the venv python: `C:\Users\kinet\xiangqi-venv\Scripts\python.exe`.

### 3.1 Dataset build (one-time)
```
python -m src.data.pipeline --raw data/raw --out data/splits --notation iccs --spot-check 500
```
→ 141,556 seen → 141,356 kept; 11.6M positions; split 113,084 / 14,135 / 14,137
games. Spot-check 500/500 legal. Writes `data/splits/{train,val,test}.jsonl`
(gitignored) + `dataset_manifest.json` (committed).

### 3.2 Agent 2 — Imitation Learning (supervised pre-training)  · 2 Oct
```
python -m src.training.il_train --splits data/splits --epochs 30 --device cuda \
    --num-workers 4 --save-every 500 --save results/checkpoints/il_agent2_phase1.pt
```
- Supervised cross-entropy (policy) + MSE (value) over 9.3M positions.
- ~17 min/epoch; one run killed at epoch 21, finished via `--resume`.
- **Result:** val top-1 move accuracy **26% → ~34% (plateau)**. First trained model.
- Logs: `il_train.log`, `il_train_resume.log` → `il_agent2_phase1.pt` (4.7 MB).

### 3.3 Agent 2 Phase 2 — PPO fine-tuning (reinforcement learning)  · 3 Oct
```
python -m src.training.ppo_train --timesteps 500000 --checkpoint-every 10000 \
    --device cuda --il-checkpoint results/checkpoints/il_agent2_phase1.pt \
    --save results/checkpoints/ppo_agent2 --log-dir results/tb/ppo_agent2
```
- **501,760 steps (~7.5 h).** ep_rew_mean 0.018 → 0.513. → `ppo_agent2.zip` (29 MB).
- Logs: `ppo_agent2.log`, curve `ppo_agent2_curve.csv`.

### 3.4 Agent 1 — PPO from scratch  · 3 Oct
```
python -m src.training.ppo_train --timesteps 500000 --save results/checkpoints/ppo_agent1 \
    --log-dir results/tb/ppo_agent1
```
- 500k steps. ep_rew_mean 0.086 → 0.381; critic barely learned from scratch.
- Logs: `ppo_agent1.log` → `ppo_agent1.zip`.

### 3.5 Self-play variants  · 4 Oct
```
python -m src.training.ppo_train --opponent self --timesteps 250000 --device cuda \
    [--il-checkpoint …] --save results/checkpoints/ppo_agent{1,2}_selfplay
```
→ `ppo_agent1_selfplay.zip`, `ppo_agent2_selfplay.zip` (kept the vs-random versions
for an ablation).

### 3.6 Agent 3 — MCTS self-play (first run, fragmented)  · 4–9 Oct
```
python -m src.training.mcts_train --iterations 20 --games-per-iter 25 \
    --simulations 100 --epochs-per-iter 2 --batch-size 256 --device cuda \
    --il-checkpoint results/checkpoints/il_agent2_phase1.pt \
    --save results/checkpoints/mcts_agent3.pt --resume
```
- Reached 20 iterations across **7 stop/resume cycles** (`mcts_agent3_resume…resume7b.log`).
- Each early `--resume` reset the replay buffer → loss spikes (see §5). → `mcts_agent3.pt`.

### 3.7 Agent 3 v2 — stabilized MCTS  · 13–21 Oct
After diagnosing the regression, added **persistent replay buffer** + **best-by-value-loss
checkpoint**, then retrained from an IL warm start:
```
python -m src.training.mcts_train --iterations 20 --games-per-iter 25 \
    --simulations 100 --epochs-per-iter 2 --batch-size 256 --device cuda \
    --il-checkpoint results/checkpoints/il_agent2_phase1.pt \
    --save results/checkpoints/mcts_agent3_v2.pt
# resume (buffer auto-reloaded):  … --resume
```
- Run **detached** (`Start-Process -WindowStyle Hidden`) to survive session exits;
  PID in `results/logs/mcts_v2_detached.pid`. 8 resume sessions
  (`mcts_agent3_v2_resume…resume8.log`), **no buffer-reset spikes**.
- **Result:** best value-loss **~0.112** → `mcts_agent3_v2.best.pt`
  (+ `mcts_agent3_v2.pt.buffer.pkl` ~269 MB, `.meta.json`).

### 3.8 Pikafish distillation (teacher = engine, not humans)  · 24 Oct
```
python -m src.data.pikafish_selfplay --games 5000 --depth 8 --opening-random 6 \
    --out data/pikafish/selfplay.jsonl --seed 100
python -m src.training.il_train --splits data/pikafish_splits --epochs 30 \
    --mirror --num-workers 4 --device cuda --save-every 2000 \
    --save results/checkpoints/distill_pikafish.pt
```
- 5000 Pikafish-vs-Pikafish games (~2.2 h, 815k positions, ~59% draws).
- **Result:** val top-1 plateaued **~0.136** (below human-IL's 0.34). → `distill_pikafish.pt`.
- Logs: `pikafish_selfplay.log`, `pikafish_distill_resume.log`.

### 3.9 Bigger-network distillation  · 25–26 Oct
```
python -m src.training.il_train --splits data/pikafish_splits --channels 128 \
    --num-blocks 15 --epochs 30 --mirror --device cuda \
    --save results/checkpoints/distill_big.pt --resume
```
- C=128 / N=15 ≈ **4.85M params** (4× the 1.15M net). Trained in cooled bursts
  (laptop overheating — see §5). **Result:** val top-1 **~0.148** — only +0.008 over
  the small net. → `distill_big.pt` (19 MB).

---

## 4. Experiments & Results (Elo)

All from `results/exp*/exp1_elo.csv`. Relative Elo, anchor 1500, 95% CI.

**Experiment 1 — 7-agent round-robin (50 games/pair):**
IL_only **1689** > ILPPO_rand 1526 ≈ Minimax_d2 1524 > PPO_s_self 1481 >
PPO_s_rand 1454 > ILPPO_self 1444 > Random 1382.
*Finding:* imitation **alone** was strongest; no PPO variant beat it; self-play
even hurt the IL agent.

**MCTS headline (5-iter) — `results/exp1_agent3/`:**
MCTS **1703** > IL_only 1557 > Minimax 1541 > ILPPO 1411 > Random 1287.
*Finding:* AlphaZero-style search over the network is the strongest agent.

**MCTS v2 (stabilized) — `results/exp1_agent3_v2/`:**
MCTS_v2 **1691** > IL_only 1597 > Minimax 1460 > ILPPO 1452 > Random 1300.
*Finding:* the stability fix recovered the regression (see §5).

**Pikafish distillation — `results/exp_distill/`:**
Pikafish_d8 **2062** (anchor) ≫ Distill_pika 1501 ≈ IL_human 1489 > Minimax 1474
> ILPPO 1301 > Random 1173. *Finding:* an engine teacher did **not** beat human
imitation at this net size.

**Bigger net — `results/exp_bignet/`:**
Minimax 1585 ≈ Distill_big 1573 ≈ IL_human 1561 ≈ Distill_small 1530 > Random 1251.
*Finding:* 4× capacity gave **no** real strength gain.

**MCTS simulation ablation — `results/exp5_mcts_v2/`:** more simulations did not
clearly help (score vs IL: 100→0.50, 200→0.25, 400→0.25, 800→0.50). The network,
not search depth, is the bottleneck.

> **Cross-cutting conclusion:** four independent levers — more search, more epochs,
> engine distillation, bigger net — all hit the same ~1690-Elo ceiling ⇒ the binding
> constraint is **model capacity + data/compute scale**, not the algorithm.

---

## 5. Problems Encountered & How They Were Solved

| Problem | Symptom | Fix |
|---|---|---|
| **Colab disconnects** | Training died mid-epoch, no checkpoint | Intra-epoch (`--save-every N`) + per-epoch checkpointing + `--resume` |
| **SB3 zip-read bug** (torch 2.14.1) | `MaskablePPO.load` "file read failed" | Robust loader: fall back to `build_model` + load policy state_dict from the zip |
| **MCTS buffer reset on resume** | Fragmented training **regressed** 1703 → 1582 Elo | Persist replay buffer to `*.buffer.pkl` + save `*.best.pt`; retrain → recovered to 1691 |
| **`move_generator` typo** (`legalcek`) | `NameError` crashed self-play | Fixed to `return legal`; verified 44 legal opening moves |
| **Local GPU setup** | `torchvision` clobbered CUDA torch | Uninstall `torchvision`, reinstall `torch==2.14.1` cu126 `--no-deps` |
| **Laptop overheating** | GPU throttling on sustained training | Cooled bursts (~4 epochs), batch 128, num-workers 2, CPU cap 85% → GPU ~60–67 °C |
| **Windows cp1252 console** | box-drawing chars failed to encode | ASCII-only console summary |
| **Scraper .exe broken** | Playwright driver assets missing | Switched to ready-made dataset (`CGLemon/chinese-chess-PGN`) |
| **Session boundaries kill runs** | Training stopped when a session ended | Detached Windows process (`Start-Process -WindowStyle Hidden -PassThru`) + PID file |

---

## 6. Trained Artifacts (on disk, gitignored)

| Checkpoint | What it is |
|---|---|
| `il_agent2_phase1.pt` | Imitation-learning net (supervised, 34% top-1) |
| `ppo_agent1.zip` / `_selfplay.zip` | PPO from scratch (vs-random / self-play) |
| `ppo_agent2.zip` / `_selfplay.zip` | IL→PPO fine-tuned |
| `mcts_agent3.pt` | MCTS v1 (fragmented 20-iter) |
| `mcts_agent3_v2.best.pt` | MCTS v2 (stabilized, strongest) |
| `distill_pikafish.pt` | Engine-distilled net (small) |
| `distill_big.pt` | Engine-distilled net (4.85M) |

---

## 7. Reproduction Quick-Reference

```bash
# 0. activate the CUDA venv
C:\Users\kinet\xiangqi-venv\Scripts\activate

# 1. build splits   2. IL   3. PPO(IL-init)   4. PPO(scratch)   5. MCTS   6. eval
python -m src.data.pipeline --raw data/raw --out data/splits --notation iccs --spot-check 500
python -m src.training.il_train  --splits data/splits --epochs 30 --device cuda --save results/checkpoints/il_agent2_phase1.pt
python -m src.training.ppo_train --timesteps 500000 --device cuda --il-checkpoint results/checkpoints/il_agent2_phase1.pt --save results/checkpoints/ppo_agent2
python -m src.training.ppo_train --timesteps 500000 --device cuda --save results/checkpoints/ppo_agent1
python -m src.training.mcts_train --iterations 20 --games-per-iter 25 --simulations 100 --device cuda --il-checkpoint results/checkpoints/il_agent2_phase1.pt --save results/checkpoints/mcts_agent3_v2.pt
python -m pytest -q      # full test suite
```

💡 **To capture future terminal sessions in full:** run
`Start-Transcript -Path "$HOME\xiangqi-terminal-log.txt" -Append` at the start of a
session (unlimited, unlike the 1000-line terminal scrollback), or raise
`terminal.integrated.scrollback` in VSCode settings.

---

*Cross-references: 06-AGENTS.md (the agents), 07-TRAINING.md (training detail),
08-EVALUATION.md (Elo), 09-EXPERIMENTS.md (experiment protocol),
results/AGENT3_MCTS_FINDING.md (the MCTS stability write-up).*
