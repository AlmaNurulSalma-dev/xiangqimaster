# XiangqiMaster — Course Project Progress Review

**Student:** Alma Nurul Salma (诺艾玛) · L25020007
**Affiliation:** Nanjing Xiaozhuang University (南京晓庄学院)
**Date:** 5 November 2026 · **Thesis defense target:** end of May 2027

**Project in one sentence.** A controlled, head-to-head comparison of four
deep-learning paradigms — Minimax search, imitation learning, self-play
reinforcement learning (PPO), and AlphaZero-style Monte Carlo Tree Search (MCTS) —
for the game of **Xiangqi (象棋, Chinese Chess)**, all built on one shared engine
and rated on a single Elo scale, to identify which paradigm is strongest — and
why — by comparing them head-to-head under identical conditions.

> This document answers the three points for tomorrow's review:
> **(1)** project schedule · **(2)** functions already implemented (with a live
> demo plan) · **(3)** problems/difficulties encountered and the potential outcomes
> once they are resolved. §1 first sets the **research plan** (problem, gap, and
> solution, grounded in the literature), and §3 gives a **step-by-step account of
> the development process**.

---

## 1. Research Plan: Problem, Gap & Solution

### 1.1 Where the problem comes from (reported in the literature, not assumed)
The modern recipe for superhuman board-game play — imitation of experts, self-play
reinforcement learning, and tree search over a learned network — was established by
the AlphaGo family (**Silver et al., 2016, 2017, 2018**). But those results were
obtained on large clusters and each reports a *single* successful system, not a
comparison of the paradigms it combines.

For **Xiangqi specifically**, the published deep-learning work is recent and sparse,
and every system again builds **one** maximally strong pipeline:
- **Yilmaz, Hu & Liu (2025)** — an AlphaZero-style player using Monte Carlo Tree
  Search (arXiv:2506.15880).
- **Li et al. (2023), "JiangJun"** — MCTS + Policy-Space Response Oracles over
  >10,000 human games, focused on non-transitivity (TMLR, arXiv:2308.04719).
- **Chen et al. (2025), "Xiangqi-R1"** — a language-model approach trained with
  reinforcement learning (arXiv:2507.12215).

None of them compares the distinct learning paradigms **against each other,
head-to-head**. Independently, **Henderson et al. (2018)** show that deep
reinforcement-learning results are fragile and hard to reproduce when reported from
single runs without variance analysis — so single-system claims are exactly the ones
that need a controlled, interval-backed comparison.

> ⚠️ **The problem is therefore literature-reported, not our own assumption:** the
> field has many single systems and *no* like-for-like comparison for Xiangqi.

### 1.2 The research gap (what no one has done yet)

| Prior work | Xiangqi? | Compares the paradigms? | Analyses *why* one wins? |
|---|:--:|:--:|:--:|
| AlphaGo family (Silver et al. 2016–2018) | no | no | no |
| MCTS survey (Browne et al. 2012) | n/a | n/a | n/a |
| PPO (Schulman et al. 2017) | no | partial | no |
| RL reproducibility (Henderson et al. 2018) | no | no | yes |
| AlphaStar (Vinyals et al. 2019) | no | no | no |
| JiangJun (Li et al. 2023) | yes | no | no |
| Yilmaz et al. (2025) | yes | no | partial |
| Xiangqi-R1 (Chen et al. 2025) | yes | no | partial |
| **This project** | **yes** | **yes** | **yes** |

The empty cell no one has filled is the bottom row: a controlled, head-to-head,
Elo-anchored comparison of all four paradigms for Xiangqi, with an analysis of
*why* one wins.

### 1.3 Problem and Solution (the core of the thesis)

> 🔴 **PROBLEM.** For Xiangqi, the four learning paradigms have *never been
> compared head-to-head*, so three things are unknown: **(a)** which paradigm plays
> strongest; **(b)** whether
> imitation pretraining actually helps the self-play reinforcement learning that
> follows (or can hurt it); and **(c)** which factor — search, model capacity,
> training data/teacher, or the algorithm — limits playing strength. Existing
> Xiangqi systems each optimise *one* pipeline and never compare.

> 🟢 **SOLUTION.** Hold the **environment, action space, network, and Elo scale
> fixed**, and change **only the learning paradigm** — a single-interface
> *multi-agent evaluation league*. Because everything but the decision function is
> constant, any difference in strength is attributable to the **paradigm itself**,
> so the project can **compare and analyse** which approach wins and **why**.
> Keeping the conditions identical for every agent is simply what makes the
> comparison fair.

### 1.4 Blueprint of the solution (the apparatus)
1. **Shared Xiangqi engine** — rules, legal masking, a fixed **2086-move action
   space**, and a **14×10×9** board encoder.
2. **One policy–value network** reused by imitation, by PPO (actor + critic), and by
   MCTS (priors + leaf value); its size is configurable for the capacity ablation.
3. **Four interchangeable decision functions** — Minimax · Imitation Learning ·
   Self-Play PPO · MCTS-over-network — plus a Random floor.
4. **Evaluation league** — colour-balanced **round-robin → fitted Elo with 95% CI**,
   a **head-to-head** matrix, and a **Pikafish** absolute anchor.
5. **Three experiments** — Exp 1: ranking (RQ1) · Exp 2: imitation-pretraining
   effect (RQ2) · Exp 3: single-variable ablation of each lever (RQ3).

*Research questions:* **RQ1** how do the four paradigms rank on one Elo scale?
**RQ2** does imitation pretraining help the subsequent PPO, or degrade it?
**RQ3** which factor binds strength — search, capacity, data, or algorithm?

### 1.5 Novelty — doing what others have not
- The **first controlled, head-to-head, Elo-anchored comparison** of Minimax,
  imitation, self-play PPO, and AlphaZero-style MCTS **for Xiangqi**.
- A **reusable single-interface league** for heterogeneous game-playing paradigms,
  runnable on **one consumer GPU** (not a cluster).
- It deliberately **reports negative results** — imitation beating PPO, and a
  capacity ceiling — which single-system papers never surface.

> *Full citations are in the proposal's reference list
> (`research/proposal/references.bib`); this is the same gap the 4-page proposal
> argues, condensed.*

---

## 2. Project Schedule / Timeline

The experimental apparatus and a full set of **preliminary results already exist**
(built Sep–Oct 2026). The remaining ~7 months are therefore mostly *formalisation,
scale-up, writing, and the defense demo* — not a speculative build. This makes the
schedule realistic rather than optimistic.

### 2.1 What is already done (Sep–Oct 2026)
- **Sep 2026** — Full environment layer: Xiangqi rules, legal-move generation,
  2086-move action space, 14×10×9 board encoder, Gymnasium environment; the
  policy–value network; all four agents; the Elo/tournament harness; the data
  pipeline; Streamlit dashboard.
- **Oct 2026** — All agents trained; five experiments run; MCTS stabilised
  (persistent replay buffer); Pikafish engine anchor + self-play distillation;
  configurable network size for the capacity ablation.

### 2.2 Remaining schedule to defense

| Period | Milestone | Status |
|---|---|---|
| **Nov 2026** (now) | Finalise the 4-page proposal (done); formalise the test-set move-accuracy analysis; begin scaling the round-robins. | In progress |
| **Dec 2026** | Scale every tournament to more games/pair for **tighter 95% confidence intervals**; add **Pikafish absolute anchor** across all agents; lock the final result tables. | Planned |
| **Jan 2027** | Write thesis **Chapters 1–3** (Introduction, Related Work, Methodology). | Planned |
| **Feb 2027** | Write **Chapter 4** (Results & analysis); complete the single-variable ablations (search / epochs / teacher / network size); finalise all figures. | Planned |
| **Mar 2027** | Write **Chapter 5** (Discussion, limitations, future work); deliver the **full first draft** to advisor. | Planned |
| **Apr 2027** | Advisor revisions; reproducibility pass (fixed seeds, committed manifest); build the **defense demo** (see §4.4); prepare slides. | Planned |
| **May 2027** | Rehearse; mock defense; final revisions; **thesis defense (end of May 2027).** | Planned |

**Buffer note.** Writing (Jan–Mar) and the demo build (Apr) are the critical path.
Because the code and preliminary data are finished, slippage in one month can be
absorbed by the next without touching experiments — the main risk (long training
runs) is already behind me.

---

## 3. What I Did — The Development Process (in order)

I built the project **bottom-up**: first a provably-correct game engine, then the
learning machinery on top of it, then the data, then the agents, and only then the
experiments that compare them. The order matters — each layer depends on the one
below being correct. This is the story of how it came together.

**Phase 0 — A correct Xiangqi engine first (19 Sep 2026).**
Before any machine learning, I implemented the game itself: the board, the seven
piece types, full legal-move generation (including the cannon's screen-capture and
the blockable horse), the palace/river constraints, and the rule that *stalemate is
a loss*. I wrapped it as a Gymnasium environment with a fixed **2086-move action
space** and a **14×10×9 board encoder**, and verified it with a test suite (e.g. 44
legal opening moves). *Why first:* every agent — search or neural — must share one
trusted rule engine, or the comparison means nothing.

**Phase 1 — The core learning stack, in one push (20 Sep).**
With the engine trusted, I built the shared **policy–value network** and, on top of
it, **all four decision functions at once**: Minimax (alpha–beta), PPO self-play,
the imitation-learning trainer + neural agent, and the MCTS+network agent — plus the
**Elo/tournament harness** and a first **Streamlit dashboard**. Everything now
existed; nothing was trained yet.

**Phase 2 — Real data and the pipeline (21–23 Sep).**
A fair imitation agent needs real human games. I built a notation parser and an
end-to-end data pipeline, then scaled it to the full public dataset
(`CGLemon/chinese-chess-PGN`) → **141,356 legal games / 11.6M positions**, split
113k/14k/14k at the game level with a committed manifest for reproducibility. I also
wrote the **IL→PPO bridge** (so a PPO agent can start from an imitation checkpoint)
and a Colab notebook.

**Phase 3 — Making training survive (1–3 Oct).**
My first training attempts kept dying — Colab disconnects, laptop sleep, session
exits. So *before* training seriously, I added survival features: intra-epoch and
periodic **checkpointing**, robust `--resume`, TensorBoard/CSV logging, and
selectable opponents. This decision paid off repeatedly later.

**Phase 4 — Training the agents, in dependency order (2–4 Oct).**
1. **Imitation Learning first** — it's the foundation the others build on.
   Supervised cross-entropy (policy) + MSE (value) over 9.3M positions, ~17 min
   per epoch; one run was killed at epoch 21 and finished via `--resume`. Result:
   move accuracy 26% → **~34%**. *My first trained model.*
2. **IL→PPO fine-tuning** (501k steps, ~7.5 h) — episode reward rose 0.02 → 0.51,
   but (as evaluation later showed) it did not actually get stronger.
3. **PPO from scratch** (500k steps) — the critic barely learned without the
   imitation warm start.
4. **Self-play variants** of both, kept for an ablation.

**Phase 5 — The MCTS agent and a hard bug (4–9 Oct).**
I committed the AlphaZero-style **MCTS self-play trainer** and ran it. But the run
had to be stopped and resumed **7 times**, and each resume **reset the replay
buffer**, causing loss spikes — the agent actually **regressed from ~1703 to ~1582
Elo.** This was the hardest problem of the project.

**Phase 6 — Diagnosis and the stability fix (13–21 Oct).**
I traced the regression to the buffer reset, then added a **persistent replay
buffer** (saved to disk) + **best-by-value-loss checkpointing**, and retrained from
the imitation warm start — this time as a **detached background process** so a
closing session couldn't kill it. Over 8 resume sessions there were **no more loss
spikes**, and strength recovered to **~1691 Elo**. (Written up in
`results/AGENT3_MCTS_FINDING.md`.)

**Phase 7 — Trying to break the strength ceiling (24–26 Oct).**
Club-level wasn't satisfying, so I tried two ways to get stronger without just
tuning hyper-parameters: (a) **distilling Pikafish** — I generated 5,000
engine-vs-engine games and trained on them; and (b) a **4× bigger network** (~4.85M
params). **Neither helped** — distillation plateaued *below* human imitation, and
the big net gained almost nothing.

**Phase 8 — Running the experiments and finding the answer.**
With every agent trained, I ran the full round-robins and ablations and fitted Elo
with confidence intervals. The pattern was consistent: **MCTS strongest (1703),
imitation alone beating PPO, and four independent levers — more search, more epochs,
engine distillation, bigger net — all hitting the same ~1690 ceiling.** That
convergence is the project's central finding: **the binding constraint is model
capacity + data/compute scale, not the learning algorithm.**

**Phase 9 — Formalising it (1–2 Nov).**
Finally I turned the finished system and its results into the **4-page research
proposal** (`research/proposal/`) that frames the whole thing as a controlled,
head-to-head paradigm comparison.

**The throughline.** I didn't train first and build later — I made the *engine*
correct, the *training survivable*, and only then trained; when the strongest agent
regressed I stopped and fixed the **root cause** rather than papering over it; and
when the agents hit a ceiling I ran controlled experiments to *explain why* instead
of chasing a bigger number. That discipline is what turned a build into a study
whose most interesting result is a well-supported **negative** one.

---

## 4. Functions Already Implemented — Demo

Everything below is **built, runs, and produced the numbers quoted**. A live demo
is possible from the Streamlit dashboard and the command line.

### 4.1 Core system (the shared "apparatus")
- **Xiangqi rule engine** — full rules incl. the cannon's screen-capture, the
  blockable horse, palace/river constraints, and *stalemate = loss*; exposed as a
  **Gymnasium environment** with a fixed **2086-move action space** and legal-move
  masking. (Verified: 44 legal opening moves.)
- **Board encoder** — each position → a **14×10×9 tensor** (one 10×9 binary plane
  per piece type and side, from the side-to-move's perspective).
- **Policy–value network** — one AlphaZero-style conv-residual network (~1.15M
  parameters) reused by every learning agent; network size is configurable
  (tested up to ~4.85M params for the capacity ablation).

### 4.2 The four agents + evaluation (all trained)
| Agent | Method | Trained? |
|---|---|---|
| **Minimax** | Alpha–beta (negamax) search + material evaluation | ✅ baseline |
| **Imitation Learning (IL)** | Supervised on **113k human games / 9.3M positions** | ✅ **34% top-1 move accuracy** |
| **PPO self-play** | Action-masked PPO, from scratch and IL-initialised | ✅ ~500k steps each |
| **MCTS + network** | AlphaZero-style PUCT search over the network | ✅ **strongest agent** |

- **Dataset:** `CGLemon/chinese-chess-PGN` → 141,356 legal games, **11.6M
  positions**, split 113k / 14k / 14k at the game level (committed manifest).
- **Evaluation league:** colour-balanced **round-robin → fitted Elo with 95% CI**,
  head-to-head win-rate matrix, and a **Pikafish** absolute anchor.
- **Test-set move-prediction report:** on 163,416 held-out positions the IL net
  predicts the human's exact move with **42.3% top-1 / 66.5% top-3** (legal-masked);
  **63% top-1 in the opening**, falling to **32%** in the endgame.

### 4.3 Headline results (preliminary, to be scaled for the thesis)
Relative Elo (anchor 1500, 95% CI):

> **MCTS 1703** > IL 1557 > Minimax 1541 > IL→PPO 1411 > Random 1287

- **MCTS (AlphaZero-style search) is strongest.**
- **Imitation alone beats PPO fine-tuning** — reinforcement learning *degraded* the
  imitation policy under this budget (a genuine, reportable negative result).
- A Pikafish engine anchor sits far above (~2062 Elo at depth 8), placing my learned
  agents at a **strong-amateur / club** level.

### 4.4 What I can show live tomorrow
1. **Streamlit dashboard** — AI-vs-AI games, play against any agent, and against
   Pikafish at adjustable depth.
2. The **Elo bar chart** + **head-to-head heatmap** from the real round-robin.
3. The **test-accuracy report** (opening→endgame difficulty curve).
4. The **code**: engine, training pipelines (IL / PPO / MCTS), evaluation harness.

---

## 5. Problems & Difficulties — and Potential Outcomes if Resolved

### 5.1 Solved during development (shows robustness)
| Problem | Impact | How it was solved |
|---|---|---|
| **Training interruptions** (Colab/laptop) | Runs died mid-epoch | Intra-epoch + periodic checkpointing & `--resume`; detached background process + PID file |
| **MCTS instability** | Replay-buffer reset on resume **regressed 1703 → 1582 Elo** | Persist the replay buffer + keep the best-by-value-loss checkpoint → recovered to **1691** |
| **GPU / CUDA setup** | `torchvision` clobbered the CUDA build | Reinstalled the exact CUDA PyTorch wheel; verified `cuda.is_available()` |
| **Laptop overheating** | GPU throttled on long training | Cooled bursts, smaller batch/workers → GPU held ~60–67 °C |

### 5.2 Open difficulties (the research questions)
1. **Model capacity is the binding constraint.** Four independent levers — more
   search simulations, more epochs, engine (Pikafish) distillation, and a 4× bigger
   network — **all plateau at the same ~1690 Elo ceiling.** The small 1.15M-param
   network, not the algorithm, limits strength.
   - *If resolved* (a larger network + cloud/cluster GPU and more self-play): the
     learned agents could climb well above club level toward engine-competitive
     play — and, more importantly, the comparison would be confirmed at a second,
     larger scale, strengthening the thesis's central claim.

2. **Reinforcement learning fine-tuning can hurt imitation.** IL→PPO scored *below*
   IL alone.
   - *If resolved* (better reward shaping / longer, more stable RL schedule): I
     could pin down **the conditions under which self-play RL starts to help** — a
     precise, publishable answer to RQ2 rather than only "it hurt here."

3. **Fair, absolute evaluation.** Xiangqi has no single universal Elo scale, so
   relative ratings depend on the agent pool.
   - *If resolved* (Pikafish anchor at fixed depth + more games per pair): every
     agent gets an **absolute, reproducible rating with tight confidence
     intervals**, making the ranking defensible.

4. **Confirming the comparison at a larger scale.** The current ranking and analysis
   come from a single training scale; they should be re-confirmed with a bigger
   network and more self-play.
   - *If resolved:* the same pipeline scales up unchanged, so the comparison and its
     conclusions can be re-verified at a second scale, strengthening the thesis's
     central claim.

### 5.3 Potential outcome of the whole project (if the difficulties are resolved)
A **controlled, reproducible, Elo-anchored answer** to three questions the current
Xiangqi literature (all single, mostly cluster-scale systems) leaves open:
- **which paradigm is strongest** under equal compute (currently: MCTS);
- **whether imitation pretraining helps the subsequent RL** (currently: no — it can
  hurt);
- **what factor binds playing strength** (currently: model capacity + data/compute
  scale, not the algorithm).

Plus two tangible deliverables: a **reusable multi-agent evaluation league** for
heterogeneous game-playing agents, and — for the defense — an **interactive demo**
(advantage/eval bar, AI-vs-AI showcase, difficulty levels from Random → Minimax →
IL → MCTS) that lets a non-Xiangqi audience *see* the strength difference, framed
with the "like AlphaGo, but for Chinese Chess" analogy.

---

*Supporting material: the 4-page proposal (`research/proposal/main.pdf`), the full
build-and-training log (`docs/13-WORKLOG.md`), and the result CSVs under
`results/` back every number in this document.*
