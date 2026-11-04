# XiangqiMaster — Direct Project Brief

*A plain, in-order statement: problem → gap → my project → solution blueprint →
what I研究 → objective → done → not done + timeline → the GPU I trained on.*

---

## 1. The Problem
Modern game-playing AI (the AlphaGo family) is built from three ideas — **imitation
of human experts, self-play reinforcement learning, and tree search over a learned
network**. These were proven on huge computing clusters, and each paper ships **one**
finished system — and the paradigms are **never compared head-to-head**. So for
**Xiangqi (象棋, Chinese Chess)** three things are **unknown**:
1. **Which** of these learning paradigms actually plays strongest?
2. **Does imitation pretraining help** the self-play reinforcement learning that
   comes after it — or can it *hurt*?
3. **What limits** playing strength — the search, the model size, the training
   data, or the learning algorithm?

## 2. The Gap
The published Xiangqi deep-learning work is recent and sparse, and **every system
builds one single pipeline** — an AlphaZero-style MCTS player (Yilmaz et al. 2025),
a population method called JiangJun (Li et al. 2023), and an LLM approach called
Xiangqi-R1 (Chen et al. 2025). **None of them compares the paradigms against each
other, head-to-head.** And reinforcement-learning results are known
to be fragile when reported from single runs (Henderson et al. 2018). So the gap is:

> **No controlled, head-to-head, Elo-anchored comparison of the four paradigms
> exists for Xiangqi.** (This is reported by the literature — not my assumption.)

## 3. How This Connects to My Project
My project, **XiangqiMaster**, *is exactly that missing comparison.* I take the four
paradigms — **Minimax search, Imitation Learning, Self-Play Reinforcement Learning
(PPO), and AlphaZero-style Monte Carlo Tree Search (MCTS)** — and run them as a
**multi-agent "league"** on one shared Xiangqi engine, rated on one Elo scale, so I
can measure which is strongest and why.

## 4. Blueprint of the Solution
The key idea: **keep everything the same except the one thing I'm studying.**
- **One shared Xiangqi engine** — rules, legal moves, a fixed 2086-move action
  space, and a 14×10×9 board encoding.
- **One policy–value neural network**, reused by every learning agent.
- **Four interchangeable "players"** (Minimax / IL / PPO / MCTS) + a Random floor —
  the *only* thing that changes between runs.
- **One evaluation league** — a colour-balanced round-robin → **Elo rating with 95%
  confidence intervals**, a head-to-head win-rate matrix, and an absolute anchor
  from the strong engine **Pikafish**.

Because only the paradigm changes, any difference in strength is caused by the
**paradigm itself** — so I can **compare and analyse** which approach wins and
**why**. Keeping the conditions identical for every agent is simply what makes the
comparison fair.

## 5. What I'm Researching (the three questions)
- **RQ1 — Ranking:** how do the four paradigms rank in Xiangqi strength on one Elo
  scale?
- **RQ2 — Imitation effect:** does imitation pretraining improve the PPO self-play
  that follows, or degrade it?
- **RQ3 — Binding constraint:** which factor most limits strength — search depth,
  model capacity, training data/teacher, or the learning algorithm?

## 6. The Objective (purpose)
To produce a **controlled, reproducible, evidence-backed answer** to "which learning
paradigm plays strongest for Xiangqi, and why" — by comparing them head-to-head —
plus a **reusable evaluation league** other researchers can reuse.

## 7. What Is Already Done ✅
- Full **Xiangqi engine** (rules, legal moves, action space, encoder, Gym env).
- The **policy–value network** + **all four agents**, all **trained**.
- The full **dataset pipeline**: 141,356 human games → 11.6M positions (split
  113k/14k/14k).
- The **evaluation league** (Elo + head-to-head + Pikafish anchor) and a
  **Streamlit dashboard**.
- **Preliminary results + the key finding**:
  - **Elo:** MCTS **1703** > IL 1557 > Minimax 1541 > IL→PPO 1411 > Random 1287.
  - MCTS is strongest; **imitation alone beats PPO fine-tuning** (RL hurt it).
  - Four different levers (more search, more epochs, engine distillation, a 4×
    bigger network) **all hit the same ~1690 Elo ceiling** → **the limit is model
    capacity + data/compute scale, not the algorithm.**
- The **4-page research proposal** (`research/proposal/`).

## 8. What Is Not Done Yet + Plan / Timeline (to defense, end of May 2027)
**Still to do:** scale the tournaments up for tighter confidence intervals, add the
Pikafish absolute anchor across all agents, finish the single-variable ablations,
write the thesis, and build the defense demo.

| Month (from now) | What I will do |
|---|---|
| **Nov 2026** | Finalise the proposal (done); formalise the test-set accuracy analysis; start scaling the round-robins. |
| **Dec 2026** | Scale every tournament for **tighter 95% CIs**; add the **Pikafish anchor**; lock the final result tables. |
| **Jan 2027** | Write thesis **Chapters 1–3** (intro, related work, method). |
| **Feb 2027** | Write **Chapter 4** (results); finish the ablations; finalise figures. |
| **Mar 2027** | Write **Chapter 5** (discussion); deliver the **full draft** to advisor. |
| **Apr 2027** | Revisions; reproducibility pass; **build the defense demo** + slides. |
| **May 2027** | Rehearse, mock defense, **defend (end of May 2027).** |

*Why this is realistic:* the code and preliminary data are already finished, so the
risky part (long training runs) is behind me; the remaining work is scale-up and
writing, where a slip in one month can be absorbed by the next.

## 9. The GPU I Trained On (explained simply)
I trained everything on **one NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB of video
memory** — inside an ordinary laptop.

**In plain terms:**
- A **GPU** (graphics card) is the chip that does the heavy parallel math AI
  training needs. The RTX 4060 Laptop is a **consumer gaming laptop card** —
  mid-range, the kind you'd find in a regular gaming laptop, **not** a specialised
  server/data-center chip.
- **"8 GB"** is its working memory — how big a model and how much data it can hold
  at once. 8 GB is modest, so it caps how large my network can be (this is directly
  why I hit the capacity ceiling).
- **For scale:** AlphaGo-type systems were trained on **dozens to thousands** of
  data-center GPUs/TPUs costing millions of dollars. Mine is **a single everyday
  laptop GPU.**

I report the GPU only for transparency and reproducibility. The **point of the
project is the comparison** — which paradigm wins and **why** — and using the same
hardware for every agent is just what keeps that comparison fair.

---

*Every number and citation here is backed by `research/08_progress_review.md`,
`research/proposal/` (incl. `references.bib`), and `docs/13-WORKLOG.md`.*
