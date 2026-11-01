# 03 — Research Framework

> The complete, locked research framework for the proposal. Every element is
> traceable to the audit (`01_project_audit.md`) and the literature review
> (`02_literature_review.md`). The IEEE proposal (`proposal/main.tex`) typesets
> this content; nothing in the paper should exceed what is specified here.

---

## 5.1 Research Title

**Primary (used in the paper, two lines):**
> **A Controlled Comparison of Deep-Learning Paradigms
> for Xiangqi under a Constrained Compute Budget**

Rationale: it names the *contribution* (a controlled comparison), the *method
context* (deep-learning paradigms, set against the classical baseline), the
*domain* (Xiangqi), and the *distinguishing condition* (a constrained compute
budget) — satisfying the brief's rule against generic "development of a Xiangqi
app" titles, while fitting cleanly onto two title lines. The heterogeneous agents
competing on one Elo ladder constitute the **multi-agent evaluation league**
framing.

---

## 5.2 Research Background
Strong board-game agents are built from three ideas established on Go and chess:
imitation of human experts, reinforcement learning by self-play, and tree search
over a learned network [`silver2016`,`silver2017`,`silver2018`]. These results,
however, were obtained at cluster scale, and the published deep-RL work on
**Xiangqi** — a larger-board, higher-branching relative of chess — consists of a
few single-system papers [`yilmaz2025`,`li2023jiangjun`,`xiangqir1`]. For the
researcher or student working on **a single consumer GPU**, it is unknown which
paradigm is worth the effort, whether the celebrated imitation→RL recipe still
pays off, and what actually limits strength at that scale. Xiangqi is an apt
research context: it is culturally major yet academically under-served, and its
rules (10×9 board, cannon screen-capture, river/palace constraints, and
*stalemate counts as a loss*) make it a non-trivial transfer target rather than
a re-run of chess.

---

## 5.3 Problem Statement
Choosing a learning paradigm for a game-playing agent under limited compute is
currently guesswork. The field's headline recipes were validated at a scale most
practitioners cannot reach, were demonstrated on Go/chess rather than Xiangqi,
and were reported as single successful systems rather than controlled
comparisons [`silver2018`,`yilmaz2025`]. Concretely:

- **What is the problem?** There is no equal-budget, like-for-like measurement of
  how classical search, pure PPO self-play, imitation-initialised PPO, and
  AlphaZero-style MCTS compare in Xiangqi playing strength, nor of whether
  imitation pretraining helps or harms the subsequent RL, nor of which factor
  bounds strength.
- **Where does it occur?** In the constrained-compute regime (one consumer GPU),
  which is where most research and teaching actually happens.
- **Why does it matter?** Practitioners waste compute on paradigms that do not
  pay off under their budget; the literature's single-run, cluster-scale claims
  may not transfer, and un-replicated RL results are known to be unreliable
  [`henderson2018`].
- **Who is affected?** Student and independent researchers, game-AI
  practitioners, and the under-served Xiangqi research community.
- **What evidence supports it?** This project's own preliminary runs (audit §14):
  under a single-GPU budget, imitation alone outranked every PPO variant, PPO
  self-play *reduced* the imitation agent's strength, and four independent
  interventions (more search, more epochs, engine distillation, 4× network size)
  all failed to pass a ~1690-Elo ceiling.

---

## 5.4 Research Questions
- **RQ1 — Ranking.** Under a fixed single-GPU compute budget, how do the four
  paradigms (classical alpha-beta Minimax, pure-self-play PPO, imitation-then-PPO,
  and AlphaZero-style MCTS+network) rank in Xiangqi playing strength on a common
  Elo scale?
- **RQ2 — The imitation→RL effect.** Does imitation-learning pretraining improve
  subsequent PPO self-play under this budget, or can RL fine-tuning degrade the
  imitation policy?
- **RQ3 — The binding constraint.** Which factor — play-time search,
  model capacity, training data/teacher, or the learning algorithm — most limits
  playing strength at this scale?

(Each RQ is answerable by the methodology in §5.8 and the already-built
infrastructure in the audit.)

---

## 5.5 Research Objectives (measurable)
- **O1.** Build a unified comparison apparatus — one rule engine, one 2086-action
  space, one policy-value network, one Elo estimator — so the *only* variable
  between measured agents is the decision function. *(Status: implemented;
  serves as feasibility evidence.)*
- **O2.** Train all four paradigms under a matched single-GPU budget and produce
  an **Elo ranking with 95% confidence intervals** and a head-to-head matrix
  (answers RQ1).
- **O3.** Quantify the **imitation→RL effect** as the signed Elo difference
  between the imitation-only agent and its PPO-refined versions (answers RQ2).
- **O4.** Run a **four-lever ablation** — simulations, training epochs,
  teacher (human vs engine distillation), and network size — and report each
  lever's Elo effect to identify the binding constraint (answers RQ3).
- **O5.** Measure how **training-stability** (replay-buffer continuity and
  best-checkpoint selection) affects final Elo, yielding reproducibility guidance
  for self-play RL (supporting contribution).

---

## 5.6 Research Hypotheses
- **H1 (RQ1).** AlphaZero-style MCTS over the learned network achieves the
  highest Elo among the four paradigms. *(Preliminary support: MCTS 1703 vs next
  best 1557.)*
- **H2 (RQ2).** Imitation pretraining raises the *starting* strength, but under
  the constrained budget PPO self-play does **not** raise the ceiling and can
  **lower** it. *(Preliminary support: IL_only 1689 > all IL+PPO and PPO
  variants; self-play fine-tuning reduced the IL agent.)*
- **H3 (RQ3).** Model capacity / training-data scale — not play-time search or
  the learning algorithm — is the binding constraint. *(Preliminary support: more
  sims, more epochs, engine distillation, and a 4× network all plateau near
  1690 Elo.)*

All three are **falsifiable** with the proposed experiments and would be
*rejected* if, e.g., a PPO variant overtakes MCTS, if IL+PPO reliably exceeds IL,
or if a single lever lifts Elo well past the ceiling.

---

## 5.7 Proposed Solution (the apparatus under test)
A single shared stack, with the paradigm as the only swapped component:

- **Environment (control).** A tested Xiangqi rule engine exposed as a Gymnasium
  env, with a fixed **2086-move action space** and legal masking, and a
  **14×10×9 side-to-move-relative** board encoding.
- **Shared network.** One AlphaZero-style policy-value network (input conv → N
  residual blocks → 2086-logit policy head + scalar value head), reused by
  imitation, PPO (actor+critic), and MCTS (priors + leaf value); network size is
  configurable for the O4 ablation.
- **The four agents.** (1) Minimax: alpha-beta negamax with material evaluation;
  (2) PPO-scratch: MaskablePPO self-play from random init; (3) IL→PPO:
  supervised imitation of 113k human games, then MaskablePPO refinement;
  (4) MCTS+NN: PUCT search over the network with value-head leaf evaluation.
- **Evaluation harness.** Colour-balanced round-robin tournaments, relative Elo
  by iterative fit (anchor 1500) with delta-method 95% CIs, and an **absolute
  anchor** via a fixed-strength engine (Pikafish) agent.

---

## 5.8 Research Methodology
- **Design.** Controlled comparative study (quasi-experimental): one independent
  variable (the paradigm / the ablated lever), one primary dependent variable
  (Elo), everything else held fixed by the shared stack.
- **Compute budget.** A matched budget per RL agent on one RTX-4060-class GPU
  (e.g. equal self-play step counts), reported explicitly so the comparison is
  "equal-effort."
- **Data.** 141,356 real human games → 11.6M positions, game-level split
  113k/14k/14k, recorded in a committed `dataset_manifest.json`; an engine
  self-play corpus for the distillation lever.
- **Procedure.**
  1. Train each paradigm to its budget, checkpointing for reproducibility.
  2. **Experiment 1 (RQ1):** round-robin among all four + Random, colours
     balanced, N games/pair; fit Elo + CIs; build head-to-head matrix.
  3. **Experiment 2 (RQ2):** isolate IL vs IL+PPO vs PPO-scratch; report the
     signed Elo deltas and learning-curve proxies (IL top-1 accuracy; PPO
     reward/value traces).
  4. **Experiment 3 / ablation (RQ3):** vary one lever at a time — MCTS
     simulations (100/200/400/800), IL epochs, teacher (human vs Pikafish
     distillation), network size (1.15M vs 4.85M) — each re-entered into the
     round-robin; report each lever's Elo effect.
  5. **Stability study (O5):** compare fragmented vs continuous vs
     buffer-persisted MCTS training; report Elo and value-loss trajectories.
- **Statistics.** 95% Elo CIs via the delta method; differences called
  significant only when CIs do not overlap; non-transitivity handled by using
  full round-robins and fitted Elo rather than single head-to-heads
  [`li2023jiangjun`].
- **Reproducibility.** Fixed seeds, committed manifest, per-run config/seed/commit
  logging, and regenerable tables/figures — directly addressing the concerns of
  Henderson *et al.* [`henderson2018`].

---

## 5.9 Evaluation Metrics
| Metric | What it measures | For |
|---|---|---|
| **Elo (relative, fitted) + 95% CI** | overall playing strength | RQ1, RQ3 |
| **Absolute Elo vs Pikafish anchor** | strength on a fixed scale | RQ1 |
| **Head-to-head win rate matrix** | pairwise dominance, non-transitivity | RQ1 |
| **Signed Elo delta (IL vs IL+PPO vs PPO)** | the imitation→RL effect | RQ2 |
| **Top-1 move-prediction accuracy** | imitation fidelity (learning proxy) | RQ2, RQ3 |
| **Per-lever Elo effect** | which factor binds strength | RQ3 |
| **Value-loss trajectory & Elo under resume patterns** | training stability | O5 |
| **Compute cost (GPU-hours / steps)** | budget accounting | all |

All are objective and measurable; none requires human judgement.

---

## 5.10 Expected Contributions
- **Technical.** A reusable, single-interface **multi-agent evaluation league**
  for comparing heterogeneous game-playing paradigms on one rule engine,
  runnable on a single consumer GPU.
- **Research (empirical).** The first **controlled, equal-budget, Elo-anchored
  comparison** of classical search, PPO, imitation+RL, and AlphaZero-style MCTS
  for Xiangqi; a quantified answer to whether imitation pretraining helps RL
  under constraint; and a **multi-lever capacity-bottleneck analysis**.
- **Software-engineering.** Evidence and guidance that **training-stability
  engineering** (replay-buffer continuity, best-checkpoint selection) materially
  affects learned-model quality — a concrete, actionable reproducibility finding
  for self-play RL systems.
- **Scope honesty.** Contributions are deliberately framed as *measurement and
  comparison under constraint*, not "a new state-of-the-art Xiangqi engine" —
  the results are strongest as a rigorous, honest study including negative
  findings.

---

## Alignment check (brief's required chain)
Problem (§5.3) → Gap (02 §3) → RQs (§5.4) → Objectives (§5.5) → Method (§5.7–5.8)
→ Evaluation (§5.9) → Contribution (§5.10). Each RQ has a matching objective, a
matching experiment, and a matching metric; no promised result lacks an
experiment that can measure it.
