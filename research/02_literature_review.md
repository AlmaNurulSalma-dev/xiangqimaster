# 02 — Literature Review & Research Gap

> **Scope.** Literature relevant to the selected research direction: a
> *controlled, equal-compute, Elo-anchored comparison of game-playing paradigms
> (classical search, policy-gradient RL, imitation+RL, and AlphaZero-style
> MCTS) for Xiangqi.* Every reference below has been verified for existence and
> bibliographic detail via public sources (arXiv, publisher records, DOIs). No
> reference, author, year, or DOI is fabricated. Where a claim could not be
> verified from a primary source it is flagged as such.

**Citation key** (used in the proposal's `references.bib`):

| Key | Reference (verified) |
|---|---|
| `silver2016` | D. Silver *et al.*, "Mastering the game of Go with deep neural networks and tree search," *Nature*, vol. 529, no. 7587, pp. 484–489, 2016. DOI 10.1038/nature16961. |
| `silver2017` | D. Silver *et al.*, "Mastering the game of Go without human knowledge," *Nature*, vol. 550, no. 7676, pp. 354–359, 2017. DOI 10.1038/nature24270. |
| `silver2018` | D. Silver *et al.*, "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, vol. 362, no. 6419, pp. 1140–1144, 2018. DOI 10.1126/science.aar6404. |
| `browne2012` | C. B. Browne *et al.*, "A survey of Monte Carlo tree search methods," *IEEE Trans. Comput. Intell. AI Games*, vol. 4, no. 1, pp. 1–43, 2012. DOI 10.1109/TCIAIG.2012.2186810. |
| `schulman2017` | J. Schulman *et al.*, "Proximal policy optimization algorithms," *arXiv:1707.06347*, 2017. |
| `vinyals2019` | O. Vinyals *et al.*, "Grandmaster level in StarCraft II using multi-agent reinforcement learning," *Nature*, vol. 575, pp. 350–354, 2019. DOI 10.1038/s41586-019-1724-z. |
| `henderson2018` | P. Henderson *et al.*, "Deep reinforcement learning that matters," in *Proc. AAAI Conf. Artif. Intell.*, vol. 32, no. 1, 2018. DOI 10.1609/aaai.v32i1.11694. |
| `yilmaz2025` | B. Yilmaz, J. Hu, and J. Liu, "Deep reinforcement learning Xiangqi player with Monte Carlo tree search," *arXiv:2506.15880*, 2025. |
| `li2023jiangjun` | Y. Li *et al.*, "Mastering Xiangqi by tackling non-transitivity in two-player zero-sum games," *Trans. Mach. Learn. Res. (TMLR)*, 2023. *arXiv:2308.04719*. |
| `xiangqir1` | *Xiangqi-R1*: "Enhancing spatial strategic reasoning in LLMs for Chinese Chess via reinforcement learning," *arXiv:2507.12215*, 2025. |

---

## 1. Thematic Survey

### 1.1 Deep RL with search for perfect-information board games
The modern template for strong board-game agents was set by AlphaGo
[`silver2016`], which combined **supervised (imitation) learning** of expert
moves, **reinforcement learning** by self-play, and **Monte Carlo Tree Search
(MCTS)** at play time. AlphaGo Zero [`silver2017`] then removed the human-data
stage, learning *tabula rasa* from self-play alone and surpassing the original.
AlphaZero [`silver2018`] generalised the same algorithm and network architecture
to chess, shogi, and Go, replacing alpha-beta search with general MCTS and
reaching superhuman strength from the rules alone. These three works establish
the two levers this project compares — *imitation initialisation* vs *pure
self-play*, and *search over a network* vs *a raw network* — but they do so at
the scale of large TPU clusters, and none targets Xiangqi.

### 1.2 Tree search: classical and neural
MCTS is surveyed comprehensively by Browne *et al.* [`browne2012`], which
documents the core selection–expansion–simulation–backpropagation loop and the
UCT/PUCT family this project's `agents/mcts.py` implements. The AlphaZero line
[`silver2017`,`silver2018`] replaced MCTS's random rollouts with a value-network
estimate — exactly the design used here. Classical Xiangqi engines
(ElephantEye, Pikafish) instead use **alpha-beta search with handcrafted
evaluation**, the tradition represented by this project's Minimax baseline
[`browne2012` contrasts the two search philosophies].

### 1.3 Policy-gradient RL and its stability
Proximal Policy Optimization [`schulman2017`] is the policy-gradient method used
for the two PPO agents (via MaskablePPO). It is valued for stability and
simplicity, yet the deep-RL reproducibility study of Henderson *et al.*
[`henderson2018`] shows that policy-gradient results vary sharply with seeds,
hyperparameters, and implementation details — motivating careful,
confidence-interval-backed evaluation and the training-stability analysis this
project contributes.

### 1.4 Imitation-then-RL and its limits
The AlphaGo pipeline [`silver2016`] demonstrated that supervised imitation of
human experts can *initialise* a policy that RL then refines. A recurring caveat
in the imitation-learning literature is that an imitator is **upper-bounded by
the demonstrator's quality**, which motivates subsequent RL. Whether that RL
step *helps* under a constrained compute budget — or whether, as this project's
preliminary results suggest, it can *erode* a sharp imitation policy — is not
settled for Xiangqi.

### 1.5 Multi-agent populations and evaluation
AlphaStar [`vinyals2019`] extended self-play to a **league (population) of
continually adapting agents**, evaluated on an Elo-style ladder — the conceptual
basis for framing this project's heterogeneous agents as a *multi-agent
evaluation league* rated on a common Elo scale. For Xiangqi specifically, JiangJun
[`li2023jiangjun`] combines MCTS with Policy-Space Response Oracles to address
**non-transitivity** (rock–paper–scissors strategy cycles), analysing >10,000
human games and reaching master level (99.41% win rate vs humans in a deployed
app). Non-transitivity is also why single head-to-head results can mislead and
why this project evaluates with full round-robins and fitted Elo.

### 1.6 Xiangqi-specific deep RL
Direct prior work on Xiangqi deep RL is **recent and sparse**. Yilmaz *et al.*
[`yilmaz2025`] present an AlphaZero-style Xiangqi player pairing a policy-value
network with MCTS self-play (a secondary review of that paper reports a
supervised-pretraining stage on ~141k historical games; the abstract confirms
the policy-value + MCTS self-play design). JiangJun [`li2023jiangjun`] targets
the game-theoretic non-transitivity problem. The most recent line, Xiangqi-R1
[`xiangqir1`], abandons the AlphaZero template entirely, using a large language
model fine-tuned with Group-Relative Policy Optimization for *reasoning* about
Xiangqi. Each of these pursues a **single, maximally strong system**; none
provides a controlled, equal-budget comparison *across* the distinct learning
paradigms.

---

## 2. Comparative Analysis (not just summaries)

| Study | Game(s) | Paradigm(s) studied | Compute scale | Comparison across paradigms? | Reports negative/limit results? | Evaluation |
|---|---|---|---|---|---|---|
| AlphaGo [`silver2016`] | Go | IL → RL + MCTS (one system) | TPU cluster | No (one pipeline) | No | Elo vs humans/programs |
| AlphaGo Zero [`silver2017`] | Go | self-play RL + MCTS | TPU cluster | Ablates IL *vs* no-IL (one game) | Partial | Elo self-ladder |
| AlphaZero [`silver2018`] | chess/shogi/Go | self-play RL + MCTS | TPU cluster | Across *games*, one paradigm | No | Elo vs engines |
| MCTS survey [`browne2012`] | many | MCTS variants | n/a (survey) | n/a | discusses open problems | n/a |
| PPO [`schulman2017`] | control/Atari | policy gradient | GPU | vs other PG methods | some | return curves |
| Henderson [`henderson2018`] | control | policy-gradient RL | GPU | *reproducibility* across configs | **Yes (the point)** | return + variance |
| AlphaStar [`vinyals2019`] | StarCraft II | league self-play | large cluster | within one paradigm | partial | Elo ladder |
| JiangJun [`li2023jiangjun`] | **Xiangqi** | MCTS + PSRO (one system) | cluster | No (one system) | exploitability analysis | vs AlphaZero-Xiangqi / humans |
| Yilmaz [`yilmaz2025`] | **Xiangqi** | AlphaZero-style (one system) | GPU | No (one system) | limited | self-play / humans |
| Xiangqi-R1 [`xiangqir1`] | **Xiangqi** | LLM + GRPO (one system) | GPU | No (one system) | limited | reasoning accuracy |
| **This project** | **Xiangqi** | **4 paradigms + Random** | **single consumer GPU (RTX 4060)** | **Yes — the contribution** | **Yes (IL>RL, capacity wall, stability)** | **unified Elo round-robin + CIs** |

Three patterns emerge. First, the strongest results all come from
**cluster-scale compute**, leaving the *constrained-budget* regime (a single
consumer GPU — the realistic setting for most researchers, students, and
practitioners) essentially unstudied. Second, every system-building paper
optimises **one pipeline** and reports where it *succeeds*; controlled
*cross-paradigm* comparison and *negative* results are rare, yet they are the
most decision-relevant findings for someone choosing a method under a budget.
Third, for **Xiangqi specifically**, the entire published deep-RL literature is
a handful of single-system papers from 2023–2025; there is no equal-budget,
Elo-anchored comparison of the classical, policy-gradient, imitation+RL, and
AlphaZero paradigms on one rule engine.

---

## 3. Research Gap

**Stated gap.** *There is no controlled, equal-compute, Elo-anchored comparison
of the major game-playing paradigms — classical alpha-beta search, pure
self-play PPO, imitation-initialised PPO, and AlphaZero-style MCTS — for Xiangqi
under a constrained single-GPU budget; consequently, two questions of direct
practical importance are unanswered for this game at this scale: (i) does
imitation pretraining actually improve RL self-play, or can it be eroded by it,
and (ii) which factor — search, model capacity, training data/teacher, or the
learning algorithm — binds playing strength.*

This gap is supported by the comparison in Section 2: the AlphaGo/AlphaZero line
[`silver2016`–`silver2018`] establishes the recipes but at cluster scale and not
for Xiangqi; the Xiangqi papers [`yilmaz2025`,`li2023jiangjun`,`xiangqir1`] each
build one system without cross-paradigm comparison; and the reproducibility
literature [`henderson2018`] warns that un-replicated, single-run RL claims are
unreliable — precisely the weakness a controlled, CI-backed comparison removes.

### 3.1 Gap classification
Using the taxonomy from the lab brief, the gap is primarily:

- **Evaluation/methodological gap (primary).** No unified, equal-budget,
  round-robin-Elo benchmark exists that places these four paradigms on one
  Xiangqi rule engine with confidence intervals. Existing Xiangqi work evaluates
  single systems, often against humans in a deployed app, without controlled
  cross-paradigm comparison.
- **Performance/knowledge gap (secondary).** The *compute–strength* relationship
  for Xiangqi at consumer-GPU scale, and the *direction* of the imitation→RL
  effect under that budget, are unquantified. This project's preliminary
  evidence (IL alone outranking every PPO variant; a ~1690-Elo ceiling that four
  independent interventions fail to breach) indicates the conventional "RL
  improves on IL / more search and bigger nets help" intuitions may not hold
  under constraint — a result worth establishing rigorously.
- **Application gap (tertiary).** Xiangqi is far less studied than Go or chess;
  transferring and *measuring* these paradigms on its larger board, different
  piece dynamics, and the stalemate-is-a-loss rule is itself an open application.

### 3.2 Why this gap is addressable here
The `xiangqimaster` codebase already provides the controlled apparatus the gap
requires — one environment, one 2086-action space, one shared network, one Elo
estimator — and has produced preliminary results for all three sub-questions
(audit Section 14). The proposed study is therefore a *rigorous completion and
formalisation* of a comparison the infrastructure already supports, not a
speculative build. This is the feasibility argument the proposal will make.

---

## 4. Citation Integrity Statement
All ten references were checked against primary or authoritative secondary
sources (arXiv abstract pages, publisher/DOI records, indexing databases).
ArXiv-only works (`schulman2017`, `yilmaz2025`, `xiangqir1`, and the preprint of
`li2023jiangjun`) are cited as preprints; `li2023jiangjun` is additionally
recorded as accepted at TMLR. One unverified detail — the exact size of the
supervised-pretraining corpus in `yilmaz2025` — is attributed to a secondary
review and is **not** stated as fact in the proposal. No DOIs, authors, venues,
or years were invented.
