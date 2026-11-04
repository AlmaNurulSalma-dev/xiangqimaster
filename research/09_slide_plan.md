# Progress-Review Slide Plan (text only — not built yet)

**Follows `10_project_brief.md`** (problem → gap → my project → blueprint → the four
paradigms → research → objective → what's built → results → demo → not-done →
timeline), **plus Xiangqi context slides** up front for people who don't know the
game. The talk is about **comparing four AI paradigms head-to-head and analysing
which is strongest and why** — not about any budget.
**Deck:** 16 slides + 6 backup · **~12 min + demo** · audience: advisor + classmates.

Each slide gives: **On-slide** (text to put on the slide) · **🎤 Say** (speaker
notes) · **📊 Visual / data needed** (chart/table/figure + where to get it).

> Most visuals already exist in `research/proposal/figures/` — reuse them.
> Accent colour: deep red (Xiangqi's Red side). ≤ 6 bullets/slide.

---

## Slide 1 — Title
**On-slide:**
- XiangqiMaster — Benchmarking Four AI Paradigms for Chinese Chess (象棋)
- Course Project Progress Review · Alma Nurul Salma (诺艾玛) · L25020007
- Nanjing Xiaozhuang University (南京晓庄学院) · Nov 2026

🎤 "I take four different ways to teach an AI to play Chinese Chess, put them in one
league, and compare them head-to-head to find **which is strongest and why**."
📊 **Visual:** `gfig1_board.png` faint in the background (from proposal).

---

## Slide 2 — What Is Xiangqi?  *(context for non-players)*
**On-slide:**
- Xiangqi (象棋) = **Chinese Chess** — same family as Western chess
- Like chess: **two armies**, turn-based, you win by **checkmating** the enemy
  **General** (its "king"); similar pieces — chariot ≈ rook, horse ≈ knight, etc.
- Different board: a **9×10 grid**, pieces sit on the **line intersections** (not
  inside squares), and a **river** runs across the middle
- The river **changes the rules**: a **Soldier** that crosses it **gains a sideways
  move**; **Elephants can never cross** it; the **General** is locked in its **palace**

🎤 "Quick primer, since not everyone plays it. It's the same idea as chess — two
armies, checkmate the enemy king — so if you know chess, you already understand the
goal. The differences: pieces sit on intersections, there's a **river** down the
middle, and that river actually changes how pieces move — a pawn that crosses it can
suddenly move sideways, and some pieces aren't allowed across at all."
📊 **Visual:** `gfig1_board.png` (board + pieces). Inset `gfig2_moves.png` to show the
river and how a Soldier's moves change after crossing.

---

## Slide 3 — Xiangqi vs Chess: Similar Game, Harder Problem
**On-slide:**
- Same family as chess — **but the rules are richer and the problem is bigger**
- More unusual rules: the **Cannon captures by jumping** over one screen piece, the
  **Horse can be blocked**, the river + palace constraints above

| Measure (bigger = harder) | Chess | Xiangqi |
|---|---|---|
| Board | 8×8 = 64 squares | **9×10 = 90 points** |
| Branching factor (moves/turn) | ~35 | **~38** |
| Average game length | ~70 plies | **~95 plies** |
| Game-tree complexity | ~10¹²³ | **~10¹⁵⁰** |

🎤 "It looks like chess, but on every axis of difficulty it's at least as hard — a
bigger board, more moves to weigh each turn, longer games, and a far larger game tree,
about 10¹⁵⁰ possible games versus 10¹²³ for chess. On top of that the rules are
quirkier: the cannon only captures by jumping over another piece, and the horse can be
physically blocked. Full rules are one click away for anyone curious."
📊 **Visual / data:** build this **comparison table** (new); numbers from Yilmaz et
al. 2025 (arXiv:2506.15880). **Put a direct link on the slide** for the special rules:
`https://en.wikipedia.org/wiki/Xiangqi` (make it a clickable QR/hyperlink).

---

## Slide 4 — Xiangqi Is a Serious AI Benchmark  *(hook)*
**On-slide:**
- **Did you know?** Xiangqi is so complex it's a live testbed in published AI research
- **JiangJun** (Li et al. 2023, journal **TMLR**) — learned on **>10,000** human games
- **Xiangqi-R1** (Chen et al. 2025) — used to **test whether LLMs can reason**
- **Yilmaz et al. 2025** — an AlphaZero-style Xiangqi player
- → **but none of them compares the paradigms against each other — that's my project**

🎤 "Here's the hook: this isn't a toy. Xiangqi is so complex that it's used in serious
AI research. **One journal paper — JiangJun, in TMLR — trained on over ten thousand
human games** just to study how to play it well. Then it gets more interesting:
**people recently used Xiangqi to test whether large language models can actually
reason** — that's the Xiangqi-R1 study — because the game forces genuine multi-step
thinking. And there's an AlphaZero-style player from Yilmaz and colleagues. So three
strong, recent works. **But here's the gap: every one of them builds a single system.
Not one of them puts the different learning methods side by side to ask which is
actually best. That empty space is exactly where my project lives.**"
📊 **Visual:** 3 small 'paper cards' (title · authors/year · venue), with an arrow
leading to a 4th card: **"This project — the comparison nobody did."** No new data.

---

## Slide 5 — The Problem
**On-slide:**
- Modern game AI mixes **three ideas**: imitation of experts · self-play RL · tree
  search over a learned network (the AlphaGo recipe; Silver et al. 2016–18)
- Every Xiangqi paper ships **one** finished system — the ideas are **never compared**
- So for Xiangqi, three things are **genuinely unknown**:
  - **which** paradigm plays strongest?
  - does **imitation help** the self-play RL that follows — or **hurt** it?
  - **what limits** a learned agent's strength?

🎤 "The best game AIs blend three ingredients, but each famous paper is one finished
dish — it never tells you which ingredient mattered. So nobody actually knows, for
Xiangqi, which method wins, whether combining them helps, or what holds strength back.
My project is built to **answer those by comparing and analysing**, not to crown one
pre-chosen method."
📊 **Visual:** icon row 🧠 imitation → 🔁 self-play → 🌳 search → ❓ "which wins?"

---

## Slide 6 — The Gap *(from the literature, not assumption)*
**On-slide — gap table:**

| Prior work | Xiangqi? | Compares the paradigms? | Analyses *why* one wins? |
|---|:--:|:--:|:--:|
| AlphaGo family (2016–18) | no | no | no |
| JiangJun (Li 2023) | yes | no | no |
| Yilmaz et al. (2025) | yes | no | no |
| Xiangqi-R1 (Chen 2025) | yes | no | no |
| **This project** | **yes** | **yes** | **yes** |

🎤 "Each Xiangqi system optimises one pipeline; none compares them, and none digs into
*why* one approach beats another. That empty bottom row is my gap — and Henderson et
al. 2018 add that single-run results are exactly the ones you can't trust, which is why
a careful side-by-side comparison with analysis is worth doing."
📊 **Visual / data:** reuse the **gap table** from the proposal (`tab:gap` in
`main.tex`), relabelled to these comparison/analysis columns.

---

## Slide 7 — Problem & Solution  ← heart of the talk
**On-slide:**
- 🔴 **PROBLEM:** for Xiangqi, nobody has compared the learning paradigms **head-to-head** —
  so which is strongest, whether imitation helps the RL, and what limits strength are
  all open.
- 🟢 **SOLUTION:** run all four in **one league** on the **same engine, rules, action
  space, network and Elo scale**, then **compare and analyse** which wins and why.
- → keeping the conditions identical for every agent is simply what makes the
  comparison **fair** — any strength gap then comes from the **paradigm itself**.

🎤 "Problem: the methods have never been compared for Xiangqi. Solution: a controlled
comparison — I hold everything identical and swap only the learning method, so the
result is a fair, analysable answer to 'which is best, and why'. Keeping conditions the
same for all of them is just the fairness rule of the experiment, not the point of it —
**the point is the comparison and the analysis.**"
📊 **Visual:** split slide, red PROBLEM box / green SOLUTION box. No new data.

---

## Slide 8 — How It Connects + Blueprint of the Solution
**On-slide:**
- My project **is** that missing comparison — a 4-agent **league** on one engine
- One Xiangqi engine · fixed **2086-move** action space · **14×10×9** encoder
- One **policy–value network** reused by every learning agent
- **Four interchangeable players** + a Random floor — only this changes between runs
- Evaluation: colour-balanced round-robin → **Elo (95% CI)** + head-to-head matrix +
  **Pikafish** reference anchor

🎤 "Here's the machine. Everything in grey is shared; only the coloured decision
function changes between runs — that's what makes every comparison apples-to-apples."
📊 **Visual:** `gfig4_framework.png` (the apparatus, from proposal).

---

## Slide 9 — The Four Paradigms I Compare  *(what each is · how it works · where it's used)*
**On-slide — table:**

| Paradigm | How it works (one line) | Used in the real world |
|---|---|---|
| **Minimax** (alpha–beta) | Looks ahead in the game tree, assumes best play on both sides, picks the best worst-case move — a hand-written evaluation, **no learning** | **IBM Deep Blue** (beat Kasparov, 1997); classical chess engines |
| **Imitation Learning** | Trains a network to **copy the move a human master played** (supervised learning) | **AlphaGo's** first policy net; **behaviour cloning** in self-driving cars & robots |
| **Self-Play RL (PPO)** | Plays **against itself**, rewarded for winning, and nudges its policy toward moves that win — **learns from scratch, no human data** | **OpenAI Five** (Dota 2), **AlphaStar** (StarCraft II); **PPO** also powers RLHF in ChatGPT-style models |
| **MCTS + network** | **"Intuition + calculation":** the network suggests moves & scores positions, then a **look-ahead search runs many simulations** before moving (AlphaZero's PUCT) | **AlphaGo / AlphaZero / MuZero** (Go, chess, shogi) |

🎤 "These are the four contestants. Minimax is the classical calculator — pure
look-ahead, no learning, the Deep Blue approach. Imitation learning copies human
experts, the way AlphaGo started and the way self-driving cars learn from human
drivers. Self-play RL learns entirely on its own by playing itself — that's OpenAI
Five and AlphaStar, and the same PPO algorithm that fine-tunes ChatGPT. And MCTS plus a
network is the full AlphaZero recipe: intuition from the network, calculation from the
search. I implemented all four on the same engine so I could compare them directly."
📊 **Visual:** `gfig5_mechanisms.png` (the 4 mechanisms side by side — exists in
`research/proposal/figures/`, not used in the 4-page proposal, perfect here).

---

## Slide 10 — What I'm Researching + Objective
**On-slide:**
- **RQ1 — Ranking:** how do the four paradigms rank on one Elo scale?
- **RQ2 — Imitation effect:** does imitation pretraining help the following PPO, or
  degrade it?
- **RQ3 — What limits strength:** search, network capacity, data/teacher, or algorithm?
- **Objective:** a controlled, reproducible, **analysed** answer to "which paradigm
  wins and why" + a **reusable evaluation league** others can use

🎤 "Three research questions — the ranking, the effect of imitation, and what actually
limits strength — and one goal: a trustworthy, analysed answer, plus a reusable tool."
📊 **Visual:** three numbered RQ cards; reuse the icon motif from Slide 5.

---

## Slide 11 — What's Already Built ✅  *(step-by-step, with dates)*
**On-slide — build timeline table:**

| When | What I did | Detail |
|---|---|---|
| **19 Sep** | Built a correct **Xiangqi engine first** | Rules, legal-move generation (cannon jump, blockable horse, palace/river, stalemate = loss), fixed **2086-move action space**, **14×10×9** encoder, Gymnasium env; verified (44 legal opening moves) |
| **20 Sep** | **"Big bang":** the whole learning stack in one day | Policy–value network + **all four agents** (Minimax, PPO, IL, MCTS) + Elo/tournament harness + first Streamlit dashboard — built, not yet trained |
| **21–23 Sep** | **Data pipeline** → real human games | Parsed the `CGLemon/chinese-chess-PGN` dataset, scaled to full size, wrote the **IL→PPO bridge** |
| **1–3 Oct** | Made training **survive interruptions** | Intra-epoch + periodic **checkpointing**, robust `--resume`, TensorBoard/CSV logging (needed because runs kept dying) |
| **2 Oct** | Trained **Imitation Learning** | 30 epochs over 9.3M positions, ~17 min/epoch (one run killed at epoch 21, resumed) → **~34% top-1** move accuracy — first trained model |
| **3 Oct** | Trained **PPO** (both ways) | IL→PPO fine-tune (**501k steps, ~7.5 h**) and PPO-from-scratch (500k steps) |
| **4–9 Oct** | **MCTS v1** — and a hard bug | Had to stop/resume **7×**; each resume reset the replay buffer → agent **regressed 1703 → 1582 Elo** |
| **13–21 Oct** | **MCTS v2** — stabilised | Persistent replay buffer + best-checkpoint, run **detached** in the background → recovered to **~1691 Elo**, no more spikes |
| **24 Oct** | **Pikafish distillation** | Generated 5,000 engine-vs-engine games, trained on them — plateaued **below** human imitation |
| **25–26 Oct** | **4× bigger network** (~4.85M params) | Gained **almost nothing** (+0.008 accuracy) |
| **1–2 Nov** | **4-page research proposal** | Framed the finished system as a controlled paradigm comparison |

**How the dataset was processed to fit the project (21–23 Sep):**
- Took **141,356 legal Xiangqi games** (public PGN dataset).
- **Parsed** each game's moves (ICCS notation) and **replayed** them through my engine,
  checking every move is legal (spot-check: 500/500 legal).
- **Expanded** games into individual board positions → **11.6M positions**; each
  position **encoded to a 14×10×9 tensor**, with the **label = the move the human
  actually played** (an index in the 2086-move action space).
- **Split at the game level** into 113k train / 14k val / 14k test (train ≈ 9.3M
  positions) so no game leaks across splits; manifest committed for reproducibility.

🎤 "This is the honest story of how it was built, in order. I built the **engine
first** — because if the rules were wrong, every comparison would be meaningless. Then
the whole learning stack in one intense day. Then I turned 141 thousand human games
into 11.6 million training positions. Then — and this matters — I made training
*survive* before I trained seriously, because my early runs kept dying. Then I trained
each agent in dependency order. The MCTS agent was the hardest part: it regressed
because of a bug, I found the root cause, fixed it, and recovered the strength."
📊 **Visual / data:** this **build-timeline table**; add a Streamlit dashboard
screenshot; optional small layered-stack diagram (engine → network → agents → league).

---

## Slide 12 — Results So Far  *(detailed comparison + analysis)*
**On-slide:**
- **Main ranking (Elo, anchor 1500, 95% CI):**
  **MCTS 1703** > IL 1557 > Minimax 1541 > IL→PPO 1411 > Random 1287
- **Imitation move accuracy:** 42.3% top-1 / 66.5% top-3 — **63% in the opening**,
  falling to **32%** in the endgame
- **Reference anchor:** Pikafish at depth 8 ≈ **2062 Elo** → my agents are
  **strong-amateur / club** level
- **Analysis — what the comparison shows:**
  - **Search wins:** MCTS (intuition + look-ahead) is clearly strongest
  - **Imitation alone beats IL→PPO** — self-play RL *degraded* the policy here
  - Pushing harder four ways (more search · more epochs · engine distillation · 4×
    bigger net) **all land at ~1690 Elo** → the limit is **network capacity + data**,
    not the learning algorithm

🎤 "Here's what the comparison actually says. First, search wins — the AlphaZero-style
MCTS agent is clearly the strongest. Second, a surprise: adding reinforcement learning
on top of imitation made it *worse*, not better — a genuine negative result that a
single-system paper would never surface. Third, when I tried four independent ways to
push strength higher, they all hit the same ceiling around 1690 — which tells me the
bottleneck is the size of the network and the data, not the algorithm. That's the kind
of analysis this whole project was built to produce."
📊 **Visual / data:** `elo_bars.png` + `headtohead.png` (Fig 4, from proposal); the
**per-phase accuracy table** from `results/test_report/` (opening 63% → endgame 32%);
optional small "four arrows → one wall at ~1690" sketch to illustrate the ceiling.

---

## Slide 13 — Live Demo
**On-slide:**
- AI vs AI — watch MCTS beat the weaker agents
- Play any agent yourself: Random → Minimax → IL → MCTS = Easy → Hard
- Compare vs Pikafish at adjustable depth · "like AlphaGo, for Chinese Chess"

🎤 *(Switch to the dashboard.)* "Let me show it actually playing."
📊 **Visual:** LIVE Streamlit dashboard. *(Fallback: short screen recording.)*

---

## Slide 14 — What's Not Done Yet
**On-slide:**
- Scale the tournaments up → **tighter confidence intervals**
- Add the **Pikafish absolute anchor** across all agents
- Finish the single-variable **ablations** (search / epochs / teacher / net size)
- Write the thesis + build the defense demo

🎤 "The comparison and the analysis exist; what's left is scaling it up for tighter
numbers, finishing the ablations, and the writing plus the demo."
📊 **Visual:** short ▶ checklist. No new data.

---

## Slide 15 — Schedule to Defense (end of May 2027)  *(detailed)*
**On-slide — timeline table:**

| Period | Milestone | Detail |
|---|---|---|
| **Nov 2026** (now) | Finalise proposal · formalise accuracy analysis · start scaling | 4-page proposal done; lock the test-set move-prediction report; begin bigger round-robins |
| **Dec 2026** | Tighter results + absolute anchor | Scale every tournament to more games/pair for **tighter 95% CIs**; add **Pikafish anchor** across all agents; lock final result tables |
| **Jan 2027** | Write **Chapters 1–3** | Introduction · Related Work · Methodology (engine, encoding, the four agents, the league) |
| **Feb 2027** | Write **Chapter 4** + finish ablations | Results & analysis; complete the single-variable ablations; finalise all figures |
| **Mar 2027** | Write **Chapter 5** + full draft | Discussion, limitations, future work; deliver the **full first draft** to advisor |
| **Apr 2027** | Revisions + build the demo | Advisor revisions; reproducibility pass (fixed seeds, committed manifest); build the **defense demo** + slides |
| **May 2027** | Rehearse & **defend** | Mock defense, final revisions, **thesis defense (end of May 2027)** |

🎤 "This is realistic because the risky part — the long training runs — is already
behind me. What's left is scale-up, writing, and the demo, so a slip in any one month
can be absorbed by the next without touching the experiments."
📊 **Visual / data:** reuse the **timeline table** from the proposal (`tab:timeline`),
or draw a horizontal Nov→May bar with a 'you are here' marker on Nov.

---

## Slide 16 — Closing
**On-slide:**
- A gap the Xiangqi literature left open — **a head-to-head comparison of the four
  paradigms** — now built, run, and analysed
- Finding so far: **MCTS strongest; imitation beats IL→PPO; capacity (not the
  algorithm) is the limit**
- On track for the **May 2027** defense
- 谢谢 · **Thank you — questions?**

🎤 "To close: I built the comparison nobody had built for Xiangqi, ran it, and analysed
*why* one method wins. The headline is that search wins and — surprisingly — more RL
hurt. I'm on track for May. Thank you — happy to take questions."
📊 **Visual:** board image + "谢谢 / Thank you". No new data.

---

### Backup slides (hidden, for Q&A)
- **B1 — Full reference list** (Silver 2016/17/18, Browne 2012, Schulman 2017,
  Vinyals 2019, Henderson 2018, Yilmaz 2025, Li 2023, Chen 2025).
- **B2 — Game-state encoding** → `gfig3_tensor.png` (14×10×9 tensor, from proposal).
- **B3 — Self-play training loop** → `gfig6_selfplay.png` (from proposal).
- **B4 — Per-phase accuracy table** (opening 63% → midgame 38% → endgame 32%) from
  `results/test_report/`.
- **B5 — MCTS stability story** (1703 → 1582 → 1691, the buffer-reset bug + fix).
- **B6 — All experiments list** (7-agent round-robin, MCTS headline, MCTS v2, Pikafish
  distillation, bigger-net, simulation ablation).

---

### Asset checklist (what to prepare)
**Already exist in `research/proposal/figures/` — reuse:** `gfig1_board.png`,
`gfig2_moves.png`, `gfig3_tensor.png`, `gfig4_framework.png`, `gfig5_mechanisms.png`,
`gfig6_selfplay.png`, `elo_bars.png`, `headtohead.png`.
**Reuse from `main.tex`:** gap table, timeline table.
**Reuse from `results/test_report/`:** move-accuracy table (per-phase).
**New to make:** (1) Xiangqi-vs-chess comparison table **+ a clickable link/QR to
`https://en.wikipedia.org/wiki/Xiangqi`** (Slide 3); (2) the 4 'paper cards' + "this
project" card (Slide 4); (3) the four-paradigms table (Slide 9); (4) the build-timeline
table (Slide 11); (5) a Streamlit dashboard screenshot; (6) optional "four arrows → one
wall at ~1690" sketch (Slide 12).

*Sources: `research/10_project_brief.md`, `research/08_progress_review.md`,
`research/proposal/` (incl. `references.bib`), `docs/13-WORKLOG.md`, and Yilmaz et al.
2025 (arXiv:2506.15880) for the chess-vs-Xiangqi numbers.*
