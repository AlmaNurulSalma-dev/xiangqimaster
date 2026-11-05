# XiangqiMaster — Presentation Study Guide

> **Purpose.** A detailed, self-contained script to study from before the
> progress-review presentation. It answers the three required questions:
> **(1) project schedule, (2) a demonstration of what is already implemented,
> (3) the problems/difficulties encountered and the potential outcomes if they
> are resolved.**
>
> **One-line pitch (memorise this).** *"XiangqiMaster builds four different
> game-playing AIs for Chinese Chess — from classical search to an AlphaGo-style
> network — and compares them head-to-head on the same board, the same code, and
> the same budget, to find out which paradigm actually plays best and why."*
>
> **Project facts.** Student: Alma Nurul Salma (诺艾玛), ID L25020007, Nanjing
> Xiaozhuang University. Game: Xiangqi (象棋 / Chinese Chess), a 9×10 board,
> 2,086-move action space. Hardware: one laptop, NVIDIA RTX 4060 (8 GB).

---

## 0. The 30-second framing (say this first)

- Xiangqi is **like chess but with more complex rules** (a river that changes how
  soldiers move, blockable horses, the cannon's screen-capture, a palace that
  traps the general). It is **harder to search** than chess: branching factor
  ≈ 38 (chess ≈ 35), game length ≈ 95 plies (chess ≈ 70), game-tree complexity
  ≈ 10¹⁵⁰ (chess ≈ 10¹²³).
- Existing papers each study **one** method (JiangJun, Xiangqi-R1, AlphaZero-style
  work). **No paper puts the four main AI paradigms side-by-side on one board and
  analyses which wins and why.** That gap is this project.
- The four paradigms (weakest → strongest idea): **Minimax search → Imitation
  Learning → Self-Play RL (PPO) → MCTS + neural network (AlphaZero recipe)**,
  with **Random** as a floor and **Pikafish** (a superhuman engine) as the ceiling.

---

## 1. Project Schedule

The project runs in two halves: **(A) the build-and-experiment phase, already
finished**, and **(B) the remaining thesis work up to the defense (end of May
2027).**

### 1.A  Completed — development milestones (from the git history / work log)

| Date | What was built |
|---|---|
| **19 Sep** | Scaffolding + the **environment layer**: board, pieces, rules, legal-move generation, the 2,086 action space, the 14×10×9 encoder, and a Gymnasium environment. |
| **20 Sep** | The "big-bang" day: the **Policy-Value network**, **Minimax** (Agent 4), **PPO self-play** (Agent 1), the **Elo / tournament** harness, the **Imitation-Learning** dataset + **NeuralAgent** (Agent 2), **MCTS + network** (Agent 3), and the first Streamlit dashboard. |
| **21–22 Sep** | **UCCI/UCI engine interface** (ElephantEye / Pikafish), experiment logging, the **end-to-end data pipeline**, scaling to 100k+ games, the lazy IL dataset, and the **IL→PPO bridge**. |
| **1–4 Oct** | **Survival features** (intra-epoch + periodic checkpointing, `--resume`), PPO logging, and the AlphaZero-style **MCTS self-play trainer**. |
| **24–26 Oct** | **Pikafish distillation** (engine as teacher) and a **configurable network size** (`--channels` / `--num-blocks`) for bigger-net experiments. |
| **Nov** | Capstone proposal + this progress review (the `research/` work). |

### 1.B  Completed — training runs (what was actually trained)

| Run | Command (short) | Outcome |
|---|---|---|
| **Dataset build** | pipeline over `CGLemon/chinese-chess-PGN` | 141,356 games → **11.6M positions**, split 113k/14k/14k. |
| **Agent 2 — Imitation Learning** | `il_train --epochs 30` | Supervised on **9.3M** positions, ~17 min/epoch. **`il_agent2_phase1.pt`.** |
| **Agent 2 P2 — IL→PPO** | `ppo_train --timesteps 500000 --il-checkpoint …` | 501,760 steps (~7.5 h), reward 0.018 → 0.513. |
| **Agent 1 — PPO from scratch** | `ppo_train --timesteps 500000` | 500k steps, reward 0.086 → 0.381. |
| **Agent 3 — MCTS self-play** | `mcts_train --iterations 20 --games-per-iter 25 --simulations 100` | strongest agent; v2 stabilised with a **persistent replay buffer** → `mcts_agent3_v2.best.pt`. |
| **Pikafish distillation** | 5,000 engine self-play games (depth 8) → `il_train --mirror` | `distill_pikafish.pt` (did **not** beat human IL). |
| **Bigger net** | `--channels 128 --num-blocks 15` (≈4.85M params) | **no real strength gain**. |

### 1.C  Remaining schedule — now → defense (end of May 2027)

| Window | Goal | Concrete deliverable |
|---|---|---|
| **Nov–Dec 2026** | Lock the experimental protocol; finalise the fair-comparison setup (same net, same budget) and the Elo methodology. | Finalised methodology chapter + reproducible scripts. |
| **Jan–Feb 2027** | **Scale-up experiments** (the heart of the thesis): larger networks, more self-play steps, more MCTS simulations — to confirm or break the ~1690-Elo ceiling with real numbers. | New Elo tables + ablation curves. |
| **Feb–Mar 2027** | **Absolute-strength calibration** against Pikafish at several depths; report each agent on a shared, human-interpretable Elo scale. | Calibrated Elo (anchored to the engine). |
| **Mar–Apr 2027** | Write the full thesis (literature, method, results, analysis, discussion). Polish the live demo. | Complete thesis draft. |
| **Apr–May 2027** | Revisions with the advisor; rehearse the defense; prepare slides + the live demo. | Final thesis + defense. |
| **End of May 2027** | **Thesis defense.** | — |

> **Talking point:** "The engineering and the first full round of experiments are
> done and reproducible. The remaining work is **scaling and calibration** — turning
> the comparison into a definitive, interpretable result."

---

## 2. Functions Already Implemented (the demonstration)

Structure the demo as **four layers**: the engine, the four AIs, the evaluation,
and the live app.

### 2.1  The game engine (the foundation — everything else depends on it)

- Full Xiangqi **rules + legal-move generation** (river, palace, blockable horse,
  cannon screen-capture, flying-general rule, stalemate = loss).
- A fixed **action space of 2,086 moves** (every geometrically possible move).
- A **14×10×9 tensor encoding** of any position (piece planes + side-to-move),
  the input to every neural agent.
- A **Gymnasium environment** so the RL agents can train against it.

### 2.2  The four paradigms — *how each one works* (this is the core of the talk)

| # | Agent | How it works | Real-world lineage |
|---|---|---|---|
| **Random** | floor | plays a uniformly random legal move | the baseline everything must beat |
| **1. Minimax (Agent 4)** | **search, no learning** | alpha-beta negamax over a hand-written **material evaluation**; looks ahead, assumes best play by both sides, prunes hopeless branches | **Deep Blue** (beat Kasparov, 1997) |
| **2. Imitation Learning (Agent 2)** | **supervised, copies humans** | encodes 9.3M human positions and trains the network to **predict the human's move** (cross-entropy on the policy + MSE on the value) | **AlphaGo's first policy net**; behaviour cloning in self-driving |
| **3. Self-Play RL (Agents 1 & 2-P2)** | **trial and error, from scratch** | **MaskablePPO** plays against itself, gets **+1 win / −1 loss**, and nudges its policy toward winning moves — no human data needed | **OpenAI Five**, **AlphaStar**, RLHF behind ChatGPT |
| **4. MCTS + network (Agent 3)** | **intuition + calculation** | the AlphaZero recipe: the network gives "intuition" (move priors + a value), then **Monte-Carlo Tree Search** (select → expand → evaluate → back-up, guided by **PUCT**) does the "calculation", and the most-visited move is played | **AlphaGo / AlphaZero / MuZero** |
| **(benchmark) Pikafish** | superhuman engine | a top open-source Xiangqi engine (UCI, NNUE) — the **ceiling** and the **teacher** for distillation | **Stockfish**'s Xiangqi cousin |

**Shared "brain".** IL, PPO, and MCTS all use the **same** Policy-Value network
(AlphaZero-style): input 14×10×9 → `C=64` channels × `N=10` residual blocks
(**≈1.15M parameters**) → two heads (policy = 2,086 logits, value = a scalar in
[−1,1]). Using one network for all three keeps the comparison fair — the only
thing that changes is the *algorithm around it*.

### 2.3  Training + evaluation pipeline (the "it's reproducible" story)

- A full **data pipeline**: raw PGN → cleaned, legal, encoded splits (train/val/test).
  *(The three data sources and the full end-to-end process are detailed in §4.)*
- Three **trainers**: supervised IL, PPO (self-play RL), and AlphaZero-style MCTS.
- **Survival features** so long runs survive laptop/Colab interruptions
  (checkpoint + `--resume`, persistent MCTS replay buffer).
- An **Elo tournament harness** (round-robin, anchored) + a **test-set
  move-prediction accuracy** report.
- A **Pikafish engine interface** for an absolute-strength benchmark and for
  distillation.

### 2.4  The live demo — *what to actually show on screen*

Two apps, both runnable from one laptop:

- **The web demo** (`webapp/`, chessgpt-style, port 8502): a **pixel-art landing
  → full-screen loading → the board**. Pick any two models, press **Play**, and
  watch them play out, with:
  - a **wooden board** with carved ebony pieces,
  - a **live advantage graph** (red area = Red ahead, blue = Black ahead) that
    shows the lead rising and falling move-by-move — *this is the "who's winning"
    visual*,
  - **sound effects** on every move and a heavier sound on captures,
  - adjustable **play speed**.
- **The Streamlit dashboard** (port 8501): the analytical view with the advantage
  bar, agent picker, and AI-vs-AI showcase.

> **Demo script (rehearse this):** open the web app → press **"Oh yeah? Game on"**
> → watch the pixel loading → on the board, run **Pikafish MAX (Red) vs Minimax d1
> (Black)** → narrate the advantage graph as Red pulls ahead → "Red wins —
> checkmate." A full game generates in **≈1–3 seconds**.

To launch the web demo:
```bash
C:\Python314\python.exe webapp/server.py      # then open http://localhost:8502
```

### 2.5  Results already obtained (have these numbers ready)

**Relative Elo (anchor 1500, round-robin):**

| Rank | Agent | Elo |
|---|---|---|
| 1 | **MCTS + network (v2)** | **≈1691–1703** |
| 2 | Imitation Learning (human) | ≈1557–1597 |
| 3 | Minimax | ≈1460–1541 |
| 4 | IL→PPO | ≈1411–1452 |
| 5 | Random | ≈1287–1300 |
| — | **Pikafish (depth 8), for scale** | **≈2062** |

**Imitation-Learning move-prediction accuracy (test set):** **≈42% top-1 /
≈66% top-3** — and it degrades from the **opening (~63%)** to the **endgame
(~32%)**, because the opening is more predictable than deep tactical endgames.

> **Headline result to say out loud:** *"MCTS over the network is the strongest of
> my four agents. Surprisingly, imitation learning **alone** beat every PPO
> self-play variant — self-play from scratch on one laptop was not enough. And an
> engine teacher (distillation) did **not** beat learning from humans at this
> network size."*

---

## 3. Problems, Difficulties, and Potential Outcomes

### 3.1  Engineering problems encountered — and how they were solved

| Problem | Symptom | Fix |
|---|---|---|
| **Training interruptions** (laptop/Colab) | runs died mid-epoch with no checkpoint | intra-epoch + per-epoch checkpointing, `--resume`, detached background process |
| **MCTS regression on resume** | fragmented training **dropped 1703 → 1582 Elo** | **persist the replay buffer** + keep the best-by-value-loss checkpoint → recovered to ~1691 |
| **SB3 / PyTorch 2.14 load bug** | `MaskablePPO.load` failed to read the zip | a robust loader (rebuild model, load the policy state-dict) |
| **GPU setup clash** | `torchvision` clobbered the CUDA PyTorch | uninstall `torchvision`, reinstall the exact CUDA wheel |
| **Laptop overheating** | GPU throttled on long training | cooled bursts (~4 epochs), smaller batch, CPU cap → 60–67 °C |
| **Broken data scraper** | Playwright assets missing | switched to a ready-made dataset (`CGLemon/chinese-chess-PGN`) |

### 3.2  The central scientific difficulty (the most important slide)

**Four independent attempts to make the agents stronger all hit the same
~1690-Elo ceiling:**

1. more MCTS **search** (100 → 800 simulations) — no clear gain,
2. more training **epochs** — plateaued,
3. **engine distillation** (a stronger teacher) — did not beat human IL,
4. a **4× bigger network** (1.15M → 4.85M params) — no real gain.

> **The conclusion:** on this scale, the binding constraint is **not the
> algorithm** — it is **model capacity + data/compute scale**. All four levers
> converge because the laptop budget (one 8 GB GPU) caps how much signal the
> network can absorb.

**Honest limitations to acknowledge:** single-laptop compute, self-play RL
under-trained from scratch, distillation weaker than expected at this net size,
and a gap of ~360 Elo to the Pikafish benchmark.

### 3.3  Potential outcomes *if these difficulties are resolved*

Frame each as **"if we remove the compute/data bottleneck, then…"**:

| If resolved (more compute / data / bigger net, properly trained) | Potential outcome |
|---|---|
| Train a **larger network** (e.g. C256/N19 ≈ 25–30M) with **more self-play steps** | break the ~1690 ceiling and climb toward the **Pikafish (~2062) / superhuman** range |
| Give **self-play RL** a realistic budget (millions more games) | test whether PPO/MCTS can finally **surpass pure imitation**, as the theory predicts |
| Run the **comparison at scale** on a multi-GPU/cloud setup | turn the current laptop-scale ranking into a **definitive, generalisable** result about which paradigm wins |
| Use the calibrated agents | a **tunable-strength Xiangqi opponent** (beginner → master) and a **teaching/analysis tool** — a reusable research platform, not just a one-off experiment |

> **The payoff sentence:** *"My contribution is the first **fair, reproducible,
> head-to-head comparison** of the four AI paradigms on Xiangqi. Even at laptop
> scale it already shows a clear ranking and a clear bottleneck. With more
> compute, the same framework directly answers how far each paradigm can go — and
> which one is worth scaling toward superhuman play."*

---

## 4. Deep-dive: the data (three sources, full process)

The project uses **three kinds of data**. One is downloaded (human games); two are
**generated by the project itself** (engine self-play and MCTS self-play). All
three end up in the **same JSONL format** — `{"o": outcome, "m": [[fr,fc,tr,tc],…]}`
— so the same training code can read any of them.

### 4.1  Data source #1 — Human games (CGLemon/chinese-chess-PGN)

This is the big supervised dataset, and it is itself a **bundle of two archives**:

| Archive | Games | Origin |
|---|---|---|
| **WXF** (World Xiangqi Federation) | **41,743** | federation game records |
| **DPXQ** (东萍象棋网) | **99,813** | **dpxq.com**, a major Xiangqi website |
| **Total** | **141,556 downloaded** | → 141,356 kept |

**How it was obtained (this is the "web" part you remembered):**
1. I **first tried to scrape** games from the web with a Playwright-based scraper —
   but it **broke** (the browser-driver assets were missing), so I abandoned it.
2. Instead I used the **ready-made CGLemon collection**, whose files are hosted on
   **Google Drive**. I downloaded the two folders with `gdown --folder <id>` into
   `data/raw/` → `WXF-41743games.pgns` and `dpxq-99813games.pgns`.
   *So the games originate from the web (dpxq.com etc.), but I got them via a
   packaged dataset, not by scraping myself.*

**How the raw games become training data** (`src/data/pipeline.py`, steps 1–6):
1. **Download** → `data/raw/`.
2. **Parse** the movetext (auto-detects **WXF** vs **ICCS** coordinate notation).
3. **Replay-validate** every game move-by-move through my real rules engine —
   any illegal/corrupt game is discarded. (**0 invalid** here — the corpus is clean.)
4. **Length filter**: keep 10–300 plies → dropped **26 too-short**, **174 too-long**.
5. **Encode** each position lazily into a **14×10×9 tensor** + a **policy label**
   (the move's index in the 2,086 space) + a **value label** (the game's result).
6. **Game-level split** 80/10/10 (seed 42) so no game leaks across splits.

**Final numbers (for the thesis methodology chapter):** 141,356 games →
**11,624,921 positions** (train 9.30M / val 1.17M / test 1.16M), average **82.24
plies/game**, outcomes **Red 53,764 / Black 39,860 / Draw 47,732**, spot-check
**500/500 legal**. All recorded in `data/splits/dataset_manifest.json`.

### 4.2  Data source #2 — Pikafish engine self-play (for distillation)

Generated, not downloaded (`src/data/pikafish_selfplay.py`):
- Play **Pikafish vs Pikafish**. The **first 6 plies are random legal moves** so the
  games don't all start identically (a deterministic engine would repeat one game);
  after that, Pikafish picks moves at **depth 8**.
- A game ends on no-legal-moves, **repetition**, or the ply cap → outcome Red/Black/Draw.
- Written **incrementally** to JSONL (crash-safe + resumable), in the **same format**
  as the human corpus, so the IL trainer reads it unchanged — only the *teacher*
  differs (a superhuman engine instead of humans).
- **Scale:** **5,000 games** (~2.2 h) → **~815k positions**, **~59% draws** → split
  into `data/pikafish_splits/`. Used to train `distill_pikafish.pt` and the bigger
  `distill_big.pt`.

### 4.3  Data source #3 — MCTS self-play replay buffer (AlphaZero training)

Also generated by the project, during Agent 3 training (`src/training/mcts_train.py`):
- The agent **plays against itself** using MCTS. For every move it stores a training
  example: the **position tensor**, the **MCTS visit-count distribution** (the policy
  target), and the **final game result** (the value target).
- Self-play uses **Dirichlet root noise** + **temperature sampling for the first 30
  plies** for exploration.
- These examples accumulate in a **persistent replay buffer**
  (`mcts_agent3_v2.pt.buffer.pkl`, ~269 MB) that the network trains on — **20
  iterations × 25 games/iter**. Persisting this buffer is exactly what fixed the
  earlier regression (losing it on resume caused the 1703 → 1582 Elo drop).

> *(Note: PPO self-play also generates experience on the fly during training, but it
> is consumed immediately and not saved as a named dataset.)*

### 4.4  Derived set — opening reference distribution

A small **derived** dataset (`src/evaluation/opening_analysis.py` → `data/openings/`):
the frequency of standard openings (e.g. 当头炮 Central Cannon, 屏风马 Screen Horse)
in the human corpus, used to **check whether the agents open like real players**.

> **One-line answer if asked "where's your data from?":** *"Human games from the
> CGLemon collection — two archives, WXF (41,743) and DPXQ from dpxq.com (99,813) —
> ~141k games, 11.6M positions. Plus two datasets I generated myself: 5,000 Pikafish
> self-play games for distillation, and the MCTS self-play replay buffer for the
> AlphaZero agent. I tried scraping first but switched to the packaged dataset."*

---

## 5. Components checklist (so nothing is missed)

Everything the project contains — skim this to be sure no examiner question catches
you off guard:

| Component | One-line summary |
|---|---|
| **Rules engine** | Full Xiangqi legality (river, palace, blockable horse, cannon screen-capture, flying-general, repetition, stalemate = loss). |
| **Action space** | A fixed **2,086-move** index; a `legal_mask` sets illegal logits to −∞ before softmax. |
| **Encoder (14×10×9)** | 14 binary planes = **7 of my piece types + 7 of the opponent's**, always from the **side-to-move's perspective**. |
| **Policy-Value network** | Shared AlphaZero-style net, C64×N10 (**~1.15M params**), two heads (policy 2,086 + value scalar). |
| **4 agents + 2 refs** | Random, Minimax, IL, Self-play PPO, MCTS, + Pikafish (benchmark/teacher). |
| **Three data sources** | Human (WXF+DPXQ), Pikafish self-play, MCTS self-play buffer (see §4). |
| **Trainers** | IL (supervised), PPO (MaskablePPO), MCTS (AlphaZero self-play) — all with checkpoint/resume. |
| **The 6 experiments** | ① 7-agent round-robin, ② MCTS headline, ③ MCTS v2 (stability fix), ④ Pikafish distillation, ⑤ bigger-net, ⑥ MCTS-simulation ablation. |
| **Elo method** | Round-robin, **50 games/pair**, relative Elo anchored at **1500**, 95% CI; plus a test-set move-accuracy report. |
| **Opening analysis** | Compares agent openings against the human opening distribution. |
| **Two apps** | Streamlit **dashboard** (analysis) + pixel **web demo** (watch AI-vs-AI with a live advantage graph). |
| **Colab workflow** | Notebooks that clone the repo, `gdown` the data, build splits, and train on a Colab GPU. |
| **Test suite** | **224 automated tests** across environment, data, models, agents, training, evaluation — the reproducibility backbone. |

---

## 6. Anticipated Q&A (prepare answers)

- **"Your model has only ~1.15M parameters. Isn't that tiny vs an 8B LLM?"**
  Yes, and that's by design. This is a **single-game policy-value net**, not a
  general language model. Even DeepMind's AlphaZero used a network of tens of
  millions of parameters — far below an LLM. Strength in board games comes from
  **search + network**, not raw parameter count — which is exactly why my MCTS
  agent (same small net) is my strongest.

- **"Can you make the model bigger?"**
  Yes — the size is configurable (`--channels` / `--num-blocks`). I already tested
  4× bigger (1.15M → 4.85M) and got **no real gain**, which is itself a finding:
  the bottleneck is **data/compute scale**, not just parameter count. Scaling
  needs more training signal, not only more weights.

- **"Why is MCTS the strongest?"**
  It combines the network's **intuition** (priors + value) with explicit
  **look-ahead search**, so it corrects the network's mistakes at move time — the
  AlphaZero insight.

- **"Why did the engine teacher (distillation) lose to human imitation?"**
  At this small network size, the net can't absorb the engine's deeper knowledge,
  and the engine's self-play games are less varied (many draws). Human games gave
  a richer, more learnable signal.

- **"Is the comparison fair?"**
  Yes — all learning agents share the **same network, the same action space, the
  same evaluation harness**, and are measured by the **same Elo tournament**. The
  only variable is the algorithm.

- **"Where did your data come from — did you scrape it?"**
  Three sources (see §4). The human games are the **CGLemon collection** — two
  archives, **WXF (41,743)** and **DPXQ from dpxq.com (99,813)** — ~141k games /
  11.6M positions, downloaded from Google Drive via `gdown`. I **did try scraping
  first**, but the scraper broke, so I used the packaged dataset. The other two
  datasets I **generated myself**: 5,000 Pikafish self-play games (distillation)
  and the MCTS self-play replay buffer (AlphaZero training).

---

*Cross-references in this repo: `research/08_progress_review.md`,
`research/09_slide_plan.md`, `research/10_project_brief.md`,
`docs/13-WORKLOG.md` (authoritative build/training facts), and the live apps in
`webapp/` and `dashboard/`.*
