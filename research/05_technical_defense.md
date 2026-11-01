# 05 — Technical Defense Preparation (2-Minute Q&A)

> Preparation for defending the proposal against student-reviewer questions.
> For each question: **(T)** a technically accurate answer, **(S)** a concise
> spoken answer for the room, and **(F)** a deeper follow-up if pressed. All
> answers are grounded in the actual project; no results are invented.

---

## Category A — Research Problem

**A1. Why did you choose this research problem?**
- **(T)** The AlphaGo/AlphaZero paradigms were validated at cluster scale and
  mostly on Go/chess; their trade-offs under a single-GPU budget, and for
  Xiangqi, are unmeasured. I already had a complete, tested four-paradigm
  Xiangqi testbed, so a *controlled* comparison was both novel and feasible.
- **(S)** Because everyone cites the AlphaGo recipe, but nobody has measured
  whether it pays off for Xiangqi on a normal GPU — and I had the exact testbed
  to measure it.
- **(F)** Feasibility mattered: the environment, network, four agents, and Elo
  harness are implemented and unit-tested, and preliminary runs exist, so this
  is formalising a comparison, not starting from zero.

**A2. Why is this problem important?**
- **(T)** Most researchers and students work under constrained compute; choosing
  the wrong paradigm wastes scarce GPU time. Single-run, cluster-scale claims may
  not transfer, and deep-RL results are known to be hard to reproduce.
- **(S)** It saves people from spending GPU-weeks on a paradigm that doesn't
  actually help at their budget.
- **(F)** Henderson et al. showed deep-RL results swing with seeds and
  implementation — so budget-aware, CI-backed guidance has real value.

**A3. What makes the problem worth investigating (why not obvious)?**
- **(T)** The "obvious" answer — RL improves on imitation, more search/bigger
  nets help — is contradicted by my preliminary data, where RL fine-tuning
  *lowered* strength and four power levers plateaued. A counter-intuitive,
  measurable effect is exactly what deserves study.
- **(S)** Because the intuitive answer turned out to be wrong in my pilot runs —
  that's the interesting part.
- **(F)** If the result had matched intuition, it would be a confirmation; the
  surprise is what makes it a contribution.

---

## Category B — Research Gap and Novelty

**B1. What is the actual research gap?**
- **(T)** No controlled, equal-compute, Elo-anchored comparison of classical
  search, pure PPO, imitation+PPO, and AlphaZero-style MCTS exists *for Xiangqi
  under a single-GPU budget*; consequently the direction of the imitation→RL
  effect and the binding constraint on strength are unquantified at this scale.
- **(S)** There's no like-for-like, same-budget comparison of these four methods
  for Xiangqi — existing Xiangqi papers each build just one system.
- **(F)** It's primarily an evaluation/methodological gap, with secondary
  performance and application dimensions (Xiangqi is under-served vs Go/chess).

**B2. How is your research different from the existing Xiangqi papers
(Yilmaz 2025, JiangJun 2023, Xiangqi-R1 2025)?**
- **(T)** Each of those optimises one pipeline (AlphaZero-style, PSRO-population,
  or an LLM) and reports its strengths. Mine holds the environment, action space,
  network, and Elo scale fixed and varies only the paradigm, enabling a
  *controlled* comparison and an ablation of what binds strength.
- **(S)** They each build one strong system; I compare four methods under
  identical conditions to see which is worth it and why.
- **(F)** I'm not competing on raw strength; I'm contributing measurement — the
  controlled comparison and the negative results they don't report.

**B3. What is the novelty of your approach?**
- **(T)** The novelty is the *controlled measurement under constraint*, not a new
  algorithm: a single-interface multi-agent league that isolates the paradigm as
  the only variable and quantifies the imitation→RL effect and the capacity
  bottleneck for Xiangqi.
- **(S)** The novelty is the controlled, honest comparison and its findings — not
  a brand-new model.
- **(F)** I deliberately avoid claiming algorithmic novelty I can't defend;
  methodological novelty and new empirical knowledge are defensible.

**B4. Why is the contribution meaningful if you didn't beat the state of the art?**
- **(T)** Decision-relevant, reproducible evidence has value independent of SOTA:
  it tells practitioners which paradigm to pick under a budget and warns that RL
  fine-tuning can backfire — a result the strength-chasing literature omits.
- **(S)** Knowing what *doesn't* work under a budget is worth as much as a new
  record, and it's reproducible.

---

## Category C — Technical Implementation

**C1. How does the proposed system work, end to end?**
- **(T)** A Gymnasium Xiangqi env exposes a 14×10×9 side-to-move encoding and a
  fixed 2086-move action space with legal masking. One policy-value network
  (conv → N residual blocks → 2086-logit policy head + scalar value head) is
  shared by imitation, PPO (actor+critic), and MCTS (priors+leaf value). Four
  agents plug into one `select_move(board, mask)` interface; a tournament runs
  round-robins and `fit_ratings` computes Elo.
- **(S)** One shared board engine and network; four interchangeable decision
  modules; one tournament that rates them all on the same Elo scale.
- **(F)** The uniform interface is what makes it fair — the tournament can't tell
  a 20-line Minimax from a GPU net apart except by the moves they return.

**C2. Why did you select MCTS/PUCT for the strongest agent?**
- **(T)** PUCT-guided MCTS with a value-head leaf evaluation is the AlphaZero
  design: it trades compute for depth, using the network's priors to focus search
  and its value head to avoid random rollouts. My pilot has it strongest
  (Elo ~1703).
- **(S)** It's the AlphaZero approach — search over the network — and in my pilot
  it's the strongest agent.
- **(F)** The move is chosen by visit count, which is more robust than raw value;
  Dirichlet root noise adds exploration during training self-play.

**C3. Why PPO rather than another RL algorithm?**
- **(T)** PPO is stable, simple to tune, and handles large discrete action spaces
  via MaskablePPO's native action masking, so no probability mass is wasted on
  illegal moves. It's also the de-facto baseline, which keeps the comparison
  interpretable.
- **(S)** PPO with action masking is stable and standard, so it's a fair,
  interpretable RL baseline.
- **(F)** Masking is essential here — vanilla PPO would waste most of its
  distribution on the ~2000 illegal actions each turn.

**C4. What are the alternatives you didn't use, and why?**
- **(T)** Alternatives include AlphaZero-only (no classical/PPO contrast),
  PSRO/population methods (JiangJun — more compute, targets non-transitivity), and
  LLM-based reasoning (Xiangqi-R1 — different problem). I include the classical
  and PPO contrasts precisely because the comparison, not one method, is the
  point.
- **(S)** I kept the classical and PPO baselines on purpose — dropping them would
  defeat the comparison.

**C5. How do you ensure correctness of the environment and agents?**
- **(T)** 48 test files unit-test rules, move generation (the opening yields
  exactly 44 legal moves), the action-space round-trip, the encoder, Elo math,
  and tournaments; a working Minimax is itself an environment-correctness check;
  every agent's `select_move` is constrained to legal moves by masking.
- **(S)** Around 175 unit tests, plus Minimax as a sanity check that the rules
  are right, and legal-move masking everywhere.
- **(F)** The "44 legal opening moves" invariant is my canonical regression
  anchor.

**C6. What are the technical limitations of the implementation?**
- **(T)** The 1.15M-param net plateaus near 34% imitation top-1; evaluation uses
  20–50 games/pair (CIs ±40–96); most Elo is relative (one run adds a Pikafish
  absolute anchor); some eval scripts live outside the committed experiment
  modules.
- **(S)** Small net, wide confidence intervals at low game counts, and mostly
  relative Elo — all things the full study tightens.

---

## Category D — Research Methodology

**D1. How will you conduct the experiments?**
- **(T)** Train each paradigm to a matched single-GPU budget; run colour-balanced
  round-robins; fit Elo with 95% CIs and build a head-to-head matrix; then vary
  one lever at a time (sims/epochs/teacher/net-size) re-entered into the
  round-robin; plus a stability study on buffer continuity.
- **(S)** Matched-budget training, round-robin tournaments, fitted Elo with
  intervals, then one-variable-at-a-time ablations.

**D2. What is your baseline?**
- **(T)** Two: a classical alpha-beta Minimax (the pre-deep-learning tradition)
  and a Random agent (the floor); PPO-from-scratch is the RL baseline for the
  imitation→RL comparison. Pikafish provides an absolute Elo anchor.
- **(S)** Minimax and Random as floors, PPO-from-scratch as the RL baseline, and
  Pikafish as the absolute yardstick.

**D3. How will you collect the data (and training corpus)?**
- **(T)** 141,356 legal human games (~11.6M positions) from public PGN
  collections, game-level split 113k/14k/14k, recorded in a committed manifest;
  an engine self-play corpus for the distillation lever; every run logs its
  config, seed, and git commit.
- **(S)** 141k human games split reproducibly, plus engine self-play games for
  one ablation, all logged with seeds.

**D4. Why these evaluation metrics?**
- **(T)** Elo (relative + absolute) is the standard skill scale and is directly
  comparable across agents; CIs quantify uncertainty; head-to-head exposes
  non-transitivity; the signed IL-vs-IL+PPO delta directly answers RQ2; per-lever
  Elo answers RQ3; top-1 accuracy is a learning proxy.
- **(S)** Elo answers "how strong," CIs answer "how sure," and the deltas answer
  the specific research questions.

**D5. How will you ensure a fair comparison?**
- **(T)** Identical environment, action space, network architecture, and Elo
  protocol; matched compute budget per RL agent; balanced colours; fixed seeds.
  The paradigm is the only deliberately varied factor.
- **(S)** Everything is held constant except the decision method, and compute is
  budget-matched.
- **(F)** That's the whole design philosophy — any strength difference is
  attributable to the paradigm, not to incidental setup differences.

---

## Category E — Expected Results

**E1. How will you decide whether an approach is "successful"?**
- **(T)** By its Elo rank and whether differences are statistically separated
  (non-overlapping 95% CIs); for RQ2, by the sign and magnitude of the IL→RL Elo
  delta; for RQ3, by which lever produces the largest Elo change.
- **(S)** Non-overlapping confidence intervals on Elo, and the sign of the
  imitation-versus-RL difference.

**E2. What if the proposed approach doesn't outperform the baseline?**
- **(T)** That *is* a result here — the study measures comparative strength, not
  a target. My pilot already shows RL fine-tuning underperforming imitation; a
  "negative" outcome answers the research question just as validly.
- **(S)** A negative result still answers the question — this study is about
  measurement, not beating a number.

**E3. What are the expected limitations of the results?**
- **(T)** Wide CIs at modest game counts; relative Elo except the Pikafish-anchored
  run; single game and single GPU class, so external validity is bounded.
- **(S)** Mostly: limited statistical power and single-setting scope — both
  stated honestly.

**E4. Can the approach generalise to other board games?**
- **(T)** The *methodology* (shared stack + Elo league + lever ablation)
  transfers directly to any two-player perfect-information game; the *numerical
  findings* are Xiangqi-specific and should be re-measured elsewhere.
- **(S)** The method generalises; the specific numbers don't, and I don't claim
  they do.

**E5. Why should the audience care about a capacity bottleneck finding?**
- **(T)** It redirects effort: if capacity/data binds strength, spending on deeper
  search or more RL epochs is wasted, and the productive lever is scale — useful,
  actionable guidance under a budget.
- **(S)** It tells you where *not* to spend your next GPU-week.

---

## Five challenging reviewer questions (rehearse these)
1. **"Your 'preliminary results' use 20–50 games per pair — aren't the
   confidence intervals so wide that your headline claims aren't significant?"**
   → Concede partially: only IL's lead and Random's trail are clearly separated;
   mid-table ties are within noise. The proposed study's first task is to add
   games until the decisive intervals separate. The *direction* of the IL→RL
   effect is consistent across multiple independent runs, which is the robust
   part.
2. **"If this infrastructure and these results already exist, what's left to
   research — isn't it done?"** → What exists is pilot-grade: relative Elo, wide
   CIs, mostly one absolute anchor, and some eval outside committed modules. The
   proposed work is the rigorous, powered, reproducible version plus the
   systematic four-lever ablation and stability study.
3. **"Relative Elo is only meaningful within your pool — how do I know your
   1703 means anything?"** → Correct; that's why one experiment anchors
   absolutely to Pikafish, and all claims are framed as within-pool ranks with
   CIs, not absolute strength.
4. **"Could the 'RL erodes IL' result just be a bug or bad hyperparameters rather
   than a real effect?"** → A fair risk; it's why the study holds the PPO config
   fixed across agents, reports the full training curves, and tests multiple
   opponent settings (random/self/minimax). If a reasonable config reverses it,
   that itself is reportable.
5. **"Why should a Software Engineering course accept an ML experiment as SE
   research?"** → The SE content is explicit: a reusable evaluation framework, a
   controlled empirical-SE methodology, and a reproducibility/training-stability
   finding (buffer continuity affects model quality) — empirical software
   engineering for ML systems.

---

## Five likely weaknesses reviewers may identify + responses
| Weakness | Response |
|---|---|
| Low statistical power (few games/pair) | Acknowledged; scaling games is objective O2's first task; report CIs always. |
| Mostly relative Elo | One experiment anchors to Pikafish; claims are scoped as within-pool ranks. |
| Single game / single GPU class | Scope stated; methodology (not numbers) is the transferable contribution. |
| Reproducibility seam (scratchpad eval scripts) | Migrate key eval into committed `experiments/` modules; manifest + seeds already committed. |
| "Not novel" (no new algorithm) | Novelty is methodological + empirical (controlled comparison, negative results), explicitly, not algorithmic. |

---

## 2-Minute defense strategy
1. **First 20s:** restate the one-sentence contribution — *a controlled,
   equal-budget, Elo-anchored comparison of four paradigms for Xiangqi,
   including negative results.* Anchors every answer.
2. **Concede fast, then pivot:** on statistics/relative-Elo questions, agree
   immediately, then state the specific mitigation (more games; Pikafish anchor).
   Conceding a true limitation builds credibility and costs nothing.
3. **Hold the line on scope:** when pushed on generality or SOTA, restate that
   the contribution is *measurement under constraint*, not a new record — don't
   over-claim.
4. **Use the surprise:** steer toward the IL>RL and capacity-bottleneck findings;
   they are your strongest, most memorable material.
5. **Never invent a number:** if asked for a value you don't have, say "that's
   exactly what the full round-robin will measure," and name the metric.
