# 04 — 3-Minute Elevator Pitch

> For the Lab 8 classroom presentation. Target: **~3 minutes** spoken at a
> natural pace (≈430–470 words). Do **not** read the paper aloud — tell the
> story. Timing cues are in brackets.

---

## Opening statement (memorable hook)
> "Everyone 'knows' the AlphaGo recipe: imitate humans, then let reinforcement
> learning make the agent superhuman. I spent a thesis testing whether that
> recipe actually pays off for Chinese Chess on a single laptop GPU — and the
> answer surprised me."

---

## Spoken script

**[0:00–0:30 — Motivation]**
Xiangqi, Chinese Chess, is played by hundreds of millions of people, yet it is
far less studied than Go or chess. The famous game-AI breakthroughs — AlphaGo,
AlphaZero — were built from three ideas: imitating human experts, reinforcement
learning through self-play, and tree search over a neural network. But those
results came from massive compute clusters. If you are a student or a researcher
with one consumer GPU, nobody has told you which of these ideas is actually
worth your time for Xiangqi.

**[0:30–1:00 — Problem and gap]**
I looked at the literature. The handful of Xiangqi deep-RL papers each build one
system and report where it succeeds. What is missing is a *controlled
comparison*: putting these paradigms side by side, under the *same* compute
budget, on the *same* Elo scale, so the only thing that differs is the learning
method. And two practical questions have never been answered for Xiangqi at this
scale: does imitation pre-training really help the reinforcement learning that
follows, and what ultimately limits how strong the agent can get?

**[1:00–1:45 — Proposed approach]**
So I built a controlled testbed. One shared Xiangqi engine, one fixed action
space, one neural network, one Elo rating system — and then I swap in four
different "brains": a classical alpha-beta search baseline, pure PPO self-play
from scratch, imitation-then-PPO, and an AlphaZero-style agent that runs
Monte-Carlo Tree Search over the network. Because everything except the decision
function is held constant, any difference in strength is caused by the paradigm
itself. I think of it as a *multi-agent league* where heterogeneous agents fight
on one ladder.

**[1:45–2:30 — Methodology and evaluation]**
I train each agent to a matched budget on a single RTX-4060 laptop, then run
colour-balanced round-robin tournaments and fit Elo ratings with 95% confidence
intervals, anchored absolutely against the Pikafish engine. Then I run a
four-lever ablation — search simulations, training epochs, teacher, and network
size — to find out which one actually moves the needle. Everything is seeded and
logged so it reproduces.

**[2:30–3:00 — Expected contribution]**
The preliminary results are already striking: the AlphaZero-style agent is
strongest — but imitation learning *alone* beats every PPO variant, and under
this budget, reinforcement fine-tuning actually *degraded* the imitation policy.
Four different ways of adding power all hit the same ceiling, which points at
capacity, not algorithm, as the bottleneck. The contribution is the first
controlled, honest, reproducible comparison of these paradigms for Xiangqi under
real-world compute — including the negative results nobody else reports.

---

## Closing statement (memorable)
> "The headline isn't 'I built the strongest Xiangqi engine.' It's: *under a real
> budget, search and imitation beat reinforcement fine-tuning — and knowing that
> saves everyone else a lot of wasted GPU hours.*"

---

## Slide-by-slide outline (6 slides, ~30s each)
| # | Slide title | Visual | Say |
|---|---|---|---|
| 1 | **Title + hook** | Xiangqi board photo; title; your name/affiliation | The opening hook |
| 2 | **The gap** | 1 line each: AlphaGo recipe ✓ cluster-scale, ✗ Xiangqi, ✗ one-GPU, ✗ controlled comparison | Motivation + gap |
| 3 | **The testbed** | The architecture figure from the paper (shared engine/net; 4 agents; Elo league) | Proposed approach |
| 4 | **Method** | Round-robin → fitted Elo + CI; Pikafish anchor; 4-lever ablation | How it's measured |
| 5 | **Preliminary Elo** | The Elo bar chart (`results/figures/exp1_agent3_elo.png`) with CI bars | The surprising results |
| 6 | **Contribution** | 3 bullets: framework · first controlled Xiangqi comparison · reproducibility/stability finding | Closing hook |

---

## Technical points to emphasise (and to be ready to defend)
1. **Controlled design** — one env/action-space/network/Elo; only the decision
   function varies. This is what makes it a *fair* comparison, not a bake-off.
2. **Elo with 95% CIs + Pikafish absolute anchor** — not single games;
   non-transitivity handled by round-robins.
3. **The surprise** — IL alone > every PPO variant; RL fine-tuning can erode IL
   under constrained compute. State it as a *measured* result, with CIs.
4. **The bottleneck** — four independent levers (search, epochs, teacher, net
   size) all plateau → capacity/data, not algorithm, binds strength.
5. **Honesty** — wide CIs at low game counts are acknowledged; the plan tightens
   them. Scope is one game, one GPU class — no over-generalisation.

## Delivery tips
- Practise to land the 1:00 and 2:30 marks; the opening and closing hooks are
  worth memorising word-for-word.
- Keep slide 5 (the result) on screen longest — it earns the questions you want.
- If you run short on time, cut the method detail (slide 4), never the gap
  (slide 2) or the result (slide 5).
