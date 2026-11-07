# Five-Minute Elevator Pitch — Slide Content and Spoken Script

> **The device.** You act out an elevator ride. You step in, your boss is
> already there, he is going to floor 27, and you have until *your* floor to
> make it count. Your floor comes before his, which is why the ending is a
> rush — and why the rush is funny instead of awkward.
>
> **The voice.** Short, simple, spoken — the way you actually talk. Sentences
> are joined with *so*, *because*, *and*, *but here's the thing*, rather than
> chopped into neat lists. Lists like "one engine, one action space, one
> network" are the thing that makes a script sound written rather than said, so
> there are none here. Read it out loud; if a line feels stiff in your mouth,
> change it to whatever you would really say.
>
> **The script is deliberately light. The slides carry the detail.** If someone
> wants depth, it is on the slide behind you and in the backup slides.
>
> **Hold the energy, not the acting.** Full acting on Slides 1 and 7. In
> between, just talk, and keep "boss" as a thread you tug now and then.
>
> **Budget.** 719 spoken words -- about **5:08 at a careful 140 words a minute,
> 4:48 at a natural talking pace of 150.** Leaves a little slack for pauses, a
> laugh at the opening, and a slow ask at the end.
>
> **Source.** Every figure and table cited is from
> `research/proposal/L25020007_Proposal_v2.pdf`.

---

## Slide plan at a glance

| # | Slide | Time | Cum. | Function | Asset from Proposal v2 |
|---|---|---|---|---|---|
| 1 | Doors open — the hook | 0:58 | 0:58 | Set the scene, name Xiangqi, earn attention | — (board image, faint) |
| 2 | Floor 6 — why this costs money | 0:46 | 1:44 | Problem as a cost | **Table I** |
| 3 | Floor 11 — the question | 0:23 | 2:07 | One sharp question | — |
| 4 | Floor 15 — what you'd be backing | 0:23 | 2:30 | Scope *(cuttable)* | — (O1–O5 as text) |
| 5 | Floor 19 — why you can trust it | 0:41 | 3:11 | The unfair advantage | **Fig. 1** |
| 6 | Floor 23 — it already runs | 0:46 | 3:57 | Traction, de-risk | **Fig. 2** |
| 7 | Floor 25 — the ask, and out | 0:51 | 4:48 | Close and exit | **Table III** |

**Design.** Dark background, high contrast. Put a small floor counter in a
corner — **6 · 11 · 15 · 19 · 23 · 25** — ticking up as you go. Tiny touch, and
it is what makes the whole device land.

---

## Slide 1 — Doors open · 0:56

**On the slide**
- Almost empty. Large: **Going up?**
- Small, bottom corner: title, your name, student ID, supervisor
- Optional: the Xiangqi board, very faint, behind the text

**You say**

*(Stand to the side of the screen. Relaxed, talking to the room — not performing yet.)*

"Okay guys — before I start, I want you to imagine something with me. I'm about
to step into a lift.

*(Mime pressing the button. Wait a second.)*

*(Sound cue: ding — doors opening.)*

*(Look up. Freeze. Then turn to the audience, stage whisper:)*

"Wait, wait — that's my boss."

*(Turn back. Straighten up. Nervous smile.)*

"H-hi! Hello, boss! Uh — which floor are you going to? …Oh, twenty-seven. Okay,
so we've got a bit of time.

*(Relax. Lean in a little, like you are sharing something you just found.)*

Boss, so my thesis is actually about Xiangqi — Chinese chess — and the AI that
plays it. And while I was reading around it, I found this: back in 1997, there
was already a computer that beat the world chess champion. And then in 2016,
another AI beat the world champion at Go, except that one worked in a
completely different way.

*(beat — this is the turn, slow down)*

But here's the thing I can't get over, boss. Those two ideas, and a couple more
like them, have never actually been compared against each other. Not once —
and definitely not on Xiangqi."

---

## Slide 2 — Floor 6: why this costs money · 0:45

**Asset: Table I** (*Where prior work leaves the gap this study fills*)

**On the slide**
- Table I, with your own bottom row highlighted
- Above it: **Every study advances one method. None compares them.**
- Small caption: *Henderson et al., AAAI 2018 — controlled comparisons are missing*

**You say**

*(The acting fades out here. Just talk, but keep the energy.)*

"And that actually costs people money, boss. Because when a team starts one of
these projects, there's no real evidence telling them which approach to pick —
so they're stuck guessing.

And the reason there's no evidence is that every paper changes everything at
once, so you can never tell whether the method won or the extra computing power
did.

*(point at the table)*

This table is from my proposal, and every row in it is good work — but each one
only does a single method. The row at the bottom is the one nobody has written
yet.

And it isn't just me saying that, because there's a well-known AAAI paper that
asked for exactly this comparison."

---

## Slide 3 — Floor 11: the question · 0:27

**On the slide**
- One large line: **Keep everything else the same. Which one wins — and what stops the others?**
- Three small lines: **RQ1** the ranking · **RQ2** does learning from humans help or hurt · **RQ3** the real bottleneck

**You say**

"So my question is pretty simple, boss. If I keep everything else exactly the
same and only change the method, which one actually wins?

And then, when one of them stops getting better, I want to know what's holding
it back — whether that's the search, or the size of the model, or just not
enough data."

---

## Slide 4 — Floor 15: what you'd be backing · 0:28

**On the slide**
- **Aim:** a controlled, reproducible, *analysed* answer — plus a league others can reuse
- Five short lines: **O1** environment · **O2** four agents · **O3** Elo league · **O4** ablations · **O5** public release

**You say**

*(Glance up at the floor counter.)*

"Floor fifteen, okay, we're still fine.

So what you'd be backing isn't just a number at the end, boss. It's the whole
setup — the environment, the four agents, and the league that rates them — and
I'm releasing all of it afterwards, so whoever comes after me can add a fifth
method without having to rebuild anything."

---

## Slide 5 — Floor 19: why you can trust it · 0:44

**Asset: Fig. 1** (the controlled apparatus) · keep **Table II** as a backup slide

**On the slide**
- Fig. 1, large
- One line beneath: **Only the decision function changes**

**You say**

"Okay boss, this is the part I'd really like you to judge.

*(point at the diagram)*

Everything underneath here is shared, so all four agents are running on exactly
the same engine, with the same board representation and the same network. The
only thing I change between them is how they decide which move to play.

Then I just let them play each other, both colours, and rate them all on one
Elo scale with confidence intervals.

*(slow right down — this is the key sentence)*

And that's really the whole trick, boss. Because only that one piece changes,
whatever difference I see has to be coming from the method itself, and not from
my coding."

---

## Slide 6 — Floor 23: it already runs · 0:48

**Asset: Fig. 2** (preliminary Elo with confidence intervals)

**On the slide**
- Fig. 2, large
- Footer strip: **Already built — 141,356 games · 11.6M positions · four agents trained · demo running**

**You say**

"Floor twenty-three. Now boss, if this were just an idea I'd understand you
being sceptical, so let me show you something.

It's already built. I've processed a hundred and forty-one thousand games,
trained all four agents, and I've got a demo running.

*(point at the chart)*

And these are the first real results. Tree search comes out on top, but look at
the second bar — learning from humans is still beating every self-play agent,
which is the opposite of what you'd expect, and that's exactly what this setup
is there to check.

I've also tried four ways to push it higher, and all of them hit the same
ceiling.

*(beat)*

So the risky part is already behind me, boss."

---

## Slide 7 — Floor 25: the ask, and out · 0:47

**Asset: Table III** (schedule to the defence)

**On the slide**
- Table III, compact
- Large above it: **The question is open — and I already have the system that can answer it**
- Corner: **Defence — end of May 2027**

**You say**

*(Sound cue: ding. Look up at the counter — small panic, you are not finished.)*

"Oh no — this is me. Twenty-five.

*(Fast now, you are genuinely out of time.)*

Okay okay — quickly, boss. The question is still open, but I've got a design
that can actually answer it, and the thing is already running. And this is my
plan from here up to the defence in May.

*(Stop moving. Look up. Let the ask land — this is the slowest moment of the talk.)*

So what I'm asking for is your support to take it all the way. And if you think
I'm heading in the wrong direction, then honestly, boss, I'd much rather hear
that now than in May.

*(Step out. Turn back and hold the doors with one hand.)*

Oh — and boss? If this works, nobody has to guess which approach to use ever
again.

*(Let the doors go. Small wave, big smile.)*

Okay! Thank you, boss! Enjoy floor twenty-seven!

*(beat, as if the boss calls back just before the doors shut — switch to a slightly different tone for his line)*

"...Don't forget to keep me posted on how it goes, okay? I'll be waiting."

*(Doors close. You are alone. Exhale. Turn to the audience — normal voice, drop the character.)*

I'd be glad to take your questions."

---

## Delivery notes

- **Commit to the opening or cut it.** A half-hearted mime is worse than none.
  Press the button, wait, let the ding land, do the double-take, do the
  stammer. Ten seconds of real commitment buys you the whole room.
- **The "floor 27" question is doing real work.** It tells everyone your time
  is limited, so the pitch feels urgent rather than rushed. Do not cut it.
- **Let the acting go after Slide 1.** Slides 2 to 6 are just you talking.
- **Four moments decide this pitch:** the ding, "never been compared — not
  once", "it's already built", and stepping out. Rehearse those four until they
  are automatic; the rest can be loose.
- **The ask is the slowest line in the talk.** Everything before it speeds up;
  that one sentence slows right down. Then stop talking and let them answer.
- **Rehearse with a timer.** You should start Slide 6 at about 3:20. Behind?
  Cut Slide 4. Never cut the exit — that is what they will remember.
- **Say "Xiangqi" early and clearly.** The whole pitch is about this game, so
  Slide 1 names it outright — don't let the 1997/2016 story run so long that
  the audience forgets what you're actually pitching.
- **The boss's last line is a soft yes.** It never promises support outright,
  but "keep me posted" reads as genuine interest — let it land before you drop
  the character.
- **Avoid:** "In this presentation I will…", "As you can see…", "Furthermore…".

---

## Backup slides (do not show unless asked)

| If they ask about… | Show |
|---|---|
| Why these choices, or why Xiangqi | **Table II** — decisions, reasons, costs accepted |
| Feasibility or the schedule | **Table III** — already on Slide 7, expand it |
| How the ratings are computed | **Equation (1)** — the Elo expected score |
| The game itself | The board and piece-movement figures from the repository |

---

## Anticipated questions

| Question | Answer |
|---|---|
| Why Xiangqi and not chess or Go? | Chess and Go are saturated — another comparison adds little. Xiangqi is harder to search than chess, has very few published studies, and has open data and an open engine, so a rigorous comparison is possible without a cluster. |
| Your agents are far weaker than a real engine. Is this still useful? | Yes — the contribution is the controlled comparison, not peak strength. The engine is included precisely so the gap is measured and reported rather than hidden. |
| Isn't AlphaZero already known to be best? | It is included, and it does win in my results. But those papers do not tell you how it compares to the alternatives on one board under one budget, and they do not say what limits the others. |
| Why is imitation beating self-play? That seems wrong. | At a large budget the literature expects self-play to win. My finding is that at the budget I can afford it is not yet competitive — a statement about the budget, not a refutation of the method, and I say so explicitly. |
| Four paradigms is a lot for one thesis. | They share one substrate, so each agent is a single function on top of code that already exists. Three of the four are already trained and measured. |
| Can your Elo be compared to a human rating? | No. It is relative within each tournament. Mapping to human ratings needs human games and raises its own validity questions — it is an open question in my proposal. |
| How do you know the data is clean? | All 141,356 games are replayed through my own rule engine and discarded if any move is illegal. Zero failed, and an independent sample of 500 replayed legally. |
| What if the scale-up finds nothing? | Then I report that, and it is still a result: a characterisation of the budget-limited regime. The design is set up so either outcome answers RQ3. |

### If you cannot answer

Do not improvise a technical answer you are unsure of. Use one of these, then
move on — committing to a concrete follow-up is far stronger than guessing:

- "That is an interesting point, and I have not examined it yet. I will look
  into it and come back to you."
- "I do not have data on that, but it would be straightforward to test with the
  league I already have, and I will add it."
- "I am not certain, and I would rather check than guess. May I follow up with
  you afterwards?"
- "That is outside what I have measured so far — may I note it as something to
  address in the next round of experiments?"

If a question exposes a real weakness, say so plainly and say what you will do
about it. Examiners respect a student who knows the limits of their own
evidence far more than one who defends everything.
