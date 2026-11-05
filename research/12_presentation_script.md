# XiangqiMaster — Presentation Script (spoken)

> **How to use this.** This is what you *say*, not what you read off the slides.
> Lines in plain text are your words. Lines in *(italics)* are delivery cues — pauses,
> tone, when to click. Don't memorise word-for-word; learn the **beats** so it sounds
> like you, not a script. Total run time ≈ **7–8 minutes**.
>
> **Three rules to not sound stiff:** (1) talk to one friendly face, then another —
> not the wall; (2) short sentences, breathe between them; (3) never read a bullet
> out loud — the slide is your poster, your voice is the story.

---

## Slide 1 — Title

*(Don't read the title. Look up, smile, then the hook. Let the first line land.)*

"In 1997, a computer beat the world chess champion. Twenty years later, a completely
different kind of AI beat the world champion at Go. Two games — two breakthroughs —
and two totally different ideas about how a machine should *think*.

*(beat)*

So here's the question I kept coming back to: if you took the biggest ideas behind
those breakthroughs, and set them loose on **Chinese Chess** — same board, same code,
same budget — which one would actually win?

*(beat, gesture to slide)*

That's this project. I'm Alma, and this is **XiangqiMaster** — a fair fight between
four AI paradigms on one Xiangqi board, to find out which one plays best, and *why*."

---

## Slide 2 — What is Xiangqi?

*(Lighter, quicker. Assume some of them know chess.)*

"Quick ground rules, in case Xiangqi is new to you. If you know chess, you're
basically 80% there. Two armies, you take turns, and you win by checkmating the enemy
General — that's their king. A lot of the pieces rhyme with chess: the chariot moves
like a rook, the horse like a knight.

*(click / point to board)*

But three things make it its own game. One — the pieces sit *on* the lines, not inside
the squares. Two — there's a **river** across the middle. And three — that river
changes the rules: a Soldier that crosses it suddenly can move sideways, the Elephants
can *never* cross it, and the General is stuck inside his palace the whole game.

*(beat)*

So: familiar enough to follow, different enough to be its own puzzle."

---

## Slide 3 — Similar game, harder problem

"And here's the thing — those quirks don't just add flavour, they make it genuinely
*harder* for a computer. Two examples. The cannon can only capture by **jumping over**
another piece — so every capture depends on what's in the way. And the horse can be
physically **blocked** by a piece right next to it.

*(beat)*

Little rules — but they blow up the number of things the AI has to think about.
Xiangqi has a bigger branching factor than chess and longer games. Same family as
chess — meaner problem. Which, for testing AI, is exactly what you want."

---

## Slide 4 — Xiangqi is a serious AI benchmark

*(Confident. This is where you earn academic credibility — then open the gap.)*

"And I'm not the only one who thinks so. Researchers recently used Xiangqi to test
whether large language models can *actually reason* — because this game forces real,
multi-step thinking, not pattern-matching. There's a paper in TMLR, called JiangJun,
that trained on over ten thousand human games just to study one way to play it well.
And there's an AlphaZero-style player from Yilmaz and colleagues.

*(beat — slow down, this is the pivot)*

So the game is taken seriously. But notice something: every one of these studies looks
at **one** method. One team does imitation. Another does AlphaZero. Nobody lines them
up on the *same* board and asks the obvious question —

*(beat)*

— which approach actually wins? That's the gap I'm filling."

---

## Slide 5 & 6 — The four paradigms

*(This is the heart of the talk. Give each one ONE clear sentence + its famous cousin.
Use your hands — count them off, one to four. Don't rush.)*

"So I built four players, each using a different big idea in AI. Think of them as going
from 'pure calculation' to 'learned intuition'.

*(1)* **Minimax.** Old-school brute-force search — no learning at all. It just looks
ahead and assumes both sides play perfectly. This is the Deep Blue idea, the one that
beat Kasparov.

*(2)* **Imitation Learning.** This one learns by **copying humans** — I showed it
millions of human moves and it learned to play like us. This is how AlphaGo first
learned Go.

*(3)* **Self-Play Reinforcement Learning.** No human data at all — it plays against
itself millions of times, wins and loses, and slowly figures out what works. This is
the family behind OpenAI's Dota bots and the training behind ChatGPT.

*(4)* **Monte Carlo Tree Search with a neural network** — the AlphaZero recipe. The
network gives it *intuition*, and then it *searches* to double-check that intuition
before every move.

*(beat)*

Plus two reference points: a random player as the floor, and Pikafish — a superhuman
engine — as the ceiling. Everybody plays everybody. One league, one scale."

---

## Slide 7 — What I'm researching + objective

"So, concretely, I'm chasing three questions.

One — **the ranking**: on a single Elo scale, how do the four actually stack up?

Two — **the imitation effect**: if I let a model learn from humans first and *then*
do self-play, does that help it, or hold it back? That turned out to be the most
surprising part — I'll get to it.

Three — **what limits strength**: when an agent hits a wall, is it the search, the
network size, the data, or the algorithm itself?

*(beat)*

And the goal isn't just a number. It's a **controlled, reproducible answer** to
'which paradigm wins and why' — plus a reusable evaluation league that the next
student can pick up and build on."

---

## Slide 8 — What's already built

*(Tone shift: proud, grounded. This is where you show it's real, not a proposal.)*

"And this isn't a plan — it's already running. I built the whole thing from the ground
up: the game engine with every Xiangqi rule, the data pipeline, all four agents, three
different training systems, the Elo tournament, and a live demo.

*(beat)*

Just to give you a sense of scale — the engine, the 2,000-move action space, over
eleven million training positions, and 224 automated tests keeping it honest. The
foundation is done. What's left is scaling it up — and I'll show you exactly what and
when."

---

## Slide 9 — Results so far

*(Slow down. This is your payoff slide. Point at the chart, then deliver the twist.)*

"Here's what came out. On the Elo scale, the **MCTS + network agent won** — the
AlphaZero-style one. No huge surprise there; intuition plus search is a strong combo.

*(beat — now the interesting part)*

But three things *did* surprise me. First: imitation learning, just copying humans,
**beat every self-play agent**. Learning from scratch on one laptop simply wasn't
enough. Second: I tried using the superhuman engine as a teacher — and even *that*
didn't beat plain human imitation. And third — this is the big one —

*(beat)*

I tried four different ways to push the agents stronger: more search, more training,
a stronger teacher, a bigger network. And all four hit the **same ceiling**, around
1690 Elo.

*(beat)*

That tells me something important. On this scale, the thing holding the AI back isn't
the *algorithm* — it's the **capacity and the compute**. And knowing *that* is exactly
what tells me where to push next."

---

## Slide 10 — Live demo

*(Switch to the web app. Keep talking while it loads — never stare at a loading screen
in silence. Run a FAST matchup: Pikafish MAX vs Minimax d1.)*

"Let me actually show you. This is the live demo — I pick two of my agents, hit play,
and we watch them fight it out.

*(click Play)*

Now, watch the graph at the bottom, because that's the whole story in one picture. When
the red area grows, Red is winning; blue means Black is winning. You can literally see
the moment one side takes over.

*(let a few moves play)*

And there it is — one side pulls ahead, the lead climbs, checkmate. That graph is how
I turn 'which AI is better' into something you can actually *see*, move by move."

---

## Slide 11 — What's not done + schedule

*(Honest and confident — examiners respect knowing your own gaps.)*

"So what's left? The foundation is solid; now it's about **scale**. Next, I take those
experiments bigger — larger networks, more self-play, more compute — to see if I can
break that 1690 ceiling. Then I calibrate everyone against the engine on one absolute
scale. Then I write it all up.

*(gesture to timeline)*

That's the road to my defense at the end of May 2027. Clear steps, real deadlines —
I know exactly where this is going."

---

## Slide 12 — Thank you

*(Land it. Don't trail off. Look up, pause, then the closer.)*

"So — four AI paradigms, one Xiangqi board, one fair fight. I already know which one
wins today, and I know exactly what's holding the rest back. The next six months are
about pushing past that ceiling.

*(beat, smile)*

Thank you — I'd love to take your questions."

---

## Quick delivery cheat-sheet

- **The hook (Slide 1)** and **the twist (Slide 9)** are the two moments that make or
  break the talk. Rehearse those until they feel effortless.
- **Pause after every big number and every surprise.** Silence makes you sound
  confident, not nervous.
- **Demo = pick a fast matchup** (Pikafish MAX vs Minimax d1) so it finishes in a
  couple of seconds. Keep narrating while it loads.
- **If you blank:** your spine is always the same four words — *engine, agents,
  results, what's next.* Find your place and keep going.
- **Phrases to avoid** (they sound canned): "In this presentation I will…",
  "Furthermore", "As you can see", "In conclusion". Just say the thing.
