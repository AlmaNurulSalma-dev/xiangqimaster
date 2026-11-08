# 3-Minute Pitch Script — "Going Up?" (8 slides)

Format: 3 minutes to present, 2 minutes for questions. One short paragraph per slide,
matched to the 8-slide deck ("GOING UP?" / elevator theme). Keep the boss/elevator
voice from the 5-minute version, but tighter — no long stories, just the beats.

Target: ~140 words/slide average is too much for 8 slides in 3 minutes, so most
slides are 2-3 sentences only. Total ≈ 430-450 words ≈ 2:50-3:00 at 150 wpm.

---

### Slide 1 — "GOING UP?"
*(doors open, step in, see the audience, do a double take)*

Okay guys, before I start, imagine I'm about to step into a lift. *(ding, doors open)* Wait, wait, that's my boss! H-hi! Hello, boss! Which floor are you going to? Oh, twenty-seven, okay, we've got a bit of time. Boss, so my thesis is actually about Xiangqi, Chinese chess, and the AI that plays it. And get this, back in 1997 a computer already beat the world chess champion, and in 2016 another AI beat the world champion at Go, a completely different approach. But here's the thing, boss, those ideas have never actually been compared against each other. Not once, and definitely not on Xiangqi.

### Slide 2 — "Controlled comparisons are missing"
So here's the problem, boss: there are four different ways people teach AI to play games like this — hand-written rules, learning from humans, self-play, and tree search. But every study only ever tests one method, on its own setup. Nobody's put all four side by side, same game, same rules, same everything else.

### Slide 3 — "Which one wins, what stops the others"
So we don't actually know which one wins, or what's holding the other three back. And there's a more specific question inside that: does learning from human games help the self-play agent later, or does it actually get in the way? Right now that's just a guess.

### Slide 4 — "The aim"
My aim is to build one shared environment, train all four agents in it, and rank them with a proper Elo league — plus run ablations to see what each one needs to get stronger. And since it's all one shared system, I can release the whole thing publicly for others to reuse.

### Slide 5 — "Only the decision function changes"
The trick, boss, is that all four agents share the exact same rules engine, same board encoding, same network — the only thing that's different between them is the decision function, the part that actually picks the move. So any gap in strength is coming from that one thing, not from some hidden setup difference.

### Slide 6 — "Already built"
And this isn't just a plan — it's already running. I've got 141,356 real games, 11.6 million board positions, all four agents trained, and a working demo you can actually play against right now.

### Slide 7 — "The system that can answer it"
So the question's still wide open in the field, boss — but I already have the system built that can answer it.

### Slide 8 — "Thank you" / exit
Okay! Thank you, boss! Enjoy floor twenty-seven! *(beat, as if the boss calls back just before the doors shut)* ...Don't forget to keep me posted on how it goes, okay? I'll be waiting. *(doors close)*

---

## Prepared non-answers (for the 2-minute Q&A)

- **Unreleased/early result ("why is X not finished yet"):** "That's a great question — it's still running, so I don't want to give you a number I'm not confident in yet. I'll follow up once it's settled."
- **Comparison to a specific other engine/paper not covered:** "I haven't benchmarked against that one specifically, but it's a fair comparison to add — I'll look into it."
- **"Why not just use the strongest method and skip the comparison?"** "Because without the controlled comparison we wouldn't actually know *why* it's strongest, or whether it's strongest for every situation or just on average."
- **Hardware/scale question:** "Right now I'm compute-limited to a single laptop GPU, so the honest answer is scale is my main open risk — that's actually one of the things I'm discussing with my supervisor."

## Word count

Slides 1-8 body text (spoken words only, stage directions excluded): **389 words** → ≈ 2:36 at 150 wpm, ≈ 2:47 at 140 wpm — comfortably fits the 3-minute budget, with a little room to spare after swapping in the richer Slide 1 intro.
