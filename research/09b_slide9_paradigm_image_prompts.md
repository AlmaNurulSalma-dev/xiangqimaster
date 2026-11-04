# Slide 9 — Image-generation prompts for the four paradigms

Four matching visuals, one per technique (Minimax · Imitation Learning · Self-Play
RL · MCTS+Network), each showing **how the technique actually works**. Designed as a
**set**: same style line, same Xiangqi motif, one signature accent colour each.

**How to use.** Paste one prompt at a time into your image generator (GPT-Image /
DALL·E / Midjourney / Firefly). Each prompt already includes the shared style, so it
works standalone. Generate at **4:3** (≈1600×1200) or **1:1**. AI image tools render
long text poorly — keep the in-image words short and add the crisp labels (the
"Overlay labels" lines) yourself in PowerPoint. Colour-code the four accents across the
whole deck so the audience tracks each paradigm.

**Shared STYLE block** (already embedded in each prompt below):
> Clean modern flat-vector technical infographic, educational diagram style, crisp thin
> outlines, soft long shadows, rounded cards and nodes, subtle gradients; dark slate-navy
> background (#0E1117) with warm amber highlights; a small stylized **Xiangqi (Chinese
> chess) board — 9×10 grid with a river and two palaces, RED discs vs BLUE discs** as a
> recurring motif; professional thesis-presentation quality, high detail, balanced
> composition, generous empty space for captions, 4:3, flat UI illustration, light
> isometric. Negative: no gibberish text, no long paragraphs, no watermark, not photo-
> realistic, **not an 8×8 chessboard — it is a 9×10 Xiangqi board**.

---

## 1 · MINIMAX  (alpha–beta search)  — accent: steel grey-blue
**Prompt:**
> Clean modern flat-vector technical infographic, educational diagram style, crisp thin
> outlines, soft shadows, rounded cards, dark slate-navy background (#0E1117) with amber
> highlights; recurring stylized 9×10 Xiangqi board motif with red vs blue discs; thesis
> quality, 4:3, flat isometric. SUBJECT: how a **Minimax (alpha–beta) search** agent picks
> a move in Xiangqi. Center: a **branching game tree growing downward** from a single root
> node that shows a small Xiangqi position; each node is a rounded card holding a tiny
> Xiangqi board. The levels **alternate and are labelled**: top level "MAX — our move" in
> red, next level "MIN — opponent" in steel-blue, alternating down 3–4 plies. At the leaf
> nodes show a **"material evaluation" balance-scale gauge** weighing red pieces vs blue
> pieces, each leaf tagged with a number. Several branches are **cut off with dashed lines
> and a small scissors icon labelled "α–β pruning"**, greyed out to show skipped work. One
> **bold highlighted path** runs from the best leaf up to the root with an arrow "best
> worst-case move". Add a small **gear / calculator icon** to signal pure calculation —
> NO brain, NO learning. Signature accent colour **cool steel grey-blue**; mood: classical,
> exhaustive, precise. Negative: no 8×8 chessboard, no gibberish text.

**Overlay labels (add in PPT):** Look ahead in the game tree · assume best play by both
sides · prune hopeless branches (α–β) · pick the move with the best worst-case ·
hand-written evaluation, **no learning**.

---

## 2 · IMITATION LEARNING  (supervised, copy the human)  — accent: teal / emerald
**Prompt:**
> Clean modern flat-vector technical infographic, educational diagram style, crisp thin
> outlines, soft shadows, rounded cards, dark slate-navy background (#0E1117) with amber
> highlights; recurring stylized 9×10 Xiangqi board motif with red vs blue discs; thesis
> quality, 4:3, flat isometric. SUBJECT: how an **Imitation-Learning** agent learns Xiangqi
> by **copying human experts (supervised learning)**, left-to-right pipeline. LEFT: a tall
> **stack of human game records** — scrolls / record cards covered in Xiangqi move notation
> — labelled "141,356 human games → 11.6M positions". An arrow pulls one position out and
> converts it into a **glowing 3-D stacked tensor cube** labelled "14×10×9 board encoding".
> CENTER: a **neural network as a tidy pipeline of stacked convolutional / residual
> blocks**. RIGHT: the network splits into **two output heads** — a "policy" head showing
> move-probability arrows laid over a Xiangqi board, and a "value" head showing a win-chance
> gauge. KEY METAPHOR: a **human hand placing a piece and the student AI mirroring the exact
> same move**, joined by a "copy / =" icon and a small target "match the human's move".
> Signature accent colour **teal / emerald green**; mood: learning-by-demonstration,
> textbooks, calm. Negative: no 8×8 chessboard, no gibberish text.

**Overlay labels:** Huge dataset of human-master games · encode each board (14×10×9) ·
neural net predicts the **human's** move · policy head + value head · trained by
supervised learning — **it copies the expert**.

---

## 3 · SELF-PLAY REINFORCEMENT LEARNING  (PPO, from scratch)  — accent: warm orange
**Prompt:**
> Clean modern flat-vector technical infographic, educational diagram style, crisp thin
> outlines, soft shadows, rounded cards, dark slate-navy background (#0E1117) with amber
> highlights; recurring stylized 9×10 Xiangqi board motif with red vs blue discs; thesis
> quality, 4:3, flat isometric. SUBJECT: how a **Self-Play Reinforcement-Learning (PPO)**
> agent trains with **NO human data**, drawn as a **big circular loop** with cycle arrows.
> TOP of the loop: **two identical robot / AI avatars face each other across a Xiangqi
> board — the agent playing a mirror copy of ITSELF** (one tinted red, one blue, a small
> "mirror / reflection" icon between them). From the finished game a **reward** flows out:
> a **trophy / +1 for the winner, −1 for the loser**. The reward travels along the loop as
> a glowing signal into a **"policy update (PPO)" module** shown as a dial + gradient arrow
> nudging a small neural network to "play winning moves more often". The loop closes back
> into another self-play game, labelled **"repeat millions of times"**. Add a small
> **"from zero / tabula-rasa" spark** to stress learning from scratch by trial and error.
> Signature accent colour **warm orange / amber**; mood: iterative, evolutionary, energetic.
> Negative: no 8×8 chessboard, no gibberish text.

**Overlay labels:** Play **against itself** · win = reward +1, loss = −1 · nudge the policy
toward winning moves (PPO) · repeat millions of games · **learns from scratch, no human
data**.

---

## 4 · MCTS + NEURAL NETWORK  (AlphaZero-style: intuition + calculation)  — accent: violet
**Prompt:**
> Clean modern flat-vector technical infographic, educational diagram style, crisp thin
> outlines, soft shadows, rounded cards, dark slate-navy background (#0E1117) with amber
> highlights; recurring stylized 9×10 Xiangqi board motif with red vs blue discs; thesis
> quality, 4:3, flat isometric. SUBJECT: how an **AlphaZero-style MCTS + neural-network**
> agent "thinks" before moving — **intuition plus calculation**. CENTER-RIGHT: a **search
> tree grown by simulations**, radiating from a root Xiangqi position, with the **four MCTS
> steps as a labelled cycle** around it: 1) **SELECT** — follow the strongest branch via a
> "PUCT" arrow that weights prior + value; 2) **EXPAND** — add a new leaf node; 3)
> **EVALUATE** — score the leaf with the network's value head; 4) **BACK-UP** — results flow
> back up the path, **thickening the good branches**. CENTER-LEFT: the **same policy-value
> neural network drawn as a glowing brain / network providing "intuition"** — move priors
> (faint arrows) and a position-value gauge — feeding the search. After many simulations the
> final move is chosen by the **MOST-VISITED branch**, drawn as the thickest, highlighted
> edge with a "pick the most-visited move" arrow. Banner motif: **"intuition (network) +
> calculation (search)"**. Signature accent colour **violet / purple**; mood: the most
> advanced, AlphaGo-like, sophisticated. Negative: no 8×8 chessboard, no gibberish text.

**Overlay labels:** Network gives **intuition** (move priors + position value) · run many
look-ahead **simulations** (select → expand → evaluate → back-up, PUCT) · choose the
**most-visited** move · the **AlphaZero recipe**.

---

### Consistency tips
- Generate all four with the **same style sentence** (and, in Midjourney, the same
  `--style`/`--sref` or a fixed seed) so they read as one family.
- Keep the **accent colours fixed** — steel-blue = Minimax, teal = IL, orange = Self-Play
  RL, violet = MCTS — and reuse those same colours for these agents in the Elo chart and
  everywhere else in the deck.
- If a generator keeps writing garbled words, add **"minimal text, labels added later"** and
  put the real captions on the slide yourself.
- Pair nicely as a **2×2 grid** on Slide 9 (or four stacked panels), left→right in order of
  increasing sophistication: Minimax → IL → Self-Play RL → MCTS.
