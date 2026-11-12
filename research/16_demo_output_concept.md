# XiangqiMaster Demo: Proposed Output Concept

Alma Nurul Salma (L25020007) - Nanjing Xiaozhuang University - November 2026

## 1. Why this output

This thesis asks several questions about the four AI approaches it compares.
One of them is: once everything else about the setup, the game, the board
representation, the network, and the evaluation protocol, is held exactly
the same, which of these four ways of choosing a move actually plays Xiangqi
best? A table of Elo numbers answers that question for a reader who already
trusts the methodology behind it. It does not answer it for a visitor who
does not play Xiangqi, or who has not read the full report, because a number
on its own does not show how four different kinds of reasoning actually
differ in practice.

The output proposed here turns that comparison into something a visitor can
experience directly: playing against each approach, watching a live
evaluation of the position, and seeing, move by move, where the four ways of
choosing a move diverge. The aim is for the output to demonstrate the
thesis's own finding, not merely to be a nice web page.

## 2. What it is

- **Play directly against any of the four paradigms**, presented as distinct
  characters. Each character's strength (Elo) is read from the project's own
  completed round-robin results, not invented for the demo.
- **A live advantage meter** showing the model's running evaluation of the
  position as the game is played.
- **A short explanation after each move**, built from the policy and value
  outputs the network already computes internally, exposed through one small
  new interface rather than a new model.
- **A side-by-side comparison mode**, showing two paradigms' chosen move from
  the identical position.
- **A post-game summary** classifying move quality across the game and
  allowing it to be replayed.

## 3. User flow

A visitor lands on the title screen, picks an opponent, plays a full game
with live feedback, and finishes on a summary screen that can loop back into
another match against a different opponent. Comparing two opponents on the
same position is an optional side branch from the gameplay loop, not a
separate destination, so it never gets in the way of simply playing.

@FIG span | flow | research/figures/demo/flow_diagram.png | User flow through the four core screens of the demo.

## 4. Screen by screen

### 4.1 Landing (Figure 1)

The first thing a visitor sees: the game's title, a one-line description,
and a single "Play Now" action. Three small badges beneath it preview what
makes this different from an ordinary play-a-bot page: a live advantage
meter, a mode to compare two models' styles, and move-by-move feedback.
Nothing here is interactive beyond the single entry point, on purpose, so a
first-time visitor is never unsure what to do next.

### 4.2 Choose Opponent (Figure 2)

This is the screen that connects the demo back to the actual research. Each
of the seven cards is not a cosmetic skin: it stands for one trained model
from this project's own experiments, and the number on the card is that
model's measured Elo rating, not a made-up difficulty label. {{t:roster}}
sets out the full mapping.

@TABLE roster | Each character, the model behind it, and why that model exists
| Character | Model it represents | Elo (indicative) | Why this model was built |
|---|---|---|---|
| The Rookie | Random-move agent | ~1300 | Learns nothing at all. Exists purely as the floor every other agent is measured against. |
| The Calculator | Minimax, alpha-beta search (depths 1-3) | ~1460 | Classical adversarial search with a hand-written evaluation function. Learns nothing, so it isolates what learning adds compared to a traditional method. |
| The Student | Imitation learning | ~1600 | Trained by supervised learning on 141,356 real human games, learning to predict the human move and the eventual result. Tests whether copying expert human patterns alone is enough to play well. |
| The Gambler | Self-play reinforcement learning (PPO) | ~1450 | Trained purely by repeatedly playing itself and learning from the outcome. Two starting points exist, from scratch and from the Student's checkpoint, specifically to test whether starting from human imitation helps or hurts self-play. |
| The Strategist | Tree search guided by the policy-value network (MCTS) | ~1690 | Combines the network's instinct with explicit lookahead search before each move, in the AlphaZero style. The strongest agent trained in this project. |
| The Apprentice | Distilled from a chess engine's self-play instead of humans | ~1500-1570 | Trained by imitating a strong existing engine's self-play rather than human games, to test whether a machine teacher beats a human one at the same network size. It did not. |
| The Grandmaster | Pikafish engine (external reference) | 2000+ | Not one of the four paradigms under comparison. Included only as an outside anchor, so a visitor has a feel for genuinely superhuman play next to this project's own models. |

Elo is relative within whichever round robin produced it, so the figures
above are indicative markers for the demo, not a single absolute scale; the
full report states each number's exact origin.

### 4.3 Gameplay (Figure 3)

The board occupies most of the screen, with the most recent move highlighted
by an arrow. To its right, a vertical advantage meter updates after every
move, a small dialogue box gives a short line reacting to what just
happened, and a scrollable move list keeps the full notation. Hint, draw,
and resign sit below the board as the only other controls, so the screen
stays a chessboard first and an instrument panel second.

### 4.4 Compare Mode (Figure 5, pending)

The same position is shown twice, once per selected character, with each
side's chosen move and evaluation displayed beneath it. This screen is still
being regenerated and will be added once it is ready.

### 4.5 Game Over and Replay (Figure 4)

After the match, both sides get an accuracy percentage, a move-by-move
quality strip (best, good, inaccuracy, mistake, blunder), and a graph of how
the evaluation moved across the whole game. A scrubber lets the match be
replayed move by move, which is where the per-move feedback from Figure 3
becomes most useful: a visitor can go back to the exact point an advantage
was won or lost.

## 5. Why this is feasible at this scope

- **No new training.** Every character on the roster is backed by a
  checkpoint that already exists from completed experiments.
- **The explanation and hint features reuse existing outputs.** The policy
  logits and value estimate are already computed inside every forward pass;
  this only exposes them through one additional method, not new modelling
  work.
- **This matches the schedule already proposed**: an interactive-platform
  block in Mar-Apr 2027, after the scale-up study, not displacing it.
- **A human-vs-human mode was deliberately excluded.** It does not
  demonstrate anything about the four paradigms being compared, and would
  add engineering scope unrelated to the research question.

This is the direction currently being pursued for the project's output,
built entirely from work already completed rather than work still to be
done.
