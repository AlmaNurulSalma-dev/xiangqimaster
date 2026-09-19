# 01 — Xiangqi Game Rules (Complete Reference)

> **Read this when:** implementing board representation, piece movement logic, legal move generation, check/checkmate detection, or win conditions. This is the authoritative rulebook for the environment implementation.

---

## 1. Overview

Xiangqi (象棋), also called Chinese Chess, is a two-player, zero-sum, perfect-information strategy board game. Two sides — **Red (红)** and **Black (黑)** — alternate moves. Red always moves first. The objective is to checkmate the opponent's General (capture is not required; trapping it is enough).

**Key game properties for the implementation:**
- Perfect information (both players see the entire board — no hidden state)
- Deterministic (no dice, no randomness in the rules)
- Zero-sum (one player's win is the other's loss)
- Average game length: ~95 plies (half-moves)
- Branching factor: ~40 legal moves per position
- Draws are possible but less frequent than in International Chess

---

## 2. The Board

- The board is a grid of **9 columns (files) × 10 rows (ranks)**.
- Pieces are placed on the **intersections** of lines, not inside squares — so there are 9 × 10 = **90 points**.
- **The River (河界):** a horizontal gap divides the board between row 4 and row 5 (0-indexed) — separating the two halves. Some pieces are affected by the river.
- **The Palace (九宫):** a 3×3 area on each side, marked by diagonal lines. The General and Advisors are confined to the palace.
  - Red palace: columns 3–5, rows 0–2 (0-indexed)
  - Black palace: columns 3–5, rows 7–9 (0-indexed)

**Coordinate convention for the codebase:**
- Use **(row, col)** with 0-indexing.
- row 0 = Red's back rank (bottom). row 9 = Black's back rank (top).
- col 0 = leftmost file. col 8 = rightmost file.
- This must be consistent across the ENTIRE codebase. Document any deviation loudly.

---

## 3. The Pieces

Each side has 16 pieces of 7 types. Below, each piece is described with its movement rules. **All movement rules must be implemented exactly.**

### 3.1 General / King (將 / 帥) — "general"
- **Count:** 1 per side
- **Starts:** center of the back rank, inside the palace
- **Movement:** one point orthogonally (horizontally or vertically). Never diagonally.
- **Restriction:** confined to the 3×3 palace. Cannot leave it.
- **Special rule — "Flying General" (飞将):** The two Generals may NOT face each other directly along an open file with no pieces between them. If a move would cause the two Generals to directly face each other with nothing in between, that move is illegal. (This effectively acts like the Generals attacking each other along the file.)

### 3.2 Advisor / Guard (士 / 仕) — "advisor"
- **Count:** 2 per side
- **Starts:** beside the General
- **Movement:** one point diagonally.
- **Restriction:** confined to the palace. Because of the palace shape, an advisor only ever has 5 reachable points.

### 3.3 Elephant / Minister (象 / 相) — "elephant"
- **Count:** 2 per side
- **Movement:** exactly two points diagonally (a fixed diagonal jump of 2×2).
- **Restriction 1 — River:** the Elephant may NOT cross the river. It stays on its own side. This gives each Elephant only 7 reachable points total.
- **Restriction 2 — "Blocking the Elephant's eye" (塞象眼):** the Elephant is blocked if there is a piece on the midpoint of its diagonal path. If the intermediate point (one step diagonally toward the destination) is occupied by any piece, the move is illegal.

### 3.4 Horse / Knight (馬 / 傌) — "horse"
- **Count:** 2 per side
- **Movement:** moves one point orthogonally, then one point diagonally outward — an "L" shape similar to the Chess knight, BUT with a critical difference below.
- **Restriction — "Hobbling the Horse's leg" (蹩马腿):** unlike the Chess knight, the Xiangqi horse can be BLOCKED. If the point directly orthogonally adjacent (the first step of the L, before the diagonal) is occupied by any piece, the horse cannot move in that direction. This means the horse does not "jump over" pieces the way a Chess knight does.

### 3.5 Chariot / Rook (車 / 俥) — "chariot"
- **Count:** 2 per side
- **Movement:** any number of points orthogonally (horizontally or vertically), like a Chess rook.
- **Restriction:** cannot jump over pieces. Movement stops at the first occupied point (captures if it's an enemy piece).
- **Note:** the Chariot is the most powerful piece — value ~9.

### 3.6 Cannon (砲 / 炮) — "cannon"
- **Count:** 2 per side
- **Movement (non-capturing):** moves like a Chariot — any number of points orthogonally over EMPTY points. Cannot jump when NOT capturing.
- **Movement (capturing):** to capture, the Cannon must jump over exactly ONE piece (called the "screen" or "cannon mount", 炮台) of either color, and land on an enemy piece directly beyond it. There must be exactly one piece between the cannon and its target — no more, no less.
- **This is the most distinctive Xiangqi rule** and the trickiest to implement correctly. Test it thoroughly.

### 3.7 Soldier / Pawn (卒 / 兵) — "soldier"
- **Count:** 5 per side
- **Movement (before crossing river):** one point straight forward only. Cannot move sideways or backward.
- **Movement (after crossing river):** one point forward OR one point sideways (left/right). Still can never move backward.
- **Promotion:** unlike Chess, there is no promotion. A soldier that reaches the last rank simply can only move sideways from then on.

---

## 4. Starting Position

```
Row 9 (Black back): 車 馬 象 士 將 士 象 馬 車   (chariot horse elephant advisor general advisor elephant horse chariot)
Row 8:              .  .  .  .  .  .  .  .  .
Row 7:              .  砲 .  .  .  .  .  砲 .   (cannons)
Row 6:              卒 .  卒 .  卒 .  卒 .  卒   (soldiers)
Row 5:              .  .  .  .  .  .  .  .  .   (Black side of river)
------------------------ RIVER ------------------------
Row 4:              .  .  .  .  .  .  .  .  .   (Red side of river)
Row 3:              兵 .  兵 .  兵 .  兵 .  兵   (soldiers)
Row 2:              .  砲 .  .  .  .  .  砲 .   (cannons)
Row 1:              .  .  .  .  .  .  .  .  .
Row 0 (Red back):   車 馬 相 仕 帥 仕 相 馬 車
```

---

## 5. Movement Summary Table

| Piece | Move pattern | Can jump? | Special restriction |
|---|---|---|---|
| General | 1 orthogonal | No | Palace only; no flying-general face-off |
| Advisor | 1 diagonal | No | Palace only |
| Elephant | 2 diagonal | No | Own side only; blocked at midpoint |
| Horse | L-shape | No | Blocked at the leg |
| Chariot | Any orthogonal | No | Stops at first piece |
| Cannon | Any orthogonal (move); jump 1 to capture | Only when capturing | Exactly one screen piece to capture |
| Soldier | 1 forward; +sideways after river | No | Never backward |

---

## 6. Check, Checkmate, and Game End

### 6.1 Check (将军)
A General is in **check** when an enemy piece threatens to capture it on the next move. A player in check MUST make a move that removes the check.

### 6.2 Checkmate (将死)
If a player is in check and has NO legal move to escape check, it is **checkmate** and that player LOSES.

### 6.3 Stalemate (困毙)
If a player has NO legal moves but is NOT in check, in Xiangqi this is ALSO a **LOSS** for the player who cannot move. (This is different from International Chess, where stalemate is a draw. Implement carefully.)

### 6.4 Draw Conditions
- **Perpetual check (长将):** a player is forbidden from checking the opponent's General perpetually (repeatedly forcing the same position). If a player perpetually checks, they are typically required to vary or they lose. For the RL implementation, use a simpler rule: if the exact same position repeats 3 times, declare a draw.
- **Mutual blockade / no progress:** if neither side can make progress, a draw may be declared. For training, use a move limit (e.g., 300 plies → draw).

### 6.5 Rules Simplification for RL Training
For the environment, use these simplified terminal conditions (document this as a known simplification in the thesis):
- Checkmate → win/loss
- Stalemate (no legal moves) → loss for the side to move
- Position repeats 3 times → draw
- Move count exceeds 300 plies → draw
- (Advanced perpetual check / chasing rules may be omitted for simplicity and noted as a limitation.)

---

## 7. Notation — WXF Format

The dataset uses **WXF (World Xiangqi Federation)** notation. Key facts the parser must handle (full spec in 04-DATA-PIPELINE.md):
- Moves are recorded relative to each player's own perspective.
- Files are numbered 1–9 from each player's right.
- Movement is described as: piece + starting file + direction (forward/backward/traverse) + destination.
- The parser must convert WXF into the internal (from_row, from_col, to_row, to_col) representation.

---

## 8. Piece Values (For Minimax Evaluation & Reward Shaping)

Standard relative values used in evaluation functions:

| Piece | Value |
|---|---|
| Chariot | 9 |
| Cannon | 4.5 |
| Horse | 4 |
| Advisor | 2 |
| Elephant | 2 |
| Soldier (before river) | 1 |
| Soldier (after river) | 2 |
| General | infinite (game-ending) |

These values are used in the Minimax baseline (Agent 4) and optionally in dense reward shaping (see 03-ENVIRONMENT and 06-AGENTS).

---

## 9. Implementation Checklist

When implementing the rules, verify each of these with unit tests:
- [ ] Each piece type moves correctly in all directions
- [ ] Chariot stops at first blocking piece
- [ ] Cannon moves without jumping, captures with exactly one screen
- [ ] Horse is blocked by the leg
- [ ] Elephant is blocked at the eye and cannot cross the river
- [ ] Advisor and General stay in the palace
- [ ] Soldier gains sideways movement only after crossing the river
- [ ] Flying-general rule prevents Generals facing each other
- [ ] Check detection works for all attacking piece types
- [ ] Checkmate correctly identified (in check + no legal escape)
- [ ] Stalemate correctly results in a loss (not draw)
- [ ] Threefold repetition detected → draw
- [ ] Move limit enforced → draw

---

*Cross-references: 03-ENVIRONMENT.md (how rules become the Gym environment), 04-DATA-PIPELINE.md (WXF parsing), 12-GLOSSARY.md (terms).*
