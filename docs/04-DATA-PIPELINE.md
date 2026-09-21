# 04 — Data Pipeline Specification

> **Read this when:** downloading the dataset, parsing WXF game files, encoding board states, building PyTorch datasets, or creating train/val/test splits. This governs everything about how raw game records become training data.

---

## 1. Purpose

The data pipeline turns raw human game records into a clean, labeled dataset used for **imitation learning** (Agent 2, Phase 1). It also provides the classical opening database used in Experiment 3 (opening analysis).

**The pipeline is only needed for imitation learning and opening analysis.** Pure self-play agents (Agent 1) and MCTS (Agent 3) generate their own data through play and do not need this pipeline for training — but they still use the same `encoder` to represent board states.

---

## 2. Data Source

### 2.1 Data source & processing

| Property | Value |
|---|---|
| Processing codebase | `github.com/kaifeiji/xiangqi` — expects PGN/XQF files in `data/raw/` |
| Game sources | dpxq.com game archive, WXF Federation records (the actual games) |
| Referenced inputs | e.g. `dpxq-99813games.pgn`, `WXF-41743games.pgn` (user-supplied) |
| Format | PGN (Roman WXF movetext) and WXF/XQF |
| Content | Professional & semi-professional games; some carry player Elo/results |
| Cost | Free |
| License | Check each source — cite appropriately in thesis |

> **Note (verified 2026-09):**
> 1. The originally-planned `donkeyid/xiangqi-dataset` (claimed ~800k games)
>    does **not exist** (GitHub 404).
> 2. `github.com/kaifeiji/xiangqi` is a **processing codebase, NOT a dataset** —
>    the PGN/XQF game files are not in the repo; you download them separately
>    (from dpxq.com / the WXF Federation) and place them in `data/raw/`.
> 3. This project ships its own loader: `src/data/pgn_adapter.py` reads Xiangqi
>    PGN (Roman WXF movetext) → `src/data/game_loader.py` replay-validates and
>    filters → `src/data/dataset.py` produces training tensors. Chinese-character
>    or ICCS-coordinate movetext would need an additional move parser.
> 4. Fallback sources: Kaggle "onlinexiangqi" (~10k blitz games), and the
>    DpXq / 01xq online databases (scrapeable). Any tens-of-thousands of games
>    is enough for imitation-learning pre-training.

### 2.2 Data Quality Filtering

Not all games are useful. Apply filters:
- **Elo filter:** where player Elo is available in the record, keep only games where both players' Elo ≥ 1800 (strong amateur+). This yields higher-quality moves for imitation. (Note: not all games in this dataset carry Elo metadata — for those, rely on the length/result/validity filters below.)
- **Length filter:** discard games shorter than ~10 moves (likely resignations/errors) or absurdly long (likely data errors).
- **Result filter:** keep only games with a clear result (Red win / Black win / draw). Discard unfinished or corrupt records.
- **Validity filter:** every game must fully replay legally (see section 5). Discard any game that produces an illegal move during replay.

---

## 3. WXF Notation Format

WXF describes each move relative to the moving player. The parser must convert WXF into the internal `(from_row, from_col, to_row, to_col)` form.

### 3.1 Structure of a WXF Move
A WXF move has four components:
1. **Piece character** — which piece is moving (e.g., R=Chariot/车, H=Horse/马, C=Cannon/炮, etc. — note: notation may be Chinese characters or Roman letters depending on the file)
2. **Starting file** — the column the piece starts on, numbered 1–9 from the mover's right side
3. **Direction indicator** — forward (+/进), backward (−/退), or traverse (=/平)
4. **Destination** — the target file (for horizontal moves) or number of steps (for vertical moves)

### 3.2 Perspective Handling (Critical)
- Red counts files from Red's right; Black counts from Black's right. These are mirror images.
- The parser must convert both to the absolute (row, col) board coordinates used internally.
- **Disambiguation:** when two identical pieces are on the same file (e.g., two Chariots), WXF uses front/back prefixes. The parser must resolve which piece moves.

### 3.3 Parser Responsibilities
The `wxf_parser.py` module must:
1. Read a game record file
2. Extract metadata (players, Elo, result, date)
3. Parse each move string into an absolute move
4. Handle both Chinese-character and Roman-letter WXF variants (detect which)
5. Handle disambiguation (front/back pieces on same file)
6. Return an ordered list of moves + metadata

---

## 4. Board State Encoding

The `encoder.py` module converts a board position into the **14×10×9 tensor** described in 03-ENVIRONMENT.md section 2. This is the SAME encoder used by the environment — it must be shared, not duplicated, to guarantee consistency between training data and self-play.

**For imitation learning, each training example is:**
- Input: 14×10×9 board tensor (from the moving player's perspective)
- Label 1 (policy): the action index of the move the human played
- Label 2 (value): the final game outcome from this player's perspective (+1 win, −1 loss, +0.1 or 0 draw)

---

## 5. Game Replay Validation

**This is the most important quality-control step.** For each game:
1. Start from the standard position
2. Apply each parsed move in sequence
3. Verify each move is legal at the time it's played (using `move_generator`)
4. Verify the final board state is consistent with the recorded result
5. If ANY move is illegal or replay fails → discard the entire game and log it

This catches parser bugs and corrupt records. A silent parser bug that produces wrong board tensors would poison the imitation dataset — replay validation prevents this.

---

## 6. From Games to Training Examples

Each game of ~95 moves produces ~95 training examples (one per position-move pair).

```
For each valid game:
  outcome = game result from Red's perspective
  board = starting position
  for each move in game:
    player = whoever is to move
    tensor = encoder.encode(board, from player's perspective)
    label_policy = action_index(move)
    label_value = outcome adjusted to current player's perspective
    store (tensor, label_policy, label_value)
    board = board.apply(move)
```

With ~350k valid games × ~95 moves ≈ **~33 million training examples**. This is large — consider:
- Storing as compressed tensors (e.g., sparse or uint8 encoding, since values are binary)
- Streaming from disk rather than loading all into RAM
- Optionally subsampling (e.g., use every game but cap total examples if compute-limited)

---

## 7. Train / Validation / Test Split

**Split by GAME, not by position.** Positions from the same game must not appear in both train and test (that would leak information).

| Split | Fraction | Purpose |
|---|---|---|
| Train | 80% | Imitation learning training |
| Validation | 10% | Monitor overfitting, early stopping |
| Test | 10% | Final move-prediction accuracy reporting |

- Split randomly at the game level, with a fixed seed for reproducibility.
- Save split indices to `data/splits/` so the split is reproducible.
- **Never train on validation or test games.**

---

## 8. PyTorch Dataset & DataLoader

The `dataset.py` module wraps the processed examples as a PyTorch `Dataset`:
- `__len__` → number of examples
- `__getitem__(idx)` → (board_tensor, policy_label, value_label)
- Support lazy loading from disk for the full 33M examples

The `data_loader.py` module builds `DataLoader`s:
- Configurable batch size (default 512 — see 07-TRAINING.md)
- Shuffle for training, no shuffle for val/test
- Multiple workers for parallel loading
- Optional data augmentation (see section 9)

---

## 9. Data Augmentation (Optional)

Xiangqi has a **horizontal (left-right) symmetry**: mirroring the board across the vertical center produces a valid, equivalent position. This can double the effective dataset.

- Mirror the board tensor horizontally AND mirror the corresponding move label
- **Caution:** the mirror must correctly transform the action index. Test that a mirrored (board, move) pair is still legal and correct.
- This is optional — enable via config and note in thesis if used.

**Note:** Xiangqi does NOT have vertical symmetry (the two sides are asymmetric due to the river and piece setup being mirrored, but the palace/soldier positions make naive vertical flips invalid). Only use horizontal mirroring.

---

## 10. Opening Database (For Experiment 3)

For opening analysis (Experiment 3, see 09-EXPERIMENTS.md), build a reference distribution of opening moves from the professional dataset:
- For each game, extract the first N moves (e.g., first 10 plies)
- Count frequency of each opening sequence
- Identify classical named openings:
  - 当头炮 (Central Cannon) — cannon to center file
  - 顺炮 (Same-Direction Cannons)
  - 列炮 / 逆炮 (Opposite Cannons)
  - 飞相局 (Flying Elephant Opening)
  - 仕角炮 (Advisor's Cannon)
- Store this reference distribution in `data/openings/` for comparison against agent behavior

---

## 11. Pipeline Execution Order

```
STEP 1: Download raw data → data/raw/
STEP 2: Parse WXF → intermediate move-sequence files
STEP 3: Replay-validate every game → discard invalid, log stats
STEP 4: Apply quality filters (Elo, length, result)
STEP 5: Encode positions → board tensors + labels → data/processed/
STEP 6: Create game-level train/val/test split → data/splits/
STEP 7: Build opening reference distribution → data/openings/
```

Run this pipeline ONCE and cache all outputs. Re-running should be deterministic (seeded).

---

## 12. Output Statistics to Record (For Thesis)

Log and report these dataset statistics in the thesis methodology chapter:
- Total games downloaded
- Games passing each filter (Elo, length, result, validity)
- Final game count and total training example count
- Distribution of game outcomes (Red win % / Black win % / draw %)
- Average game length
- Elo distribution of retained games

---

## 13. Testing the Data Pipeline

- [ ] Parser correctly converts known WXF moves to expected coordinates
- [ ] Disambiguation (front/back same-file pieces) handled correctly
- [ ] Replay validation catches deliberately corrupted games
- [ ] Encoder output matches the environment's encoder exactly (shared code)
- [ ] Train/val/test splits share no games
- [ ] Horizontal mirror augmentation produces legal positions
- [ ] Opening reference distribution sums to 1.0 and matches known Xiangqi opening frequencies (sanity check: 当头炮 should be very common)

---

*Cross-references: 01-GAME-RULES.md (WXF notation, rules), 03-ENVIRONMENT.md (shared encoder), 06-AGENTS.md (IL agent consumes this data), 09-EXPERIMENTS.md (opening analysis uses opening database).*
