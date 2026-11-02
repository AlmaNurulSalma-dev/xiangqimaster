# Test-Set Report — Move-Prediction Accuracy (Imitation-Learning Net)

Quantitative evaluation of the imitation-learning network
(`results/checkpoints/il_agent2_phase1.pt`, C=64 / N=10, ~1.15M params) on the
**held-out test split**, analogous to the board-move accuracy tables reported by
recent Xiangqi papers. The metric is how often the network predicts the **human
expert's actual move**.

- **Sample:** 2,000 test games → **163,416 positions** (the test split is fully
  disjoint from train/val at the game level).
- **Metrics:**
  - `top1_unmasked` — argmax over all 2086 actions (the exact metric used during
    training, so it is directly comparable to the ~34% validation number),
  - `top1_masked` / `top3_masked` — after masking illegal moves (how the agent
    actually selects — see `NeuralAgent`).
- **Phases** by ply: opening (<20), midgame (20–59), endgame (≥60).
- **Script:** `scripts/eval_test_move_accuracy.py` · **Data:** `test_move_accuracy.csv`.

## Results

| Phase | Positions | Top-1 (unmasked) | Top-1 (masked) | Top-3 (masked) |
|---|--:|--:|--:|--:|
| **Overall** | 163,416 | **34.3%** | **42.3%** | **66.5%** |
| Opening (ply <20) | 39,998 | 62.2% | 63.2% | 87.8% |
| Midgame (20–59) | 72,425 | 29.5% | 37.7% | 61.8% |
| Endgame (≥60) | 50,993 | 19.1% | 32.3% | 56.5% |

## Findings

1. **The metric checks out.** Overall top-1 (unmasked) = **34.3%**, matching the
   ~34% validation accuracy seen during training — the net generalises to unseen
   test games with no drop, confirming the split is clean and the number is real.
2. **Legal masking lifts it meaningfully:** overall top-1 **42.3%** and top-3
   **66.5%** — i.e. two-thirds of the time the human's move is in the net's top
   three legal choices.
3. **Clear phase pattern — strong opening, hard endgame.** The net reproduces
   human opening theory well (top-1 **63%**, top-3 **88%**) but degrades through
   the midgame (top-1 38%) into the endgame (top-1 32%, top-3 57%).
4. **Same difficulty curve the LLM papers report.** Xiangqi-R1 (Chen *et al.*,
   2025) likewise reports accuracy falling in the midgame/endgame and attributes
   it to limited model capacity — the identical pattern here, from a ~1.15M-param
   convolutional net, is consistent with this project's broader
   *capacity-is-the-bottleneck* finding.

## Relation to the journal metrics
Xiangqi-R1 reports `legal@1`, a 3-class position assessment, and `best@1`
(match to Pikafish's *best* move). The table above is the direct analog of its
**board-move accuracy** but measured against the **human expert's move** rather
than the engine's. A `best@1`-style match-rate against Pikafish can be added as a
follow-up (slower, since it queries the engine per position).

## Reproduce
```bash
C:\Users\kinet\xiangqi-venv\Scripts\python.exe scripts/eval_test_move_accuracy.py \
    --splits data/splits --checkpoint results/checkpoints/il_agent2_phase1.pt \
    --limit-games 2000 --device cuda --out results/test_report
```
Swap `--checkpoint` to evaluate other nets (e.g. `distill_pikafish.pt`,
`distill_big.pt`) for a cross-model comparison.
