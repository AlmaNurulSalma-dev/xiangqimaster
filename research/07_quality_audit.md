# 07 — Senior Quality Audit of the Lab 8 Deliverables

> An independent, critical review of every research deliverable, written in the
> stance of a senior reviewer. Each finding has a **severity**
> (Critical / Major / Minor), a **recommendation**, and a **status**
> (Fixed / Flagged / Verify-with-author). "Fixed" items were corrected in the
> current revision; "Flagged" items are left for the author's decision; the
> final section is the prioritised pre-submission action list.

**Audit date:** 2026-11-01 · **Scope:** `research/01`–`06`, `research/README.md`,
`proposal/` (LaTeX + PDF).

---

## A. Defects corrected in this revision (Fixed)

| # | Finding | Severity | What was done |
|---|---|:--:|---|
| A1 | **Wrong affiliation** — "NXU" was written as *Ningxia University*; it is **Nanjing Xiaozhuang University (南京晓庄学院)**. | **Critical** | Corrected in `main.tex`, `01`, `03`, `proposal/README.md`. A factual error on the title page is the worst kind — fixed first. |
| A2 | **UII / "Universitas Islam Indonesia" / "Dual-Degree Programme"** present on the title page and in docs, against the author's instruction. | Major | Removed from `main.tex` author block and from `01`, `03`, `proposal/README.md`. |
| A3 | **Missing Chinese name.** | Major | Added **诺艾玛** beside the romanised name (`main.tex`, `01`, `README`). Required a CJK font (see B6). |
| A4 | **Title too long** (ran ~4 title lines). | Minor | Replaced with a 2-line title: *"A Controlled Comparison of Deep-Learning Paradigms for Xiangqi under a Constrained Compute Budget."* |
| A5 | **Under length** — the paper was 3 pp.; the brief asks for 4. | Major | Expanded to a full **4 pp.** with substance, not filler: an explicit RQ list, a RQ→objective→experiment→metric traceability table, an "imitation ceiling" related-work paragraph, a compute-budget definition, and a decision-guidance paragraph. |
| A6 | **Build instructions were wrong.** `main.tex`'s header comment and `README` said compile with **pdfLaTeX**, but `fontspec` (and now CJK) **require XeLaTeX**. pdfLaTeX would fail. | Major | Corrected the in-file comment and rewrote the README toolchain section to specify XeLaTeX / LuaLaTeX / Tectonic. |
| A7 | **Stale page count** — `research/README.md` still advertised "3 pp." | Minor | Updated to "4 pp." |
| A8 | **Unfalsifiable superlatives** — "the first controlled… comparison" appeared 4×. | Minor | Softened: one hedged "to our knowledge the first" in the abstract; the other three now read "a controlled…". |

---

## B. Issues flagged for your decision

**Implementation status (updated 2026-11-01):** **B1** and **B3** are now
**applied** to the proposal (see the notes on each below). B2 and B7 the audit
recommends leaving as they are; B6 is already documented in `proposal/README.md`;
B4 and B5 are thesis-stage or scope notes, not proposal-text edits. Nothing
further is therefore pending on the proposal itself.

**B1 — Preliminary Elo table uses the 5-iteration MCTS (1703).** *(Minor.)*
Your own `results/AGENT3_MCTS_FINDING.md` records that this early checkpoint has a
wide CI and that a later, *stabilised* run (MCTS v2) measured **1691**. Both are
defensible; 1703 is the documented headline. **Recommendation:** keep 1703 (the
table is explicitly labelled "preliminary, CI-wide"), but be ready in the defence
to explain the 5-iter vs v2 story — reviewers may probe it. Optionally switch the
table to the v2 number (1691) as the more conservative figure. **Update —
applied:** the table caption now cites the stabilised rerun (MCTS ≈1691) beside
the 1703 headline, so both are visible to the reader.

**B2 — `xiangqir1` (LLM/GRPO) is the weakest citation.** *(Minor.)*
It is tangential to a four-*paradigm* comparison (it is a reasoning-LLM paper).
It currently does useful work showing the field's breadth and that each Xiangqi
paper builds one system. **Recommendation:** keep it, but it is the first thing to
cut if you ever need space or tighter focus.

**B3 — "11.6M positions" vs the 9.3M training split.** *(Minor.)*
The abstract says preliminary runs span "11.6M positions"; 11.6M is the *whole*
dataset, while the *training* split is 9.3M (the manifest's figures). Not wrong,
but loose. **Recommendation:** either say "≈11.6M positions (9.3M in the training
split)" or just "9.3M training positions" for precision. **Update — applied:**
the abstract now reads "a 9.3M-position training split from 141k human games."

**B4 — Reproducibility seam (for the eventual thesis, not this proposal).**
*(Major for the thesis; N/A for Lab 8.)* Per the audit (`01` §10, §13), several
headline evaluations were produced by throwaway scratchpad scripts rather than
committed `experiments/` modules, and `data/raw/` is empty by default.
**Recommendation:** before the thesis write-up, migrate the key eval into
`experiments/` so every table regenerates from a committed script. The proposal
already lists this honestly as a limitation.

**B5 — The `01`/`03` planning docs are supporting material, not required
submission artifacts.** *(Minor / scope.)* The lab's graded deliverables are the
**4-page PDF**, **two peer reviews**, and the pitch/defence. `01`–`03` and `07`
are excellent scaffolding but are not what the rubric scores. **Recommendation:**
keep them in `research/` (they show rigour and traceability) but do not expect
them to be marked; lead your submission with `final_proposal.pdf`.

**B6 — `main.tex` is no longer self-contained for recompilation.** *(Minor.)*
It now needs XeLaTeX and a 24 MB CJK font (`NotoSerifSC.ttf`), which is
git-ignored. The committed **PDF embeds the glyphs**, so the deliverable is fine,
but a fresh `tectonic main.tex` from the repo will fail without the font.
**Recommendation (documented in `proposal/README.md`):** on Overleaf, set XeLaTeX
and load the CJK font by name (`\newfontfamily\cjkfont{Noto Serif CJK SC}`); or
keep a `NotoSerifSC.ttf` next to `main.tex` locally.

**B7 — Cross-pool Elo is not directly comparable.** *(Minor.)*
The preliminary table is one round-robin pool; your other pools (7-agent exp1
with IL at 1689; the v2 pool; the Pikafish-anchored pool) use different fields, so
their Elo numbers are not directly comparable across tables. **Recommendation:**
when you present multiple tables later, add a one-line note that relative Elo is
only comparable *within* a pool. (The proposal avoids this trap by showing a
single pool.)

---

## C. Verify with the author (facts I could not confirm)

- **C1 — Romanised full name.** "Alma Nurul Salma" was inferred from your email
  and GitHub handle, not stated explicitly. **Confirm the exact spelling/order**
  as it should appear on the paper.
- **C2 — Chinese name 诺艾玛** — confirm this is the exact, complete form you want
  on the title page.
- **C3 — Student ID `L25020007`** — taken from the project `README.md`; confirm it
  is current and correct.

---

## D. Checked and found sound (no action needed)

- **Citations:** all 10 references verified against arXiv/DOI; authors, years, and
  venues correct; the one unverifiable detail (corpus size in Yilmaz et al.) is
  not asserted as fact.
- **No fabrication:** no invented results, references, or peer reviews; the
  preliminary Elo numbers are the project's real measurements, labelled as such.
- **Argument alignment:** Problem → Gap → RQ → Objective → Method → Evaluation →
  Contribution is complete and consistent; every RQ has a matching experiment and
  metric (now shown explicitly in the traceability table).
- **Formatting:** IEEEtran two-column, Times body, **exactly 4 pages**, no text
  overflow (Overfull boxes resolved), the gap table fits its column, the TikZ
  figure and the Elo equation render, and all `\cite` references resolve.
- **CJK rendering:** no "Missing character" warnings — all eight Chinese glyphs
  render from the embedded Noto Serif SC.

---

## E. Prioritised pre-submission action list

1. **Confirm C1–C3** (name spelling, Chinese name, student ID) — 2 minutes, and
   the only remaining *factual* risk on the title page.
2. **Decide B1** (keep 1703 or switch to the conservative 1691) and rehearse the
   answer either way for the defence.
3. **Optional B3** wording tweak for precision in the abstract.
4. **Do the peer reviews** once you have the two classmates' proposals and the
   official ICSE rubric (template in `06`).
5. **Build the slides** from `04` (6-slide outline) for the 3-minute pitch.
6. **(Thesis-stage, not Lab 8) B4** — migrate eval scripts into `experiments/`.

**Overall verdict.** After this revision the proposal is **submission-ready**: the
one critical factual error (affiliation) is fixed, the author identity is correct
and complete, it meets the 4-page IEEE requirement cleanly, and its claims are
honest and evidence-backed. The remaining items are either author confirmations
(C1–C3) or optional refinements (B1–B3, B7); none blocks submission.
