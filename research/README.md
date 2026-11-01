# Research — Lab 8 Capstone Proposal Deliverables

This directory holds the **Lab 8: Capstone Proposal Submission & Peer Defense**
deliverables for the XiangqiMaster project. It is kept separate from the
application code; nothing in `src/`, `experiments/`, or `dashboard/` was modified
for this assignment.

## Research direction
**A controlled, Elo-anchored comparison of four deep-learning paradigms for
Xiangqi under a constrained compute budget** — framed as a multi-agent
evaluation league (classical alpha-beta search vs pure PPO self-play vs
imitation-then-PPO vs AlphaZero-style MCTS), on one shared environment, action
space, network, and Elo scale.

## Contents
| File / dir | Deliverable |
|---|---|
| `01_project_audit.md` | Evidence-based audit of the whole codebase (what exists, verified against source + results + git). |
| `02_literature_review.md` | Literature survey, comparison table, and the classified research gap. All 10 references verified. |
| `03_research_framework.md` | Title, background, problem, RQs, objectives, hypotheses, methodology, metrics, contributions. |
| `proposal/` | The IEEE two-column LaTeX proposal and its compiled PDF. |
| `proposal/main.tex` | Proposal source (IEEEtran `conference`). |
| `proposal/references.bib` | 10 verified BibTeX references. |
| `proposal/final_proposal.pdf` | **Camera-ready compiled PDF** (4 pp., within the 4-page limit). |
| `proposal/README.md` | How to compile (pdfLaTeX / latexmk / Tectonic / Overleaf). |
| `04_elevator_pitch.md` | 3-minute pitch: script, 6-slide outline, timing, hooks. |
| `05_technical_defense.md` | 20+ Q&A (categories A–E) + 5 challenging questions + 5 weaknesses + 2-min strategy. |
| `06_peer_review_template.md` | Reusable ICSE-style peer-review template. |
| `07_quality_audit.md` | Senior quality audit: fixed defects, flagged items, and the pre-submission action list. |
| `peer_reviews/` | Where the two completed peer reviews go (stubs await classmates' proposals + the official rubric). |

## Status checklist
- [x] Phase 1 — Project audit (`01_project_audit.md`)
- [x] Phase 2 — Research direction selected (controlled comparative study)
- [x] Phase 3 — Literature review + research gap (`02_literature_review.md`)
- [x] Phase 4 — Research framework (`03_research_framework.md`)
- [x] Phase 5 — IEEE proposal written **and compiled to PDF** (`proposal/`)
- [x] Phase 6 — Elevator pitch (`04_elevator_pitch.md`)
- [x] Phase 7 — Technical defense (`05_technical_defense.md`)
- [x] Phase 8 — Peer-review template (`06_peer_review_template.md`)
- [x] Quality audit — fixed defects + flagged items (`07_quality_audit.md`)
- [ ] Peer reviews — completed after classmates' proposals + official rubric are provided
- [ ] Slides — build from `04_elevator_pitch.md` (6 slides)

## Integrity notes
- Every claim about the codebase is traceable to a source file or result table.
- Every academic reference was verified (arXiv/DOI); none fabricated.
- Preliminary Elo numbers are the project's **actual** results, presented in the
  proposal as *feasibility/preliminary* evidence — never as final findings.
- No peer reviews are fabricated; only the template is provided.
