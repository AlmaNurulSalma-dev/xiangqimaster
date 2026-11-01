# IEEE Research Proposal — LaTeX Source

Camera-ready 4-page Software Engineering research proposal for **Lab 8**, built
on the `xiangqimaster` project.

**Title:** *Search, Imitation, or Reinforcement? A Controlled, Elo-Anchored
Comparison of Deep Learning Paradigms for Xiangqi under a Constrained Compute
Budget.*

## Files
| File | Purpose |
|---|---|
| `main.tex` | The proposal (IEEEtran `conference` class, two-column). |
| `references.bib` | 10 verified references (BibTeX). |
| `figures/` | Figure assets (the architecture figure is inline TikZ, so this is a placeholder). |
| `final_proposal.pdf` | Compiled output (generated — see below). |

## Author information
Author name, student ID (L25020007), and the UII × NXU dual-degree affiliation
are taken from the project's own `README.md`. Edit the `\author{...}` block in
`main.tex` if any detail needs correcting before submission.

## Compiling
The proposal uses the standard IEEE toolchain. Any of the following works:

```bash
# Option A — classic sequence (TeX Live / MiKTeX)
pdflatex main
bibtex main
pdflatex main
pdflatex main

# Option B — latexmk (handles the passes automatically)
latexmk -pdf main.tex

# Option C — Tectonic (self-contained, fetches packages on first run)
tectonic main.tex
```

Then rename/copy the output to `final_proposal.pdf`.

### Overleaf (no local install needed)
Upload `main.tex` and `references.bib` to a new Overleaf project, set the
compiler to **pdfLaTeX**, and compile. Overleaf already has IEEEtran, TikZ, and
booktabs.

## Dependencies
`IEEEtran` (document class + `IEEEtran.bst`), `cite`, `amsmath`, `amssymb`,
`graphicx`, `booktabs`, `tikz` (libraries `arrows.meta`, `positioning`, `fit`,
`backgrounds`), `hyperref`. All are in a standard TeX Live / MiKTeX / Tectonic
distribution.

## Page-limit note
The content is written to fit the strict 4-page IEEE limit (including
references). If a future edit pushes it over, trim Section IV's preliminary
table or tighten Section II before adding length elsewhere.
