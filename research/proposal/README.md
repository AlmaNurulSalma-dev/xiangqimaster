# IEEE Research Proposal — LaTeX Source

Camera-ready 4-page Software Engineering research proposal for **Lab 8**, built
on the `xiangqimaster` project.

**Title:** *Comparing Deep-Learning Paradigms for Xiangqi (象棋): A Controlled
Study under a Constrained Compute Budget.*

## Files
| File | Purpose |
|---|---|
| `main.tex` | The proposal (IEEEtran `conference` class, two-column, **XeLaTeX**). |
| `references.bib` | 10 verified references (BibTeX). |
| `figures/` | Figure assets (the architecture figure is inline TikZ, so this is a placeholder). |
| `final_proposal.pdf` | Compiled output — **4 pages**. |
| `NotoSerifSC.ttf` | CJK font for the Chinese name/affiliation (local, git-ignored — see below). |

## Author information
The `\author{...}` block in `main.tex` reads **Alma Nurul Salma (诺艾玛)**,
student ID **L25020007**, **Nanjing Xiaozhuang University (南京晓庄学院)**. Edit
that block if any detail needs correcting before submission.

## Toolchain — XeLaTeX required
The proposal uses `fontspec` (Times via TeX Gyre Termes, plus a CJK font for the
Chinese text), so it must be compiled with **XeLaTeX** (or LuaLaTeX / Tectonic) —
**not** pdfLaTeX.

```bash
# Option A — XeLaTeX sequence
xelatex main
bibtex main
xelatex main
xelatex main

# Option B — latexmk
latexmk -xelatex main.tex

# Option C — Tectonic (self-contained; what this PDF was built with)
tectonic main.tex
```

Then copy the output to `final_proposal.pdf`.

## CJK font
`main.tex` loads the Chinese font by relative path:
`\newfontfamily\cjkfont{NotoSerifSC}[Path=./, Extension=.ttf]`, expecting
`NotoSerifSC.ttf` (OFL Noto Serif SC, ~24 MB) in this folder. That file is
**git-ignored** (too large to commit); the camera-ready glyphs are already
embedded in `final_proposal.pdf`, so the PDF itself needs nothing extra.

To recompile elsewhere:
- **Locally:** drop any `NotoSerifSC.ttf` here, or edit the `\newfontfamily` line
  to point at any installed CJK font.
- **Overleaf:** set the compiler to **XeLaTeX** and change the `\newfontfamily`
  line to load a CJK font by name, e.g.
  `\newfontfamily\cjkfont{Noto Serif CJK SC}` (Overleaf provides it), or upload a
  `.ttf` and keep the `Path=./` form.

## Dependencies
`IEEEtran` (+ `IEEEtran.bst`), `fontspec` (needs XeLaTeX), `cite`, `amsmath`,
`amssymb`, `graphicx`, `booktabs`, `tikz` (libraries `arrows.meta`,
`positioning`, `fit`, `backgrounds`), `hyperref`, plus TeX Gyre Termes and a Noto
Serif SC font. All ship with a standard TeX Live / Tectonic distribution (CJK
font as noted above).

## Page-limit note
The proposal fills the strict **4-page** IEEE limit (including references). If a
future edit pushes it to a 5th page, trim the preliminary-results table (Sec. IV)
or a Related-Work paragraph before adding length elsewhere.
