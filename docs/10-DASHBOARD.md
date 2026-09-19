# 10 — Dashboard Specification

> **Read this when:** building the Streamlit evaluation dashboard — its pages, layout, visualizations, and how it connects to trained agents and results. The dashboard is the demonstration interface for the thesis defense and presentation.

---

## 1. Purpose

The dashboard is the **presentation layer** — the interactive tool shown during the thesis defense and included as a project deliverable. It has two jobs:
1. **Demonstrate** the agents playing live (the "wow factor" for the audience)
2. **Present** the experimental results in an interactive, explorable way

It reads trained agent checkpoints and result files; it does NOT train anything.

---

## 2. Technology

- **Framework:** Streamlit (Python-native — no separate frontend code)
- **Charts:** Plotly (interactive) and Matplotlib (static, for exported figures)
- **Board rendering:** a custom Xiangqi board renderer (reused from `utils/render.py`)
- **Deployment:** Streamlit Community Cloud (free) or run locally during defense

---

## 3. Dashboard Structure

A multi-page Streamlit app. `dashboard/app.py` is the entry point; each page is a module in `dashboard/pages/`; reusable widgets live in `dashboard/components/`.

| Page | File | Purpose |
|---|---|---|
| Home / Overview | `app.py` | Project summary, navigation, key results at a glance |
| 1. Live Game | `pages/01_live_game.py` | Watch any agent play live |
| 2. Elo Comparison | `pages/02_elo_comparison.py` | Compare agent strength |
| 3. Learning Curves | `pages/03_learning_curves.py` | Training progress over time |
| 4. Opening Analysis | `pages/04_opening_analysis.py` | Opening move distributions |
| 5. Game Explorer | `pages/05_game_explorer.py` | Step through recorded games |

---

## 4. Page 1 — Live Game Viewer

### 4.1 Purpose
The centerpiece demo: let the audience watch a trained agent play in real time. This is what makes the presentation memorable.

### 4.2 Features
- **Agent selector:** dropdown to choose which agent plays (PPO / IL+RL / MCTS+NN / Minimax)
- **Opponent selector:** choose the opponent (another agent, an ElephantEye level, or a human clicking moves)
- **Board display:** the Xiangqi board, updated move by move, clearly showing pieces (Chinese characters or clear icons)
- **Play controls:** Start, Pause, Step (one move), Reset, and a speed slider
- **Move log:** running list of moves played (in readable notation)
- **Live info panel:** for neural agents, optionally show the agent's value estimate ("I'm winning: +0.3") and top candidate moves with probabilities

### 4.3 Performance Note
For MCTS, use a modest simulation count (e.g., 200–400) so moves appear reasonably fast during a live demo. Pre-compute or cache if needed to avoid awkward pauses.

### 4.4 Human-vs-AI Mode (Optional, High Impact)
Let an audience member play against the agent by clicking pieces. Highly engaging for a defense. Requires click-to-move interaction on the board.

---

## 5. Page 2 — Elo Comparison

### 5.1 Purpose
Present Experiment 1 results: how the agents rank in strength.

### 5.2 Features
- **Bar chart:** final Elo per agent with 95% confidence interval error bars
- **Table:** win rates of each agent against each ElephantEye level
- **Head-to-head matrix:** interactive heatmap of agent-vs-agent win rates
- **Filter/toggle:** show/hide specific agents or opponent levels

### 5.3 Data Source
Reads the results tables produced by Experiment 1 (`results/tables/`).

---

## 6. Page 3 — Learning Curves

### 6.1 Purpose
Present Experiment 2 results: the central RQ1 finding (IL vs from-scratch convergence).

### 6.2 Features
- **Line chart:** Elo vs training games, with one line per agent (Agent 1 pure PPO vs Agent 2 IL+RL)
- **Interactive:** hover to read exact values, zoom into regions
- **Annotations:** mark where each agent crosses key Elo milestones
- **Table:** games-to-reach-milestone comparison

### 6.3 Data Source
Reads the logged training-time Elo history (from wandb export or saved CSVs).

---

## 7. Page 4 — Opening Analysis

### 7.1 Purpose
Present Experiment 3: whether agents rediscover classical openings.

### 7.2 Features
- **Opening frequency chart:** bar/heatmap of how often each agent plays each classical opening, side by side with the professional reference distribution
- **KL divergence display:** how far each agent's opening style is from professionals
- **First-move heatmap:** visualize where each agent tends to make its first move on the board
- **Named opening callouts:** highlight 当头炮, 顺炮, etc., and note any novel openings discovered by self-play

### 7.3 Data Source
Reads Experiment 3 outputs and the opening reference database.

---

## 8. Page 5 — Game Explorer

### 8.1 Purpose
Let the user step through any recorded game move by move — for analysis and to inspect agent decisions.

### 8.2 Features
- **Game selector:** pick from saved games (e.g., notable wins, specific matchups)
- **Move-by-move stepping:** forward/back through the game with the board updating
- **Per-move info:** for neural agents, show the value estimate and top candidate moves at each position
- **Move list:** clickable notation list to jump to any move

### 8.3 Data Source
Reads saved game records (from evaluation/tournament runs).

---

## 9. Home / Overview Page

The landing page (`app.py`) should:
- State the project title and one-line description
- Show 2–3 headline results (e.g., best agent's Elo, the key RQ1 finding)
- Provide clear navigation to all pages
- Briefly explain what Xiangqi is and what the project studies (for audience members unfamiliar with either)

---

## 10. Reusable Components

Build these once in `dashboard/components/` and reuse across pages:
- **Board renderer:** draws a Xiangqi board given a board state (shared with `utils/render.py`)
- **Move-list widget:** displays and navigates a sequence of moves
- **Elo chart widget:** standardized Elo bar/line chart styling
- **Agent-info panel:** shows an agent's value estimate and top moves

---

## 11. Design Principles

- **Clarity over flashiness:** the board and results must be immediately readable to non-experts (your lecturers may not all know Xiangqi notation)
- **Fast:** avoid long computations on page load; pre-compute and cache results
- **Self-explanatory:** every chart has a title and short caption explaining what it shows
- **Defense-ready:** the live game page must work reliably offline during the presentation (don't depend on a live internet connection if avoidable)

---

## 12. What the Dashboard Does NOT Do

- It does not train agents (training is offline, in Colab)
- It does not run the full expensive experiments (it displays their saved results)
- It does not require a database — it reads from result files and checkpoints

Keeping the dashboard read-only and lightweight makes it robust for the live defense.

---

## 13. Testing the Dashboard

- [ ] Each page loads without error given the expected result files
- [ ] Live game plays a full game to completion for every agent
- [ ] Board renders correctly for arbitrary positions
- [ ] Charts display correctly when result files are present
- [ ] Graceful message shown when a result file is missing (rather than crashing)
- [ ] Works offline (for the defense)

---

*Cross-references: 06-AGENTS.md (agents played live), 08-EVALUATION.md (Elo results shown), 09-EXPERIMENTS.md (experiment results visualized), 02-ARCHITECTURE.md (dashboard layer rules).*
