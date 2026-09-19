# 11 — Coding Standards & Conventions

> **Read this when:** writing ANY code for this project. These conventions keep the codebase consistent, reproducible, and easy for both humans and AI agents to navigate. Following them prevents ambiguity and bugs.

---

## 1. General Principles

1. **Clarity over cleverness** — readable code beats clever code. This is a research/thesis codebase that must be understood and defended.
2. **Reproducibility is mandatory** — every result must be regenerable. Seed everything, log everything.
3. **One responsibility per module** — each file does one clear thing (see 02-ARCHITECTURE.md).
4. **No magic numbers** — every tunable value lives in config, not hardcoded inline.
5. **Fail loudly** — when something is wrong (illegal move, shape mismatch), raise a clear error rather than silently continuing.

---

## 2. Language & Style

- **Python 3.10+**
- Follow **PEP 8** style
- Use an autoformatter: **Black** (line length 88 or 100 — pick one, apply everywhere)
- Use **isort** for import ordering
- Use a linter: **ruff** or **flake8**
- Use **type hints** on all function signatures — this is critical for clarity and catches bugs

---

## 3. Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Modules / files | lowercase_with_underscores | `move_generator.py` |
| Classes | PascalCase | `PolicyValueNetwork` |
| Functions / methods | lowercase_with_underscores | `select_move` |
| Variables | lowercase_with_underscores | `legal_mask` |
| Constants | UPPERCASE_WITH_UNDERSCORES | `ACTION_SPACE_SIZE` |
| Private helpers | leading underscore | `_compute_advantage` |

**Domain-specific naming (use consistently everywhere):**
- Board coordinates: always `(row, col)`, 0-indexed
- Players: `RED` and `BLACK` (constants), never "player1/player2"
- The state tensor: `board_tensor` (shape 14×10×9)
- A move as an integer: `action_index` or `action`
- A move as coordinates: `(from_row, from_col, to_row, to_col)`
- The legal move mask: `legal_mask` (boolean array of size 2086)

---

## 4. Configuration Convention

- **All hyperparameters and constants live in config**, never hardcoded in logic.
- Global constants (board size, action space size, piece values, channel count) → `src/utils/config.py`
- Experiment-specific settings (learning rate, games, MCTS sims) → `configs/<experiment>.yaml`
- Training/experiment scripts load a config at startup and log the full config.
- **Rule:** if a reviewer might ask "why this number?", it belongs in config where it's visible and adjustable — not buried in code.

---

## 5. Reproducibility Requirements

Every training or experiment run MUST:
1. **Set all random seeds:** Python `random`, NumPy, PyTorch (CPU and CUDA), and the environment. Provide a single `set_seed(seed)` utility in `utils/seeding.py` that seeds all of them.
2. **Log the seed** used.
3. **Log the full config** (all hyperparameters).
4. **Log the git commit hash** so the exact code version is known.
5. **Use deterministic data splits** (04-DATA-PIPELINE.md).

Note: full determinism with GPU + PyTorch can require extra flags (deterministic algorithms) and may cost speed — document the chosen level of determinism.

---

## 6. Docstrings & Comments

- **Every module** starts with a docstring: what it does and which doc(s) specify it (e.g., "Implements the environment per docs/03-ENVIRONMENT.md").
- **Every public function/class** has a docstring stating: purpose, arguments (with shapes for tensors), return value (with shape), and any important side effects.
- **For tensors, always document the shape** in the docstring or an inline comment, e.g., `# board_tensor: (batch, 14, 10, 9)`.
- Comments explain WHY, not WHAT (the code shows what; comments explain reasoning and non-obvious decisions).
- Reference the relevant doc when implementing a spec, e.g., `# PUCT formula, see docs/06-AGENTS.md section 5.3`.

---

## 7. Tensor Shape Discipline

Shape bugs are the most common deep-learning error. Prevent them:
- Document expected input/output shapes in every function that handles tensors.
- Add shape assertions at function boundaries during development, e.g., assert the board tensor is (batch, 14, 10, 9).
- Never silently reshape — be explicit about permutations and flattens.
- Keep a consistent convention: channels-first (C, H, W) = (14, 10, 9), matching PyTorch Conv2D.

---

## 8. Error Handling

- **Illegal moves:** the environment/agents must never silently accept an illegal move. Raise a clear exception (this should never happen if masking is correct — treat it as a bug).
- **Shape mismatches:** assert early with a descriptive message.
- **Missing files/checkpoints:** fail with a clear message telling the user what's missing and where it was expected.
- **Never use bare `except:`** — catch specific exceptions.

---

## 9. Testing Requirements

- Every core module in `environment/`, `data/`, `models/`, and `agents/` has tests in `tests/`.
- **Environment tests are the highest priority** — a bug in the rules corrupts everything.
- Test categories:
  - **Unit tests:** individual functions (a piece's moves, the encoder, the Elo formula)
  - **Integration tests:** full games play to completion, a training step reduces loss
  - **Sanity tests:** a strong agent beats a weak one, IL reduces validation loss
- Run tests before committing. A failing environment test blocks everything downstream.
- Each doc's "Testing" checklist section lists the specific tests for that component.

---

## 10. Version Control (Git)

- **Commit often** with clear messages describing WHAT changed and WHY.
- **Never commit** large files: raw data, processed tensors, model checkpoints, wandb logs. Put them in `.gitignore`.
- Use Google Drive / external storage for large artifacts.
- Tag or note the commit hash used for each reported experiment result.
- Keep the `main` branch working; use branches for experimental changes.

---

## 11. Logging

- Use Python's `logging` module (not `print`) for program logs, via a shared setup in `utils/logging_utils.py`.
- Use **wandb** for experiment metrics (losses, Elo, win rates) — see 07-TRAINING.md.
- Log enough that a run can be understood after the fact without re-running it.

---

## 12. Dependency Management

- All dependencies pinned in `requirements.txt` (with versions) and/or `environment.yml`.
- Document any non-pip dependency (like the ElephantEye binary) with install instructions.
- Keep dependencies minimal — every added library is a maintenance and reproducibility cost.

---

## 13. Working With Claude Code (Practical Notes)

Since Claude Code will write much of this codebase:
- **Point it to the relevant doc** before a task (e.g., "implement the encoder per docs/03-ENVIRONMENT.md and docs/04-DATA-PIPELINE.md").
- **Ask it to write tests alongside code**, using the testing checklists in each doc.
- **Have it document shapes and reference the spec doc** in docstrings.
- **Review generated code against the spec** — the human (you) owns correctness and must understand every module for the defense.
- **Keep modules small** so each fits comfortably in context and has one clear job.

---

## 14. The Non-Negotiables (Quick Checklist)

Before considering any code "done":
- [ ] Type-hinted and docstringed
- [ ] Tensor shapes documented
- [ ] No hardcoded magic numbers (they're in config)
- [ ] Seeds set for any randomness
- [ ] Tests written and passing
- [ ] References the relevant spec doc
- [ ] Follows the module dependency rules (02-ARCHITECTURE.md)
- [ ] Large artifacts gitignored

---

*Cross-references: 02-ARCHITECTURE.md (module structure & dependency rules), and every other doc's "Testing" checklist section.*
