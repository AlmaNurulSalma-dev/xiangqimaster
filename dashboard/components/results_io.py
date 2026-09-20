"""Read experiment result CSVs for the dashboard (docs/10-DASHBOARD.md).

The dashboard is read-only: it displays saved results. These loaders return the
rows of a results CSV, or ``None`` when the file is missing, so pages can show a
graceful "run the experiment first" message instead of crashing (docs/10 13).
"""

from __future__ import annotations

import csv
import os

Rows = list[dict[str, str]]


def load_table(path: str) -> Rows | None:
    """Load a CSV as a list of row dicts, or ``None`` if the file is absent."""
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def default_results_dir() -> str:
    """Repo ``results/tables`` directory (relative to this file)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "results", "tables")


__all__ = ["Rows", "load_table", "default_results_dir"]
