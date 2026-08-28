"""Group-level rich eye-tracking features (blink rate, gaze dispersion, gaze
velocity) recovered from the existing paper HMM matrix — no raw Tobii files
needed.

`tools/features/extract_eyetracking_features.py` (the richer per-participant
gaze/blink extractor) cannot be run in this workspace because the raw
`et/*_acq-P*_tobii.tsv.gz` files are not present here. However, its group-level
aggregate (mean/std across P1-P4) was already computed for the paper's HMM
analysis and survives in `icmi_paper/results/enhanced_features_final.tsv`,
covering all 10 groups (grp-07..grp-16), tasks T1-T3, ~85% window coverage.

This script pulls just the `group_et_*` columns (and identifying/attendance
columns) out of that matrix and writes them as a standalone group-level ET
feature table, plus a task-level rollup (mean across windows per group x task).

This is GROUP-level data only (already averaged across P1-P4 upstream) — it
cannot be disaggregated back to individual participants. True per-participant
blink rate / gaze dispersion / gaze velocity would require re-running
`tools/features/extract_eyetracking_features.py` against the original raw ET
files (not present in this workspace).

Outputs (written to analysis/results/):
    group_eyetracking_features_window_30s.tsv  — one row per group x task x 30s window
    group_eyetracking_features_task.tsv         — one row per group x task (mean across windows)

Usage:
    python analysis/extract_group_eyetracking_features.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "icmi_paper" / "results" / "enhanced_features_final.tsv"
OUT_WINDOW_PATH = REPO_ROOT / "analysis" / "results" / "group_eyetracking_features_window_30s.tsv"
OUT_TASK_PATH = REPO_ROOT / "analysis" / "results" / "group_eyetracking_features_task.tsv"

ID_COLS: list[str] = ["group_id", "task_id", "window_index", "window_start_s", "n_participants"]


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SRC_PATH}")

    matrix = pd.read_csv(SRC_PATH, sep="\t")
    logger.info("Loaded %d window rows from %s", len(matrix), SRC_PATH.name)

    et_cols = [c for c in matrix.columns if c.startswith("group_et_")]
    if not et_cols:
        raise ValueError(f"No group_et_* columns found in {SRC_PATH.name}")

    window_out = matrix[ID_COLS + et_cols].copy()
    window_out = window_out.sort_values(["group_id", "task_id", "window_index"]).reset_index(drop=True)

    OUT_WINDOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    window_out.to_csv(OUT_WINDOW_PATH, sep="\t", index=False)
    logger.info(
        "Wrote %d window rows (%d groups) -> %s",
        len(window_out),
        window_out["group_id"].nunique(),
        OUT_WINDOW_PATH,
    )

    task_out = (
        window_out.groupby(["group_id", "task_id"])[et_cols]
        .mean()
        .reset_index()
    )
    n_windows = (
        window_out.groupby(["group_id", "task_id"])
        .size()
        .reset_index(name="n_windows")
    )
    task_out = task_out.merge(n_windows, on=["group_id", "task_id"])
    task_out = task_out.sort_values(["group_id", "task_id"]).reset_index(drop=True)

    task_out.to_csv(OUT_TASK_PATH, sep="\t", index=False)
    logger.info(
        "Wrote %d task rows (%d groups) -> %s",
        len(task_out),
        task_out["group_id"].nunique(),
        OUT_TASK_PATH,
    )


if __name__ == "__main__":
    main()
