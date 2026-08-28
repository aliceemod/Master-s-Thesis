"""Per-participant audio/prosodic features aggregated to task level from the existing
window-level per-participant GeMAPS table (`icmi_paper/results/_audio_window_partial.tsv`).

Uses the same curated 4-feature mapping as `icmi_paper/analysis/audio_window_features.ipynb`
(the group-level version) — just aggregated per participant instead of averaged across P1-P4.

Output: analysis/results/participant_audio_features.tsv
    columns: group_id, task, participant, audio_energy_mean, audio_pitch_mean,
             audio_pitch_sd, audio_hnr_mean, n_windows
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "icmi_paper" / "results" / "_audio_window_partial.tsv"
OUT_PATH = REPO_ROOT / "analysis" / "results" / "participant_audio_features.tsv"

GEMAP_MAP = {
    "gemap_loudness_sma3_amean": "audio_energy_mean",
    "gemap_F0semitoneFrom27.5Hz_sma3nz_amean": "audio_pitch_mean",
    "gemap_F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "audio_pitch_sd",
    "gemap_HNRdBACF_sma3nz_amean": "audio_hnr_mean",
}


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SRC_PATH}")

    windows = pd.read_csv(SRC_PATH, sep="\t")
    logger.info("Loaded %d participant-window rows from %s", len(windows), SRC_PATH.name)

    windows = windows.rename(columns=GEMAP_MAP)
    result = (
        windows.groupby(["group_id", "task_id", "participant_id"])[list(GEMAP_MAP.values())]
        .mean()
        .reset_index()
        .rename(columns={"task_id": "task", "participant_id": "participant"})
    )
    n_windows = (
        windows.groupby(["group_id", "task_id", "participant_id"])
        .size()
        .reset_index(name="n_windows")
        .rename(columns={"task_id": "task", "participant_id": "participant"})
    )
    result = result.merge(n_windows, on=["group_id", "task", "participant"])
    result = result.sort_values(["group_id", "task", "participant"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, sep="\t", index=False)
    logger.info("Wrote %d rows (%d groups) -> %s", len(result), result["group_id"].nunique(), OUT_PATH)


if __name__ == "__main__":
    main()
