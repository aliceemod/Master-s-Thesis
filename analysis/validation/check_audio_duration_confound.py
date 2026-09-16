"""Check whether participant-level audio prosody features (energy, pitch, pitch SD, HNR)
are confounded by how long a group's discussion phase lasted (`discussion_duration_s`,
from `analysis/results/session_timing/task_timing.tsv`).

Two checks, per audio feature:
    1. Direct correlation with discussion_duration_s (is the feature itself duration-dependent?).
    2. Correlation with each self-report target, both raw and after residualizing the audio
       feature on discussion_duration_s (does controlling for duration change the target
       correlation?).

Context: user asked (2026-09-02) whether audio features might depend on how long a group
took in a task, given that discussion-only window filtering was already applied. This
script is the reproducible check for that question — see docs/feature_catalog.md and
docs/session_timing_audit.md for the discussion-window fix itself.

Output: analysis/results/audio_duration_confound_check.tsv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = REPO_ROOT / "analysis" / "results" / "participant_audio_features.tsv"
TIMING_PATH = REPO_ROOT / "analysis" / "results" / "session_timing" / "task_timing.tsv"
SELFREPORT_PATH = (
    REPO_ROOT / "analysis" / "results" / "collective_features_v20260812" / "collective_task_selfreport.tsv"
)
OUT_PATH = REPO_ROOT / "analysis" / "results" / "audio_duration_confound_check.tsv"

AUDIO_FEATURES = ["audio_energy_mean", "audio_pitch_mean", "audio_pitch_sd", "audio_hnr_mean"]
TARGETS = [
    "cooperative_mean",
    "team_coordination_mean",
    "voice_inclusion_mean",
    "mental_demand_mean",
    "idea_quality_mean",
]
MIN_N = 8


def main() -> None:
    audio = pd.read_csv(AUDIO_PATH, sep="\t")
    audio_g = (
        audio.groupby(["group_id", "task"])[AUDIO_FEATURES].mean().reset_index().rename(columns={"task": "task_id"})
    )

    timing = pd.read_csv(TIMING_PATH, sep="\t")
    duration = timing[timing["task"].isin(["T1", "T2", "T3"])][
        ["group_id", "task", "discussion_duration_s"]
    ].rename(columns={"task": "task_id"})

    selfreport = pd.read_csv(SELFREPORT_PATH, sep="\t")

    df = audio_g.merge(duration, on=["group_id", "task_id"]).merge(
        selfreport, on=["group_id", "task_id"], how="left"
    )

    print(f"N group-task rows with audio + duration: {len(df)}")
    print("\n=== Audio feature vs. discussion_duration_s (direct confound check) ===")
    direct_rows = []
    for feat in AUDIO_FEATURES:
        sub = df[[feat, "discussion_duration_s"]].dropna()
        if len(sub) < MIN_N:
            continue
        r, p = stats.pearsonr(sub[feat], sub["discussion_duration_s"])
        direct_rows.append({"audio_feature": feat, "target": "discussion_duration_s", "r_raw": r, "p_raw": p,
                             "r_resid": np.nan, "p_resid": np.nan, "n": len(sub)})
        print(f"  {feat:22s} r={r:+.3f} p={p:.3f} n={len(sub)}")

    print("\n=== Audio-target correlation: raw vs. duration-residualized ===")
    target_rows = []
    for feat in AUDIO_FEATURES:
        for target in TARGETS:
            sub = df[[feat, "discussion_duration_s", target]].dropna()
            if len(sub) < MIN_N:
                continue
            r_raw, p_raw = stats.pearsonr(sub[feat], sub[target])
            slope, intercept = np.polyfit(sub["discussion_duration_s"], sub[feat], 1)
            resid = sub[feat] - (slope * sub["discussion_duration_s"] + intercept)
            r_resid, p_resid = stats.pearsonr(resid, sub[target])
            target_rows.append({"audio_feature": feat, "target": target, "r_raw": r_raw, "p_raw": p_raw,
                                 "r_resid": r_resid, "p_resid": p_resid, "n": len(sub)})

    out = pd.DataFrame(direct_rows + target_rows)
    print(out.round(3).to_string(index=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
