from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).parent / "results"
df = pd.read_csv(RESULTS_DIR / "participant_model_df.tsv", sep="\t")

FEATS = [
    "tr_speaking_time_s", "tr_n_turns", "tr_backchannel_count", "tr_laughter_count",
    "tr_overlap_count", "tr_overlap_time_s",
    "audio_energy_mean", "audio_pitch_mean", "audio_pitch_sd", "audio_hnr_mean",
    "hr_mean_bpm", "hrv_rmssd_ms", "eda_phasic_rate_hz", "temp_mean",
    "pupil_mean", "pupil_std", "pupil_slope_per_s",
]
Y = "perceived_effectiveness_z"

print("--- Univariate Spearman corr with target (pairwise-complete) ---")
rows = []
for f in FEATS:
    sub = df[[f, Y]].dropna()
    if len(sub) < 10:
        continue
    rho, p = spearmanr(sub[f], sub[Y])
    rows.append((f, len(sub), rho, p))
res = pd.DataFrame(rows, columns=["feature", "n", "rho", "p"]).sort_values("p")
print(res.to_string(index=False))

print("\n--- Target variance decomposition ---")
total_var = df[Y].var()
between_group = df.groupby("group_id")[Y].mean().var()
between_task = df.groupby("task")[Y].mean().var()
between_participant = df.groupby("participant")[Y].mean().var()
between_group_task = df.groupby(["group_id", "task"])[Y].mean().var()
print(f"total var: {total_var:.3f}")
print(f"between-group var (share): {between_group:.3f} ({100*between_group/total_var:.1f}%)")
print(f"between-task var (share): {between_task:.3f} ({100*between_task/total_var:.1f}%)")
print(f"between-participant(P1-4) var (share): {between_participant:.3f} ({100*between_participant/total_var:.1f}%)")
print(f"between-group*task var (share): {between_group_task:.3f} ({100*between_group_task/total_var:.1f}%)")

print("\n--- Within-group-task spread of target across P1-P4 (residual individual variation) ---")
within = df.groupby(["group_id", "task"])[Y].std()
print(f"mean within-group-task std across participants: {within.mean():.3f} (vs total std {df[Y].std():.3f})")
