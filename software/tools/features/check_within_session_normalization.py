"""Compare latent-state results with and without within-session normalization.

This script runs the same latent-state pipeline in two modes:
1) baseline: current global scaling pipeline
2) within_session_norm: feature normalization within each group/session before
   global scaling (to reduce static between-group offsets)

Outputs under --out-dir:
- normalization_comparison.tsv
- baseline_state_profiles_zscore.tsv
- within_session_norm_state_profiles_zscore.tsv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _viterbi_smooth(log_probs: np.ndarray, transition_penalty: float = 1.5) -> np.ndarray:
    t_len, n_states = log_probs.shape
    v = np.full((t_len, n_states), -np.inf)
    ptr = np.zeros((t_len, n_states), dtype=int)

    v[0] = log_probs[0] - np.log(n_states)
    trans = np.full((n_states, n_states), -transition_penalty)
    np.fill_diagonal(trans, 0.0)

    for t in range(1, t_len):
        for s in range(n_states):
            scores = v[t - 1] + trans[:, s]
            ptr[t, s] = int(np.argmax(scores))
            v[t, s] = scores[ptr[t, s]] + log_probs[t, s]

    path = np.zeros(t_len, dtype=int)
    path[-1] = int(np.argmax(v[-1]))
    for t in range(t_len - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]
    return path


def _feature_list(windows: pd.DataFrame) -> list[str]:
    physio_features = [
        "group_hr_mean_bpm_mean",
        "group_hr_mean_bpm_std",
        "group_hrv_rmssd_ms_mean",
        "group_eda_tonic_mean_mean",
        "group_eda_tonic_mean_std",
        "group_eda_phasic_rate_hz_mean",
        "group_temp_mean_mean",
    ]
    transcript_features = [
        "tr_spk_duration_s",
        "tr_silence_duration_s",
        "tr_backchannel_count",
        "tr_overlap_count",
        "tr_competitive_overlap",
        "tr_laughter_count",
        "tr_n_active_speakers",
        "tr_speaking_entropy",
        "tr_overlap_time_s",
    ]
    et_features = [
        "group_et_pupil_mean_mean",
        "group_et_blink_rate_per_min_mean",
        "group_et_gaze_dispersion_mean",
    ]
    all_features = physio_features + transcript_features + et_features
    return [c for c in all_features if c in windows.columns]


def _prepare_matrix(
    windows: pd.DataFrame,
    feature_cols: list[str],
    within_session_norm: bool,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    data = windows.dropna(subset=["tr_spk_duration_s"]).copy()
    keep = [c for c in feature_cols if data[c].isna().mean() <= 0.6]

    imputer = SimpleImputer(strategy="mean")
    x_raw = imputer.fit_transform(data[keep])
    work = pd.DataFrame(x_raw, columns=keep, index=data.index)

    if within_session_norm:
        # Normalize each feature within group_id to reduce static group offsets.
        work_norm = work.copy()
        for grp, idx in data.groupby("group_id").groups.items():
            sub = work.loc[idx, keep]
            mu = sub.mean(axis=0)
            sd = sub.std(axis=0).replace(0.0, 1.0)
            work_norm.loc[idx, keep] = (sub - mu) / sd
        work = work_norm

    scaler = StandardScaler()
    x = scaler.fit_transform(work.to_numpy())
    return data, x, keep


def _group_entropy(row: pd.Series) -> float:
    probs = row[row > 0].to_numpy(dtype=float)
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _run_mode(
    windows: pd.DataFrame,
    selfreport: pd.DataFrame,
    mode_name: str,
    within_session_norm: bool,
    transition_penalty: float,
    out_dir: Path,
) -> dict[str, object]:
    feature_cols = _feature_list(windows)
    data, x, keep = _prepare_matrix(windows, feature_cols, within_session_norm)

    pca = PCA(n_components=min(10, len(keep)))
    x_pca = pca.fit_transform(x)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cumvar, 0.80)) + 1
    x_reduced = x_pca[:, :n_pcs]

    k_values = list(range(2, 8))
    sil_scores: list[float] = []
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(x_reduced)
        sil_scores.append(float(silhouette_score(x_reduced, labels)))
    best_k = int(k_values[int(np.argmax(sil_scores))])

    km = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    kmeans_state = km.fit_predict(x_reduced)

    meta = data[["group_id", "task_id", "window_index", "window_start_s"]].reset_index(drop=True)
    meta["kmeans_state"] = kmeans_state

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        n_init=5,
        random_state=42,
        means_init=km.cluster_centers_,
    )
    gmm.fit(x_reduced)

    hmm_labels = np.full(len(meta), -1, dtype=int)
    for grp in sorted(meta["group_id"].unique()):
        for task in sorted(meta["task_id"].unique()):
            mask = (meta["group_id"] == grp) & (meta["task_id"] == task)
            idx = meta[mask].sort_values("window_index").index
            if len(idx) < 2:
                hmm_labels[idx] = gmm.predict(x_reduced[idx])
                continue
            log_prob = np.log(gmm.predict_proba(x_reduced[idx]) + 1e-12)
            hmm_labels[idx] = _viterbi_smooth(log_prob, transition_penalty)

    meta["hmm_state"] = hmm_labels
    ari = float(adjusted_rand_score(meta["kmeans_state"], meta["hmm_state"]))

    # Between-group concentration check.
    ctab_group = pd.crosstab(meta["group_id"], meta["kmeans_state"], normalize="index")
    mean_max_state_share = float(ctab_group.max(axis=1).mean())
    mean_group_state_entropy = float(ctab_group.apply(_group_entropy, axis=1).mean())

    # State profile table (for interpretation).
    feat_df = pd.DataFrame(x, columns=keep)
    feat_df["kmeans_state"] = meta["kmeans_state"].values
    profiles = feat_df.groupby("kmeans_state")[keep].mean()
    profiles.to_csv(out_dir / f"{mode_name}_state_profiles_zscore.tsv", sep="\t")

    # Self-report association summary (directional check).
    state_props = (
        meta.groupby(["group_id", "task_id", "hmm_state"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda row: row / row.sum(), axis=1)
        .reset_index()
    )
    state_props.columns = ["group_id", "task_id"] + [f"state_{s}_prop" for s in range(best_k)]
    val_df = pd.merge(state_props, selfreport, on=["group_id", "task_id"], how="inner")

    sr_items = [
        "vad_valence_mean",
        "vad_arousal_mean",
        "engagement_mean",
        "voice_inclusion_mean",
        "satisfaction_mean",
        "mental_demand_mean",
        "team_coordination_mean",
        "psych_safety_mean",
        "decision_confidence_mean",
        "perceived_control_mean",
    ]
    sr_items = [c for c in sr_items if c in val_df.columns]

    rhos: list[float] = []
    pvals: list[float] = []
    for state_col in [f"state_{s}_prop" for s in range(best_k)]:
        for sr_col in sr_items:
            pair = val_df[[state_col, sr_col]].dropna()
            if len(pair) < 4:
                continue
            rho, p_val = spearmanr(pair[state_col], pair[sr_col])
            if not math.isnan(rho) and not math.isnan(p_val):
                rhos.append(abs(float(rho)))
                pvals.append(float(p_val))

    return {
        "mode": mode_name,
        "n_windows": len(meta),
        "n_features": len(keep),
        "kmeans_best_k": best_k,
        "pca_components_for_80pct": n_pcs,
        "mean_silhouette": float(np.max(sil_scores)),
        "kmeans_hmm_ari": ari,
        "mean_max_state_share_by_group": mean_max_state_share,
        "mean_group_state_entropy_bits": mean_group_state_entropy,
        "selfreport_assoc_mean_abs_rho": float(np.mean(rhos)) if rhos else np.nan,
        "selfreport_assoc_max_abs_rho": float(np.max(rhos)) if rhos else np.nan,
        "selfreport_assoc_n_p_lt_0_10": int(sum(p < 0.10 for p in pvals)),
    }


def run_analysis(
    window_file: Path,
    selfreport_file: Path,
    out_dir: Path,
    transition_penalty: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(window_file, sep="\t")
    selfreport = pd.read_csv(selfreport_file, sep="\t")

    rows = []
    rows.append(
        _run_mode(
            windows=windows,
            selfreport=selfreport,
            mode_name="baseline",
            within_session_norm=False,
            transition_penalty=transition_penalty,
            out_dir=out_dir,
        )
    )
    rows.append(
        _run_mode(
            windows=windows,
            selfreport=selfreport,
            mode_name="within_session_norm",
            within_session_norm=True,
            transition_penalty=transition_penalty,
            out_dir=out_dir,
        )
    )

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "normalization_comparison.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-file",
        type=Path,
        default=Path("features_after") / "collective_window_features.tsv",
        help="Window-level feature file.",
    )
    parser.add_argument(
        "--selfreport-file",
        type=Path,
        default=Path("features_after") / "collective_task_selfreport.tsv",
        help="Task-level self-report file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "latent_states_normalization_check",
        help="Output directory.",
    )
    parser.add_argument(
        "--transition-penalty",
        type=float,
        default=1.5,
        help="Viterbi transition penalty.",
    )
    args = parser.parse_args()

    run_analysis(
        window_file=args.window_file.resolve(),
        selfreport_file=args.selfreport_file.resolve(),
        out_dir=args.out_dir.resolve(),
        transition_penalty=args.transition_penalty,
    )


if __name__ == "__main__":
    main()
