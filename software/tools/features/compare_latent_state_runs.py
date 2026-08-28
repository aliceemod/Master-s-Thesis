"""Compare latent-state runs with and without task-level VAD features.

Produces side-by-side metrics and figures so results are directly comparable.

Outputs (under --out-dir):
- run_comparison_metrics.tsv
- run_comparison_task_state_entropy.tsv
- run_comparison_tension_by_task.tsv
- compare_run_metrics.png
- compare_state_entropy_by_task.png
- compare_tension_by_task.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def _prepare_features(
    windows: pd.DataFrame,
    selfreport: pd.DataFrame,
    include_vad: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    physio = [
        "group_hr_mean_bpm_mean",
        "group_hr_mean_bpm_std",
        "group_hrv_rmssd_ms_mean",
        "group_eda_tonic_mean_mean",
        "group_eda_tonic_mean_std",
        "group_eda_phasic_rate_hz_mean",
        "group_temp_mean_mean",
    ]
    transcript = [
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
    et = [
        "group_et_pupil_mean_mean",
        "group_et_blink_rate_per_min_mean",
        "group_et_gaze_dispersion_mean",
    ]

    data = windows.dropna(subset=["tr_spk_duration_s"]).copy()

    if include_vad:
        vad_map = {
            "vad_valence_mean": "task_vad_valence_mean",
            "vad_arousal_mean": "task_vad_arousal_mean",
            "vad_dominance_mean": "task_vad_dominance_mean",
        }
        sr = selfreport[["group_id", "task_id"] + [k for k in vad_map if k in selfreport.columns]].copy()
        for src, dst in vad_map.items():
            if src in sr.columns:
                sr[dst] = pd.to_numeric(sr[src], errors="coerce")
        keep_cols = ["group_id", "task_id"] + [v for v in vad_map.values() if v in sr.columns]
        data = pd.merge(data, sr[keep_cols], on=["group_id", "task_id"], how="left")
        vad = ["task_vad_valence_mean", "task_vad_arousal_mean", "task_vad_dominance_mean"]
    else:
        vad = []

    all_features = [c for c in (physio + transcript + et + vad) if c in data.columns]
    all_features = [c for c in all_features if data[c].isna().mean() <= 0.6]

    x_raw = SimpleImputer(strategy="mean").fit_transform(data[all_features])
    x = StandardScaler().fit_transform(x_raw)
    return data, x_raw, x, all_features


def _run_once(
    windows: pd.DataFrame,
    selfreport: pd.DataFrame,
    include_vad: bool,
    transition_penalty: float,
) -> dict[str, object]:
    data, x_raw, x, features = _prepare_features(windows, selfreport, include_vad)

    pca = PCA(n_components=min(10, len(features)))
    x_pca = pca.fit_transform(x)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cumvar, 0.80)) + 1
    x_reduced = x_pca[:, :n_pcs]

    k_values = list(range(2, 8))
    sil = []
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(x_reduced)
        sil.append(float(silhouette_score(x_reduced, labels)))
    best_k = int(k_values[int(np.argmax(sil))])

    km = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    kmeans_labels = km.fit_predict(x_reduced)

    meta = data[["group_id", "task_id", "window_index", "window_start_s"]].reset_index(drop=True)
    meta["kmeans_state"] = kmeans_labels

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

    # Task-level state entropy (normalized by log(K))
    ent_rows = []
    for task, sub in meta.groupby("task_id"):
        p = sub["hmm_state"].value_counts(normalize=True).values
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        entropy_norm = entropy / np.log(best_k)
        ent_rows.append({"task_id": task, "state_entropy_norm": entropy_norm})
    entropy_df = pd.DataFrame(ent_rows)

    # Tension index by task
    tension_cols = [c for c in ["tr_competitive_overlap", "tr_overlap_count", "tr_overlap_time_s"] if c in data.columns]
    z = pd.DataFrame(index=data.index)
    for col in tension_cols:
        vals = pd.to_numeric(data[col], errors="coerce")
        z[col] = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-12)
    data = data.copy()
    data["tension_index"] = z.mean(axis=1)
    tension_df = (
        data.groupby("task_id", as_index=False)["tension_index"]
        .agg(tension_mean="mean", tension_std="std")
    )

    return {
        "meta": meta,
        "n_windows": len(meta),
        "n_features": len(features),
        "best_k": best_k,
        "n_pcs_80": n_pcs,
        "ari": ari,
        "silhouette_best": max(sil),
        "entropy_df": entropy_df,
        "tension_df": tension_df,
    }


def run_compare(repo_root: Path, out_dir: Path, transition_penalty: float) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(repo_root / "features" / "collective_window_features.tsv", sep="\t")
    selfreport = pd.read_csv(repo_root / "features" / "collective_task_selfreport.tsv", sep="\t")

    no_vad = _run_once(windows, selfreport, include_vad=False, transition_penalty=transition_penalty)
    with_vad = _run_once(windows, selfreport, include_vad=True, transition_penalty=transition_penalty)

    # Summary metrics table
    metrics = pd.DataFrame(
        [
            {
                "run": "no_vad",
                "n_windows": no_vad["n_windows"],
                "n_features": no_vad["n_features"],
                "best_k": no_vad["best_k"],
                "n_pcs_80": no_vad["n_pcs_80"],
                "kmeans_hmm_ari": no_vad["ari"],
                "best_silhouette": no_vad["silhouette_best"],
            },
            {
                "run": "with_vad",
                "n_windows": with_vad["n_windows"],
                "n_features": with_vad["n_features"],
                "best_k": with_vad["best_k"],
                "n_pcs_80": with_vad["n_pcs_80"],
                "kmeans_hmm_ari": with_vad["ari"],
                "best_silhouette": with_vad["silhouette_best"],
            },
        ]
    )
    metrics.to_csv(out_dir / "run_comparison_metrics.tsv", sep="\t", index=False)

    # Task entropy comparison
    ent = pd.concat(
        [
            no_vad["entropy_df"].assign(run="no_vad"),
            with_vad["entropy_df"].assign(run="with_vad"),
        ],
        ignore_index=True,
    )
    ent.to_csv(out_dir / "run_comparison_task_state_entropy.tsv", sep="\t", index=False)

    # Tension comparison
    tens = pd.concat(
        [
            no_vad["tension_df"].assign(run="no_vad"),
            with_vad["tension_df"].assign(run="with_vad"),
        ],
        ignore_index=True,
    )
    tens.to_csv(out_dir / "run_comparison_tension_by_task.tsv", sep="\t", index=False)

    # Figure 1: metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.barplot(data=metrics, x="run", y="best_k", ax=axes[0])
    axes[0].set_title("Best k")
    axes[0].set_xlabel("")

    sns.barplot(data=metrics, x="run", y="kmeans_hmm_ari", ax=axes[1])
    axes[1].set_title("KMeans-HMM ARI")
    axes[1].set_xlabel("")

    sns.barplot(data=metrics, x="run", y="best_silhouette", ax=axes[2])
    axes[2].set_title("Best Silhouette")
    axes[2].set_xlabel("")

    fig.suptitle("Latent Run Comparison: No VAD vs With VAD", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_run_metrics.png", dpi=220)
    plt.close(fig)

    # Figure 2: task entropy
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=ent, x="task_id", y="state_entropy_norm", hue="run", ax=ax)
    ax.set_title("Task-wise State Entropy (Normalized)")
    ax.set_ylabel("entropy / log(K)")
    ax.set_xlabel("task")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_state_entropy_by_task.png", dpi=220)
    plt.close(fig)

    # Figure 3: tension by task
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=tens, x="task_id", y="tension_mean", hue="run", ax=axes[0])
    axes[0].set_title("Tension Index Mean by Task")
    axes[0].set_xlabel("task")

    sns.barplot(data=tens, x="task_id", y="tension_std", hue="run", ax=axes[1])
    axes[1].set_title("Tension Index Variability by Task")
    axes[1].set_xlabel("task")

    fig.tight_layout()
    fig.savefig(out_dir / "compare_tension_by_task.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "latent_states_compare",
    )
    parser.add_argument("--transition-penalty", type=float, default=1.5)
    args = parser.parse_args()

    run_compare(args.repo_root.resolve(), args.out_dir.resolve(), args.transition_penalty)


if __name__ == "__main__":
    main()
