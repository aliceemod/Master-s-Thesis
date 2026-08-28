"""Export latent-state interpretation artefacts and figures.

This script reproduces the current latent collective-state analysis for groups
with transcript annotations (grp-07, grp-10, grp-14, grp-15) and tasks T1-T3,
then saves publication-ready figures and summary tables.

Outputs (under --out-dir):
- pca_scree.png
- pca_scatter_by_group.png
- pca_scatter_by_task.png
- pca_scatter_by_kmeans_state.png
- pca_scatter_by_hmm_state.png
- kmeans_silhouette.png
- kmeans_state_profiles_heatmap.png
- kmeans_distribution_by_group.png
- kmeans_distribution_by_task.png
- kmeans_trajectories_grid.png
- hmm_distribution_by_task.png
- hmm_trajectories_grid.png
- task_tension_variability_boxplots.png
- selfreport_state_correlations.png
- dominance_scatter.png
- state_profiles_zscore.tsv
- run_summary.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _viterbi_smooth(log_probs: np.ndarray, transition_penalty: float = 1.5) -> np.ndarray:
    """Decode the most likely state path using a simple Viterbi pass."""
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


def _load_and_prepare(
    window_file: Path,
    selfreport_file: Path,
    include_vad_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load input files and build scaled matrices for clustering."""
    windows = pd.read_csv(window_file, sep="\t")
    selfreport = pd.read_csv(selfreport_file, sep="\t")

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

    vad_features = [
        "task_vad_valence_mean",
        "task_vad_arousal_mean",
        "task_vad_dominance_mean",
    ]

    if include_vad_features:
        vad_map = {
            "vad_valence_mean": "task_vad_valence_mean",
            "vad_arousal_mean": "task_vad_arousal_mean",
            "vad_dominance_mean": "task_vad_dominance_mean",
        }
        sr_cols = ["group_id", "task_id"] + [c for c in vad_map if c in selfreport.columns]
        sr_vad = selfreport[sr_cols].copy()
        for src, dst in vad_map.items():
            if src in sr_vad.columns:
                sr_vad[dst] = pd.to_numeric(sr_vad[src], errors="coerce")
        keep_cols = ["group_id", "task_id"] + [c for c in vad_map.values() if c in sr_vad.columns]
        windows = pd.merge(windows, sr_vad[keep_cols], on=["group_id", "task_id"], how="left")

    all_features = [
        col for col in (physio_features + transcript_features + et_features) if col in windows.columns
    ]
    if include_vad_features:
        all_features.extend([col for col in vad_features if col in windows.columns])

    data = windows.dropna(subset=["tr_spk_duration_s"]).copy()
    keep = [col for col in all_features if data[col].isna().mean() <= 0.6]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_raw = imputer.fit_transform(data[keep])
    x = scaler.fit_transform(x_raw)

    meta = data[["group_id", "task_id", "window_index", "window_start_s"]].reset_index(drop=True)
    return data, selfreport, meta, x_raw, x, keep


def run_export(
    repo_root: Path,
    out_dir: Path,
    transition_penalty: float = 1.5,
    include_vad_features: bool = False,
) -> None:
    """Run analysis and export figures + tables."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_file = repo_root / "features" / "collective_window_features.tsv"
    selfreport_file = repo_root / "features" / "collective_task_selfreport.tsv"
    transcript_participant_file = repo_root / "features" / "transcript_participant_task.tsv"

    data, selfreport, meta, x_raw, x, features = _load_and_prepare(
        window_file,
        selfreport_file,
        include_vad_features,
    )

    # PCA
    pca = PCA(n_components=min(10, len(features)))
    x_pca = pca.fit_transform(x)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cumvar, 0.80)) + 1
    x_reduced = x_pca[:, :n_pcs]

    # PCA figures
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(cumvar) + 1), pca.explained_variance_ratio_, color="#4c78a8", alpha=0.8)
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="#e45756", label="Cumulative")
    ax.axhline(0.80, ls="--", color="gray", lw=1, label="80%")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained Variance")
    ax.set_title("PCA Scree Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scree.png", dpi=200)
    plt.close(fig)

    group_colors = {"grp-07": "#1f77b4", "grp-10": "#ff7f0e", "grp-14": "#2ca02c", "grp-15": "#d62728"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for grp, sub in meta.groupby("group_id"):
        idx = sub.index
        ax.scatter(
            x_pca[idx, 0],
            x_pca[idx, 1],
            c=group_colors.get(grp, "gray"),
            alpha=0.55,
            s=28,
            label=grp,
            edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Scatter by Group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scatter_by_group.png", dpi=200)
    plt.close(fig)

    task_colors = {"T1": "#4e9af1", "T2": "#f1a94e", "T3": "#7bc950"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for task, sub in meta.groupby("task_id"):
        idx = sub.index
        ax.scatter(
            x_pca[idx, 0],
            x_pca[idx, 1],
            c=task_colors.get(task, "gray"),
            alpha=0.55,
            s=28,
            label=task,
            edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Scatter by Task")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scatter_by_task.png", dpi=200)
    plt.close(fig)

    # KMeans model selection
    k_values = list(range(2, 8))
    sil_scores: list[float] = []
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(x_reduced)
        sil_scores.append(float(silhouette_score(x_reduced, labels)))

    best_k = int(k_values[int(np.argmax(sil_scores))])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, sil_scores, "o-", color="#e45756")
    ax.axvline(best_k, ls="--", color="gray", label=f"Best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_title("KMeans Model Selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_silhouette.png", dpi=200)
    plt.close(fig)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    meta = meta.copy()
    meta["kmeans_state"] = km.fit_predict(x_reduced)
    groups = sorted(meta["group_id"].unique())
    tasks = sorted(meta["task_id"].unique())

    # PCA scatter colored by discovered KMeans state.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.scatterplot(
        x=x_pca[:, 0],
        y=x_pca[:, 1],
        hue=meta["kmeans_state"].astype(str),
        palette="tab10",
        alpha=0.72,
        s=34,
        edgecolor="white",
        linewidth=0.25,
        ax=ax,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Scatter by KMeans State")
    ax.legend(title="KMeans state", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scatter_by_kmeans_state.png", dpi=220)
    plt.close(fig)

    # KMeans profiles
    feat_df = pd.DataFrame(x, columns=features)
    feat_df["kmeans_state"] = meta["kmeans_state"].values
    profiles = feat_df.groupby("kmeans_state")[features].mean()
    profiles.to_csv(out_dir / "state_profiles_zscore.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.65), best_k * 0.75 + 2))
    sns.heatmap(
        profiles,
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Z-score"},
        ax=ax,
    )
    ax.set_title("KMeans State Profiles")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_state_profiles_heatmap.png", dpi=220)
    plt.close(fig)

    # Distributions
    state_palette = sns.color_palette("tab10", best_k)
    ctab_group = pd.crosstab(meta["group_id"], meta["kmeans_state"], normalize="index")
    fig, ax = plt.subplots(figsize=(7, 4))
    ctab_group.plot(kind="bar", stacked=True, color=state_palette, edgecolor="white", ax=ax)
    ax.set_ylabel("Proportion")
    ax.set_title("KMeans State Distribution by Group")
    ax.legend(title="State", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_distribution_by_group.png", dpi=200)
    plt.close(fig)

    ctab_task = pd.crosstab(meta["task_id"], meta["kmeans_state"], normalize="index")
    fig, ax = plt.subplots(figsize=(7, 4))
    ctab_task.plot(kind="bar", stacked=True, color=state_palette, edgecolor="white", ax=ax)
    ax.set_ylabel("Proportion")
    ax.set_title("KMeans State Distribution by Task")
    ax.legend(title="State", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_distribution_by_task.png", dpi=200)
    plt.close(fig)

    # Task-level variability diagnostics for tension-related features
    tension_candidates = [
        "tr_competitive_overlap",
        "tr_overlap_count",
        "tr_overlap_time_s",
        "group_eda_tonic_mean_std",
        "group_eda_phasic_rate_hz_mean",
    ]
    tension_cols = [col for col in tension_candidates if col in data.columns]
    if tension_cols:
        long_df = data[["task_id"] + tension_cols].copy().melt(
            id_vars=["task_id"],
            value_vars=tension_cols,
            var_name="feature",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(max(10, len(tension_cols) * 2.2), 5))
        sns.boxplot(data=long_df, x="feature", y="value", hue="task_id", ax=ax)
        ax.set_title("Task-Level Variability on Tension-Relevant Features")
        ax.set_xlabel("")
        ax.set_ylabel("value")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "task_tension_variability_boxplots.png", dpi=220)
        plt.close(fig)

    # KMeans trajectories (group x task grid)
    fig, axes = plt.subplots(
        len(groups),
        len(tasks),
        figsize=(4.6 * len(tasks), 2.5 * len(groups)),
        sharey=True,
    )
    if len(groups) == 1 and len(tasks) == 1:
        axes = np.array([[axes]])
    elif len(groups) == 1:
        axes = np.array([axes])
    elif len(tasks) == 1:
        axes = np.array([[ax] for ax in axes])

    for r, grp in enumerate(groups):
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            subset = meta[(meta["group_id"] == grp) & (meta["task_id"] == task)].sort_values(
                "window_index"
            )
            if subset.empty:
                ax.set_visible(False)
                continue
            t_min = subset["window_start_s"].to_numpy() / 60.0
            s = subset["kmeans_state"].to_numpy()
            ax.step(t_min, s, where="post", color="#2f2f2f", lw=1.25)
            ax.scatter(
                t_min,
                s,
                c=[state_palette[int(v)] for v in s],
                s=28,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            ax.set_ylim(-0.5, best_k - 0.5)
            ax.set_yticks(range(best_k))
            if c == 0:
                ax.set_ylabel(f"{grp}\nstate")
            if r == 0:
                ax.set_title(task)
            if r == len(groups) - 1:
                ax.set_xlabel("time (min)")

    fig.suptitle("KMeans State Trajectories by Group and Task", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_trajectories_grid.png", dpi=220)
    plt.close(fig)

    # HMM-like smoothing via GMM + Viterbi
    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        n_init=5,
        random_state=42,
        means_init=km.cluster_centers_,
    )
    gmm.fit(x_reduced)

    hmm_labels = np.full(len(meta), -1, dtype=int)
    for grp in groups:
        for task in tasks:
            mask = (meta["group_id"] == grp) & (meta["task_id"] == task)
            idx = meta[mask].sort_values("window_index").index
            if len(idx) < 2:
                hmm_labels[idx] = gmm.predict(x_reduced[idx])
                continue
            log_prob = np.log(gmm.predict_proba(x_reduced[idx]) + 1e-12)
            hmm_labels[idx] = _viterbi_smooth(log_prob, transition_penalty)

    meta["hmm_state"] = hmm_labels
    ari = float(adjusted_rand_score(meta["kmeans_state"], meta["hmm_state"]))

    # PCA scatter colored by HMM-smoothed state.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.scatterplot(
        x=x_pca[:, 0],
        y=x_pca[:, 1],
        hue=meta["hmm_state"].astype(str),
        palette="tab10",
        alpha=0.72,
        s=34,
        edgecolor="white",
        linewidth=0.25,
        ax=ax,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Scatter by HMM State")
    ax.legend(title="HMM state", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_scatter_by_hmm_state.png", dpi=220)
    plt.close(fig)

    ctab_hmm_task = pd.crosstab(meta["task_id"], meta["hmm_state"], normalize="index")
    fig, ax = plt.subplots(figsize=(7, 4))
    ctab_hmm_task.plot(kind="bar", stacked=True, color=state_palette, edgecolor="white", ax=ax)
    ax.set_ylabel("Proportion")
    ax.set_title("HMM State Distribution by Task")
    ax.legend(title="State", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "hmm_distribution_by_task.png", dpi=200)
    plt.close(fig)

    # HMM trajectories (group x task grid)
    fig, axes = plt.subplots(
        len(groups),
        len(tasks),
        figsize=(4.6 * len(tasks), 2.5 * len(groups)),
        sharey=True,
    )
    if len(groups) == 1 and len(tasks) == 1:
        axes = np.array([[axes]])
    elif len(groups) == 1:
        axes = np.array([axes])
    elif len(tasks) == 1:
        axes = np.array([[ax] for ax in axes])

    for r, grp in enumerate(groups):
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            subset = meta[(meta["group_id"] == grp) & (meta["task_id"] == task)].sort_values(
                "window_index"
            )
            if subset.empty:
                ax.set_visible(False)
                continue
            t_min = subset["window_start_s"].to_numpy() / 60.0
            s = subset["hmm_state"].to_numpy()
            ax.step(t_min, s, where="post", color="#2f2f2f", lw=1.25)
            ax.scatter(
                t_min,
                s,
                c=[state_palette[int(v)] for v in s],
                s=28,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            ax.set_ylim(-0.5, best_k - 0.5)
            ax.set_yticks(range(best_k))
            if c == 0:
                ax.set_ylabel(f"{grp}\nstate")
            if r == 0:
                ax.set_title(task)
            if r == len(groups) - 1:
                ax.set_xlabel("time (min)")

    fig.suptitle("HMM State Trajectories by Group and Task", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "hmm_trajectories_grid.png", dpi=220)
    plt.close(fig)

    # Self-report correlation heatmap
    state_props = (
        meta.groupby(["group_id", "task_id", "hmm_state"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda row: row / row.sum(), axis=1)
        .reset_index()
    )
    state_props.columns = ["group_id", "task_id"] + [f"state_{s}_prop" for s in range(best_k)]
    val_df = pd.merge(state_props, selfreport, on=["group_id", "task_id"], how="inner")

    state_cols = [f"state_{s}_prop" for s in range(best_k)]
    selfreport_items = [
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
    selfreport_items = [c for c in selfreport_items if c in val_df.columns]

    corr_rows: list[dict[str, object]] = []
    for state_col in state_cols:
        for sr_col in selfreport_items:
            pair = val_df[[state_col, sr_col]].dropna()
            if len(pair) < 4:
                continue
            rho, p_val = spearmanr(pair[state_col], pair[sr_col])
            corr_rows.append(
                {
                    "state_prop": state_col,
                    "self_report": sr_col,
                    "rho": float(rho),
                    "p": float(p_val),
                }
            )

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        pivot = corr_df.pivot(index="self_report", columns="state_prop", values="rho")
        p_pivot = corr_df.pivot(index="self_report", columns="state_prop", values="p")
        annot = pivot.round(2).astype(str).where(p_pivot >= 0.10, pivot.round(2).astype(str) + "*")

        fig, ax = plt.subplots(figsize=(best_k * 1.7 + 2, len(selfreport_items) * 0.65 + 2))
        sns.heatmap(
            pivot,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=annot,
            fmt="",
            linewidths=0.5,
            cbar_kws={"label": "Spearman rho"},
            ax=ax,
        )
        ax.set_title("State Proportion vs Self-Report (* p < .10)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "selfreport_state_correlations.png", dpi=220)
        plt.close(fig)

    # Dominance scatter
    transcript_pt = pd.read_csv(transcript_participant_file, sep="\t")
    transcript_pt = transcript_pt[transcript_pt["task_id"].isin(["T1", "T2", "T3"])].copy()

    dom_cols = [c for c in selfreport.columns if c.startswith("perceived_dominance_")]
    if dom_cols:
        perc_long = selfreport[["group_id", "task_id"] + dom_cols].melt(
            id_vars=["group_id", "task_id"],
            value_vars=dom_cols,
            var_name="participant_id",
            value_name="perceived_dominance",
        )
        perc_long["participant_id"] = perc_long["participant_id"].str.replace(
            "perceived_dominance_", "", regex=False
        )

        merged = pd.merge(
            transcript_pt[["group_id", "task_id", "participant_id", "speaking_share"]],
            perc_long,
            on=["group_id", "task_id", "participant_id"],
            how="inner",
        ).dropna()

        if len(merged) >= 4:
            rho, p_val = spearmanr(merged["speaking_share"], merged["perceived_dominance"])
            fig, ax = plt.subplots(figsize=(6, 5))
            for pid, sub in merged.groupby("participant_id"):
                ax.scatter(
                    sub["speaking_share"],
                    sub["perceived_dominance"],
                    alpha=0.75,
                    s=58,
                    label=pid,
                )
            ax.set_xlabel("Behavioural Speaking Share")
            ax.set_ylabel("Perceived Dominance")
            ax.set_title(f"Dominance Alignment (rho={rho:.2f}, p={p_val:.3f})")
            ax.legend(title="Participant")
            fig.tight_layout()
            fig.savefig(out_dir / "dominance_scatter.png", dpi=220)
            plt.close(fig)
        else:
            rho, p_val = np.nan, np.nan
    else:
        rho, p_val = np.nan, np.nan

    # Summary table
    summary = pd.DataFrame(
        {
            "metric": [
                "n_windows",
                "n_features",
                "kmeans_best_k",
                "pca_components_for_80pct",
                "kmeans_hmm_ari",
                "include_vad_features",
                "dominance_rho",
                "dominance_p",
            ],
            "value": [
                len(meta),
                len(features),
                best_k,
                n_pcs,
                ari,
                int(include_vad_features),
                float(rho) if not np.isnan(rho) else np.nan,
                float(p_val) if not np.isnan(p_val) else np.nan,
            ],
        }
    )
    summary.to_csv(out_dir / "run_summary.tsv", sep="\t", index=False)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing features/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "latent_states",
        help="Output directory for figures and summary tables.",
    )
    parser.add_argument(
        "--transition-penalty",
        type=float,
        default=1.5,
        help="Viterbi transition penalty for HMM smoothing.",
    )
    parser.add_argument(
        "--include-vad-features",
        action="store_true",
        help=(
            "Include task-level VAD means (valence/arousal/dominance) as clustering features. "
            "These values are constant within each task for a group."
        ),
    )
    args = parser.parse_args()

    run_export(
        args.repo_root.resolve(),
        args.out_dir.resolve(),
        args.transition_penalty,
        args.include_vad_features,
    )


if __name__ == "__main__":
    main()
