"""Regenerate three stale/broken figures with correct feature names and k=4 state assignments.

Figures produced:
  - figures/feature_importance_f_statistics.png  (F-stats, correct feature names)
  - figures/clusters_pca_updated_overlaps.png    (PCA scatter, k=4 states)
  - figures/state_persistence_heatmap_per_task.png (task x state prevalence, categorical)
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent.parent
FEAT_FILE = BASE / "results" / "hmm_input_features_final.tsv"
ASGN_FILE = BASE / "results" / "hmm_cluster_assignments_k5_updated_overlaps.tsv"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
CLEAN_FEATS = [
    "group_hr_mean_bpm_mean",
    "group_eda_phasic_rate_hz_mean",
    "group_temp_mean_mean",
    "group_et_pupil_mean_mean",
    "tr_silence_duration_s",
    "tr_backchannel_count",
    "tr_laughter_count",
    "tr_n_active_speakers",
    "tr_ovl_count",
    "tr_ovl_contested",
    "tr_ovl_time_s",
    "tr_ovl_collaboration_index",
]

PHYSIO_FEATS = [
    "group_hr_mean_bpm_mean",
    "group_eda_phasic_rate_hz_mean",
    "group_temp_mean_mean",
    "group_et_pupil_mean_mean",
]

LOG1P_FEATS = [
    "tr_silence_duration_s",
    "tr_backchannel_count",
    "tr_laughter_count",
    "tr_ovl_count",
    "tr_ovl_contested",
    "tr_ovl_time_s",
    "tr_ovl_collaboration_index",
]

DISPLAY_NAMES = {
    "group_hr_mean_bpm_mean":         "HR (z)",
    "group_eda_phasic_rate_hz_mean":  "EDA phasic (z)",
    "group_temp_mean_mean":           "Temp (z)",
    "group_et_pupil_mean_mean":       "Pupil diam (z)",
    "tr_silence_duration_s":          "Silence [log]",
    "tr_backchannel_count":           "Backchannels [log]",
    "tr_laughter_count":              "Laughter [log]",
    "tr_n_active_speakers":           "Active speakers",
    "tr_ovl_count":                   "Overlap count [log]",
    "tr_ovl_contested":               "Contested OVL [log]",
    "tr_ovl_time_s":                  "Overlap time [log]",
    "tr_ovl_collaboration_index":     "Collab index [log]",
}

STATE_NAMES = {
    0: "Transitional",
    1: "Active/Floor-Contested",
    2: "Aroused-Engaged",
    3: "Positive Social",
}

STATE_COLORS = {
    0: "#7f7f7f",   # gray
    1: "#d62728",   # red
    2: "#1f77b4",   # blue
    3: "#2ca02c",   # green
}


# ── load and preprocess ────────────────────────────────────────────────────────
def load_data():
    feats = pd.read_csv(FEAT_FILE, sep="\t")
    asgn = pd.read_csv(ASGN_FILE, sep="\t")

    # within-group z-score for physio
    for col in PHYSIO_FEATS:
        feats[col] = feats.groupby("group_id")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    # log1p for count/duration features
    for col in LOG1P_FEATS:
        feats[col] = np.log1p(feats[col])

    # merge with state assignments (order-preserving)
    asgn = asgn.reset_index(drop=True)
    feats = feats.reset_index(drop=True)
    df = feats[["group_id", "task_id"] + CLEAN_FEATS].copy()
    df["state"] = asgn["hmm_state"].values
    return df


# ── Figure 1: Feature importance (F-statistics) ───────────────────────────────
def plot_feature_importance(df):
    groups = [df.loc[df["state"] == s, CLEAN_FEATS].values for s in sorted(df["state"].unique())]

    f_stats, p_vals = [], []
    for j, col in enumerate(CLEAN_FEATS):
        col_groups = [g[:, j] for g in groups]
        f, p = stats.f_oneway(*col_groups)
        f_stats.append(f)
        p_vals.append(p)

    result = pd.DataFrame({
        "feature": CLEAN_FEATS,
        "display": [DISPLAY_NAMES[f] for f in CLEAN_FEATS],
        "F": f_stats,
        "p": p_vals,
    }).sort_values("F", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#c0392b" if p < 0.001 else "#e67e22" if p < 0.01 else "#3498db"
              for p in result["p"]]
    bars = ax.barh(result["display"], result["F"], color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("F-statistic (one-way ANOVA across 4 states)", fontsize=10)
    ax.set_title("Feature Discriminability Across HMM States (k=4)", fontsize=11, fontweight="bold")
    ax.axvline(x=10, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    legend_handles = [
        mpatches.Patch(color="#c0392b", label="p < 0.001"),
        mpatches.Patch(color="#e67e22", label="p < 0.01"),
        mpatches.Patch(color="#3498db", label="p < 0.05"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "feature_importance_f_statistics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: PCA scatter (k=4 states) ────────────────────────────────────────
def plot_pca_scatter(df):
    X = df[CLEAN_FEATS].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for s in sorted(df["state"].unique()):
        mask = df["state"].values == s
        ax.scatter(
            pcs[mask, 0], pcs[mask, 1],
            c=STATE_COLORS[s], label=f"S{s}: {STATE_NAMES[s]}",
            alpha=0.55, s=35, edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=10)
    ax.set_title("PCA Projection of 526 Windows — Coloured by HMM State (k=4)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, markerscale=1.4, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG_DIR / "clusters_pca_updated_overlaps.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: State prevalence by task (categorical heatmap) ──────────────────
def plot_state_task_heatmap(df):
    # Compute % windows in each state per task
    task_order = ["T1", "T2", "T3"]
    task_labels = {"T1": "T1: Decision", "T2": "T2: Negotiation", "T3": "T3: Ideation"}
    state_order = [0, 1, 2, 3]

    pct_data = {}
    for task in task_order:
        sub = df[df["task_id"] == task]
        total = len(sub)
        pct_data[task] = {s: (sub["state"] == s).sum() / total * 100 for s in state_order}

    heat = pd.DataFrame(pct_data, index=state_order).T  # tasks × states
    heat.index = [task_labels[t] for t in task_order]
    heat.columns = [STATE_NAMES[s] for s in state_order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.5, 1]})

    # Left: grouped bar chart
    ax = axes[0]
    x = np.arange(len(task_order))
    width = 0.2
    for i, s in enumerate(state_order):
        vals = [pct_data[t][s] for t in task_order]
        bars = ax.bar(x + i * width, vals, width, label=STATE_NAMES[s],
                      color=STATE_COLORS[s], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([task_labels[t] for t in task_order], fontsize=10)
    ax.set_ylabel("% of windows", fontsize=10)
    ax.set_title("State Prevalence by Task", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", alpha=0.3)

    # Right: heatmap
    ax2 = axes[1]
    sns.heatmap(
        heat, ax=ax2, annot=True, fmt=".1f", cmap="Blues",
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 10},
        cbar_kws={"label": "% of windows"},
    )
    ax2.set_title("% Windows per State × Task", fontsize=11, fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = FIG_DIR / "state_persistence_heatmap_per_task.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4: State 1 vs State 3 distinction ─────────────────────────────────
def plot_state1_vs_state3(df):
    import numpy as np

    feats_to_plot = [
        ("tr_silence_duration_s",        "Silence [log]"),
        ("tr_n_active_speakers",          "Active speakers"),
        ("group_hr_mean_bpm_mean",        "HR (z)"),
        ("group_eda_phasic_rate_hz_mean", "EDA phasic (z)"),
        ("tr_laughter_count",             "Laughter [log]"),
        ("tr_backchannel_count",          "Backchannels [log]"),
        ("tr_ovl_count",                  "Overlap count [log]"),
        ("tr_ovl_contested",              "Contested OVL [log]"),
    ]

    # compute cross-state z-score for plotting
    sub = df[df["state"].isin([1, 3])].copy()
    all4 = df.copy()

    results = {}
    for col, label in feats_to_plot:
        mu = all4[col].mean()
        sd = all4[col].std() + 1e-9
        results[label] = {
            1: ((all4.loc[all4["state"] == 1, col] - mu) / sd).mean(),
            3: ((all4.loc[all4["state"] == 3, col] - mu) / sd).mean(),
        }

    labels_ordered = [l for _, l in feats_to_plot]
    s1_vals = [results[l][1] for l in labels_ordered]
    s3_vals = [results[l][3] for l in labels_ordered]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: grouped bar chart
    ax = axes[0]
    x = np.arange(len(labels_ordered))
    width = 0.35
    ax.barh(x + width/2, s1_vals, width, color=STATE_COLORS[1],
            label=f"State 1: {STATE_NAMES[1]}", edgecolor="white")
    ax.barh(x - width/2, s3_vals, width, color=STATE_COLORS[3],
            label=f"State 3: {STATE_NAMES[3]}", edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels(labels_ordered, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Z-score (relative to all-state mean)", fontsize=10)
    ax.set_title("State 1 vs State 3: Multimodal Profile Comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Right: scatter contested OVL vs backchannels
    ax2 = axes[1]
    for s, label in [(1, STATE_NAMES[1]), (3, STATE_NAMES[3])]:
        mask = df["state"] == s
        ax2.scatter(df.loc[mask, "tr_ovl_contested"], df.loc[mask, "tr_backchannel_count"],
                    color=STATE_COLORS[s], alpha=0.45, s=40, label=f"S{s}: {STATE_NAMES[s]}", edgecolors="none")
    ax2.set_xlabel("Contested OVL [log]", fontsize=10)
    ax2.set_ylabel("Backchannels [log]", fontsize=10)
    ax2.set_title("Clear Separation: Contested OVL vs Backchannels", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "state1_vs_state3_distinction.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 5: Task-state distribution + contested OVL by state (k=4) ─────────
def plot_task_state_ovl(df):
    import numpy as np

    task_order = ["T1", "T2", "T3"]
    task_labels = {"T1": "T1", "T2": "T2", "T3": "T3"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: stacked bar by task
    ax = axes[0]
    bottom = np.zeros(3)
    x = np.arange(3)
    for s in [0, 1, 2, 3]:
        vals = [
            (df[df["task_id"] == t]["state"] == s).sum()
            for t in task_order
        ]
        ax.bar(x, vals, bottom=bottom, color=STATE_COLORS[s],
               label=f"State {s}: {STATE_NAMES[s]}", edgecolor="white", linewidth=0.5)
        bottom += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([task_labels[t] for t in task_order], fontsize=11)
    ax.set_ylabel("Number of windows", fontsize=10)
    ax.set_title("HMM State Distribution by Task (k=4)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Right: boxplot of contested OVL by state
    ax2 = axes[1]
    data_by_state = [df.loc[df["state"] == s, "tr_ovl_contested"].values for s in [0, 1, 2, 3]]
    bp = ax2.boxplot(data_by_state, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2))
    for patch, s in zip(bp["boxes"], [0, 1, 2, 3]):
        patch.set_facecolor(STATE_COLORS[s])
        patch.set_alpha(0.8)
    ax2.set_xticklabels([f"S{s}\n{STATE_NAMES[s]}" for s in [0, 1, 2, 3]], fontsize=8.5)
    ax2.set_ylabel("Contested OVL [log]", fontsize=10)
    ax2.set_title("Contested Overlaps by State (k=4)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "task_state_competitive_overlap_analysis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 6: Standalone state profiles heatmap (z-scored) ───────────────────
def plot_state_profiles_heatmap(df):
    # Compute per-state mean for each feature, then z-score across states
    prof = df.groupby("state")[CLEAN_FEATS].mean()
    prof_z = (prof - prof.mean()) / (prof.std() + 1e-9)
    prof_z.index = [f"S{i}: {STATE_NAMES[i]}" for i in prof_z.index]
    prof_z.columns = [DISPLAY_NAMES[c] for c in CLEAN_FEATS]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(
        prof_z, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
        linewidths=0.4, ax=ax, vmin=-2, vmax=2,
        cbar_kws={"label": "Z-score (across states)", "shrink": 0.6},
        annot_kws={"size": 9},
    )
    ax.set_title("HMM State Profiles (k=4) — Feature Means Z-scored Across States",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    plt.tight_layout()
    out = FIG_DIR / "state_profiles_heatmap_zscore.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} windows, states: {sorted(df['state'].unique())}")

    print("\nGenerating figures...")
    plot_feature_importance(df)
    plot_pca_scatter(df)
    plot_state_task_heatmap(df)
    plot_state1_vs_state3(df)
    plot_task_state_ovl(df)
    plot_state_profiles_heatmap(df)

    print("\nDone. All 6 figures regenerated.")
