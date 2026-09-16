"""Generate new figures for paper/thesis_modeling_chapter.tex from already-computed,
post-BIC-fix (2026-09-10, ADR-0008) pipeline outputs.

Does NOT refit the HMM, KMeans, or any prediction model with new randomness where a
saved artifact already exists — it either replays the exact deterministic PCA/KMeans
steps from analysis/hmm_collective_states_updated_overlaps.ipynb (Sections 1-3, 11),
or reads existing TSVs directly. Run with the audio_venv interpreter.

Output: analysis/results/thesis_eda/<name>.png (matches existing thesis_eda figures'
sns.set_theme(style='whitegrid', font_scale=1.1), dpi=300, bbox_inches='tight' style).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

sns.set_theme(style="whitegrid", font_scale=1.1)

OUT_DIR = "results/thesis_eda"
COLLECTIVE_DIR = "results/collective_features_v20260905"
HMM_DIR = "results/hmm_states_v20260905"

# ---------------------------------------------------------------------------
# 1. Reproduce the exact preprocessing from hmm_collective_states_updated_overlaps.ipynb
#    (Sections 1-3): load -> filter T1-T3+transcript -> physio z-score + log1p -> impute
#    -> standardize -> PCA. This is fully deterministic (no random_state involved).
# ---------------------------------------------------------------------------
windows = pd.read_csv(f"{COLLECTIVE_DIR}/collective_window_features.tsv", sep="\t")
data = windows[windows["task_id"].isin(["T1", "T2", "T3"])].copy()
data = data.dropna(subset=["tr_spk_duration_s"])

data_clean = data.copy()
PHYSIO_Z = [
    "group_hr_mean_bpm_mean", "group_eda_phasic_rate_hz_mean", "group_temp_mean_mean",
    "group_et_pupil_mean_mean",
]
for col in PHYSIO_Z:
    if col in data_clean.columns:
        data_clean[col] = data_clean.groupby("group_id")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

LOG_FEATS = [
    "tr_silence_duration_s", "tr_overlap_count", "tr_competitive_overlap",
    "tr_laughter_count", "tr_overlap_time_s",
]
for col in LOG_FEATS:
    if col in data_clean.columns:
        data_clean[col] = np.log1p(data_clean[col].clip(lower=0))

CLEAN_FEATS = [
    "group_hr_mean_bpm_mean", "group_eda_phasic_rate_hz_mean", "group_temp_mean_mean",
    "tr_silence_duration_s", "tr_backchannel_count", "tr_overlap_count",
    "tr_competitive_overlap", "tr_laughter_count", "tr_n_active_speakers",
    "group_et_pupil_mean_mean",
]

X_all = SimpleImputer(strategy="mean").fit_transform(data_clean[CLEAN_FEATS])
X_all = StandardScaler().fit_transform(X_all)

pca = PCA()
X_pca = pca.fit_transform(X_all)
n80 = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.80) + 1)
X_pca_80 = X_pca[:, :n80]

evr = pca.explained_variance_ratio_
print(f"n80={n80}; PC1-5 = {[round(v * 100, 1) for v in evr[:5]]} "
      f"(chapter text: 30.2, 12.8, 10.2, 9.8, 9.2)")
assert data_clean.shape[0] == 616, f"expected 616 windows, got {data_clean.shape[0]}"

# ---------------------------------------------------------------------------
# Figure A: PCA scree plot (bar = per-component variance, line = cumulative)
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7, 4.5))
n_show = len(evr)
ax1.bar(np.arange(1, n_show + 1), evr * 100, color=sns.color_palette("deep")[0],
        label="Explained variance per PC")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Explained variance (%)")
ax1.set_xticks(np.arange(1, n_show + 1))
cum = np.cumsum(evr) * 100
ax2 = ax1.twinx()
ax2.plot(np.arange(1, n_show + 1), cum, color=sns.color_palette("deep")[3],
         marker="o", label="Cumulative variance")
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
ax2.set_ylabel("Cumulative explained variance (%)")
ax2.set_ylim(0, 105)
ax2.axvline(n80 + 0.5, color="gray", linestyle=":", linewidth=1)
ax2.annotate(f"{n80} components\nreach 80%", xy=(n80, 80), xytext=(n80 + 0.9, 35),
             fontsize=9, color="gray")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.set_title("PCA scree plot: 10 clean features -> retained components (80% threshold)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_scree_v20260905.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved pca_scree_v20260905.png")

# ---------------------------------------------------------------------------
# Figure B: PC1 vs PC2 scatter, colored by task
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "task": data_clean["task_id"].values,
})
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="task", palette="deep",
                 alpha=0.6, s=28, ax=ax)
ax.set_title(f"Window-level PCA space (PC1 vs.\\ PC2, {evr[0]*100:.1f}% / {evr[1]*100:.1f}% variance)\nby task")
ax.legend(title="Task", loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_scatter_by_task_v20260905.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved pca_scatter_by_task_v20260905.png")

# ---------------------------------------------------------------------------
# Figure C: K-Means silhouette plot (k=6, random_state=42, n_init=10 -- matches
# the exact call in hmm_collective_states_updated_overlaps.ipynb Section 11)
# ---------------------------------------------------------------------------
K_HMM = 6
km = KMeans(n_clusters=K_HMM, random_state=42, n_init=10)
km_labels = km.fit_predict(X_pca_80)
sil_avg = silhouette_score(X_pca_80, km_labels)
sil_vals = silhouette_samples(X_pca_80, km_labels)
print(f"K-Means silhouette_score = {sil_avg:.3f} (chapter appendix reports 0.178)")

fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
palette = sns.color_palette("deep", K_HMM)
for i in range(K_HMM):
    vals_i = np.sort(sil_vals[km_labels == i])
    y_upper = y_lower + len(vals_i)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_i, facecolor=palette[i],
                      edgecolor=palette[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * len(vals_i), str(i), fontsize=9)
    y_lower = y_upper + 10
ax.axvline(sil_avg, color="red", linestyle="--", label=f"Mean silhouette = {sil_avg:.3f}")
ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster (K-Means state)")
ax.set_yticks([])
ax.set_title("K-Means silhouette plot (k=6, matched to HMM state count)")
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/kmeans_silhouette_v20260905.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved kmeans_silhouette_v20260905.png")

# ---------------------------------------------------------------------------
# Figure D: HMM x K-Means contingency heatmap (from saved per-window assignments)
# ---------------------------------------------------------------------------
hmm_assign = pd.read_csv(f"{HMM_DIR}/hmm_window_state_assignments.tsv", sep="\t")
km_assign = pd.read_csv(f"{HMM_DIR}/kmeans_window_state_assignments.tsv", sep="\t")
merged = hmm_assign.merge(km_assign, on=["group_id", "task_id", "window_index"])
conf = pd.crosstab(merged["hmm_state"], merged["kmeans_state"])

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "# windows"}, ax=ax)
ax.set_xlabel("K-Means cluster")
ax.set_ylabel("HMM state")
ax.set_title("HMM state vs.\\ K-Means cluster: window-level contingency table")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/hmm_kmeans_contingency_v20260905.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved hmm_kmeans_contingency_v20260905.png")

# ---------------------------------------------------------------------------
# Figure E: H6 per-fold ARI / agreement bar chart (from saved stability results)
# ---------------------------------------------------------------------------
stab = pd.read_csv(f"{HMM_DIR}/hmm_state_stability_v20260908.tsv", sep="\t")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(stab))
width = 0.35
ax.bar(x - width / 2, stab["ARI"], width, label="ARI", color=sns.color_palette("deep")[0])
ax.bar(x + width / 2, stab["agreement_after_label_matching"], width,
       label="Agreement (label-matched)", color=sns.color_palette("deep")[2])
ax.axhline(stab["ARI"].mean(), color=sns.color_palette("deep")[0], linestyle="--",
           linewidth=1, alpha=0.7)
ax.axhline(stab["agreement_after_label_matching"].mean(), color=sns.color_palette("deep")[2],
           linestyle="--", linewidth=1, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {r.fold}\n(held out {r.holdout_groups})" for r in stab.itertuples()],
                    fontsize=8)
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("H6 stability check: per-fold ARI and label-matched agreement\n"
             "(leave-2-groups-out refit vs.\\ all-10-groups reference)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/hmm_stability_per_fold_v20260908.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved hmm_stability_per_fold_v20260908.png")

# ---------------------------------------------------------------------------
# Figure F: Matched-representation (HMM / K-Means / Continuous PCA) R2 comparison
# ---------------------------------------------------------------------------
pred = pd.read_csv("results/prediction_model_comparison_v20260905.tsv", sep="\t")
MATCHED = {
    "HMM states new K5 (v20260905)": "HMM states (K=6)",
    "K-Means clusters (no temporal, v20260905)": "K-Means clusters (K=6)",
    "Continuous PCA (no discretization, v20260905)": "Continuous PCA",
}
sub = pred[pred["Model"].isin(MATCHED) & (pred["CV"] == "LOGO")].copy()
sub["Model"] = sub["Model"].map(MATCHED)
targets = ["team_coordination", "voice_inclusion", "mental_demand", "coord+coop", "satisfaction"]
sub["Target"] = pd.Categorical(sub["Target"], categories=targets, ordered=True)
sub = sub.sort_values("Target")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=sub, x="Target", y="R2", hue="Model", palette="deep", ax=ax)
for i, row in enumerate(sub.itertuples()):
    if row.q < 0.05:
        pass  # significance stars added below per-bar via q lookup
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("$R^2$ (LOGO cross-validated)")
ax.set_xlabel("")
ax.set_title("Matched non-temporal comparison: HMM vs.\\ K-Means vs.\\ continuous PCA\n"
             "(identical 6-component PCA input, LOGO CV)")
ax.legend(title=None, loc="lower left", fontsize=9)
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/matched_representation_comparison_v20260905.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
print("saved matched_representation_comparison_v20260905.png")

print("\nNOTE: source table analysis/results/prediction_model_comparison_v20260905.tsv "
      "still labels the HMM model 'HMM states new K5 (v20260905)' even though the "
      "underlying occupancy features are the post-BIC-fix K=6 set (hmm_state_proportions_task.tsv "
      "has 6 state columns S0_pct..S5_pct, generated 2026-09-10 16:09, before this table's "
      "2026-09-10 21:12 generation). This script relabels it 'HMM states (K=6)' for the new "
      "figure only; the stale 'K5' string in the source TSV/notebook should be fixed separately.")
