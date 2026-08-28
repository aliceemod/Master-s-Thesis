# Analysis Folder — PAPER-READY NOTEBOOK

⭐ **THIS FOLDER CONTAINS THE ONLY PAPER-READY NOTEBOOK**

## Notebook

**`hmm_statistical_analysis_complete.ipynb`** — ✅ USE THIS FOR PAPER

**Status:**
- ✅ Uses corrected 10-group data source
- ✅ Implements updated (timing-based) overlap classification  
- ✅ Generates all paper-ready figures and tables
- ✅ Paths verified and working for icmi_paper/ location
- ✅ All 545 windows (T1-T3 with transcripts)
- ✅ 5 HMM states with BIC=11019.6

**What this notebook does:**
1. Loads feature matrix from `../../_tmp_collective_fullsample_export/`
2. Loads HMM cluster assignments with updated overlaps
3. Computes task-state associations (χ², Cramér's V)
4. Calculates feature importance (F-statistics)
5. Generates state profiles and multimodal comparisons
6. Exports figures → `../figures/`
7. Exports tables → `../results/`

**To regenerate all paper outputs:**
```bash
jupyter notebook hmm_statistical_analysis_complete.ipynb
# Kernel → Restart & Run All
# All figures and tables update automatically
```

---

## Data Source

- **Features:** `../../_tmp_collective_fullsample_export/features/collective_window_features.tsv`
- **HMM Assignments:** `../results/hmm_cluster_assignments_k5_updated_overlaps.tsv`
- **Overlap Classification:** Timing-based (July 15, 2026)

---

## Output Files

### Figures (saved to `../figures/`)
- `feature_importance_f_statistics.png`
- `hmm_bic_selection_updated_overlaps.png`
- `hmm_profiles_transitions_updated_overlaps.png`
- `task_state_competitive_overlap_analysis.png`
- `state1_vs_state3_distinction.png`
- `state_profiles_heatmap_zscore.png`
- `clusters_pca_updated_overlaps.png`

### Tables (saved to `../results/`)
- `table_state_summary.tsv` → **Table 1 in paper**
- `table_task_state_association.tsv` → **Chi-square results**
- `table_feature_importance.tsv` → **Top 10 features**
- `hmm_cluster_assignments_k5_updated_overlaps.tsv` → Window assignments
- `clustering_results_updated_overlaps.tsv` → Additional metrics

---

## ⚠️ Other Notebooks (NOT FOR PAPER)

Old/exploratory notebooks have been archived in:
- `../_archive_exploratory_old/` — DO NOT USE FOR PAPER

These use outdated data sources and incomplete analysis.

---

**Last Updated:** July 15, 2026  
**Status:** ✅ Ready for paper submission
