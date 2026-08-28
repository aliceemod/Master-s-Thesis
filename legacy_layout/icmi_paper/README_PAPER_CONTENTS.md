# ICMI 2026 Paper — Clean Output Structure

**IMPORTANT:** This folder contains ONLY the final paper-ready outputs generated from the **updated overlap classification** (timing-based, not semantic). 

All files here use the corrected 10-group feature matrix and HMM analysis with updated overlap labels.

---

## Folder Organization

### `icmi_paper/`
**Main paper and metadata**
- `ICMI_2026_Collective_Affective_States_DRAFT.md` — Main paper draft (8pp, double-blind)
- `feature_inventory.md` — Complete feature definitions
- `APPENDIX_A_Feature_Definitions.md` — Feature reference for methods
- `STRUCTURE.md` — Paper organization guide

### `icmi_paper/analysis/` ⭐ **PAPER-READY**
**The ONLY analysis notebook for this paper**
- `README_ANALYSIS.md` — Quick guide for this folder
- `hmm_statistical_analysis_complete.ipynb` — ✅ **USE THIS**
  - ✅ Updated overlap classification (timing-based)
  - ✅ Loads from corrected 10-group data source (`../../_tmp_collective_fullsample_export/`)
  - ✅ All paths verified for icmi_paper/ location
  - ✅ Generates all 7 figures & 5 tables used in paper
  - ✅ Ready to run: `jupyter notebook` → `Kernel → Restart & Run All`

### `icmi_paper/figures/` ⭐
**Paper-ready visualizations (7 PNG files)**

**USE THESE IN PAPER:**
1. `feature_importance_f_statistics.png` — Feature ranking (F-stats)
2. `hmm_bic_selection_updated_overlaps.png` — BIC curve (k=2-6 states)
3. `hmm_profiles_transitions_updated_overlaps.png` — State profiles heatmap
4. `task_state_competitive_overlap_analysis.png` — Task×State distribution
5. `state1_vs_state3_distinction.png` — State comparison analysis
6. `state_profiles_heatmap_zscore.png` — Z-scored multimodal profiles
7. `clusters_pca_updated_overlaps.png` — PCA projection

**STATUS:** All figures generated with updated overlap labels ✅

### `icmi_paper/results/` ⭐
**Paper-ready data tables (TSV format)**

**USE THESE IN PAPER:**
1. `table_state_summary.tsv` → **Table 1** in Results section
   - Columns: State, Label, n (windows), % prevalence, Competitive OVL, HR, Overlaps
   
2. `table_task_state_association.tsv` → Task×State contingency table
   - χ² = 264.21, p = 1.67e-52, Cramér's V = 0.4923 (STRONG association)
   
3. `table_feature_importance.tsv` → Feature ranking
   - Top 10 discriminative features by F-statistic
   
4. `hmm_cluster_assignments_k5_updated_overlaps.tsv` → State assignments for all 545 windows
   - For supplementary/reproducibility
   
5. `clustering_results_updated_overlaps.tsv` — Additional clustering metrics

**STATUS:** All tables generated with updated overlap labels ✅

### `icmi_paper/appendices/`
**Supplementary documentation**
- `APPENDIX_A_Feature_Definitions.md` — Feature inventory and definitions
- `APPENDIX_B_Overlap_Classification_Methodology.md` — Timing-based overlap thresholds

---

## Data Source

All outputs in this folder derive from:
- **Feature matrix:** `../../_tmp_collective_fullsample_export/features/collective_window_features.tsv`
  - 1521 windows, 10 groups, 44 features
  - Filtered to T1-T3 tasks with complete transcripts: **545 windows**
  
- **Overlap classification:** Timing-based (updated July 15, 2026)
  - 52 `needs_review` entries resolved
  - 95+ `collaborative` entries reclassified
  - Full audit trail: `RELABEL_OVERLAPS_20260714.md` (at project root)

- **HMM Analysis:** 10-group collective affective states
  - k=5 states selected by BIC (score = 11019.6)
  - Viterbi decoding for state assignments
  - Cross-validated with multimodal feature profiles

---

## State Names (Updated)

The analysis reveals 5 distinct affective states. Original placeholder names have been replaced with data-driven labels:

| State | New Name | n | % | Key Features |
|-------|----------|---|---|----|
| 0 | Quiet-Calm | 87 | 16% | HR=65.7 (lowest), EDA=0.124, High silence |
| 1 | Competitive/Tense | 92 | 17% | Comp OVL=1.15 (highest), High EDA, Low backchannels |
| 2 | Energetic-Aroused | 149 | 27% | HR=78.3 (highest), Temp=36.4 |
| 3 | Collaborative-Celebratory | 92 | 17% | Laughter=1.18 (highest), Backchannels=3.11 (highest) |
| 4 | Relaxed-Disengaged | 125 | 23% | HR=67.5, EDA=0.088 (lowest) |

---

## ⚠️ DO NOT MIX WITH OLD FILES

This folder is deliberately isolated to prevent accidental inclusion of stale outputs. 

- **Old figures** (from original HMM with wrong data) are in `../figures/` — **DO NOT USE**
- **Old analysis** notebooks are in `../analysis/` — **DO NOT USE**  
- **Old results** tables are in `../results/` — **DO NOT USE**

Only use files from `icmi_paper/figures/`, `icmi_paper/results/`, and `icmi_paper/analysis/`.

---

---

## ⚠️ ARCHIVED EXPLORATORY NOTEBOOKS

Old/exploratory notebooks are in `_archive_exploratory_old/` — **DO NOT USE FOR PAPER**

- `collective_state_discovery.ipynb` — Uses outdated data source
- `robustness_and_fairness_analysis.ipynb` — Incomplete analysis

These notebooks use the wrong feature matrix and incomplete overlap classification. They are kept for reference only.

---

## Quick Start

**To include a figure in the paper:**
1. Open a figure from `icmi_paper/figures/`
2. All 7 figures are paper-ready (updated overlaps, 10 groups)

**To include a table in the paper:**
1. Open a table from `icmi_paper/results/`
2. All 5 TSV tables are paper-ready
3. `table_state_summary.tsv` → **Table 1** in Results section

**To regenerate all figures and tables:**
1. Open `icmi_paper/analysis/hmm_statistical_analysis_complete.ipynb`
2. Run all cells: `Kernel → Restart & Run All`
3. Outputs update to `../figures/` and `../results/`

**In Methods section:**
- Refer to timing-based overlap classification (Appendix B)
- Sample size: 545 windows from 10 groups (T1-T3 tasks)
- HMM: k=5 states selected by BIC

**In Results section:**
- Use figures from `icmi_paper/figures/`
- Use tables from `icmi_paper/results/`
- State names: Use the 5 data-driven names above (not original placeholders)

**In Appendices:**
- Feature definitions: Appendix A
- Overlap classification methodology: Appendix B

---

## Running the Analysis

To regenerate all figures and tables:

```bash
cd icmi_paper/analysis
jupyter notebook hmm_statistical_analysis_complete.ipynb
# Run all cells (Kernel → Restart & Run All)
# All outputs update automatically to ../figures/ and ../results/
```

All paths in the notebook are already corrected for the `icmi_paper/` location.

---

**Last updated:** July 15, 2026  
**Status:** ✅ All paper-ready outputs consolidated, paths verified, ready for submission
