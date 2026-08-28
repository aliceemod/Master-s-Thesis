# ICMI 2026 Paper — Updated Overlaps Analysis

**⚠️ CRITICAL: This folder contains ONLY files generated from the updated overlap classification.**

Use figures and results **ONLY** from this folder for the paper. Do NOT use figures from `../figures/` or `../results/` folders — those contain old analysis.

## Folder Structure

```
icmi_paper/
├── ICMI_2026_Collective_Affective_States_DRAFT.md    ← Main paper
├── analysis/
│   └── hmm_statistical_analysis_complete.ipynb      ← Analysis notebook (all figures generated here)
├── figures/                                           ← Paper-ready figures (ONLY from updated overlaps)
│   ├── hmm_bic_selection_updated_overlaps.png
│   ├── hmm_profiles_transitions_updated_overlaps.png
│   ├── feature_importance_f_statistics.png
│   ├── state1_vs_state3_distinction.png
│   ├── task_state_competitive_overlap_analysis.png
│   └── state_profiles_heatmap_zscore.png
├── results/                                           ← Paper-ready data tables (ONLY from updated overlaps)
│   ├── table_state_summary.tsv                       ← Table 1 (for Results section)
│   ├── table_task_state_association.tsv              ← Task × State contingency
│   ├── table_feature_importance.tsv                  ← Top 10 features
│   └── hmm_cluster_assignments_k5_updated_overlaps.tsv
└── appendices/
    ├── APPENDIX_A_Feature_Definitions.md
    └── APPENDIX_B_Overlap_Classification_Methodology.md
```

## What Changed

✅ **Updated Overlaps Applied**
- All 329 overlaps reclassified using timing-based methodology
- 52 `needs_review` entries resolved → concrete labels
- 95+ collaborative overlaps reclassified
- 10 groups, 545 windows analyzed
- 5 states identified by BIC

✅ **Figures Generated From This Updated Data**
All `.png` files in `figures/` are from the updated overlap analysis.

## Do NOT Use

❌ Anything in `../figures/` — old analysis
❌ Anything in `../results/` — old analysis  
❌ Old notebooks in `../analysis/` — different data source

## State Names (Final)

| State | Label | Interpretation |
|-------|-------|-----------------|
| 0 | Quiet-Calm | Low HR, high silence, minimal interaction |
| 1 | Competitive/Tense | High competitive overlaps, stress, argumentative |
| 2 | Energetic-Aroused | Highest HR, elevated physiology, moderate engagement |
| 3 | Collaborative-Celebratory | High backchannels, laughter, supportive tone |
| 4 | Relaxed-Disengaged | Low EDA/HR, minimal all interactions |

## Next Steps

1. Run `icmi_paper/analysis/hmm_statistical_analysis_complete.ipynb` to regenerate all figures/tables
2. Update paper with final state names in Table 1
3. Use only figures from `icmi_paper/figures/` in the paper
4. Include Appendix B (Overlap Classification Methodology) as reproducibility documentation
