# ICMI Paper Analysis — Collective Affective States

## Overview

This folder contains the complete analysis for discovering and characterising collective affective states in multimodal group conversation data.

**Main finding:** 5 interpretable latent states emerge from temporal sequence of multimodal features, strongly structured by task phase.

## Key Results

### State Discovery Method
- **Algorithm:** Gaussian Hidden Markov Model (HMM)
- **Why HMM?** Models temporal transitions in 30-second sliding windows within group conversation sessions
- **k selection:** BIC-based, k=5 chosen with clear elbow at 11108.1
- **Feature engineering:** Within-group z-scoring (physio confounds), log transforms (skew), redundancy removal

### 5 Discovered States

| State | Label | N | Signature |
|---|---|---|---|
| 0 | Aroused-active | 182 (33%) | High HR, temp, laughter, active speakers |
| 1 | Silence-dominated | 103 (19%) | High silence but with competitive overlaps |
| 2 | High-conflict | 61 (11%) | High EDA, floor-fighting (competitive overlaps, backchannels) |
| 3 | Low-engagement | 134 (25%) | All features low, quiet deliberation |
| 4 | Cool-attentive | 65 (12%) | Physiologically calm but attentive (high pupil) |

### Task Structuring

- **χ² = 296.92, p = 1.86×10⁻⁵⁹, Cramér's V = 0.522** — states are strongly task-dependent
- **T1 (Hiring decision)** → mostly States 3 & 4 (deliberative)
- **T2 (Negotiation)** → shift to States 0 & 1 (more activity, engagement)
- **T3 (Idea generation)** → peak in State 2 (competitive floor-fighting for ideas)

### State Persistence

Transition matrix shows high diagonal values (0.79–0.94): **states persist for multiple windows**, confirming they are meaningful stable modes, not noise.

## Folder Structure

```
icmi_paper/
├── notebooks/
│   └── collective_state_discovery.ipynb          # Main analysis notebook
├── figures/                                       # All publication-ready figures
│   ├── pca_clean_*.png                           # PCA diagnostics
│   ├── hmm_*.png                                 # HMM results
│   ├── diag_*.png                                # Feature engineering diagnostics
│   └── ...
├── results/                                       # Exported tables & summaries
│   ├── analysis_summary.json                     # Structured results + metadata
│   ├── hmm_profiles.csv                          # State feature profiles (if exported)
│   ├── hmm_transitions.csv                       # Transition matrix (if exported)
│   └── ...
├── consolidate_results.py                        # This script
└── README.md                                      # This file
```

## Reproduction

1. **Run the analysis:**
   ```bash
   jupyter notebook analysis/collective_state_discovery.ipynb
   ```
   - Executes all diagnostic cells (feature validation)
   - Runs PCA on cleaned features
   - Fits HMMs for k=2..6, selects best via BIC
   - Generates all figures

2. **Consolidate outputs (optional):**
   ```bash
   python icmi_paper/consolidate_results.py
   ```
   - Copies figures and tables to icmi_paper/results and icmi_paper/figures
   - Generates this README

## Key Figures for Paper

### Figure 1: Feature Engineering & Diagnostics
- `diag_within_between_variance.png` — why each feature was included
- `diag_distributions.png` — skew justifying log transforms
- `diag_task_discriminability.png` — task-relevant features

### Figure 2: PCA on Clean Features
- `pca_clean_scree_loadings.png` — PC1 (interaction intensity), PC2 (arousal)
- `pca_clean_scatter_density.png` — continuous spectrum, no discrete clusters (motivates HMM)

### Figure 3: HMM State Discovery
- `hmm_bic_selection.png` — k=5 as clear BIC minimum
- `hmm_profiles_transitions.png` — state signatures + transition matrix

### Figure 4: Task Dynamics
- `hmm_task_dist_trajectory.png` — state distribution by task + example group trajectory

## Data Files

All analysis uses:
- `features/collective_window_features.tsv` — 545 windows of multimodal features
- `features/transcript_group_task.tsv` — task/group labels
- `features/collective_task_selfreport.tsv` — post-task questionnaires (for validation)

## Technical Notes

### Feature Set (11 features, after cleaning)
- **Physio (z-scored within-group):** HR, EDA phasic, Temperature
- **Transcript (log-transformed where skewed):** Silence dur, Backchannels, Overlaps, Competitive OVL, Laughter
- **Interaction:** Active speakers
- **Eye-tracking (z-scored within-group):** Pupil diameter

### HMM Details
- Pure NumPy/SciPy implementation (Baum-Welch EM)
- Diagonal Gaussian emissions
- Fitted on per-group-task sequences (28 sequences, mean=19.5 windows)
- Viterbi decoding for state assignment

### Statistical Tests
- **Task-state:** Pearson χ² goodness of fit with Cramér's V effect size
- **State profiles:** Kruskal-Wallis H-test (non-parametric ANOVA) on each feature
- **BIC k-selection:** Bayesian Information Criterion across models

---

**Contact:** See AGENTS.md for paper-writing and analysis specialist contacts.
