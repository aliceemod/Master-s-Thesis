# HMM State Prediction Analysis

> **Date:** 2026-08-13  
> **Goal:** Predict task effectiveness from HMM state proportions

---

## Overview

This analysis tests whether the distribution of HMM-identified interaction states during a task can predict self-reported task effectiveness. The HMM was trained on 17 multimodal features (physio + conversation + lexical) and identified 5 distinct states.

---

## Target Variable: Composite Task Effectiveness

A task-appropriate effectiveness measure was constructed:

| Task | Outcome Variable | Rationale |
|------|------------------|-----------|
| **T1** (Hidden Profile Decision) | `team_coordination_mean` | Decision quality depends on information sharing |
| **T2** (Mini-Negotiation) | `cooperative_mean` | Negotiation success requires cooperation |
| **T3** (Nominal Group Technique) | `team_coordination_mean` | Idea generation benefits from coordination |

Both measures are self-reported (1–7 Likert scale), aggregated across the 4 participants per group.

---

## Features

**Input:** 5 HMM state proportions per task instance  
- `S0_pct` — proportion of 30s windows in State 0 (Focused Dialogue)
- `S1_pct` — State 1 (Playful Exchange)
- `S2_pct` — State 2 (Silent/Thinking)
- `S3_pct` — State 3 (Active Discussion)
- `S4_pct` — State 4 (High Engagement)

**Samples:** 27 task instances (9–10 groups × 3 tasks)

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | Ridge Regression (α = 1.0) |
| Preprocessing | StandardScaler on features |
| Validation | Leave-One-Group-Out (LOGO) |
| Folds | 10 (one per group) |

LOGO ensures predictions for each group are made without seeing any data from that group, preventing information leakage.

---

## Results

### Primary Metrics

| Metric | Value |
|--------|-------|
| **Spearman ρ** | **0.751** (p < 0.001) |
| **Pearson r** | 0.735 (p < 0.001) |
| **R²** | 0.538 |
| **MAE** | 0.614 (on 1–7 scale) |

### State Coefficients (Standardized)

| State | β | Direction | Interpretation |
|-------|---|-----------|----------------|
| **S1 (Playful Exchange)** | **+0.417** | ↑ better | Laughter + agreement predicts success |
| **S2 (Silent/Thinking)** | **+0.364** | ↑ better | Pauses for reflection help |
| S0 (Focused Dialogue) | −0.338 | ↓ worse | Sustained focus without play hurts |
| S3 (Active Discussion) | −0.341 | ↓ worse | High activity ≠ effectiveness |
| S4 (High Engagement) | −0.216 | ↓ worse | Peak intensity not beneficial |

### Per-Group Predictions

| Group | Actual | Predicted | Error |
|-------|--------|-----------|-------|
| grp-07 | 5.67 | 5.11 | +0.55 |
| grp-08 | 4.58 | 4.26 | +0.32 |
| grp-09 | 3.88 | 4.65 | −0.77 |
| grp-10 | 5.17 | 5.25 | −0.08 |
| grp-11 | 5.33 | 5.26 | +0.08 |
| grp-12 | 4.38 | 4.28 | +0.10 |
| grp-13 | 4.38 | 3.88 | +0.49 |
| grp-14 | 5.67 | 5.58 | +0.09 |
| grp-15 | 5.17 | 5.50 | −0.33 |
| grp-16 | 4.92 | 5.16 | −0.25 |

---

## Model Comparison

Alternative models tested under the same LOGO cross-validation:

| Model | Spearman ρ | R² |
|-------|------------|-----|
| **SVR (linear)** | **0.766** | 0.544 |
| Ridge (α=1.0) | 0.751 | 0.538 |
| Ridge (α=0.1) | 0.742 | 0.526 |
| ElasticNet | 0.736 | 0.507 |
| SVR (RBF) | 0.710 | 0.469 |
| Lasso (α=0.1) | 0.695 | 0.451 |
| Random Forest | 0.645 | 0.354 |

Linear models outperform non-linear ones, likely due to small sample size (n=27).

---

## Permutation Test

To verify that prediction is not by chance:

| Metric | Value |
|--------|-------|
| True R² | ~0.54 |
| Permutation p-value | **0.003** |
| Permutation mean R² | −3.31 |
| Permutation std | 2.29 |

The model significantly outperforms random label shuffling.

---

## Feature Set Comparison

Tested whether adding raw features improves over state proportions alone:

| Target | States Only ρ | Full Features ρ |
|--------|---------------|-----------------|
| composite_target | **+0.751** | +0.772 |
| team_coordination | +0.407 | +0.151 |
| cooperative | +0.548 | +0.166 |
| voice_inclusion | +0.597 | +0.416 |

**Finding:** State proportions alone perform well; adding raw features helps composite target slightly but hurts individual targets (overfitting with 17+ features on n=27).

---

## Interpretation

### Key Finding

Groups that spend more time in **Playful Exchange** (S1) and **Silent/Thinking** (S2) achieve better task outcomes than those constantly in high-activity states.

### Theoretical Implications

1. **Playful Exchange (S1 → better):** Laughter and agreement markers signal psychological safety and rapport, enabling constructive collaboration.

2. **Silent/Thinking (S2 → better):** Pauses may indicate:
   - Cognitive processing of information
   - Turn-taking coordination
   - Allowing space for all members to contribute

3. **High-intensity states (S0, S3, S4 → worse):** Constant talking without pauses or play may indicate:
   - Dominance by few speakers
   - Lack of reflection
   - Information overload

### Practical Implication

Effective teams balance engagement with reflection. Intervention design could target increasing playful moments and structured pauses.

---

## Output Files

| File | Description |
|------|-------------|
| `icmi_paper/results/hmm_prediction_results.tsv` | Per-task predictions with state proportions |
| `icmi_paper/results/hmm_prediction_plot.png` | Actual vs predicted scatter + coefficient bar chart |
| `icmi_paper/results/hmm_input_features_with_lexical.tsv` | 17-feature input matrix (526 windows) |
| `icmi_paper/results/hmm_cluster_assignments_k5_with_lexical.tsv` | State assignments per window |

---

## Code Reference

The prediction analysis was run inline; key steps:

```python
# 1. Load state assignments
states_df = pd.read_csv('icmi_paper/results/hmm_cluster_assignments_k5_with_lexical.tsv', sep='\t')

# 2. Compute state proportions per group-task
state_props = states_df.groupby(['group_id', 'task_id'])['hmm_state_lexical'].apply(
    lambda x: pd.Series({f'S{s}_pct': (x==s).mean() for s in range(5)})
).unstack().reset_index()

# 3. Create composite target
final['composite_target'] = np.where(
    final['task_id'].isin(['T1', 'T3']),
    final['team_coordination_mean'],
    final['cooperative_mean']
)

# 4. Ridge regression with LOGO CV
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

model = Ridge(alpha=1.0)
y_pred = cross_val_predict(model, X_scaled, y, cv=LeaveOneGroupOut(), groups=groups)
```

---

## Next Steps

1. **Validate on held-out groups** (if more data becomes available)
2. **Test temporal dynamics** — do state transitions predict outcomes?
3. **Add lexical embeddings** — sentence-BERT features for richer content representation
4. **Cross-task generalization** — train on T1/T3, predict T2 (different task type)
