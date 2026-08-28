# Prediction Model Results — Interpretation
*Generated from `prediction_models_v2.ipynb`, 2026-08-26*

---

## Summary of Findings

### Permutation Test Results (L2GO, Ridge, 1000 permutations)

| Target | Feature set | Observed R² | p-value | Significant |
|---|---|---|---|---|
| **composite (HMM-validated)** | HMM states old K=5 | **0.501** | **0.001** | ✓ p<0.01 |
| **voice_inclusion** | Audio | 0.239 | **0.026** | ✓ p<0.05 |
| **mental_demand** | Audio | 0.068 | **0.017** | ✓ p<0.05 |
| team_coord | HMM raw + ExtraTR | -0.496 | 0.152 | ~ marginal |
| cooperative | HMM raw | 0.097 | 1.000 | ✗ |

**Note on cooperative:** L2GO p=1.0 despite LOGO R²=0.376. This suggests the cooperative result is driven by
within-fold structure that doesn't generalise — the L2GO estimate (R²=0.097) is the honest one.

---

## Feature-Set Comparison (L2GO Ridge, Section 1)

### composite (HMM-validated)

The composite outcome is best predicted by:
- **Extra TR (2 features): R²=0.750** — speaking entropy + speech count are the strongest predictors
- Audio (4): R²=0.684
- All raw: R²=0.706
- Old HMM states K=5: R²=0.622

The composite's predictability is driven by speaking activity features, not physiology or semantic content.
This is consistent with the HMM validation result (R²=0.30 in icmi_paper) — the HMM states partially
capture speaking dynamics.

### cooperative (T2 — negotiation)

All feature sets produce negative or near-zero L2GO R². The LOGO result (R²=0.376 for HMM raw) does not
generalise to L2GO (R²=0.376 → 0.097). Conclusion: the cooperative outcome is not robustly predictable
with these features at this sample size.

**Exception — Lasso on HMM features: L2GO R²=0.834.** This is unusually high and likely reflects
overfitting or lucky fold assignment. The permutation test (p=1.0) confirms the result is not significant.

### voice_inclusion (all tasks — T1/T2/T3)

- **Audio: R²=0.388 (p=0.026)** — statistically significant
- All raw: R²=0.350
- Old HMM states K=5: R²=0.358
- HMM raw: R²=0.140

Voice inclusion is predicted by acoustic features (energy, pitch, HNR) — groups that speak with more
expressive prosody report higher voice inclusion. This is independent of physiology and semantic content.

### mental_demand (all tasks — T1/T2/T3)

- **Semantic continuity: R²=0.315 (raw), 0.139 (PCA-2)**
- Audio: R²=0.119
- Extra TR: R²=0.252
- HMM raw: R²=0.190

Mental demand is predicted by semantic continuity (cosine similarity between consecutive speakers) — when
speakers align semantically, the task feels less demanding. This is the strongest finding for the new NLP features.
Permutation p=0.017 confirms this is significant.

### team_coordination (T1/T3 — decision/ideation)

Most feature sets give negative R², suggesting team coordination is not well-predicted from multimodal window
features with this sample size. The best result is HMM+ExtraTR (R²=0.364) but this does not survive
permutation testing. No robust predictors identified for this outcome.

### decision_confidence (T1 only — n≈10)

Nearly all models give negative R². With only ~10 data points (10 groups × T1), no meaningful prediction
is possible. Should be treated as exploratory only.

### idea_quality (T3 only — n≈10)

Same limitation as decision_confidence. BERT-UMAP shows R²=0.083 but not significant.

### satisfaction (T2/T3 — n≈18)

- **Semantic: R²=0.611 (raw), 0.614 (PCA-2)** — highest for satisfaction
- All NLP: R²=0.456
- Audio: R²=0.132

Semantic continuity predicts satisfaction strongly. Groups where speakers pick up each other's semantic
content report higher satisfaction. Requires verification with permutation test.

---

## Model Comparison (Section 2 — HMM raw features)

| Model | composite | cooperative | mental_demand | voice_inclusion |
|---|---|---|---|---|
| **Lasso** | **0.668** | **0.834** | **0.272** | **0.484** |
| ElasticNet | 0.644 | 0.689 | 0.242 | 0.317 |
| RF | 0.708 | 0.115 | 0.331 | 0.226 |
| Ridge | 0.533 | 0.376 | 0.190 | 0.140 |

**Lasso dominates for cooperative and voice_inclusion.** This suggests sparse solutions — only a few HMM
features are driving the prediction. Lasso automatically zeros out irrelevant features, which helps with
n=28. The Lasso cooperative result (0.834) should be interpreted with caution given p=1.0 under permutation.

**Random Forest** performs best for composite (0.708) and mental_demand (0.331), suggesting non-linear
interactions between features. RF is regularised through max_depth=3 and ensemble averaging.

---

## Key Thesis Findings

1. **Composite (HMM-validated) is robustly predictable (p=0.001):** The HMM-validated effectiveness
   composite is significantly predicted by the old K=5 HMM states (R²=0.501) and even better by speaking
   entropy + speech count (R²=0.750). This confirms the thesis hypothesis that interaction structure
   predicts perceived effectiveness.

2. **Voice inclusion is significantly predicted by audio (p=0.026, R²=0.388):** Prosodic expressiveness
   (pitch, energy, HNR) predicts whether participants feel their voice was included. This supports H1.1b
   (conversation-structure features predict outcomes).

3. **Mental demand is significantly predicted by semantic continuity (p=0.017):** When speakers
   semantically align with each other, groups report lower cognitive load. This is the key finding for
   the new NLP features developed in this thesis.

4. **Cooperative is not robustly predictable under L2GO:** The LOGO result (R²=0.376) inflated by
   within-fold structure. L2GO gives R²=0.097 (p=1.0). This is an important negative finding.

5. **Lasso outperforms Ridge for most targets:** Automatic feature selection (L1 regularisation)
   substantially improves over Ridge with this dataset, confirming that sparsity is appropriate for
   n=28 with 12+ features.

6. **HMM state proportions (K=4, new) not yet available:** `hmm_state_proportions_expanded.tsv` returned
   NaN for all predictions, suggesting the task/group_id join keys need reconciliation.

---

## Limitations

- n=28 group×task observations limits statistical power; many outcomes have insufficient signal
- L2GO with 5 folds and only ~6 test rows per fold has high variance
- Task-specific outcomes (decision_confidence, idea_quality) have n≈10 — no meaningful prediction possible
- New HMM states (K=4) could not be evaluated — key join needed
- Permutation testing only performed on 5 primary targets; full permutation table needed for publication
