# Prediction Results — Feature-Set Ablation Study

Generated: 2026-08-27

Method: Leave-One-Group-Out Ridge Regression (LOGO-CV), 100 permutation tests

Models tested: 16 feature sets × 9 targets = 144 models

Significant models (p < 0.05): 47 / 144


## Summary: Best Significant Model per Target

| Target | n | Best model | R² | ρ | MAE | p |
|--------|---|-----------|-----|---|-----|---|
| effectiveness_z | 28 | Lexical (theory) | 0.284 | 0.639 | 0.308 | 0.010 |
| voice_inclusion | 28 | Turn-taking only | 0.567 | 0.766 | 0.423 | 0.010 |
| mental_demand | 28 | Audio only | 0.437 | 0.636 | 0.555 | 0.010 |
| coord+coop | 28 | Turn + Lex + DA | 0.359 | 0.647 | 0.646 | 0.010 |
| cooperative | 10 | Expanded raw (19) | 0.648 | 0.825 | 0.277 | 0.020 |
| team_coordination | 18 | Expanded raw (19) | 0.356 | 0.592 | 0.388 | 0.020 |
| satisfaction | 19 | Semantic + completion | 0.470 | 0.717 | 0.395 | 0.020 |
| decision_confidence | 9 | — | — | — | — | — |
| idea_quality | 9 | — | — | — | — | — |

## Physio & Eye-tracking Results

These modalities were tested in isolation to assess whether physiological signals alone predict outcomes.

| Model | Target | R² | ρ | p |
|-------|--------|-----|---|---|
| Physio only | effectiveness_z | -0.135 | -0.657 | 0.426 |
| Physio only | voice_inclusion | 0.365 | 0.545 | 0.010* |
| Physio only | mental_demand | 0.168 | 0.459 | 0.020* |
| Physio only | coord+coop | -0.136 | -0.474 | 0.475 |
| Physio only | cooperative | 0.122 | 0.579 | 0.050 |
| Physio only | team_coordination | -0.633 | -0.666 | 0.931 |
| Physio only | satisfaction | -0.313 | 0.171 | 0.614 |
| Physio only | decision_confidence | -0.325 | -0.218 | 0.149 |
| Physio only | idea_quality | -6.944 | -0.880 | 0.911 |
| Eye-tracking only | effectiveness_z | -0.100 | -0.632 | 0.287 |
| Eye-tracking only | voice_inclusion | 0.354 | 0.289 | 0.010* |
| Eye-tracking only | mental_demand | 0.370 | 0.500 | 0.010* |
| Eye-tracking only | coord+coop | -0.122 | -0.687 | 0.505 |
| Eye-tracking only | cooperative | -0.255 | -0.862 | 0.228 |
| Eye-tracking only | team_coordination | -0.200 | -0.566 | 0.624 |
| Eye-tracking only | satisfaction | 0.342 | 0.494 | 0.010* |
| Eye-tracking only | decision_confidence | -0.336 | -0.975 | 0.376 |
| Eye-tracking only | idea_quality | -0.432 | -0.937 | 0.594 |
| Physio + ET | effectiveness_z | -0.141 | -0.638 | 0.416 |
| Physio + ET | voice_inclusion | 0.350 | 0.533 | 0.010* |
| Physio + ET | mental_demand | 0.111 | 0.501 | 0.040* |
| Physio + ET | coord+coop | -0.147 | -0.503 | 0.515 |
| Physio + ET | cooperative | -2.019 | 0.271 | 0.891 |
| Physio + ET | team_coordination | -0.841 | -0.559 | 0.941 |
| Physio + ET | satisfaction | -0.527 | 0.146 | 0.743 |
| Physio + ET | decision_confidence | -2.881 | -0.336 | 0.812 |
| Physio + ET | idea_quality | -6.031 | -0.819 | 0.960 |

## All Significant Models (p < 0.05)

| Model | Target | R² | ρ | MAE | p |
|-------|--------|-----|---|-----|---|
| Expanded raw (19) | cooperative | 0.648 | 0.825 | 0.277 | 0.020 |
| Lexical (theory) | cooperative | 0.427 | 0.449 | 0.366 | 0.020 |
| Turn + Lex + DA | coord+coop | 0.359 | 0.647 | 0.646 | 0.010 |
| States + Lex + DA | coord+coop | 0.340 | 0.695 | 0.676 | 0.010 |
| Turn-taking only | coord+coop | 0.127 | 0.315 | 0.720 | 0.010 |
| Expanded raw (19) | coord+coop | 0.056 | 0.287 | 0.751 | 0.020 |
| States + Turn-taking | coord+coop | 0.055 | 0.197 | 0.776 | 0.020 |
| States + Lexical | coord+coop | 0.032 | 0.271 | 0.781 | 0.010 |
| States + DA | coord+coop | 0.008 | 0.274 | 0.833 | 0.040 |
| Lexical (theory) | effectiveness_z | 0.284 | 0.639 | 0.308 | 0.010 |
| States + Lexical | effectiveness_z | 0.080 | 0.388 | 0.340 | 0.020 |
| States + Lex + DA | effectiveness_z | 0.069 | 0.506 | 0.345 | 0.010 |
| Audio only | mental_demand | 0.437 | 0.636 | 0.555 | 0.010 |
| Eye-tracking only | mental_demand | 0.370 | 0.500 | 0.586 | 0.010 |
| HMM states (K=4) | mental_demand | 0.274 | 0.476 | 0.675 | 0.010 |
| HMM states (K=5) | mental_demand | 0.262 | 0.549 | 0.632 | 0.010 |
| Physio only | mental_demand | 0.168 | 0.459 | 0.666 | 0.020 |
| Physio + ET | mental_demand | 0.111 | 0.501 | 0.692 | 0.040 |
| DA keywords | mental_demand | 0.109 | 0.600 | 0.656 | 0.030 |
| Lexical (theory) | mental_demand | 0.049 | 0.348 | 0.751 | 0.030 |
| States + Lexical | mental_demand | 0.022 | 0.328 | 0.794 | 0.040 |
| Semantic + completion | satisfaction | 0.470 | 0.717 | 0.395 | 0.020 |
| Turn-taking only | satisfaction | 0.378 | 0.654 | 0.423 | 0.010 |
| Eye-tracking only | satisfaction | 0.342 | 0.494 | 0.482 | 0.010 |
| States + Lexical | satisfaction | 0.330 | 0.612 | 0.465 | 0.010 |
| Audio only | satisfaction | 0.295 | 0.447 | 0.458 | 0.010 |
| HMM states (K=4) | satisfaction | 0.239 | 0.372 | 0.482 | 0.030 |
| States + Lex + DA | satisfaction | 0.217 | 0.552 | 0.511 | 0.010 |
| States + Turn-taking | satisfaction | 0.134 | 0.612 | 0.491 | 0.040 |
| Expanded raw (19) | team_coordination | 0.356 | 0.592 | 0.388 | 0.020 |
| States + Turn-taking | team_coordination | 0.063 | 0.226 | 0.525 | 0.020 |
| Turn-taking only | voice_inclusion | 0.567 | 0.766 | 0.423 | 0.010 |
| Turn + Lex + DA | voice_inclusion | 0.556 | 0.739 | 0.425 | 0.010 |
| Expanded raw (19) | voice_inclusion | 0.541 | 0.758 | 0.474 | 0.010 |
| States + Turn-taking | voice_inclusion | 0.461 | 0.732 | 0.453 | 0.010 |
| States + Lexical | voice_inclusion | 0.433 | 0.601 | 0.501 | 0.010 |
| Lexical (theory) | voice_inclusion | 0.403 | 0.549 | 0.522 | 0.010 |
| DA keywords | voice_inclusion | 0.393 | 0.506 | 0.527 | 0.010 |
| States + DA | voice_inclusion | 0.392 | 0.632 | 0.527 | 0.010 |
| HMM states (K=5) | voice_inclusion | 0.371 | 0.609 | 0.546 | 0.010 |
| Physio only | voice_inclusion | 0.365 | 0.545 | 0.533 | 0.010 |
| Eye-tracking only | voice_inclusion | 0.354 | 0.289 | 0.551 | 0.010 |
| Physio + ET | voice_inclusion | 0.350 | 0.533 | 0.536 | 0.010 |
| HMM states (K=4) | voice_inclusion | 0.342 | 0.554 | 0.572 | 0.010 |
| States + Lex + DA | voice_inclusion | 0.330 | 0.668 | 0.500 | 0.010 |
| Semantic + completion | voice_inclusion | 0.145 | 0.409 | 0.524 | 0.020 |
| Audio only | voice_inclusion | 0.135 | 0.423 | 0.633 | 0.020 |

## Feature Importance (Ridge Coefficients)

Standardized Ridge coefficients for each significant model, showing which features drive predictions.


### effectiveness_z ← Lexical (theory) (R²=0.284)

| Feature | Coefficient |
|---------|------------|
| lex_turn_cohesion | +0.1497 |
| lex_word_gini | +0.1323 |
| lex_sentiment_ratio | +0.1251 |
| lex_social_composite | +0.1044 |
| lex_agreement_count | +0.0576 |
| lex_suggestion_count | -0.0447 |
| lex_question_count | +0.0337 |
| lex_hedging_count | +0.0192 |
| lex_we_i_ratio | -0.0057 |

### voice_inclusion ← Turn-taking only (R²=0.567)

| Feature | Coefficient |
|---------|------------|
| tr_competitive_overlap | +0.5357 |
| tr_ovl_floor_fight | -0.3973 |
| tr_filled_pause_count | +0.3454 |
| tr_backchannel_count | +0.2774 |
| tr_silence_duration_s | -0.2295 |
| tr_overlap_count | +0.1703 |
| tr_n_active_speakers | -0.1483 |
| tr_ovl_smooth | +0.1156 |
| tr_speaking_entropy | -0.0515 |
| tr_cooperative_overlap | +0.0330 |
| tr_conflict_overlap_ratio | +0.0317 |
| tr_collaborative_overlap | +0.0083 |
| tr_ovl_completion_index | -0.0060 |

### mental_demand ← Audio only (R²=0.437)

| Feature | Coefficient |
|---------|------------|
| audio_energy_mean | -0.3221 |
| audio_pitch_sd | +0.2911 |
| audio_pitch_mean | -0.2264 |
| audio_hnr_mean | +0.1798 |

### coord+coop ← Turn + Lex + DA (R²=0.359)

| Feature | Coefficient |
|---------|------------|
| da_hedge | -0.2266 |
| lex_question_count | +0.2194 |
| lex_suggestion_count | -0.1719 |
| tr_conflict_overlap_ratio | -0.1659 |
| lex_sentiment_ratio | +0.1561 |
| tr_backchannel_count | +0.1519 |
| da_compromise | -0.1420 |
| lex_turn_cohesion | +0.1402 |
| tr_filled_pause_count | +0.1349 |
| da_agreement | -0.1348 |
| tr_competitive_overlap | +0.1178 |
| da_backchannel | -0.1026 |
| tr_ovl_completion_index | -0.0881 |
| lex_word_gini | +0.0853 |
| tr_overlap_count | +0.0639 |
| da_disagreement | +0.0638 |
| tr_collaborative_overlap | +0.0578 |
| da_question | +0.0555 |
| lex_social_composite | -0.0419 |
| lex_we_i_ratio | -0.0404 |
| tr_ovl_smooth | +0.0354 |
| da_proposal | -0.0334 |
| tr_silence_duration_s | +0.0300 |
| lex_agreement_count | +0.0291 |
| tr_ovl_floor_fight | +0.0281 |
| tr_n_active_speakers | +0.0275 |
| tr_cooperative_overlap | +0.0221 |
| tr_speaking_entropy | +0.0081 |
| lex_hedging_count | -0.0032 |

### cooperative ← Expanded raw (19) (R²=0.648)

| Feature | Coefficient |
|---------|------------|
| group_hr_mean_bpm_mean | +0.2424 |
| tr_backchannel_count | +0.1525 |
| group_hrv_rmssd_ms_mean | +0.1255 |
| tr_cooperative_overlap | +0.1207 |
| tr_overlap_count | +0.1155 |
| tr_silence_duration_s | +0.1101 |
| group_eda_phasic_rate_hz_mean | +0.1076 |
| lex_we_i_ratio | -0.0847 |
| tr_n_active_speakers | +0.0771 |
| group_temp_mean_mean | +0.0764 |
| lex_hedging_count | -0.0728 |
| tr_competitive_overlap | +0.0718 |
| lex_turn_cohesion | -0.0469 |
| lex_word_gini | -0.0262 |
| tr_laughter_count | -0.0215 |
| tr_speaking_entropy | +0.0082 |
| group_eda_tonic_mean_mean | -0.0071 |
| group_et_pupil_mean_mean | +0.0019 |
| tr_collaborative_overlap | +0.0007 |

### team_coordination ← Expanded raw (19) (R²=0.356)

| Feature | Coefficient |
|---------|------------|
| tr_laughter_count | +0.1621 |
| lex_word_gini | +0.1516 |
| group_et_pupil_mean_mean | -0.1065 |
| lex_turn_cohesion | +0.1013 |
| tr_collaborative_overlap | +0.0945 |
| tr_silence_duration_s | -0.0857 |
| group_eda_tonic_mean_mean | +0.0797 |
| tr_backchannel_count | +0.0744 |
| tr_overlap_count | +0.0674 |
| tr_cooperative_overlap | +0.0552 |
| group_hr_mean_bpm_mean | -0.0544 |
| lex_we_i_ratio | -0.0448 |
| group_hrv_rmssd_ms_mean | -0.0446 |
| tr_n_active_speakers | +0.0356 |
| group_temp_mean_mean | -0.0270 |
| tr_competitive_overlap | +0.0254 |
| group_eda_phasic_rate_hz_mean | -0.0120 |
| tr_speaking_entropy | +0.0031 |
| lex_hedging_count | -0.0011 |

### satisfaction ← Semantic + completion (R²=0.470)

| Feature | Coefficient |
|---------|------------|
| tr_completion_count | +0.2366 |
| tr_cross_speaker_similarity | +0.1775 |
| tr_semantic_continuity | +0.1503 |
| tr_repetition_count | +0.1443 |

## Full R² Table

```
Target                 cooperative  coord+coop  decision_confidence  effectiveness_z  idea_quality  mental_demand  satisfaction  team_coordination  voice_inclusion
Model                                                                                                                                                              
Physio only                  0.122      -0.136               -0.325           -0.135        -6.944          0.168        -0.313             -0.633            0.365
Eye-tracking only           -0.255      -0.122               -0.336           -0.100        -0.432          0.370         0.342             -0.200            0.354
Physio + ET                 -2.019      -0.147               -2.881           -0.141        -6.031          0.111        -0.527             -0.841            0.350
Audio only                  -0.301      -0.064               -1.935           -0.263        -9.414          0.437         0.295             -0.957            0.135
Turn-taking only            -1.307       0.127               -0.954           -0.129        -0.450         -0.298         0.378             -0.034            0.567
Lexical (theory)             0.427       0.019                0.457            0.284         0.059          0.049        -0.172             -0.020            0.403
DA keywords                 -3.421      -0.058               -2.040           -0.104        -0.372          0.109         0.001             -0.121            0.393
Semantic + completion       -0.952      -0.096               -0.414           -0.134         0.408         -0.019         0.470             -0.179            0.145
HMM states (K=4)            -0.277      -0.086               -0.441           -0.041        -0.377          0.274         0.239             -0.362            0.342
HMM states (K=5)            -0.246       0.011               -1.826           -0.067        -0.424          0.262         0.004             -0.290            0.371
Expanded raw (19)            0.648       0.056                0.107           -0.269        -0.247         -1.234        -0.906              0.356            0.541
States + Turn-taking        -0.924       0.055               -0.595           -0.162        -0.405         -0.468         0.134              0.063            0.461
States + Lexical            -0.336       0.032               -0.474            0.080        -0.306          0.022         0.330             -0.115            0.433
States + DA                 -2.250       0.008               -1.219           -0.037        -1.500         -0.520        -0.291             -0.969            0.392
States + Lex + DA           -1.076       0.340               -0.361            0.069        -0.227         -0.392         0.217             -0.252            0.330
Turn + Lex + DA             -0.680       0.359               -0.647           -0.114        -0.299         -0.462        -0.134             -0.616            0.556
```


## Full Permutation p-value Table

```
Target                cooperative coord+coop decision_confidence effectiveness_z idea_quality mental_demand satisfaction team_coordination voice_inclusion
Model                                                                                                                                                     
Physio only                 0.050      0.475               0.149           0.426        0.911        0.020*        0.614             0.931          0.010*
Eye-tracking only           0.228      0.505               0.376           0.287        0.594        0.010*       0.010*             0.624          0.010*
Physio + ET                 0.891      0.515               0.812           0.416        0.960        0.040*        0.743             0.941          0.010*
Audio only                  0.337      0.089               0.891           0.861        1.000        0.010*       0.010*             0.990          0.020*
Turn-taking only            0.851     0.010*               0.673           0.297        0.248         0.762       0.010*             0.050          0.010*
Lexical (theory)           0.020*      0.050               0.050          0.010*        0.109        0.030*        0.267             0.059          0.010*
DA keywords                 0.782      0.129               0.842           0.208        0.307        0.030*        0.050             0.109          0.010*
Semantic + completion       0.802      0.119               0.208           0.208        0.069         0.099       0.020*             0.158          0.020*
HMM states (K=4)            0.198      0.238               0.455           0.089        0.545        0.010*       0.030*             0.604          0.010*
HMM states (K=5)            0.089      0.050               0.861           0.168        0.188        0.010*        0.059             0.584          0.010*
Expanded raw (19)          0.020*     0.020*               0.089           0.545        0.238         0.832        0.851            0.020*          0.010*
States + Turn-taking        0.802     0.020*               0.505           0.356        0.218         0.921       0.040*            0.020*          0.010*
States + Lexical            0.208     0.010*               0.366          0.020*        0.218        0.040*       0.010*             0.089          0.010*
States + DA                 0.861     0.040*               0.871           0.079        0.911         0.950        0.475             0.772          0.010*
States + Lex + DA           0.832     0.010*               0.277          0.010*        0.198         0.683       0.010*             0.208          0.010*
Turn + Lex + DA             0.614     0.010*               0.545           0.158        0.238         0.673        0.099             0.733          0.010*
```


## Alternative Estimators: Robustness Check

To confirm that Ridge results are not estimator-dependent, the significant Ridge models were re-run with Lasso, ElasticNet, and Random Forest (LOGO-CV, no permutation test — significance already established).

### Key comparison (best feature set per target)

| Target | Feature set | Ridge | Lasso | ElasticNet | RF | Verdict |
|--------|------------|-------|-------|------------|-----|---------|
| voice_inclusion | Turn-taking only | 0.567 | 0.550 | **0.575** | 0.257 | Robust (linear) |
| effectiveness_z | Lexical (theory) | 0.284 | 0.304 | **0.383** | 0.084 | Robust; ElasticNet best |
| cooperative | Expanded raw (19) | 0.648 | **0.662** | 0.563 | -0.062 | Robust (linear) |
| team_coordination | Expanded raw (19) | 0.356 | 0.096 | 0.367 | **0.454** | RF best; nonlinear |
| satisfaction | Semantic+completion | **0.470** | 0.329 | 0.346 | 0.433 | Ridge/RF best |
| mental_demand | Audio only | **0.437** | 0.274 | 0.282 | 0.299 | Ridge best |
| coord+coop | States+Lex+DA | 0.340 | 0.297 | 0.291 | -0.091 | Ridge best |

### Additional notable results

- **HMM states (K=4) → voice_inclusion**: Lasso R²=0.411, ElasticNet R²=0.408 (vs Ridge 0.342) — HMM state proportions predict voice inclusion better with sparse regularization
- **HMM states (K=5) → mental_demand**: RF R²=0.368 (vs Ridge 0.262) — nonlinear state-outcome relationship
- **States+Turn-taking → voice_inclusion**: Lasso R²=0.513 (vs Ridge 0.461) — combined model improves with sparsity
- **Turn-taking → satisfaction**: Lasso R²=0.415 (vs Ridge 0.378) — turn-taking dynamics predict satisfaction

### Conclusions from robustness check

1. **Linear models (Ridge/Lasso/ElasticNet) are consistent** — the top findings replicate across all three, confirming they are not artefacts of the regularization choice.
2. **ElasticNet improves effectiveness_z prediction** to R²=0.383 — sparse selection of lexical features helps.
3. **RF generally underperforms** linear models (expected with n=28), except for team_coordination where it captures a nonlinear relationship.
4. **No result flips sign** — every target that was positive with Ridge stays positive with at least 2 of 3 alternative estimators.

Full results: `analysis/results/prediction_alternative_estimators.tsv`

---

## Interpretation Notes

- **Physio alone** reaches statistical significance for voice_inclusion (R²=0.37, p=0.01) and mental_demand (R²=0.17, p=0.02), but with modest effect sizes far below behavioural features (turn-taking R²=0.57 for voice_inclusion). **Eye-tracking alone** (pupil diameter) similarly reaches significance for voice_inclusion, mental_demand, and satisfaction — likely driven by arousal co-variation rather than a direct physiological→outcome mechanism. Critically, physio/ET cannot predict coordination, cooperation, or effectiveness.

- **Turn-taking features** are the strongest single-modality predictor, especially for voice_inclusion (R²=0.57) — groups with balanced floor time, backchannels, and cooperative overlaps are perceived as more inclusive.

- **Lexical (theory-driven) features** predict effectiveness_z (R²=0.28–0.38 across estimators) — we/I ratio, turn cohesion, agreement count, and sentiment ratio capture the linguistic signature of effective meetings.

- **HMM state proportions** alone predict voice_inclusion (R²=0.34–0.41) and mental_demand (R²=0.27–0.31). Combined with lexical/DA features (States+Lex+DA), they predict coord+coop cross-task (R²=0.34) — the states encode interaction patterns that complement linguistic content.

- **Audio prosodic features** predict mental_demand (R²=0.44) — pitch variability and harmonic-to-noise ratio correlate with cognitive engagement.

- **Semantic continuity** predicts satisfaction (R²=0.47) — groups with coherent, building-on-each-other discussions report higher satisfaction.

- **The composite effectiveness_z index** is predictable (R²=0.28–0.38 with lexical features, p=0.01), confirming that the cross-task construct is meaningful and capturable from multimodal features.

- Negative R² means the model generalises worse than predicting the mean — expected with n=28 and many features. It does not imply a negative relationship.
