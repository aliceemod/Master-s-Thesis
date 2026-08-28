# HMM State Discovery — Methods & Interpretation Guide

*AffectAI thesis — last updated 2026-08-26*

---

## 1. What we are doing and why

The goal is to discover **latent interaction states** — recurring, qualitatively distinct modes of collective behaviour that groups cycle through during a task. Rather than analysing individual features one at a time, we let a data-driven model find natural clusters of windows that tend to co-occur across many signals simultaneously.

We use 30-second windows as the unit of analysis. Each window is described by ~30 features spanning five modalities:

| Modality | Features |
|---|---|
| Physio | Group mean HR, HR spread, HRV, EDA tonic mean/spread, EDA phasic rate, temperature |
| Transcript timing | Speaking density, silence, backchannels, laughter, active speakers, speaking entropy, overlap count/time, conflict-overlap ratio, four overlap-subtype timing measures |
| Eye-tracking | Group mean pupil, blink rate, gaze dispersion |
| Lexical | TTR, sentiment, we/I ratio, Gini participation, turn cohesion, elaboration, hedging, social composite |
| NLP | BERT-UMAP dims 1–5 (semantic topic), keyword DA (hedge, question, backchannel, statement, agreement), NLI DA (disagreement, compromise, off-task), semantic continuity, cross-speaker similarity |

The discovered states are then used in two ways: (1) to describe group dynamics qualitatively, and (2) as features in effectiveness prediction models.

---

## 2. Pipeline step by step

### Step 1 — Data loading and merging

Window-level features from five separate files are merged on `(group_id, task_id, window_index)`. A left-join is used so windows without NLP data still appear (with NaN for those features). Only windows that have transcript data (`tr_spk_duration_s` is not NaN) are kept, since transcript features define the interaction spine.

### Step 2 — Feature selection and preprocessing

Features with >60% missing values are dropped. The remainder are mean-imputed (simple strategy is acceptable here because missingness is mostly due to sparse NLP coverage, not structural absence). All features are then z-score standardised so they contribute equally regardless of original scale.

### Step 3 — PCA

PCA is applied to find the number of components needed to explain 80% of variance. This reduces noise and makes the subsequent clustering more stable. We retain `n_pcs` components (typically 6–8 with the expanded feature set).

### Step 4 — K selection via BIC

We fit a `GaussianMixture` (full covariance) for k = 2–8 and record the **Bayesian Information Criterion (BIC)** for each. BIC penalises model complexity — lower is better. The k that minimises BIC is `K_CHOSEN`.

**Important caveat:** BIC can over-select K with small datasets (n ≈ 700 windows, 10 groups). The check for empty states (Section 5 below) is essential.

### Step 5 — K-Means initialisation

K-Means (n_init=30) is run first to get stable centroids. These are passed to `GaussianMixture(means_init=...)` to avoid degenerate solutions. K-Means labels are also kept for comparison.

### Step 6 — Gaussian HMM via Baum-Welch + Viterbi

A plain GMM clusters windows independently of their temporal order. To impose temporal coherence we fit a true **Gaussian HMM** using the Expectation-Maximisation (Baum-Welch) algorithm:

1. **E-step (forward-backward):** for each group × task sequence, compute the posterior probability of being in each state at each time step (γ) and the posterior probability of each state-to-state transition (ξ), given the current GMM emissions and transition matrix.
2. **M-step:** update the K×K transition matrix A[i,j] = Σ_t ξ(t,i,j) / Σ_t γ(t,i) and the initial state distribution π.
3. Iterate until log-likelihood converges (typically < 10 iterations on this dataset).
4. **Viterbi decoding** with the *learned* transition log-probabilities to assign the most probable state sequence to each group × task.

The learned transition matrix is itself interpretable: it shows which states tend to follow which, and which states are "sticky" (high diagonal) vs. transitional.

### Step 7 — State characterisation

For each state we compute the mean z-score of every feature. The resulting heatmap shows which features are elevated (positive z, red) or suppressed (negative z, blue) relative to the overall mean. This is the primary interpretive output.

### Step 8 — Validation against self-report

We aggregate the proportion of windows in each state per group × task (a row in the prediction dataset). We then compute Spearman correlations between each state proportion and each post-task self-report item. A significant correlation (p < .10 given n=28) suggests the state captures something behaviourally meaningful.

### Step 9 — Export

State proportions (`S0_pct`–`SK_pct`) are exported to `analysis/results/hmm_state_proportions_expanded.tsv` for use as features in the effectiveness prediction models.

---

## 3. How to interpret the state heatmap

Each row is a state (0 to K−1). Each column is a feature. The colour shows the z-score: **red = high relative to the overall mean, blue = low**.

Look for clusters of consistently red or blue features that form a coherent narrative. For example:

- High `tr_conflict_overlap_ratio` + high `tr_ovl_competitive_timing` + high `tr_ovl_floor_fight` + low `tr_silence_duration_s` → **competitive/contentious speech floor**
- High `tr_speaking_entropy` + high `tr_backchannel_count` + high `tr_ovl_smooth` + high `lex_we_i_ratio` → **egalitarian co-construction**
- Low `tr_spk_duration_s` + high `tr_silence_duration_s` + low `tr_n_active_speakers` → **low-engagement / disengaged silence**
- High `group_eda_phasic_rate_hz_mean` + high `group_hr_mean_bpm_mean` + high `tr_ovl_simultaneous` → **physiological arousal with simultaneous speech**
- High `bert_umap*` dims in a specific direction → **semantic topic shift** (interpretation depends on what that UMAP dimension captures)

Do not over-interpret individual features. Treat the state as a *gestalt* pattern.

---

## 4. How to interpret the learned transition matrix

The Baum-Welch algorithm produces a K×K matrix A where A[i,j] is the estimated probability of transitioning from state i to state j. Look for:

- **High diagonal entries (> 0.6):** sticky states — once entered, the group tends to stay. These represent sustained interaction modes.
- **High off-diagonal entries:** common pathways between states. If A[2,4] is high, state 2 frequently gives way to state 4 (possibly an escalation or de-escalation pattern).
- **Degenerate rows (one entry near 1.0):** numerical artefact when a state appears only at sequence boundaries; interpret with caution.

---

## 5. How to interpret the temporal trajectories

Each subplot shows one group × one task. The x-axis is time in minutes; the y-axis is the state label. The step plot shows when the group is in each state. Look for:

- **Stable periods**: long stretches in one state → the group has settled into a mode
- **Transitions**: how often and when does the group switch? Early task vs. late task?
- **Group differences**: do some groups spend most time in state 0 (likely the dominant/baseline state) while others cycle more?
- **Task differences**: does the state distribution shift systematically from T1 to T2 to T3?

---

## 6. Critical check — empty states

After running, always check the mean state proportion across all group × task rows. If any state has **0% usage across the entire dataset**, BIC over-selected K and that state is a phantom (a Gaussian component that the GMM placed in an empty region of feature space). In that case, override `K_CHOSEN` to K−1 and re-run from the K-Means cell. Do not override to a round number without this evidence — BIC is the justification.

**Current run result (2026-08-26, 28 clean features):** BIC selects **K=5** (ΔBIC=109 vs K=4). All five states are populated. S3 appears in only 9/28 group×task pairs (mean 1.4% of windows) — it is a real but rare interaction mode, not a phantom.

---

## 7. Interpreting the self-report validation heatmap

Rows are self-report items; columns are state proportions. Values are Spearman ρ; cells marked `*` have p < .10.

- A **large positive ρ** between state_k_prop and `voice_inclusion_mean` means groups that spend more time in state k tend to report higher voice inclusion → state k is likely a participatory state.
- A **large negative ρ** with `mental_demand_mean` means spending time in that state is associated with lower cognitive load → possibly a smooth, routine coordination state.
- States with **no significant correlations** may still be structurally real (they represent a genuine interaction mode) but may not directly drive self-report outcomes.
- With n=28 rows, p < .10 corresponds to |ρ| ≈ 0.32. Treat these as exploratory signals, not confirmatory tests.

---

## 8. Connection to effectiveness prediction

The per-group × task state proportions (exported TSV) feed directly into the prediction models as a feature set labelled "HMM states new Kx". The logic is: if a state captures a high-quality interaction mode, then spending more time in it should predict better outcomes. This is tested via Ridge/Lasso/RF with Leave-Two-Groups-Out cross-validation. Permutation tests (1000 permutations) provide honest p-values.

---

## 9. Limitations

- **Small dataset**: n=28 group × task rows drives all aggregate-level analyses. Spearman correlations are illustrative.
- **BIC bias**: full-covariance GMM with many features on ~700 windows can over-fit, leading to phantom states (as observed above).
- **Temporal model is a true HMM but small-sample**: Baum-Welch is applied across only 30 sequences (10 groups × 3 tasks, with some missing), so transition estimates for rare states can be unreliable. Specifically, S0 (the least-used state, ~12% of windows) converges to a self-loop of 1.0 in the transition matrix — a numerical artefact caused by S0 windows clustering at sequence boundaries where outgoing transitions are unobservable.
- **NLP features have partial coverage**: BERT, NLI DA, and semantic continuity features have <100% window coverage. Mean imputation fills the gaps but may reduce discriminability for those features.
- **States are unlabelled**: the integer labels (0, 1, …) are arbitrary and change between runs. Do not compare state numbers across different notebook runs.
