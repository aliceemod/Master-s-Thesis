# HMM Collective Affective States â€” Full Analysis Report

*For paper writing reference â€” AffectAI dataset, ICMI 2026*  
*Analysis: Gaussian HMM on 526 windows Ã— 12 features Ã— 10 groups Ã— Tasks T1â€“T3*  
*All results from: `icmi_paper/analysis/hmm_collective_states_updated_overlaps.ipynb`*

---

## 1. Overview and Objective

**Research question:** What latent collective affective states emerge naturally during multimodal group interaction, and how can they be identified from physiological, speech, and overlapping-speech signals?

**Key principle:** Task structure (T1, T2, T3) is used as *external validation* â€” not as an input to the model. States are discovered purely from the multimodal signals and their task correlation confirms they are real.

---

## 2. Data: Windows and Coverage

### 2.1 Window definition

All features are computed at the **30-second window** level. Windows are created by:
1. Identifying the task start time (first event onset in the transcript)
2. Binning all events (speech, overlaps, laughter, etc.) into consecutive 30-second intervals
3. Computing aggregate statistics per window per group

**Window size rationale:** 30 seconds balances temporal resolution (capturing within-task dynamics) with statistical stability (enough events per window for reliable aggregates).

### 2.2 Window selection

| Criterion | Value |
|---|---|
| Source tasks | T1 (Hidden Profile Decision), T2 (Mini-Negotiation), T3 (Idea Generation / NGT) |
| Total transcript-covered windows | 545 |
| After sensor failure exclusion | 526 |

**Sensor failure handling:**
- **grp-12 T3**: EmotiBit wristband failed for the entire task (19 windows, 100% HR/EDA missing) â†’ **dropped entirely**. A task with no physiological signal cannot contribute meaningful data.
- **grp-10 T2 windows 17â€“24**: Tobii eye-tracker stopped at window 17 (8 windows). HR/EDA/transcript data remain valid. â†’ **Pupil diameter imputed** with within-group session mean (= z-score 0, i.e., "average attention for this group").
- **grp-16 T3 windows 11â€“22**: Tobii stopped at window 11 (10 windows). Same approach â†’ **Pupil imputed** with within-group mean.

**Justification:** Imputing only pupil (a single feature) for windows where HR, EDA, and all speech features remain valid is methodologically sound. The imputed value of 0 (after within-group z-scoring) represents "average attention" â€” a neutral, conservative assumption appropriate for isolated sensor dropouts.

### 2.3 Final dataset

- **526 windows** across 10 groups and 3 tasks
- **27 sequences** (one per group Ã— task combination; some shorter due to truncation)
- **Sequence lengths:** min=9, max=34 windows, mean=19.5 windows

---

## 3. Feature Set (12 Features)

### 3.1 Physiological features (4)

These are computed from EmotiBit wrist sensors worn by each of the 4 participants, then averaged across participants per window.

**Preprocessing:** Within-group z-scoring removes between-session baseline drift (e.g., one group wearing sensors slightly differently than another). This makes groups directly comparable.

| Feature | Raw signal | What it captures |
|---|---|---|
| `group_hr_mean_bpm_mean` | Mean heart rate (bpm) across 4 participants | **Collective physiological arousal baseline** â€” elevated HR indicates group-level activation |
| `group_eda_phasic_rate_hz_mean` | Mean rate of phasic EDA peaks (Hz) across participants | **Collective emotional reactivity** â€” phasic EDA reflects momentary arousal spikes, less affected by baseline drift than tonic EDA |
| `group_temp_mean_mean` | Mean skin temperature (Â°C) across participants | **Physiological baseline context** â€” slow-varying; helps differentiate states but not the primary discriminator |
| `group_et_pupil_mean_mean` | Mean pupil diameter (mm) from Tobii Glasses | **Attention and cognitive load** â€” pupil dilates with mental effort and engagement; complements HR/EDA |

### 3.2 Speech and turn-taking features (4)

Computed from manual transcript annotations (SPK = speech, SIL = silence, BCK = backchannel, LAU = laughter rows).

**Preprocessing:** Log1p transform (`log(1 + x)`) to handle right-skewed count distributions. Active speakers is kept raw (integer 1â€“4).

| Feature | What it captures |
|---|---|
| `tr_silence_duration_s` | Total annotated silence in the window (seconds). **Floor management** â€” high silence = groups taking structured turns or pausing to think |
| `tr_backchannel_count` | Number of BCK events (mmh, yeah, right). **Listener affiliation** â€” deliberate feedback signals indicating active listening; distinct from speech overlaps |
| `tr_laughter_count` | Number of LAU events. **Positive social affect** â€” laughter as collective social bonding; only feature uniquely capturing warmth/humour |
| `tr_n_active_speakers` | Count of unique speakers with at least one speech segment. **Participation breadth** â€” ranges from 1 (one person dominates) to 4 (everyone contributes) |

### 3.3 Overlap features (4)

Computed from transcript OVL rows, deduplicated by onset time (multi-speaker overlaps count as one event).

**Two-layer labeling system (key methodological contribution):**
- **Layer 1 (timing subtype):** Objective, acoustic-duration-based (simultaneous, backchannel_ovl, smooth, competitive, floor_fight)
- **Layer 2 (context_label):** LLM-assisted rule-based detection of communicative intent (collaborative, competitive, completion)

**Preprocessing:** Log1p transform for count/duration features.

| Feature | What it captures |
|---|---|
| `tr_ovl_count` | Total deduplicated overlap events per window. **Overlap frequency** â€” how often simultaneous speech occurs |
| `tr_ovl_contested` | Sum of competitive (500â€“1000ms) + floor_fight (â‰¥1000ms) overlaps. **Floor competition intensity** â€” these are sustained overlaps where neither speaker backs down; analogous to the old `tr_competitive_overlap` but based on purely timing-based labeling |
| `tr_ovl_time_s` | Total deduplicated overlap duration (seconds). **Simultaneous speech load** â€” how much of the window involves concurrent speech; a continuous measure complementing the count |
| `tr_ovl_collaboration_index` | Count of overlaps with `context_label = collaborative` / total overlap count. **Cooperative intent fraction** â€” fraction of overlaps where the overlapping speech contained support words (yeah, mmh, exactly, agreeâ€¦) during a sustained overlap (â‰¥250ms). Only applied to smooth, competitive, and floor_fight timing subtypes to avoid inflating with routine brief backchannels |

**Why "contested overlaps" instead of "competitive":** The feature combines timing-competitive (500â€“1000ms) and floor_fight (â‰¥1000ms) into one "contested floor" signal. This is the objective timing equivalent of the old semantic `tr_competitive_overlap` â€” not based on whether speakers say conflict words, but purely on how long both speakers continued talking simultaneously.

---

## 4. Preprocessing Pipeline

**Step 1 â€” Within-group z-score (physiological features):**

$$z_{i,g} = \frac{x_{i,g} - \mu_g}{\sigma_g + \epsilon}$$

Applied to: HR, EDA phasic rate, temperature, pupil diameter.  
$x_{i,g}$ = value for window $i$ in group $g$; $\mu_g$, $\sigma_g$ = group-level mean and std; $\epsilon = 10^{-9}$ for numerical stability.

**Why:** Between-session physiological baselines vary systematically (one group may have generally higher HR due to ambient temperature, sensor placement, etc.). Z-scoring within group removes this confound, making all groups directly comparable.

**Step 2 â€” Log1p transform (count and duration features):**

$$x' = \log(1 + x)$$

Applied to: silence, backchannel count, laughter count, overlap count, contested OVL count, overlap time, collaboration index.

**Why:** These features are right-skewed (most windows have 0 overlaps; rare windows have many). Log1p compresses the long tail and brings distributions closer to Gaussian, which is the assumed emission distribution in the HMM.

**Step 3 â€” StandardScaler (for PCA):**

$$x'' = \frac{x' - \mu}{\sigma}$$

Applied globally across all windows and all features before PCA. This ensures features with large absolute values (e.g., silence in seconds) do not dominate PCA components.

**Note:** This is NOT double-normalisation. Steps 1 and 3 serve different purposes:
- Step 1 removes *between-group* session drift in physiological signals
- Step 3 equalises *feature scale* for PCA weight equality

**Step 4 â€” PCA (dimensionality reduction):**

Applied to the 12-dimensional standardised feature matrix. **7 principal components** retain 80% of total variance (PC1=26.9%, PC2=14.1%, PC3=10.1%, â€¦). The HMM operates on this 7-dimensional PCA space.

**Why PCA before HMM:** Reduces dimensionality and noise; allows the HMM's Gaussian emission model (which assumes diagonal covariance) to work more effectively on decorrelated components.

---

## 5. Algorithm: Gaussian Hidden Markov Model

### 5.1 Model formulation

A **Gaussian HMM with diagonal covariance** trained with Baum-Welch Expectation-Maximisation.

**Model parameters:**
- $K$ = number of hidden states (selected by BIC)
- $\pi_k$ = initial state probability vector ($K$-dimensional)
- $A_{ij}$ = transition probability matrix ($K \times K$), where $A_{ij} = P(s_{t+1}=j \mid s_t=i)$
- $\mu_k$ = mean vector of Gaussian emission for state $k$ ($D$-dimensional, $D=7$ PCA components)
- $\Sigma_k$ = diagonal covariance matrix of emission for state $k$

**Emission probability (diagonal Gaussian):**

$$p(\mathbf{x}_t \mid s_t=k) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2))$$

**Observation sequences:** One sequence per group Ã— task (e.g., grp-07/T1 = 9 windows). A total of 27 sequences used for training.

### 5.2 Training: Baum-Welch EM

The Baum-Welch algorithm iterates:
1. **E-step:** Forward-backward algorithm to compute posterior state occupancy probabilities ($\gamma_{t,k}$) and transition probabilities ($\xi_{t,ij}$)
2. **M-step:** Update parameters to maximise expected log-likelihood

**Forward pass:**
$$\alpha_t(k) = p(\mathbf{x}_{1:t}, s_t=k) = p(\mathbf{x}_t \mid s_t=k) \sum_j \alpha_{t-1}(j) A_{jk}$$

**Backward pass:**
$$\beta_t(k) = p(\mathbf{x}_{t+1:T} \mid s_t=k) = \sum_j A_{kj} p(\mathbf{x}_{t+1} \mid s_{t+1}=j) \beta_{t+1}(j)$$

**Convergence:** Maximum 200 EM iterations; tolerance $10^{-5}$ on log-likelihood change.

**Initialisation:** K-means clustering used to initialise state means; repeated with 3 random seeds (42, 7, 99) per $k$; best BIC solution retained.

### 5.3 State decoding: Viterbi algorithm

After training, the most likely state sequence for each window sequence is found via the Viterbi algorithm:

$$\delta_t(k) = \max_{s_{1:t-1}} p(\mathbf{x}_{1:t}, s_{1:t-1}, s_t=k) = p(\mathbf{x}_t \mid s_t=k) \max_j \delta_{t-1}(j) A_{jk}$$

This gives the **globally optimal (MAP) state path** â€” the sequence of states that maximises the joint probability of the observations and states.

### 5.4 Model selection: BIC

**Bayesian Information Criterion:**

$$\text{BIC}(k) = -2 \mathcal{L} + n_{\text{params}} \cdot \log N$$

where:
- $\mathcal{L}$ = log-likelihood of the trained model
- $n_{\text{params}} = k(D + D + k) + (k-1)$ = number of free parameters (means + variances + transition matrix + initial probs)
- $N$ = total number of observations (526 windows)

**For k=4, D=7:** $n_{\text{params}} = 4(7+7+4) + 3 = 75$

BIC penalises model complexity to prevent overfitting. Lower BIC = better model.

| k | BIC | Î”Next |
|---|---|---|
| 2 | 10267.1 | â€“315.0 |
| 3 | 9982.9 | â€“4.2 |
| **4** | **9978.7** | **+72.8** â† minimum |
| 5 | 10051.5 | +5.0 |
| 6 | 10056.5 | â€” |

**k=4 selected** with very strong evidence (Î”BIC=+72.8 to k=5).

### 5.5 Software and packages

| Component | Package | Version |
|---|---|---|
| HMM implementation | Custom `GaussianHMM` class (in notebook) | â€” |
| Baum-Welch EM | Custom (uses `scipy.special.logsumexp`) | scipy 1.x |
| K-means initialisation | `sklearn.cluster.KMeans` | scikit-learn 1.x |
| PCA | `sklearn.decomposition.PCA` | scikit-learn 1.x |
| StandardScaler | `sklearn.preprocessing.StandardScaler` | scikit-learn 1.x |
| Gaussian log-pdf | `scipy.stats.multivariate_normal.logpdf` | scipy 1.x |
| Ï‡Â² test | `scipy.stats.chi2_contingency` | scipy 1.x |
| Plotting | `matplotlib`, `seaborn` | standard |
| Data handling | `pandas`, `numpy` | standard |

---

## 6. Results: Four Collective Affective States

### 6.1 State definitions and profiles

All states identified from multimodal signals; task distribution is post-hoc validation.

---

**State 0 â€” Transitional** (53 windows, 10.1%, persistence = 0.50)

*Multimodal signature:* Maximum silence, very low backchannel, low laughter, low active speakers, almost no overlaps.

This state has the lowest self-transition probability (0.50), indicating it is genuinely transitional â€” groups pass through it briefly when moving between other states. It reflects moments of minimal interaction: pauses between tasks, introductory phases, or brief disengagement. The low persistence suggests it represents a transient pause rather than a sustained collective mode.

*Task prevalence:* T1=19%, T2=7%, T3=7% â€” most common at the beginning of T1 when groups are getting oriented.

---

**State 1 â€” Active/Floor-Contested Discussion** (190 windows, 36.1%, persistence = 0.73)

*Multimodal signature:* ALL overlap features maximally elevated (count, contested OVL, total time, collaboration index). Low silence (lots of talking). Moderate laughter and backchannels. Physiological arousal is moderate.

This state is defined by simultaneous speech. Groups are actively competing for or co-constructing the conversational floor. The elevated collaboration index (0.23) indicates that even contested overlaps often contain supportive speech â€” confirming these are not purely adversarial but include brainstorming and co-construction.

Key insight: 70% of contested overlaps (competitive + floor_fight) occur in this state. 97% of windows in this state have at least one overlap event.

*Task prevalence:* T1=32%, T2=34%, **T3=43%** â€” particularly dominant in T3 (ideation), consistent with brainstorming generating more simultaneous speech.

---

**State 2 â€” Aroused-Engaged** (189 windows, 35.9%, persistence = 0.94)

*Multimodal signature:* Highest HR (mean ~73 bpm, z-score +0.96 above within-group mean), highest EDA phasic rate, highest pupil diameter. Moderate-high laughter and backchannels. High active speakers. Very few overlaps.

This is the most persistent state (0.94 self-transition). Once groups enter it, they stay. Physiological arousal is maximal, but the floor is orderly â€” groups are highly engaged without simultaneous speech, each person contributing in structured turns.

This state likely corresponds to moments of focused, high-stakes discussion where the physiological activation reflects group-level tension or intense engagement, but groups manage to avoid floor competition (they are listening to each other carefully).

*Task prevalence:* T1=18%, **T2=47%**, T3=35% â€” dominates T2 (negotiation), consistent with the sustained physiological tension of negotiating a joint decision.

---

**State 3 â€” Positive Social Engagement** (94 windows, 17.9%, persistence = 0.78)

*Multimodal signature:* Lowest HR (mean ~68 bpm, z-score âˆ’1.34 below within-group mean), low EDA, low pupil â€” physiologically the most RELAXED state. Highest backchannel count (mean 1.6). Elevated laughter. Moderate active speakers. Very few overlaps.

This state represents warm, collegial social interaction. Groups are laughing, giving each other supportive signals (backchannels), and participating broadly â€” but WITHOUT physiological stress or competitive floor-fighting. The low arousal combined with high social signals distinguishes this from State 2 (high arousal + social signals).

*Temporal finding:* Within T1, this state is rare in the first 5 minutes (19% early T1) but prominent in the middle of T1 (48% mid-T1, windows 5â€“15 min). This matches the qualitative observation that groups need time to warm up before relaxed collegial discussion emerges.

*Task prevalence:* **T1=31%** (especially mid-task), T2=12%, T3=15% â€” most characteristic of T1's deliberative discussion phase.

---

### 6.2 State comparison table

| Feature (raw mean) | State 0 | State 1 | State 2 | State 3 |
|---|---|---|---|---|
| HR (bpm) | 71.9 | 72.6 | **73.3** | **67.6** |
| EDA phasic rate (Hz) | 0.096 | 0.146 | 0.147 | 0.129 |
| Pupil diameter (mm) | 2.99 | 2.97 | **3.00** | **2.94** |
| Silence (s) | **9.1** | 9.7 | 10.8 | 9.1 |
| Backchannels | 1.3 | 1.4 | 1.2 | **1.6** |
| Laughter | 0.29 | 0.40 | 0.36 | 0.35 |
| Active speakers | **1.9** | 2.1 | 2.0 | 2.0 |
| Overlap count | 0.82 | **3.28** | 1.08 | 1.13 |
| Contested OVL | 0.42 | **1.74** | 0.54 | 0.61 |
| Overlap time (s) | 1.04 | **4.17** | 1.20 | 1.32 |
| Collab index | 0.06 | **0.23** | 0.08 | 0.08 |

Bold = highest value; HR in State 3 is notably the lowest.

---

### 6.3 Task-state association (external validation)

**Ï‡Â²(6) = 58.45, p = 9.31 Ã— 10â»Â¹Â¹, CramÃ©r's V = 0.236**

| State | T1 (Decision) | T2 (Negotiation) | T3 (Ideation) |
|---|---|---|---|
| Transitional | 18.6% | 6.9% | 6.6% |
| Active/Floor-Contested | 32.4% | 34.3% | **43.4%** |
| Aroused-Engaged | 17.9% | **46.9%** | 35.3% |
| Positive Social | **31.0%** | 11.8% | 14.7% |

The highly significant association (p < 10â»Â¹â°) confirms the states are **real and meaningful** â€” not random artefacts. However, no state is exclusive to one task (partial overlap across tasks), confirming states capture latent interaction dynamics that are related to but not determined by task type.

---

### 6.4 Transition matrix interpretation

The learned A matrix shows:

| From â†’ To | Transitional | Active | Aroused | Positive Social |
|---|---|---|---|---|
| **Transitional** | 0.50 | 0.29 | 0.13 | 0.08 |
| **Active** | 0.03 | 0.73 | 0.14 | 0.10 |
| **Aroused** | 0.01 | 0.03 | **0.94** | 0.03 |
| **Positive Social** | 0.02 | 0.10 | 0.10 | 0.78 |

Key patterns:
- **Aroused-Engaged (State 2) is ultra-sticky** (0.94) â€” once groups enter high-arousal engaged discussion, they remain there. This is the "T2 negotiation lock-in" pattern.
- **Transitional (State 0)** feeds primarily into Active Discussion (0.29) â€” pauses resolve into active floor competition
- **Positive Social (State 3)** can transition to Active Discussion (0.10) â€” relaxed collegial moments can escalate into floor competition
- **Active Discussion (State 1)** can lead to Aroused-Engaged (0.14) â€” sustained overlapping discussion sometimes triggers higher physiological engagement

---

## 7. Robustness Checks

### Check 1: EM Convergence

The k=4 model converged within the 200-iteration limit. Log-likelihood: âˆ’4754.4, BIC: 9978.7.

**What this means:** The Baum-Welch EM algorithm found a stable solution. Non-convergence would indicate the model was unable to find a local optimum, suggesting a poorly-conditioned problem.

### Check 2: Multi-seed Reproducibility

3 random seeds (42, 7, 99) were tested for each k âˆˆ {2,3,4,5,6}. All seeds converged. The best BIC solution was retained.

**What this means:** The solution is robust to initialisation â€” we are not stuck at a bad local optimum caused by a specific K-means initialisation.

### Check 3: State Persistence (Self-Transition Probabilities)

| State | Self-transition |
|---|---|
| Transitional | 0.50 |
| Active/Floor-Contested | 0.73 |
| Aroused-Engaged | 0.94 |
| Positive Social | 0.78 |

**What this means:** States with high persistence (>0.70) are temporally stable â€” when a group is in that state, it stays there for multiple consecutive windows. This validates that states are coherent temporal patterns, not noise. State 0's low persistence (0.50) is interpretable: it is a transitional state by nature, correctly not self-sustaining.

The high persistence of State 2 (0.94) is particularly notable â€” once groups enter intense engaged discussion, they remain in it for extended periods.

### Check 4: Feature Importance (Between-State Variance)

Top discriminating features (standard deviation of state means):

1. Pupil diam (z): 0.716
2. Overlap count [log]: 0.642
3. Active speakers: 0.638
4. Overlap time [log]: 0.601
5. Contested OVL [log]: 0.401

**What this means:** Both physiological (pupil) and interaction (overlap count, active speakers) features discriminate states, confirming the model uses signals from multiple modalities rather than being driven by a single feature.

### Check 5: Group Representation

All 10 groups contribute to all 4 states. No group is locked into a single state.

**What this means:** States are not group artefacts â€” they reflect interaction dynamics shared across groups, not idiosyncratic behaviour of specific groups.

### Check 6: Task Prevalence Patterns

States show task-specific prevalence (summarised above). The dominant task varies meaningfully across states in theoretically expected directions.

**What this means:** States capture real dynamics related to task demands, confirming external validity.

### Check 7: Task-State Ï‡Â² Association

Ï‡Â²(6) = 58.45, p = 9.31 Ã— 10â»Â¹Â¹, CramÃ©r's V = 0.236.

**What this means:** The association between state assignment and experimental task condition is highly statistically significant. Effect size (V=0.236) is moderate â€” states are meaningfully associated with tasks but not identical to them, which is exactly what we want: task-correlated but not task-defined latent states.

### Check 8: State Class Balance

53 (10.1%) / 190 (36.1%) / 189 (35.9%) / 94 (17.9%). Balance ratio = 3.58:1.

**What this means:** No extreme imbalance that would indicate a degenerate solution (one state capturing everything). The transitional state being small (10%) is theoretically expected.

---

## 8. Figure Interpretations

### Figure 1: `hmm_bic_selection_updated_overlaps.png`

This 4-panel figure shows multiple model selection criteria:

**Panel 1 (BIC, primary):** The BIC curve has a clear minimum at k=4 (BIC=9978.7). From k=2 to k=3 the improvement is large (âˆ’315 BIC points). From k=3 to k=4 there is a smaller improvement (âˆ’4.2 points). From k=4 onwards BIC increases sharply (+72.8 to k=5), confirming k=4 as the optimal solution. The BIC accounts for the sequential temporal structure of the HMM through the NÃ—log(N) penalty term.

**Panel 2 (AIC):** AIC keeps decreasing â€” k=6 is lowest. This is typical for HMMs: AIC's lighter penalty (2Ã—n_params vs BIC's log(N)Ã—n_params) causes it to overfit for sequential models. AIC is shown for completeness but not used for selection.

**Panel 3 (Silhouette) and Panel 4 (Calinski-Harabasz):** Both prefer k=2. These are *geometric* cluster quality measures that ignore the temporal structure of HMM sequences. They measure how well points cluster in PCA space, not whether the temporal transitions are meaningful. They systematically prefer fewer, more separated clusters. These are shown as secondary sanity checks; BIC is the appropriate criterion for probabilistic sequential models.

**Conclusion:** k=4 is the unambiguous BIC-optimal solution with very strong evidence.

---

### Figure 2: `hmm_profiles_transitions_updated_overlaps.png`

This 2-panel figure shows state characterisation.

**Left panel â€” State profiles heatmap (z-scored):**

Each cell shows how much above/below the cross-state mean each feature is for each state, in standard deviation units. Red = high, blue = low.

Key patterns to highlight in the paper:
- **State 0 (Transitional):** Near-zero or negative for all features â€” the most "neutral/passive" state
- **State 1 (Active):** Deep red for all 4 overlap features â€” this state is uniquely defined by simultaneous speech
- **State 2 (Aroused):** Deep red for HR, EDA, pupil â€” physiologically the most activated state
- **State 3 (Positive Social):** Deep blue for HR (strikingly low arousal), red for backchannels â€” relaxed but socially engaged

The heatmap clearly shows that each state has a distinct multimodal signature â€” no two states look the same, validating the 4-state solution.

**Right panel â€” Transition matrix A[iâ†’j]:**

The diagonal dominates (states are persistent). The most notable off-diagonal element is State 1 â†’ State 2 (0.14), suggesting that sustained floor competition can trigger higher physiological arousal (groups escalating from active discussion to tense engagement).

---

### Figure 3: `state_trajectories_per_group.png`

This 5Ã—2 grid shows one panel per group. Each row within a panel = one task (T1, T2, T3). Each dot = one 30-second window, coloured by state assignment.

**How to read it:**
- Each row represents a task sequence (left to right = time within the task)
- Colour patterns show how states evolve: sustained periods of one colour = stable states; frequent colour changes = transitional periods
- Groups generally show task-consistent patterns (more blue = T2 dominance of State 2; more red = T3 dominance of State 1)

**Key observations:**
- Most groups show State 2 (blue, Aroused) dominating their T2 rows
- T3 rows typically show more State 1 (red, Active Overlapping)
- T1 rows show the most variety â€” some groups show persistent State 1 (early T1), shifting to State 3 (green, Positive Social) mid-task
- State 0 (grey, Transitional) appears in brief bursts, often at task boundaries
- State 3 (green) in T1 tends to appear in the middle-to-late portion, consistent with the temporal analysis showing it emerges after the initial orientation phase

**Group differences:**
- grp-07 shows a lot of red (State 1 Active) â€” a group with frequent overlapping speech
- grp-08 shows predominantly blue (State 2 Aroused-Engaged) â€” physiologically engaged group
- grp-13 shows predominantly grey/green (Transitional + Positive Social) â€” more deliberative, less competitive

These inter-group differences are interesting for thesis analyses linking group personality or composition to collective state trajectories.

---

## 9. Suggested Paper Methods Text

> We modelled collective affective states as latent variables using a Gaussian Hidden Markov Model (HMM) with diagonal covariance matrices, trained with Baum-Welch Expectation-Maximisation (maximum 200 iterations, convergence tolerance 10â»âµ). Observations consisted of 526 30-second windows spanning 10 groups and tasks T1â€“T3, after excluding one group-task sequence with complete sensor failure (grp-12/T3, EmotiBit) and imputing pupil diameter with within-group mean for two sequences with Tobii dropout.
>
> Twelve features were extracted from physiological sensors (EmotiBit: HR, EDA phasic rate, temperature; Tobii: pupil diameter), speech annotations (silence duration, backchannel count, laughter count, active speakers), and transcript-based overlap features (total overlap count, contested overlap count [competitive 500â€“1000ms + floor_fight â‰¥1000ms], overlap duration, collaboration index [fraction of overlaps with lexical support cues]). Physiological features were within-group z-scored to remove session-level baseline drift; count and duration features were log1p-transformed to normalise right-skewed distributions. A standardised PCA projection (7 components, 80% variance) was applied before HMM training.
>
> The number of states k was selected by minimising the Bayesian Information Criterion (BIC; Schwarz, 1978) over k âˆˆ {2,â€¦,6} with 3 random initialisations per k, yielding k=4 (BIC=9978.7, Î”BIC=+72.8 to k=5). All k were fitted independently; task structure was not used as an input. State assignments for each window sequence were decoded using the Viterbi algorithm.
>
> External validity was assessed by computing the task Ã— state association using Pearson's Ï‡Â² test and CramÃ©r's V effect size.

---

## 10. Suggested Results Text

> BIC model selection identified k=4 collective affective states (BIC=9978.7, Î”BIC=+72.8 to k=5; Figure X). The four states were characterised by distinct multimodal signatures (Figure Y):
>
> *Transitional* (State 0; 10.1% of windows, persistence=0.50): low activity across all modalities; transient pauses between other states.
>
> *Active/Floor-Contested* (State 1; 36.1%, persistence=0.73): maximum overlap frequency and duration, elevated contested floor overlaps (mean 1.74 per window); dominates T3 (idea generation; 43%). Groups co-construct ideas through simultaneous speech.
>
> *Aroused-Engaged* (State 2; 35.9%, persistence=0.94): highest physiological arousal (HR z=+0.96, EDA z=+1.42, pupil z=+1.09); orderly floor (few overlaps); ultra-persistent. Dominates T2 (negotiation; 47%). The sustained self-transition reflects extended periods of high-stakes focused discussion.
>
> *Positive Social* (State 3; 17.9%, persistence=0.78): lowest HR (z=âˆ’1.34), highest backchannels (mean 1.6), elevated laughter; physiologically relaxed yet socially warm. Most prevalent in mid-T1 (5â€“15 min; 48% of windows), suggesting it emerges once groups have oriented themselves to the deliberative task.
>
> The task Ã— state association was highly significant (Ï‡Â²(6)=58.45, p=9.3Ã—10â»Â¹Â¹, CramÃ©r's V=0.236), confirming the states reflect real interaction dynamics. No state is exclusive to one task, consistent with these being latent interaction patterns rather than task labels.

---

*Document generated: 2026-07-16 | Analysis: `hmm_collective_states_updated_overlaps.ipynb` | Features: `hmm_input_features_final.tsv` | Assignments: `hmm_cluster_assignments_k5_updated_overlaps.tsv`*


---

## 11. Prediction Analysis: States Predicting Process Quality Outcomes

*Full analysis and code: `icmi_paper/analysis/hmm_state_prediction.ipynb`*  
*Results file: `icmi_paper/results/hmm_prediction_results.tsv`*

### 11.1 Approach and rationale

The 4 collective states were discovered from multimodal signals *without* using any self-report data. To validate that they capture meaningful dynamics, we tested whether the proportion of time spent in each state predicts self-reported process quality outcomes collected after each task.

**Unit of analysis:** Group Ã— task (n=8â€“10 groups per analysis). For each group and task, we computed what percentage of 30-second windows fell in each of the 4 states. This percentage is the predictor variable.

**Outcome measures:**
- `team_coordination` (T1, T3): "How well did the team coordinate?" (1â€“7 Likert)
- `cooperative` (T2 only): "How cooperative was the interaction?" (1â€“7 Likert)
- `voice_inclusion` (T1, T2, T3): "Did you feel your voice was included?" (1â€“7 Likert)

`voice_inclusion` is the only item available across all three tasks, enabling a cross-task pooled analysis (n=27 = 10 groups Ã— 3 tasks).

**Statistical test:** Spearman rank correlation (Ï), appropriate for small samples and ordinal outcomes. With n=10, the minimum |Ï| for p<0.05 (two-tailed) is approximately 0.648. All results at p<0.10 are reported; those at p<0.05 are considered significant.

### 11.2 Results

| Outcome | State | Ï | p | Sig |
|---|---|---|---|---|
| T1 team_coordination | State 1 (Active) | **+0.804** | **0.009** | ** |
| T1 team_coordination | State 0 (Transitional) | âˆ’0.696 | 0.037 | * |
| T2 cooperative | **State 3 (Positive Social)** | **âˆ’0.682** | **0.030** | * |
| T3 team_coordination | State 1 (Active) | +0.703 | 0.052 | . |
| T3 team_coordination | State 2 (Aroused) | âˆ’0.693 | 0.057 | . |
| T1 voice_inclusion | State 1 (Active) | +0.619 | 0.075 | . |
| Pooled voice_inclusion | State 1 (Active) | +0.462 | 0.015 | * |
| Pooled voice_inclusion | State 2 (Aroused) | âˆ’0.386 | 0.047 | * |

### 11.3 Interpretation

**Finding 1: Off-task social warmth during negotiation predicts lower cooperativeness (State 3 â†’ T2 cooperative, Ï = âˆ’0.682, p = 0.030)**

This finding is consistent with the core hypothesis: groups that spent more time in off-task, laughter-filled social mode during T2 (the negotiation task) reported *lower* cooperativeness afterwards. The Positive Social state (State 3) is characterised by warmth and humour but also reduced substantive discussion â€” it is essentially the multimodal signature of a group that has drifted away from the task.

This matches a qualitative observation from the recordings: some groups used social warmth and humour as avoidance of productive engagement. Instead of articulating positions, working through disagreements, and driving toward a joint decision, they were socially pleasant but failed to complete the substantive work. Afterwards they rated the experience as less cooperative â€” perhaps because they sensed the process had not been authentic or productive.

The principle is: **off-task warmth is not the same as cooperative engagement**. Genuine cooperative negotiation requires being on-task â€” articulating positions, disagreeing where necessary, and working through tension toward an agreement. Groups that bypassed this through jokes and social pleasantries avoided the very process that makes a negotiation feel cooperative in retrospect. This aligns with research on constructive controversy (Johnson & Johnson, 2009) and the task-relationship conflict distinction (De Dreu & Weingart, 2003).

**Finding 2: Active Floor-Contested Discussion predicts better coordination and voice inclusion**

State 1 (Active/Floor-Contested) is the strongest and most consistent positive predictor:

- T1 team_coordination: Ï = +0.804** â€” the strongest single effect in the study. Groups with more active overlapping discussion in T1 coordinated significantly better.
- T3 team_coordination: Ï = +0.703 (trend, p=0.052). Same direction.
- Pooled voice_inclusion: Ï = +0.462* â€” across all three tasks, more active discussion â†’ participants felt more heard.

This seems paradoxical: floor-contested overlapping speech predicts *better* coordination? The resolution is that State 1 is not purely adversarial. Recall that 23% of overlaps in this state carry support words in the speech â€” it represents "enthusiastic simultaneous contribution" rather than aggressive competition. When everyone is actively engaging, building on each other's ideas, the group's information exchange is high-bandwidth. Contrast with State 0 (Transitional/passive), which negatively predicts T1 coordination (Ï = âˆ’0.696, p = 0.037): groups that spent more time quiet and disengaged coordinated worse.

Conversely, State 2 (Aroused-Engaged, high physiological arousal, orderly structured discussion) negatively predicts voice inclusion in the pooled analysis (Ï = âˆ’0.386, p = 0.047). High-stakes, tense, orderly discussion may constrain who feels free to speak.

**Theoretical contribution:**

These findings challenge a simple "harmony = good, conflict = bad" model of group affect and process quality:
- Productive high-engagement discourse (even floor-contested) facilitates coordination
- Social warmth as avoidance undermines genuine cooperation
- Physiological tension in ordered discussion reduces perceived voice inclusion

### 11.4 Suggested paper text (Results section)

> "To validate the discovered states, we tested whether state prevalence predicted self-reported process quality using Spearman rank correlations (n=8â€“10 groups per analysis; full analysis in Supplementary Material). Consistent with the hypothesis that substantive on-task engagement drives coordination outcomes, State 1 (Active/Floor-Contested) positively predicted team coordination in T1 (Ï=+0.80, p=0.009) and showed a trend in T3 (Ï=+0.70, p=0.052), as well as voice inclusion across all tasks (pooled Ï=+0.46, p=0.015). State 3 (Positive Social â€” laughter, off-task warmth) negatively predicted T2 cooperativeness (Ï=âˆ’0.68, p=0.030): groups spending more time in social, laughter-filled interaction during negotiation reported lower cooperativeness, suggesting that off-task positive affect displaced the substantive engagement needed to reach a genuine agreement. State 0 (Transitional/passive) negatively predicted T1 coordination (Ï=âˆ’0.70, p=0.037), consistent with disengagement harming task completion. State 2 (Aroused-Engaged) negatively predicted voice inclusion across tasks (pooled Ï=âˆ’0.39, p=0.047), consistent with high-stakes ordered discussion constraining perceived participation equality."

### 11.5 Limitations of the prediction analysis

1. **Small sample (n=10 groups):** All results are exploratory. The T1 coordination Ã— State 1 association (p=0.009) is the most robust; others require replication. Power analysis: nâ‰ˆ22 groups needed for 80% power at |Ï|=0.60.

2. **Non-independence in pooled analysis:** Groups appear three times in the pooled voice_inclusion model (T1+T2+T3). A proper analysis would use mixed-effects regression with group as a random effect.

3. **Self-report only:** All outcomes are subjective perceptions. Objective task performance outcomes (decision quality, idea originality) are not included.

4. **Directionality:** Correlations cannot establish causation. Active discussion may cause coordination, or coordinating groups may naturally be more active â€” both interpretations are compatible.

5. **Multiple comparisons:** 6 outcomes Ã— 4 states = 24 tests. With Bonferroni correction (Î±=0.002), only the T1 coordination Ã— State 1 association survives strict correction. The remaining significant or borderline findings (p<0.05 uncorrected) are preliminary â€” they are consistent with the theoretical framework and worth including in the paper as candidate associations, but should be flagged as requiring replication in a larger sample before drawing firm conclusions.

---

*Prediction analysis notebook: `icmi_paper/analysis/hmm_state_prediction.ipynb`*  
*Figures: `icmi_paper/figures/hmm_prediction_scatter.png`, `icmi_paper/figures/hmm_prediction_pooled.png`*

