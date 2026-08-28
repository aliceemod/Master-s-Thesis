# APPENDIX A: Conversation Feature Definitions & Extraction

## A.1 Conversation-Level Features (Primary)

### Overlaps (OVL)
**Definition:** Two or more speakers active simultaneously.

Overlaps are characterized by a **two-layer annotation scheme** separating objective timing from communicative function:

#### Layer 1 — Timing subtype (`subtype`)
Assigned purely from acoustic duration and speaker start-time difference; no text analysis:

| Subtype | Rule | Interpretation |
|---|---|---|
| `simultaneous` | Start diff < 200 ms | Both speakers begin at the same moment — joint floor entry |
| `backchannel_ovl` | Duration < 250 ms | Very brief overlap — listener signal or quick interjection |
| `smooth` | Duration 250–500 ms | Managed turn handoff with short overlap |
| `competitive` | Duration 500–1000 ms | Contested floor — one speaker continues while another attempts to take over |
| `floor_fight` | Duration ≥ 1000 ms | Sustained simultaneous speech — extended floor contest |

#### Layer 2 — Context label (`context_label`)
Assigned by LLM-assisted rule-based detection on the text of concurrent speech segments; empty when no clear signal is found:

| Value | Meaning | Detection |
|---|---|---|
| `collaborative` | Supportive overlap — agreement, affirmation, or backchannel signal during another's turn | Support cues in overlapping speech (e.g., *yeah, mmh, exactly, agree*) |
| `completion` | Collaborative sentence completion — one speaker finishes another's trailing utterance | Prior speaker's text ends with ellipsis/dash (`…`, `—`) AND incoming speech contains content words |
| `competitive` | Conflictual intent — challenge or contradiction | Conflict cues (e.g., *no, wait, hold on, but*) on `competitive`/`floor_fight` timing overlap |
| *(empty)* | No clear lexical signal | — |

**Key design decision:** Context-sensitive words (*right, ok, okay, yes, sure*) are only counted as support signals when the speaker's full utterance is ≤ 4 words, preventing false positives from discourse markers mid-sentence (e.g., "...relevant for all of us, right?" is NOT a support signal).

**Measurement:** Per-speaker timestamps from WhisperX diarization; overlap duration = end\_time(earlier) − start\_time(later). Labeled by `tools/relabel_overlaps.py` (July 2026). Cue word lists developed with LLM assistance and applied deterministically via regex.

**Aggregation (60-sec window, HMM analysis):**

| Feature | Column | Transformation |
|---|---|---|
| Count of simultaneous overlaps | `tr_ovl_simultaneous` | log1p |
| Count of smooth overlaps | `tr_ovl_smooth` | log1p |
| Count of timing-competitive overlaps | `tr_ovl_competitive_timing` | log1p |
| Count of floor-fight overlaps | `tr_ovl_floor_fight` | log1p |
| Collaborative context fraction | `tr_ovl_collaboration_index` | log1p |

---

### Laughter (LAU)
**Definition:** Audible laughter vocalizations.

**Subtypes:**
- **Individual laughter**: Single speaker laughs
- **Group laughter**: Multiple speakers laugh simultaneously

**Measurement:** Detected via audio classifier (MFCCs + SVM trained on AffectAI corpus) with manual verification on subset.

**Aggregation (30-sec window):** Count of laughter events; % group vs. individual.

---

### Active Speakers (ASPR)
**Definition:** Number of unique speakers who contributed at least one speech segment.

**Measurement:** From diarization; counts unique speaker IDs per 30-sec window.

**Aggregation:** Direct count (range 1–4 in a group of 4).

---

### Backchannel (BCK)
**Definition:** Brief supportive interjections (yeah, mm-hmm, right) while another speaker holds the floor.

**Measurement:** Annotated in transcript; duration < 250 ms; no attempt to take floor.

**Aggregation (30-sec window):** Count of backchannel events; treats as indicator of attentiveness and affiliation.

---

### Silence Duration (SIL_DUR)
**Definition:** Periods when no speaker is active (pause > 500 ms).

**Measurement:** Complement of speech activity; detected from diarization silence gaps.

**Aggregation (30-sec window):** Total silence duration (sec); also computed as % of window.

---

### Competitive Floor-Fighting (COMP_OVL)
**Definition:** Sustained overlaps (> 500 ms) with conflict cues (disagreement markers, challenge phrases).

**Measurement:** Subset of OVL subtypes classified as "competitive" or "floor_fight."

**Aggregation (30-sec window):** Count of competitive overlaps; proxy for conflict intensity.

---

## A.2 Physiological Features

### Heart Rate (HR)
**Source:** EmotiBit PPG sensor worn on wrist.

**Preprocessing:**
1. Raw PPG → heart rate (bpm) via peak detection
2. Within-group z-score normalization (removes session-level baseline drift)
3. 30-sec windowing: mean HR per window

**Rationale for normalization:** Baseline HR varies across sessions and individuals (e.g., 55–95 bpm). Within-group z-score preserves inter-group differences while removing session confounds.

---

### Electrodermal Activity — Phasic Component (EDA_PHASIC)
**Source:** EmotiBit EDA (galvanic skin response) sensor.

**Preprocessing:**
1. Raw EDA → decomposition into tonic (slow) + phasic (fast) components via convex optimization (cvxEDA)
2. Within-group z-score normalization
3. 30-sec windowing: mean phasic component per window

**Rationale:** Phasic EDA is more sensitive to emotional arousal events (peaks during conflict, excitement) than tonic baseline.

---

### Skin Temperature (TEMP)
**Source:** EmotiBit thermistor.

**Preprocessing:**
1. Within-group z-score normalization
2. 30-sec windowing: mean temperature per window

**Rationale:** Temperature rises during arousal/engagement; within-group normalization removes baseline drift (e.g., ambient lab temp variation).

---

## A.3 Eye-Tracking Feature

### Pupil Diameter (PUPIL_DIAM)
**Source:** Tobii Pro Glasses 2 (sampling: 100 Hz).

**Preprocessing:**
1. Raw pupil diameter (mm) → within-group z-score normalization
2. 30-sec windowing: mean pupil diameter per window

**Rationale:** Pupil dilates during arousal, cognitive load, and attention. Within-group normalization removes inter-individual baseline differences (some people have larger pupils naturally).

---

## A.4 Feature Engineering & Selection

### Transformations Applied

1. **Log(x+1) for skewed counts:** Overlaps, competitive OVL, laughter, active speakers, backchannel counts follow power-law distributions. Log transform reduces skew, improves HMM Gaussian assumption.

2. **Within-group z-score for physiology & gaze:** 
   - Formula: z = (x - group_mean) / group_std
   - Applied to: HR, EDA_phasic, TEMP, PUPIL_DIAM
   - Removes inter-session and inter-individual baselines

3. **Dropped features (reasons):**
   - Speech duration (highly correlated with active speakers, r=0.92)
   - Blink rate (unreliable tracking, too many missing values)
   - Heart rate variability (confounded by movement)
   - Gaze dispersion (noisy at group level)

### Dimensionality Reduction

**PCA applied post-scaling:**
- Input: 11 features (after dropping above)
- Variance explained by first 7 PCs: 80%
- HMM trained on 7-dim PCA space
- Improves model stability and reduces overfitting

---

## A.5 Window Selection (30 seconds)

**Rationale:** 
- Physiological changes (HR, EDA) emerge over ~20–40 sec
- Conversation episodes (turns, overlaps, transitions) complete within ~30 sec
- Short enough to capture state transitions (~10–15 windows per 5-min task phase)
- Long enough to average high-frequency noise

**Overlap:** Windows slide at 10-sec intervals (67% overlap) to smooth state boundaries.

---

## A.6 Summary Statistics from Main Analysis

| Feature | Mean | SD | Min | Max | Skew (pre-transform) |
|---------|------|-----|-----|-----|-----|
| Overlaps (count) | 4.2 | 3.1 | 0 | 18 | 1.8 (high) → log transformed |
| Laughter (count) | 1.3 | 1.5 | 0 | 8 | 2.1 (high) → log transformed |
| Active speakers (n) | 2.8 | 1.1 | 1 | 4 | −0.5 (symmetric) |
| Backchannel (count) | 0.8 | 1.2 | 0 | 6 | 1.6 (high) → log transformed |
| Silence duration (sec) | 8.3 | 6.2 | 0 | 27 | 0.9 (moderate) → log transformed |
| Competitive OVL (count) | 0.4 | 0.8 | 0 | 4 | 2.2 (high) → log transformed |
| HR (z-scored) | 0.0 | 1.0 | −2.8 | 3.1 | 0.1 (normalized) |
| EDA phasic (z-scored) | 0.0 | 1.0 | −1.9 | 3.4 | 0.2 (normalized) |
| Temp (z-scored) | 0.0 | 1.0 | −2.5 | 2.9 | 0.0 (normalized) |
| Pupil diam (z-scored) | 0.0 | 1.0 | −2.1 | 3.0 | 0.1 (normalized) |
| PCA dim 1 | 0.0 | 1.0 | −3.2 | 3.8 | 0.1 (by design) |

---

## A.7 Data Quality & Missingness

### Transcript Coverage by Group and Task (Full Dataset, T1–T3 scope)

Transcripts are available for all 10 groups (grp-07–grp-16). Within the T1–T3 analysis window, two task-sessions have no transcript:

| Group | T1 | T2 | T3 |
|-------|----|----|-----|
| grp-07 | ✓ | ✓ | ✓ |
| grp8  | ✓ | ✓ | ✓ |
| grp9  | **✗** | ✓ | ✓ |
| grp-10 | ✓ | ✓ | ✓ |
| grp11 | ✓ | ✓ | ✓ |
| grp12 | ✓ | ✓ | ✓ |
| grp13 | ✓ | ✓ | **✗** |
| grp-14 | ✓ | ✓ | ✓ |
| grp-15 | ✓ | ✓ | ✓ |
| grp-16 | ✓ | ✓ | ✓ |

**28 of 30 task-sessions** have complete transcripts. Missing: grp9 T1 and grp13 T3.

### Handling of Missing Transcripts

The two sessions without transcripts (grp9 T1, grp13 T3) were **retained in the analysis** rather than excluded. For windows falling within these sessions, all conversation features (`tr_*`) are absent (NaN). Physiological (HR, EDA, temperature) and eye-tracking (pupil diameter) features remain fully available for those windows and contribute normally to the HMM. This approach preserves the full temporal structure of those sessions and avoids discarding valid multimodal data due to a single missing modality.

### Resolution of Ambiguous Overlaps (needs_review → timing-based labels)

All overlaps are classified based on duration alone (primary label) with optional lexical context hints:

**Primary label (`subtype`) — duration-based, no exceptions:**
- start_diff < 200 ms → `simultaneous`
- overlap < 250 ms → `backchannel_ovl` (brief, typically supportive)
- overlap 250–500 ms → `smooth` (short handoff overlap)
- overlap 500–1000 ms → `competitive` (floor-fight territory)
- overlap ≥ 1000 ms → `floor_fight` (sustained mutual overlap)

**Secondary hint (`context_label`) — lexical cues, written only when contradicting primary label:**
- `collaborative`: primary label is competitive/floor_fight, but support words detected (e.g., "yes", "right", "and")
- `competitive`: primary label is backchannel_ovl/smooth, but conflict words detected (e.g., "no", "wait", "but")
- Empty: timing and lexical signals agree, or no strong cues present

Previously, overlaps flagged as `needs_review` (annotator-ambiguous) were left unlabeled. Now they receive timing-based labels to ensure all overlaps contribute to feature aggregation. The `context_label` column provides transparency when the lexical evidence suggests a different interpretation than the duration hint.

### Other Missingness

- **Conversation features (within transcribed sessions):** < 2% (diarization alignment failures)
- **Physiological signals:** HR/EDA/Temp ≤ 3% (sensor dropouts); interpolated linearly within gaps ≤ 2 min; longer gaps excluded
- **Gaze:** ≤ 5% (calibration loss); windows with > 50% pupil data missing excluded

### Impact on Results

State discovery is robust to the two missing sessions: the remaining 28 task-sessions provide sufficient coverage across all three tasks and all 10 groups, and the physio/ET signal for the affected windows remains fully available for HMM inference. The inclusion of all resolvable overlaps (including previously ambiguous ones) ensures that the full spectrum of group conversational dynamics informs the latent state model.

---

---

## A.8 Hidden Markov Model — Rationale, Theory, and Implementation

### Why HMM?

Group affective dynamics are **temporal and sequential**, not static snapshots. A group does not just "have" a state — it transitions between states over time in a structured way. This requires a model that:

1. **Captures latent structure** — the "true" affective state is not directly observable; we only see physiological and behavioural proxies
2. **Models temporal dependencies** — the current state is more likely to persist or transition to adjacent states than to jump randomly
3. **Discovers states unsupervised** — we do not label states in advance; we want the model to find them from the data

HMM satisfies all three. Alternatives were considered and rejected:

| Alternative | Reason not used |
|-------------|-----------------|
| K-means / GMM | No temporal structure; treats each window independently; states can flicker arbitrarily |
| PCA/factor analysis | Dimensionality reduction, not state discovery; no temporal modeling |
| LSTM/RNN | Requires labelled training data; not suitable for unsupervised discovery on N=10 groups |
| Change-point detection | Finds transitions only, not the content of states |
| Recurrence quantification | Describes dynamics but does not identify discrete states |

HMM is well-established for physiological state modelling (Schuller et al., 2015; Rabiner, 1989) and has been applied to affective computing (Trigeorgis et al., 2016) and collaborative learning dynamics (Järvelä et al., 2017).

---

### Preprocessing Pipeline: From Raw Features to HMM Input

The journey from raw multimodal data to HMM input involved three key steps:

**Step 1: Feature assembly (11 features)**
Extracted all conversation, physiological, and eye-tracking features at the 30-second window level. See sections A.1–A.3.

**Step 2: Within-group z-score normalization**
For each group separately, computed mean and std of each feature across all windows in that group. Then normalized:

$$x'_{i,g} = \frac{x_{i,g} - \mu_g}{\sigma_g}$$

where $i$ indexes windows, $g$ indexes groups. This removes between-group and session-level baseline drift while preserving within-group temporal dynamics. For example, one group's baseline HR might be 65 bpm, another's 85 bpm — normalization removes this confound so the HMM sees only *changes* within each group.

**Step 3: PCA dimensionality reduction**
Fitted PCA on all 545 windowed observations (pooled across groups). Set a threshold of **80% cumulative explained variance**, then retained the minimum number of principal components needed to reach that threshold. This resulted in **7 principal components**. This step:
- Decorrelates features (PCs are orthogonal)
- Focuses on directions of maximum variance (removes noise)
- Matches assumptions of Gaussian HMM with diagonal covariance (features are approximately independent given state)
- Improves model stability (avoids overfitting with 11 features on only 545 observations)

**Is 80% a lot or a little?** It is a reasonable choice. PCA at 80% variance is a standard rule of thumb in dimensionality reduction — high enough to retain signal, low enough to discard noise. The threshold was applied objectively: take the minimum number of components needed to reach cumulative variance ≥ 0.80, which yielded 7 PCs. Alternatives (70%, 90%) would yield 5–6 or 8–9 PCs respectively; 80% balances signal retention with model simplicity.

The **PCA loadings** (eigenvector weights) show that:
- PC1 captures overall arousal (high loading on HR, EDA rate, temperature)
- PC2 captures conversation dominance (high loading on overlaps, active speakers)
- PCs 3–7 capture finer structure (silence, backchannel, pupil diameter, etc.)

**Step 4: HMM training**
Input to HMM: 545 observations × 7 PCA dimensions, grouped by session.

---

### What HMM Does

An HMM assumes the observed multimodal signal is generated by an underlying sequence of discrete hidden states $S = \{s_1, s_2, \ldots, s_K\}$. At each time step $t$, the system is in one state $z_t \in S$, which:
- **Persists** with probability $a_{ii} = P(z_{t+1} = i \mid z_t = i)$ (self-transition)
- **Switches** to another state $j$ with probability $a_{ij} = P(z_{t+1} = j \mid z_t = i)$
- **Emits** the observed feature vector $\mathbf{x}_t$ according to an emission distribution $P(\mathbf{x}_t \mid z_t = i)$

The model thus learns:
- **$\pi$** — initial state probabilities
- **$A$** — $K \times K$ transition matrix ($a_{ij}$ values)
- **$B$** — emission parameters (Gaussian means and covariances per state)

The key insight: states are **not observable directly**. The HMM infers them from patterns in the data, guided by the constraint that transitions must be temporally smooth.

---

### How HMM Works (Training)

Training uses the **Baum-Welch algorithm** (a form of Expectation-Maximisation):

1. **E-step:** Given current parameters, compute the probability of being in each state at each time step (forward-backward algorithm)
2. **M-step:** Update $\pi$, $A$, and $B$ to maximise the expected log-likelihood
3. Repeat until convergence

**Emission model:** We used **Gaussian emissions with diagonal covariance**:

$$P(\mathbf{x}_t \mid z_t = i) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$$

Diagonal covariance assumes features are conditionally independent given the state — a reasonable approximation given that features were already decorrelated via PCA.

**Model selection:** The number of states $K$ is a hyperparameter. We fitted models for $K = 2, 3, 4, 5, 6$ using Baum-Welch EM with three random restarts per k (seeds 42, 7, 99), retaining the highest log-likelihood run. Selected $K = 5$ via the **Bayesian Information Criterion (BIC)**:

$$\text{BIC} = -2 \ln \hat{L} + p \ln n$$

where $\hat{L}$ is the maximised log-likelihood, $p$ is the number of free parameters (K(D+D+K)+(K-1) for state means, diagonal covariances, and transitions), and $n=545$ is the total number of windows.

**BIC values and elbow:** K=2 (BIC≈12,340) to K=4 (BIC≈11,892) show steep improvement; K=5 yields BIC=11,108.1 — a substantial 48-point drop. Beyond K=5, the curve flattens (K=6 BIC≈11,155): the additional state does not substantially improve fit. This clear **elbow at K=5** is the primary evidence for model selection. BIC balances fit and complexity, and the elbow indicates that 5 states represent a parsimonious, well-supported summary of the dynamics.

---

### How We Used It (Implementation)

**Library:** `hmmlearn` (v0.3, Python) — `GaussianHMM` with `covariance_type='diag'`

**Input:** 7-dimensional PCA-reduced feature vectors, one per 30-second window (545 windows total across all groups and tasks)

**Training:** All groups concatenated into a single observation sequence with `lengths` parameter indicating group boundaries — the HMM sees within-group temporal continuity but not across groups

**State sequence decoding:** After training, the **Viterbi algorithm** recovers the most probable state sequence $z_1^*, z_2^*, \ldots, z_T^*$ for each session:

$$z^* = \arg\max_{z_1,\ldots,z_T} P(z_1,\ldots,z_T \mid \mathbf{x}_1,\ldots,\mathbf{x}_T; \hat{\theta})$$

**Transition matrix:** The learned $A$ matrix was inspected directly. High diagonal values ($a_{ii} > 0.75$) indicate **persistent states** — groups dwell in a state for multiple consecutive windows before transitioning. Mean diagonal = 0.82 ± 0.06.

**State labelling:** States were labelled post-hoc by inspecting the emission means $\boldsymbol{\mu}_i$ — which features are elevated or suppressed in each state. Labels (e.g., "Aroused-active", "High-conflict") are interpretive, not part of the model.

**Reproducibility:** Random seed fixed (`random_state=42`); results are deterministic given fixed initialisation. Ten random restarts were run; the solution reported is the highest log-likelihood run.

---

## A.9 Alternative Approaches Attempted & Why HMM Won

During the discovery process, we evaluated and prototyped two alternative clustering approaches before settling on the Hidden Markov Model. This section documents the exploration, rationale for rejection, and why HMM was the best fit for our temporal multimodal data.

### Hierarchical Clustering (Agglomerative)

**Goal:** Identify natural groupings by building a dendrogram from pairwise distances.

**Approach:**
- Ward linkage on top 7 PCs (80% variance threshold)
- Subsampled 200 windows for visualization (dendrogram complexity)
- Saved figure: `icmi_paper/figures/hierarchical_dendrogram.png`

**Findings:**
The dendrogram revealed a **2-branch structure** with a large distance gap (~25 units) at the first split, suggesting k=2 as the most natural structural split. Further merges were gradual, indicating no compelling higher-order clustering.

**Why rejected:**
1. **Ignores temporal structure**: Hierarchical clustering treats each window as an independent point; it does not account for the fact that windows are sequential observations within a group conversation.
2. **No transition dynamics**: The resulting partition has no model of how groups move between states. The clustering is static.
3. **Dendrogram ambiguity**: While k=2 is structurally natural, it is too coarse for our interpretive goals (we need to distinguish different types of engagement, not just "high activity" vs. "low activity").

### k-Means Clustering (K-Selection via Multiple Criteria)

**Goal:** Partition the PCA space into k clusters; use model-selection metrics to choose k.

**Approach:**
- Tested k ∈ {2, 3, 4, 5, 6, 7, 8}
- Computed five selection criteria side by side:
  1. **Silhouette score** (higher = better; range [-1, 1]; measure of cluster cohesion and separation)
  2. **Calinski-Harabasz index** (higher = better; ratio of between-cluster to within-cluster variance)
  3. **Davies-Bouldin index** (lower = better; average similarity between cluster and its most similar neighbor)
  4. **Inertia** (sum of squared distances to nearest centroid; lower = better; decreases monotonically with k)
  5. **BIC** (Bayesian Information Criterion fitted via Gaussian Mixture Model; lower = better; balances model fit and complexity)
- Saved figure: `icmi_paper/figures/k_all_criteria.png`

**Results (k=2–8 tested):**

| k | Silhouette | Calinski-Harabasz | Davies-Bouldin | Inertia | BIC |
|---|---|---|---|---|---|
| 2 | **0.68** (peak) | 520.2 (peak) | 0.45 (low) | 8142 | 12,340 |
| 3 | 0.61 | 401.5 | 0.51 | 6871 | 12,108 |
| 4 | 0.54 | 328.7 | 0.58 | 5889 | **11,892** (peak) |
| 5 | 0.49 | 286.1 | 0.68 | 5124 | 11,980 |

**Summary of disagreement:** Silhouette and Calinski-Harabasz both peak at **k=2**; Davies-Bouldin and Inertia improve monotonically; **BIC peaks at k=4**. No consensus.

**Chosen k=3 (pragmatic middle ground):**
- Justified as a 3-way split that refines the 2-branch dendrogram structure into an interpretable hierarchy
- Represents "low engagement", "collaborative engagement", and "competitive engagement"
- Preserves enough granularity for the paper's interpretive story while remaining parsimonious

**k=3 State profiles (clean features; from notebook):**

| State | N | % | Label | Key signals |
|---|---|---|---|---|
| 0 | 289 | 53% | Low-engagement | All interaction low; low HR/EDA; silence dominant; few speakers. Dominant in T1 (hiring decision). |
| 1 | 168 | 31% | Collaborative | High active speakers, backchannels, overlap time; moderate physio. Cooperative floor participation. Dominant in T2 (negotiation). |
| 2 | 88 | 16% | Competitive-conflict | Very high competitive OVL (H=408, p≈10⁻⁸⁹); high overlap time; elevated HR/EDA. Floor competition — people talking over each other. Increases in T3 (idea generation). |

**k-means + task association (k=3):**
χ² = 47.3, p = 7×10⁻⁷, Cramér's V = 0.177 — states significantly task-structured, but groups traverse all 3 states within each task.

**Why rejected:**
1. **No temporal model**: k-means assigns each window independently to the nearest centroid. It has no awareness that windows follow a temporal sequence. A group could jump wildly between states (State 0 → State 2 → State 0) with no penalty.
2. **No concept of state persistence**: In reality, when a group enters a state (e.g., "high-conflict discussion"), it tends to dwell there for multiple consecutive windows before transitioning. k-means cannot capture this.
3. **Arbitrary k selection**: The lack of consensus among criteria meant k=3 was somewhat arbitrary. We needed a principled way to prefer smooth, persistent state sequences over jumping-around artifacts.
4. **Continuous-spectrum problem**: If the true generative process is more of a continuous spectrum (e.g., gradually escalating from collaborative to competitive) rather than discrete clusters, k-means will still force discrete partitions and ignore the underlying dynamics.

---

### Why Hidden Markov Model (HMM) Was the Right Choice

**Core insight:** The windows are **not independent points**—they form a **temporal sequence** within each group conversation. The HMM explicitly models:
- **Emission distribution**: What multivariate feature values are likely from each state
- **Transition dynamics**: How probable is it to move from state $i$ to state $j$ in the next window

This addresses all the shortcomings above:
1. **Temporal structure**: The transition matrix $A$ ensures that states persist (high diagonal) or transition smoothly (off-diagonal probabilities learned from data).
2. **Viterbi decoding**: Given the sequence of observations, the model finds the state path that is both **likely under the emissions** and **consistent with the learned dynamics**. This means the model naturally prefers persistent, realistic state transitions.
3. **Principled k selection**: BIC is a well-founded criterion that balances model fit and parameter count across different k. It replaces ad hoc threshold-based heuristics.
4. **Handles continuous spectra**: If the true dynamics are more of a smooth continuum, the HMM can model it via soft transition probabilities and overlapping state clusters in feature space.

**Key empirical results supporting HMM:**
- **State persistence**: Diagonal of transition matrix (self-transition probabilities) ranges 0.79–0.94, mean 0.82±0.06. States are highly persistent, confirming the temporal structure is real and important.
- **Task-state alignment (HMM with K=5)**: χ² = 296.92, p ≈ 1.86×10⁻⁵⁹, Cramér's V = 0.522. This is **much stronger** than k-means (V=0.177), indicating that HMM states capture task-aligned dynamics more robustly.

**Diagnostic figures saved during exploration:**
- `pca_clean_scree_loadings.png` — PCA scree plot and PC loadings heatmap
- `pca_clean_scatter_density.png` — PC1 vs PC2 scatter colored by task/group, with KDE density contours
- `hierarchical_dendrogram.png` — Ward dendrogram (200-sample subset)
- `k_all_criteria.png` — Five k-selection criteria plotted side-by-side (k=2–8)
- `diag_*.png` (6 files) — Feature diagnostics (within/between variance, distributions, correlation, task discriminability, timeseries CV, feature boxplots)

All figures are located in: `icmi_paper/figures/` (copied from `figures/latent_states_k_comparison/`)

---

**References:**
- Diarization: Diez et al. (2019, "Speaker diarization review")
- cvxEDA: Greco et al. (2016, "cvxEDA: a convex optimization approach to electrodermal activity processing")
- LSL: Delorme & Makeig (2004, "EEGLAB")
- Rabiner, L.R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition." *Proceedings of the IEEE*, 77(2), 257–286.
- Schuller, B. et al. (2015). "The INTERSPEECH 2015 computational paralinguistics challenge." *Interspeech*.
- Trigeorgis, G. et al. (2016). "Adieu features? End-to-end speech emotion recognition using a deep convolutional recurrent network." *ICASSP*.
- EmotiBit: [https://www.emotibit.com/pages/research](https://www.emotibit.com/pages/research)
- Tobii: [https://www.tobii.com/](https://www.tobii.com/)
