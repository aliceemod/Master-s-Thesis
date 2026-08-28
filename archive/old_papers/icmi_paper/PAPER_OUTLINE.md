# ICMI Paper Outline — Collective Affective States in Group Conversation

## Paper Structure (ICMI 2026, Long Paper, 8pp ACM sigconf)

### 1. Introduction (1 pp)
- Motivation: Understanding collective dynamics in group social interaction
- Gap: Existing work focuses on individual affect; collective states under-studied
- Contribution: **Automatic discovery of 5 latent collective states from multimodal data**
- Key result teaser: States strongly predict task phase (χ² p≈10⁻⁶⁰)

**Figures for this section:** None yet (or motivational figure from related work)

---

### 2. Related Work (1 pp)
- Individual affect recognition (survey)
- Group dynamics + conversation analysis (e.g., Pentland, Waller)
- Hidden Markov Models in behavior analysis
- Multimodal fusion for group tasks

**Figures for this section:** None

---

### 3. Data & Methodology (1.5 pp)

#### 3.1 Multimodal Collective Features
- **Physiology:** HR, HRV, EDA, skin temp (EmotiBit) — 8 groups × 3 tasks
- **Conversation:** Overlaps, competitive turns, laughter, backchannels, silence, active speakers (transcript + audio)
- **Eye-tracking:** Pupil diameter, blink rate, gaze dispersion (Tobii)
- **Outcome:** 545 windows (30s each) across 10 groups × 3 tasks

**Figure 1 (Results panel):** Diagnostic heatmaps
- (a) Within/between variance ratio → justifies feature selection
- (b) Distribution skew before/after log transform
- (c) Feature discriminability across tasks (Kruskal-Wallis H-values)

#### 3.2 Feature Engineering
- Within-group z-scoring to remove session-level baselines
- Log(x+1) transforms for count/duration features to reduce skew
- Redundancy removal (correlation r>0.7 analysis)
- PCA to 10 dimensions (83% variance)

**Figure 2 (Results panel):** PCA diagnostics
- (a) Scree plot + loadings (PC1 = interaction intensity, PC2 = arousal)
- (b) PC1 vs PC2 scatter colored by task
- (c) PC1-4 pairplot → motivates HMM (no discrete clusters, continuous spectrum)

#### 3.3 Hidden Markov Model
- Why HMM: Models temporal transitions in window sequences
- Gaussian emissions with diagonal covariance
- Baum-Welch EM fitting
- k selection via BIC across k=2..6

**Figure 3 (Results panel):**
- (a) BIC curves, k=5 elbow
- (b) Example dendrogram (hierarchical clustering for context)

---

### 4. Results (3 pp)

#### 4.1 State Discovery
**BIC k-selection:** k=5 with elbow at BIC=11108

**5 Discovered States** (from HMM profile heatmap + transition matrix):

| State | N | Features | Interpretation |
|---|---|---|---|
| 0 | 182 | ↑HR, temp, laughter, active speakers, overlap time | Aroused-active |
| 1 | 103 | ↑Silence, ↑competitive OVL, ↑laughter | Silence-dominated / tense |
| 2 | 61 | ↑EDA phasic, ↑competitive OVL, ↑backchannels | High-conflict / floor-fighting |
| 3 | 134 | All features low | Low-engagement / deliberation |
| 4 | 65 | ↓HR, ↓temp, ↑pupil diam, ↓interaction | Cool-attentive / listening |

**Figure 4: State Profiles & Transitions**
- (a) Heatmap: z-scored features by state (clear patterns)
- (b) Transition matrix: diagonal dominance (0.79–0.94) confirms state persistence

#### 4.2 Task Structuring
**χ² = 296.92, p = 1.86×10⁻⁵⁹, Cramér's V = 0.522**

Strong evidence that states encode task phases:
- **T1 (Hiring decision):** Dominant States 3 (40%) & 4 (30%) — deliberative, low engagement
- **T2 (Negotiation):** Shift toward States 0 (35%) & 1 (25%) — more activity, engagement
- **T3 (Idea generation):** Peak State 2 (22%) + State 0 (32%) — competitive floor-fighting for ideas

**Figure 5: Task Distribution & Example Trajectory**
- (a) Stacked bar: state distribution by task (shows clear shift)
- (b) Time series: one group across T1→T2→T3, showing state transitions aligned to task boundaries

#### 4.3 Validation
- Kruskal-Wallis test on discriminating features: Competitive OVL (H=408, p≈10⁻⁹⁰), Overlap time (H=349), Active speakers (H=260)
- Post-task self-report validation (optional, if correlation with VAD/arousal self-report exists)

---

### 5. Discussion (1.5 pp)
- **What the states mean:** Five interpretable modes of collective engagement
- **Why HMM succeeded where k-means failed:** Temporal structure reveals meaningful latent dynamics even without feature-space clusters
- **Task-state alignment:** States naturally evolve with conversational task demands
- **Broader implications:** Automatic collective state detection could improve group collaboration tools, meeting facilitation, team interventions
- **Limitations:** N=10 groups (small), controlled lab setting, transcript quality effects
- **Future work:** Real-time decoding, transfer to other group tasks, personalised state interpretation

---

### 6. Conclusion (0.5 pp)
- Summary: 5 latent states from multimodal group data
- Novelty: First automatic HMM-based collective state discovery with this level of task structuring
- Impact: Opens door to group-level affect recognition systems

---

## File Mapping

| Section | Relevant Files in icmi_paper/ |
|---|---|
| 3.2 Feature Engineering | figures/diag_*.png (diagnostics) |
| 3.3 HMM Methodology | README.md (technical details) |
| 4.1 State Discovery | figures/hmm_profiles_transitions.png |
| 4.2 Task Structuring | figures/hmm_task_dist_trajectory.png |
| 4.3 Validation | results/analysis_summary.json (Kruskal-Wallis H values) |
| General Reproducibility | notebooks/collective_state_discovery.ipynb |

---

## Page Budget (4 pages total) ⚠️ TIGHT CONSTRAINTS

- Intro: 0.4 pp (compress motivation + contribution)
- Related: 0.3 pp (merge into intro or supplementary)
- Method: 0.7 pp (dense: feature table only, HMM in 2 sentences, k=5 stated)
- Results: 2 pp (2–3 key figures only; state table inline; task breakdown brief)
- Discussion: 0.4 pp (findings only; defer applications to future work)
- Conclusion: 0.2 pp (1 paragraph summary)
- References: inline/minimal

**What to CUT for 4-page version:**
- ❌ Related work section (move to supplementary or cite minimally)
- ❌ Detailed feature engineering subsection (state as applied in methods)
- ❌ Full diagnostic figures (keep only 2: PCA + HMM key results)
- ❌ State-by-state interpretations (use table only)
- ❌ Robustness subsection (keep 1 sentence: "Cross-validation confirms generalization")
- ❌ Fairness analysis (mention in limitations; expand in supplementary)

---

## Key Messaging for Abstract

"We discover and characterize five latent collective affective states from naturalistic multimodal group conversation data using Hidden Markov Models. States reflect task-phase dynamics (χ² p≈10⁻⁶⁰) and persist over multiple conversation windows (transition stability >79%), enabling real-time group affect recognition for collaborative applications."

---

## Figures to Prepare

1. **Diagnostic summary** — 3×3 grid (within/between, dist, task discrim, time series, corr, etc.)
2. **PCA scree + pairplot** — PC structure motivation
3. **BIC k-selection** — HMM model choice
4. **State profiles heatmap + transition matrix** — main result
5. **Task distribution + trajectory example** — validation
6. **All figures should have high-res PNG + vector PDF for final submission**

---

## Notes for Writing

- **Tone:** Technical but accessible (multimodal learning community)
- **Avoid:** Jargon from specific modalities; frame everything in terms of collective dynamics
- **Emphasise:** Automatic (no manual labeling), temporal modeling (HMM not k-means), strong task structuring (high V=0.522)
- **Cite:** Recent HMM papers (hmm-learn, Rabiner tutorial), multimodal fusion reviews, group affect (if any)
- **Ethics:** Mention anonymization (P1–P4 IDs), IRB approval, no sensitive PHI stored

