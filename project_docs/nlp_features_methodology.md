# NLP Feature Methodology — Session Notes (2026-08-25)

This document records the methodology decisions, implementation details, and results for all
NLP-based feature work done in this session. It covers three feature families:
semantic embeddings (BERT-UMAP), dialogue acts, and conversational synchrony.

---

## 1. Semantic Embeddings (BERT-UMAP)

### Why utterance-level pooling, not text concatenation

The original approach concatenated all speakers' text within a 30s window into a single string
and passed it to the Sentence-BERT model. This was wrong for two reasons:

1. **Truncation**: `all-MiniLM-L6-v2` has a hard 256-token limit. Concatenated 4-speaker
   30s windows typically exceed this and are silently truncated — only the beginning of the
   window is represented.
2. **Speaker identity lost**: Concatenation treats all 4 speakers as a single voice.

**Fix**: each utterance is embedded individually, then embeddings are mean-pooled per window.
Individual utterances average ~15 tokens (p99 ≈ 79 tokens), safely within the 256-token limit.

```
transcripts → individual utterances → BERT (384-dim each) → mean-pool per 30s window
```

### Model

`all-MiniLM-L6-v2` (sentence-transformers). Pre-trained on NLI + STS data. Produces
384-dimensional sentence embeddings. **Not fine-tuned** on dialogue or group dynamics data
— this is intentional: it serves as a *semantic topic* baseline, not a group affect measure.

### Dimensionality reduction: PCA whitening → UMAP

PCA alone is inefficient for sentence embeddings: 50 PCA components explain only 72.9% of
variance; 80% requires >50 components. This is expected — BERT embeddings are deliberately
distributed across many dimensions.

Pipeline used:
1. `StandardScaler` — unit variance per dimension
2. `PCA(n_components=50)` — whitening step (noise reduction, faster UMAP)
3. `UMAP(n_components=10, metric='cosine', n_neighbors=15, min_dist=0.1, random_state=42)`

**Why cosine metric in UMAP**: sentence embeddings are trained with cosine similarity as the
objective. Euclidean distance in the raw embedding space is less meaningful.

**Output columns**: `bert_umap1` … `bert_umap10`  
**Output file**: `analysis/results/bert_embeddings_window_30s.tsv`

### What BERT-UMAP captures (η² decomposition)

After computing embeddings, we ran a variance decomposition to understand what the 10 UMAP
dimensions discriminate:

| | Mean η²_task | Mean η²_group |
|---|---|---|
| BERT-UMAP (10 dims) | **0.36** | 0.06 |

Interpretation: BERT-UMAP primarily discriminates **task type** (T1 vs T2 vs T3 — different
topics) rather than group differences. PC3 specifically has η²_task ≈ 0.54. PC1–2 carry modest
group-level signal (η²_group ≈ 0.08–0.11).

**Thesis framing**: BERT-UMAP is a *semantic topic* representation. It answers "what topic is
being discussed" more than "how this group interacts." This is complementary to, not a
substitute for, the interaction-structure features (DA, lexical).

---

## 2. Dialogue Act Features

### Codebook

11-category DA taxonomy grounded in group decision-making theory:

| Label | Definition |
|---|---|
| `statement` | Asserts fact, opinion, or personal view |
| `proposal` | Introduces new option, plan, or action |
| `agreement` | Endorses a prior idea or proposal |
| `disagreement` | Rejects or pushes back on an idea |
| `compromise` | Blends competing options to find middle ground |
| `question` | Requests information, clarification, or decision |
| `answer` | Responds directly to a prior question |
| `backchannel` | Signals attention without taking the floor (mm-hmm, yeah) |
| `hedge` | Expresses uncertainty or softens a claim |
| `decision` | Moves group toward final decision or vote |
| `offtask` | Jokes, social talk, unrelated content |

### Method A — Keyword heuristic (runs locally, immediate)

Each utterance scored against cue-word lists from the codebook. Score = proportion of cues
matched, clipped to [0, 1]. Per-window feature = mean score across utterances.

**Coverage** (proportion of utterances with ≥1 cue match):
- `backchannel`: 30.3%, `question`: 14.6%, `hedge`: 10.0%
- `compromise`: 0.4%, `disagreement`: 0.8% (very sparse)

**η² decomposition** (keyword DA features):
- Mean η²_task = 0.016, Mean η²_group = **0.030**
- Notable: η²_group > η²_task for most categories — DA features are more group-discriminating
  than task-discriminating (opposite of BERT-UMAP)
- `hedge` (η²_group = 0.058) and `decision` (η²_group = 0.042) exceed the 0.05 threshold

**Output file**: `analysis/results/da_keyword_window_30s.tsv`

### Method B — Zero-shot NLI (DeBERTa, Colab)

Each utterance classified against the 11 codebook hypotheses using
`MoritzLaurer/deberta-v3-base-zeroshot-v1` (purpose-built zero-shot NLI model, ~500MB).
The model runs NLI entailment: "Does this utterance ENTAIL the hypothesis [label description]?"

**Pipeline**: utterances → DeBERTa (11 NLI checks per utterance) → probability distribution
→ mean-pool per window.

**Implementation note**: must run in a visible terminal (not a hidden process). The
`TOKENIZERS_PARALLELISM=false` environment variable must be set before importing transformers
to prevent deadlock.

**η² results (NLI DA features)**:
- `disagreement` η²_group = 0.055 ✓, `compromise` η²_group = 0.059 ✓
- Mean η²_task = 0.034, Mean η²_group = **0.056**
- Stronger group-level signal than keyword approach, especially for conflict/compromise

**Output file**: `analysis/results/da_window_30s.tsv`  
**Colab notebook**: `analysis/dialogue_act_features_colab.ipynb`

### Key finding

Both DA methods show η²_group > η²_task. Combined with the BERT result (η²_task >> η²_group),
this confirms the core thesis claim:

> **Semantic content (BERT) is primarily task-driven. Interaction structure (DA) is primarily
> group-driven.**

---

## 3. Prediction Models

### Feature sets

| Feature set | Columns | Source file |
|---|---|---|
| HMM baseline | 12 physio + ET + overlap features | `icmi_paper/results/hmm_input_features_final.tsv` |
| Audio | 4 prosodic features | `icmi_paper/results/audio_window_features.tsv` |
| BERT-UMAP | `bert_umap1`–`bert_umap5` (5 PCs) | `bert_embeddings_window_30s.tsv` |
| Keyword DA | `da_hedge`, `da_question`, `da_backchannel`, `da_statement`, `da_agreement` | `da_keyword_window_30s.tsv` |
| NLI DA | `nli_da_disagreement`, `nli_da_compromise`, `nli_da_hedge`, `nli_da_backchannel`, `nli_da_offtask` | `da_window_30s.tsv` |

**Note**: only 5 features per modality used to avoid overfitting with n=28 group×task rows.

### CV strategy

Two strategies compared (professor recommendation):

- **LOGO** (Leave-One-Group-Out): 10 folds, ~3 test rows each — standard but optimistic
- **L2GO** (GroupKFold n=5, ~2 groups per fold): stricter, ~6 test rows per fold — recommended

All models: `StandardScaler → RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])`.  
Task dummies always included as covariates.

### Key results (L2GO, raw features)

| Target | Best model | L2GO R² | Notes |
|---|---|---|---|
| `cooperative` (T2) | HMM baseline | **0.376** | Robust; drops from LOGO=0.694 |
| `voice_inclusion` | Audio | **0.388** | Improves under L2GO |
| `mental_demand` | Audio | 0.119 | Modest; was inflated at LOGO=0.437 |
| `team_coordination` (T1+T3) | HMM baseline | 0.117 | Weakly positive |
| `coord+coop` (cross-task) | HMM+KwDA | 0.050 | Marginal |

**Lesson**: LOGO was optimistic. L2GO is the honest estimate and should be the primary reported CV.
New features (BERT, DA) do not improve prediction over HMM baseline under L2GO — they overfit.
PCA-2 within modality helps for `team_coordination` (HMM baseline: 0.124 → 0.290).

---

## 4. Collaborative Completion and Repetition Features

### Motivation

Collaborative completions (B finishes A's sentence) signal high group cohesion and mutual
understanding. Repetitions/echoes signal active listening and affiliation. Neither is captured
by existing features.

### Candidate extraction

**Round 1** (82 candidates): utterances ending with `...` or `-` markers, next speaker within 2s.  
After manual labelling by the researcher: 3 completions, 3 repetitions confirmed.
**Conclusion**: the `...` heuristic is too noisy (~4% precision).

**Round 2** (37 candidates, pre-filtered): candidates filtered by:
- Completion candidates: A's last word is a syntactic function word AND B's text is substantive
- Repetition candidates: Jaccard word overlap between A and B ≥ 0.30

**Pending labelling**: `analysis/results/completion_candidates_round2.tsv`

### Planned features

Once labels are collected, three approaches will produce window-level features:

1. **Label-based** (from manually labelled instances):
   - `tr_completion_rate` = completions per minute per window
   - `tr_repetition_rate` = repetitions per minute per window

2. **BERT cosine similarity** (semantic continuity, no labelling needed):
   - `tr_semantic_continuity` = mean cos(utt_A, utt_B) for consecutive cross-speaker pairs
   - `tr_cross_speaker_similarity` = mean cos across all cross-speaker pairs in window
   - Computed in `analysis/semantic_continuity_colab.ipynb`

3. **NLI completion score** (DeBERTa, continuous):
   - Score each overlap candidate: P("B is completing A's sentence")
   - `tr_nli_completion_score` = mean score per window
   - Computed in `analysis/semantic_continuity_colab.ipynb`

**Output files** (pending Colab run):
- `analysis/results/semantic_continuity_window_30s.tsv`
- `analysis/results/nli_window_30s.tsv`
- `analysis/results/completion_scores_nli.tsv`

---

## 5. Missing Features Audit

Cross-checking `collective_window_features.tsv` (59 columns) against features used in
prediction revealed significant gaps. The following are available but not yet used in models:

| Feature | Why relevant |
|---|---|
| `group_hrv_rmssd_ms_mean` | HRV — parasympathetic tone, not in HMM baseline |
| `group_hr_mean_bpm_std` | Within-group HR spread — physiological synchrony proxy |
| `tr_speaking_entropy` | Participation equity — used in HMM but not prediction |
| `tr_conflict_overlap_ratio` | Direct conflict measure |
| `tr_ovl_ctx_collaborative` | Collaborative overlap context |
| `tr_ovl_ctx_completion` | Completion overlap context (unreliable labels) |
| `group_et_blink_rate_per_min_mean` | Cognitive load / attentional state |
| `group_et_gaze_dispersion_mean` | Attention coordination |
| `group_et_pupil_slope_per_s_mean` | Dynamic cognitive load |

**Recommendation**: add `group_hrv_rmssd_ms_mean`, `tr_speaking_entropy`,
`tr_conflict_overlap_ratio`, and `group_et_blink_rate_per_min_mean` to the HMM_FEATS baseline
in `effectiveness_prediction_models.ipynb` for a more complete feature set.

---

## File Index

| File | Purpose | Status |
|---|---|---|
| `analysis/semantic_embeddings_exploration.ipynb` | BERT-UMAP pipeline, η² analysis | ✅ Complete |
| `analysis/dialogue_act_keyword.ipynb` | Keyword DA features | ✅ Complete |
| `analysis/dialogue_act_features_colab.ipynb` | NLI DA features (Colab) | ✅ Run |
| `analysis/semantic_continuity_colab.ipynb` | Cosine + NLI completion (Colab) | 🔄 Pending run |
| `analysis/effectiveness_prediction_models.ipynb` | Prediction models (LOGO + L2GO) | ✅ Complete |
| `analysis/results/bert_embeddings_window_30s.tsv` | BERT-UMAP window features | ✅ |
| `analysis/results/da_keyword_window_30s.tsv` | Keyword DA window features | ✅ |
| `analysis/results/da_window_30s.tsv` | NLI DA window features | ✅ |
| `analysis/results/completion_candidates_round2.tsv` | Candidates for manual labelling | 🔄 Pending labels |
| `analysis/results/semantic_continuity_window_30s.tsv` | Cosine continuity features | 🔄 Pending |
| `analysis/results/nli_window_30s.tsv` | NLI window-level scores | 🔄 Pending |
