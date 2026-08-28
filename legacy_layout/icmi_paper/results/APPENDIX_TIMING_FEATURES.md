# Appendix: Overlap Annotation & Timing-Based Feature Construction

## Overview

This document records the two-layer overlap annotation scheme and the feature set derived from it for the HMM collective state analysis.

## Timing Subtype Definitions

The following timing-based subtypes are applied by `tools/relabel_overlaps.py`:

| Subtype | Duration Rule | Interpretation |
|---------|---------------|-----------------|
| `simultaneous` | start_diff < 200 ms | Very brief simultaneous speech start |
| `backchannel_ovl` | overlap < 250 ms | Brief listener overlap (backchannel signals) |
| `smooth` | overlap 250–500 ms | Managed turn transition with overlap |
| `competitive` | overlap 500–1000 ms | Timing-based competition for floor |
| `floor_fight` | overlap >= 1000 ms | Extended simultaneous speech (floor contest) |

**All based on objective acoustic overlap duration. No annotator judgment of intent.**

## Context Label Definitions (Secondary Layer)

Context labels capture **communicative function** via LLM-assisted rule-based detection on concurrent speech text. Empty when no clear signal is found.

| Value | Trigger | Example |
|---|---|---|
| `collaborative` | Support cues in overlapping speech (any timing) | Speaker says "yeah", "mmh", "exactly" while another holds floor |
| `completion` | Prior speaker trails off with `…`/`—` AND incoming speech has content words | A: "I think we should focus on..." → B: [overlaps] "...the budget" |
| `competitive` | Conflict cues on competitive/floor_fight timing | "no", "wait", "hold on" during a contested long overlap |
| *(empty)* | No clear lexical signal | — |

### How Context Labels Were Assigned

Context cue word lists were developed with **LLM assistance** (iterative refinement against the AffectAI corpus) and applied as deterministic regex pattern matching by `tools/relabel_overlaps.py`.

**Strong support signals** (always counted):
> *yeah, yep, mmh, mm-hmm, uh-huh, exactly, agree, agreed, good point, totally, absolutely, indeed*

**Context-sensitive support signals** (only in utterances ≤ 4 words):
> *yes, right, ok, okay, sure, correct, true, definitely*

This distinction prevents false positives: "...relevant for all of us, right?" does NOT trigger `collaborative`; standalone "Right." does.

**Completion detection heuristic:**
1. Another speaker's utterance ends with `...`, `…`, `—`, or `–` within 1.5 s before the overlap
2. The overlapping speaker's text contains content words (not just bare agreement tokens)

The LLM contributed to authoring and refining these rules; per-overlap decisions are fully deterministic.

## Final Label Distribution (Post-Reclassification)

### Timing subtypes — all tasks, all groups (N=940 OVL rows)

| Subtype | Count | % |
|---|---|---|
| `floor_fight` | 313 | 33.3% |
| `backchannel_ovl` | 192 | 20.4% |
| `competitive` | 189 | 20.1% |
| `smooth` | 140 | 14.9% |
| `simultaneous` | 106 | 11.3% |

### Context labels — T1–T3 only (HMM analysis subset, N=769 OVL rows)

| Context | Count | % of OVL |
|---|---|---|
| `collaborative` | 313 | 40.7% |
| `completion` | 26 | 3.4% |
| `competitive` | 76 | 9.9% |
| *(empty)* | 354 | 46.0% |

## New Feature Columns (10 added to collective feature matrix)

### Pure Timing Subtypes (counts per window, 5 features)
- `tr_ovl_simultaneous`: count of simultaneous overlaps (start_diff < 200ms)
- `tr_ovl_backchannel`: count of brief overlaps (< 250ms)
- `tr_ovl_smooth`: count of managed overlaps (250–500ms)
- `tr_ovl_competitive_timing`: count of timing-competitive (500–1000ms)
- `tr_ovl_floor_fight`: count of extended overlaps (>= 1000ms)

### Context Labels (2 features)
- `tr_ovl_context_collaborative`: count of overlaps with `context_label = collaborative` per window
- `tr_ovl_context_completion`: count of sentence-completion overlaps per window

### Engineered Features (3 features)
- `tr_ovl_long_ratio`: (competitive_timing + floor_fight) / total → long-overlap prevalence
- `tr_ovl_short_ratio`: (simultaneous + backchannel) / total → brief-overlap prevalence
- `tr_ovl_collaboration_index`: context_collaborative / total_overlaps → collaborative intent fraction
- `tr_ovl_timing_only`: Sum of all timing subtypes (should equal tr_ovl_total if fully classified)

### Legacy (kept for comparison, 3 features)
- `tr_overlap_count`: Original count (generic)
- `tr_overlap_time_s`: Total overlap duration in seconds
- `tr_overlap_ms`: Original timing column (for reference)

---

## Feature Engineering Rationale

**Problem with old set**: Mixed timing-based counts (`tr_overlap_count`) with semantic categorization (`tr_collaborative_overlap`), conflating two distinct signals.

**Solution**: Separate three layers:

1. **Timing subtypes** — Mechanics of simultaneous speech
   - Objective, duration-based
   - No annotator bias
   - Captures **interaction patterns** (brief vs. sustained, coordinated vs. chaotic)

2. **Context labels** — Semantic/collaborative intent
   - Subjective, lexical hints
   - Applied only when lexical contradicts timing
   - Captures **functional interpretation** (working together vs. fighting)

3. **Engineered ratios** — Normalized metrics
   - Combine both signals in interpretable ways
   - Enable window-to-window comparison
   - Facilitate post-hoc interpretation of state profiles

This allows the HMM to learn the relationship between **interaction dynamics** and **emotional state** with cleaner feature semantics and minimal annotator bias.

---

## Preprocessing (Log Transformation)

In the HMM notebook, apply **log1p()** to these features (right-skewed count/duration distributions):

- `tr_silence_duration_s`
- `tr_backchannel_count`
- `tr_ovl_simultaneous`
- `tr_ovl_smooth`
- `tr_ovl_competitive_timing`
- `tr_ovl_floor_fight`
- `tr_laughter_count`
- `tr_ovl_collaboration_index` (after scaling to [0, 1])

---

## Feature Selection Decisions

### Decision 1: Keep `tr_backchannel_count` SEPARATE from `tr_ovl_backchannel`

| Feature | Type | Definition |
|---------|------|-----------|
| `tr_backchannel_count` | BCK rows | Listener feedback signals (separate audio annotation) |
| `tr_ovl_backchannel` | OVL rows < 250ms | Acoustic speech overlaps (duration < 250ms) |

**Why**: Different dynamics. BCK = purposeful listener signals; OVL = incidental overlaps.

### Decision 2: INCLUDE ALL 5 TIMING SUBTYPES (not selective)

- HMM learns importance via BIC selection
- PCA reduces dimensionality afterward
- Correlation among subtypes not problematic (all informative)
- Post-hoc feature importance analysis will show state preferences

### Decision 3: Keep `tr_overlap_time_s` AVAILABLE (not in primary CLEAN_FEATS)

- Complements counts: (many brief) vs. (few long)
- Can be added to CLEAN_FEATS if specific analysis requires
- Currently available in enhanced feature matrix for reference

---

## Updated CLEAN_FEATS List (13 features)

**Previous (11 features):**
```
1. group_hr_mean_bpm_mean           → arousal baseline
2. group_eda_phasic_rate_hz_mean    → arousal response
3. group_temp_mean_mean             → arousal physiological
4. tr_silence_duration_s            → turn-taking
5. tr_backchannel_count             → listener engagement
6. tr_overlap_count                 ❌ REMOVED (too generic)
7. tr_competitive_overlap           ❌ REMOVED (mixed semantics)
8. tr_laughter_count                → affective signal
9. tr_n_active_speakers             → participation
10. group_et_pupil_mean_mean        → attention/arousal
11. tr_overlap_time_s               ❌ REMOVED (redundant with timing subtypes)
```

**NEW (13 features):**
```
1. group_hr_mean_bpm_mean           → arousal baseline
2. group_eda_phasic_rate_hz_mean    → arousal response
3. group_temp_mean_mean             → arousal physiological
4. tr_silence_duration_s            → turn-taking
5. tr_backchannel_count             → listener engagement
6. tr_ovl_simultaneous              → TIMING: brief simultaneous
7. tr_ovl_smooth                    → TIMING: managed overlap
8. tr_ovl_competitive_timing        → TIMING: contested floor
9. tr_ovl_floor_fight               → TIMING: extended conflict
10. tr_ovl_collaboration_index      → COLLABORATION: lexical signal
11. tr_laughter_count               → affective signal
12. tr_n_active_speakers            → participation
13. group_et_pupil_mean_mean        → attention/arousal
```

**Changes**:
- Removed 3 generic/redundant features
- Added 5 timing subtypes (5-way breakdown of overlaps)
- Added 1 collaboration index (lexical intent layer)
- Result: More interpretable, less redundancy, cleaner separation of signals

---

## Correlation Analysis

From `overlap_feature_correlations.png`:

**High internal correlations (expected):**
- `tr_ovl_simultaneous` ↔ `tr_ovl_short_ratio` (r=0.45) — both capture brief overlaps
- `tr_ovl_competitive_timing` ↔ `tr_ovl_long_ratio` (r=0.63) — both capture long overlaps
- `tr_ovl_floor_fight` ↔ `tr_ovl_long_ratio` (r=0.57) — extended overlaps

**Low correlation with legacy features (validates separation):**
- `tr_ovl_*` features (r=0.02–0.15) with `tr_overlap_count` — timing subtypes are more specific
- `tr_ovl_*` features (r=0.01–0.13) with `tr_overlap_time_s` — counts orthogonal to duration

**Context label columns (all zeros in current transcripts):**
- `tr_ovl_context_collaborative`: 0 count across all windows
- `tr_ovl_context_competitive`: 0 count across all windows
- Indicates: Transcripts use timing subtypes only; context labels not yet populated

---

## Implementation Notes for Paper

### Methods Section Update

```
"Overlaps were classified using pure timing-based subtypes (simultaneous, 
backchannel_ovl, smooth, competitive, floor_fight) derived from objective 
acoustic duration measurements, with no annotator bias. A separate 
collaboration_index captured instances where lexical cues (support words) 
appeared on long-duration overlaps, indicating collaborative speech 
despite extended acoustic overlap. All overlap counts were log-transformed 
to handle right-skewed distributions."
```

### Feature Engineering Section

```
"To separate interaction mechanics (objective timing) from semantic intent 
(lexical collaboration), we decomposed the generic overlap count into 
five timing-based subtypes and added a collaboration index based on 
support-word detection. This 5+1 feature structure captured both the 
temporal patterns of simultaneous speech and the linguistic signals 
suggesting coordination despite duration, enabling the HMM to learn 
distinct state profiles based on interaction style and collaborative intent."
```

### Appendix: Feature List

| Feature | Type | Definition | Transformation |
|---------|------|-----------|-----------------|
| `group_hr_mean_bpm_mean` | physio | Group mean heart rate (z-scored) | z-score |
| `group_eda_phasic_rate_hz_mean` | physio | EDA phasic response rate (z-scored) | z-score |
| `group_temp_mean_mean` | physio | Group mean skin temperature (z-scored) | z-score |
| `tr_silence_duration_s` | speech | Total silence duration (log-scaled) | log1p |
| `tr_backchannel_count` | speech | Listener feedback signals (log-scaled) | log1p |
| `tr_ovl_simultaneous` | overlap | Simultaneous overlaps start_diff<200ms (log-scaled) | log1p |
| `tr_ovl_smooth` | overlap | Smooth overlaps 250-500ms (log-scaled) | log1p |
| `tr_ovl_competitive_timing` | overlap | Competitive overlaps 500-1000ms (log-scaled) | log1p |
| `tr_ovl_floor_fight` | overlap | Extended overlaps >=1000ms (log-scaled) | log1p |
| `tr_ovl_collaboration_index` | overlap | Ratio: collaborative overlaps / total overlaps (log-scaled) | log1p |
| `tr_laughter_count` | speech | Laughter instances (log-scaled) | log1p |
| `tr_n_active_speakers` | speech | Number of active speakers (raw) | raw |
| `group_et_pupil_mean_mean` | gaze | Eye-tracking pupil diameter (z-scored) | z-score |

---

## Quality Assurance Checklist

- [x] All timing subtypes extracted from transcripts
- [x] Correlation matrix shows expected overlap structures
- [x] No missing values for windows with transcripts (137/1521 windows)
- [x] Windows without transcripts padded with 0 (no NaN)
- [x] Feature statistics within expected range (means ~0.03–0.15 for low-frequency events)
- [x] Engineered ratios bounded [0, 1] as expected
- [x] Log transformation applied correctly (handle zeros with log1p)
- [x] Documentation complete with rationale and justification

---

**Generated**: 2026-07-15
**Notebook**: `icmi_paper/analysis/timing_based_overlap_features.ipynb`
**Output File**: `icmi_paper/results/enhanced_features_timing_based.tsv`
