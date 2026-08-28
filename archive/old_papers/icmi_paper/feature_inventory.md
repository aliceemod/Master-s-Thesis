# AffectAI Feature Inventory — Collective Window Analysis

**Granularity:** 30-second windows, group-level aggregates across P1–P4  
**Analysis subset:** T1–T3 tasks, n=545 complete-transcript windows (507 with all HMM features)  
**Full matrix shape:** 1521 windows × 58 columns (`icmi_paper/results/enhanced_features_final.tsv`)  
*Last updated: 2026-07-15 — reflects two-layer overlap annotation (July 2026 reclassification)*

**HMM status legend:**

| Symbol | Meaning |
|---|---|
| ✅ | Selected for HMM analysis (CLEAN_FEATS) |
| ⚠️ | Available, useful for other analyses, not in HMM |
| ❌ | Excluded — reason given |

---

## 1. Physiological Features (EmotiBit wrist sensors)

Source: EmotiBit PPG/EDA/temperature sensors worn on the wrist by each participant.  
Aggregation: mean and std across 4 participants per window. Coverage: 96–100% of all windows.

> **Thesis note:** These features capture collective physiological arousal. Within-group z-scoring is essential — raw values carry between-session baseline drift (especially EDA tonic). HRV captures regulatory capacity / stress, complementary to HR arousal.

### Heart Rate

| Feature | Description | Units | Coverage | HMM | Preprocessing |
|---|---|---|---|---|---|
| `group_hr_mean_bpm_mean` | Mean HR across participants | bpm | 96% | ✅ | z-score within group |
| `group_hr_mean_bpm_std` | Std of HR — physiological synchrony proxy | bpm | 90% | ⚠️ | z-score |
| `group_hrv_rmssd_ms_mean` | Mean HRV (RMSSD) — stress/regulation | ms | 88% | ⚠️ | z-score |
| `group_hrv_rmssd_ms_std` | Std of HRV across participants | ms | 75% | ❌ | — |

`group_hr_mean_bpm_mean` ✅ — core arousal indicator, high coverage.  
`group_hr_mean_bpm_std` ⚠️ — captures whether participants' HR converged (synchrony); good thesis measure of group cohesion.  
`group_hrv_rmssd_ms_mean` ⚠️ — stress/regulation; lower coverage (12% missing); good sensitivity test.  
`group_hrv_rmssd_ms_std` ❌ — 27% missing; too much data loss.

### EDA (Electrodermal Activity)

| Feature | Description | Units | Coverage | HMM | Preprocessing |
|---|---|---|---|---|---|
| `group_eda_phasic_rate_hz_mean` | Mean phasic EDA event rate — emotional reactivity | Hz | 96% | ✅ | z-score within group |
| `group_eda_phasic_rate_hz_std` | Std of phasic EDA rate | Hz | 90% | ⚠️ | z-score |
| `group_eda_tonic_mean_mean` | Mean EDA tonic level — slow arousal baseline | µS | 96% | ❌ | — |
| `group_eda_tonic_mean_std` | Std of EDA tonic | µS | 90% | ❌ | — |

`group_eda_phasic_rate_hz_mean` ✅ — phasic rate = momentary reactivity spikes; high coverage; least affected by baseline drift.  
`group_eda_tonic_mean_mean` ❌ — slow-varying; mainly separates groups/sessions, not within-session states. Requires careful detrending. **Thesis use:** session-level stress covariate.

### Temperature

| Feature | Description | Units | Coverage | HMM | Notes |
|---|---|---|---|---|---|
| `group_temp_mean_mean` | Mean skin temperature | °C | 100% | ❌ | log1p variance ≈ 0.015 — near-constant |
| `group_temp_mean_std` | Std of temperature | °C | 96% | ❌ | Same issue |

❌ Both excluded — temperature changes on a timescale of minutes, not 30-second windows. **Thesis use:** session-level sustained stress indicator.

---

## 2. Eye-Tracking Features (Tobii Glasses Pro 2)

Source: Tobii Glasses worn by each participant. Aggregated per window. Coverage: 85–87%.

> **Thesis note:** Pupil diameter = cognitive load + attentional engagement. Blink rate = fatigue/disengagement. Gaze dispersion = active scanning vs. fixed focus. Together these reveal *where* attention goes and *how much* cognitive effort is deployed.

| Feature | Description | Units | Coverage | HMM | Preprocessing |
|---|---|---|---|---|---|
| `group_et_pupil_mean_mean` | Mean pupil diameter | mm | 85% | ✅ | z-score within group |
| `group_et_pupil_mean_std` | Std of pupil — attention synchrony | mm | 85% | ⚠️ | z-score |
| `group_et_pupil_std_mean` | Mean within-participant pupil variability | mm | 85% | ⚠️ | z-score |
| `group_et_pupil_slope_per_s_mean` | Pupil dilation trend per second | mm/s | 85% | ⚠️ | z-score |
| `group_et_blink_rate_per_min_mean` | Mean blink rate — fatigue/disengagement | blinks/min | 84% | ⚠️ | log1p |
| `group_et_blink_rate_per_min_std` | Std of blink rate | blinks/min | 81% | ❌ | — |
| `group_et_gaze_dispersion_mean` | Mean gaze spatial dispersion | norm. | 85% | ⚠️ | z-score |
| `group_et_gaze_dispersion_std` | Std of gaze dispersion | norm. | 85% | ❌ | — |
| `group_et_gaze_valid_frac_mean` | Mean data validity fraction | [0,1] | 86% | ❌ | — |
| `group_et_gaze_velocity_mean_mean` | Mean gaze velocity | units/s | 85% | ⚠️ | z-score |
| `group_et_gaze_velocity_mean_std` | Std of gaze velocity | units/s | 85% | ❌ | — |

`group_et_pupil_mean_mean` ✅ — best single ET feature; captures attention + cognitive load.  
`group_et_blink_rate_per_min_mean` ⚠️ — complementary to pupil (fatigue/disengagement); good robustness test.  
`group_et_gaze_dispersion_mean` ⚠️ — discriminates "reading materials" vs. "looking at each other" states; thesis: attention coordination.  
Std features ❌ — redundant; add noise.

---

## 3. Speech & Turn-Taking Features (transcript annotations)

Source: Manual transcript annotations (SPK/SIL/BCK/LAU/FP rows).  
Coverage: **36% of all windows** (T1–T3 only; non-transcript windows filled with 0).

> **Thesis note:** These features capture the *structure* of conversation. Silence = thinking/processing or disengagement. Backchannel = listener affiliation (deliberate feedback, distinct from acoustic overlap). Laughter = social bonding. Active speakers = who's engaged. Together they describe *how* the group organises its floor.

| Feature | Description | Units | Coverage | HMM | Preprocessing |
|---|---|---|---|---|---|
| `tr_silence_duration_s` | Total annotated silence in window | s | 36% | ✅ | log1p |
| `tr_backchannel_count` | Listener feedback signals (BCK rows) | count | 36% | ✅ | log1p |
| `tr_laughter_count` | Laughter events (LAU rows) | count | 36% | ✅ | log1p |
| `tr_n_active_speakers` | Unique speakers in window | count | 36% | ✅ | log1p |
| `tr_speaking_entropy` | Shannon entropy of speaking-time distribution | bits | 36% | ❌ | — |
| `tr_spk_count` | Number of speech segments | count | 36% | ❌ | — |
| `tr_spk_duration_s` | Total speech duration | s | 36% | ❌ | — |
| `tr_filled_pause_count` | Filled pauses (um, uh) — hesitation | count | 36% | ⚠️ | log1p |

`tr_silence_duration_s` ✅ — highest log1p variance (0.978) of all features; most discriminative.  
`tr_backchannel_count` ✅ — deliberately separate from OVL: BCK = purposeful listener signal.  
`tr_laughter_count` ✅ — qualitatively unique; no other feature captures positive social affect.  
`tr_n_active_speakers` ✅ — participation breadth (1=monologue, 4=everyone engaged).  
`tr_speaking_entropy` ❌ — highly correlated with `tr_n_active_speakers`; redundant.  
`tr_filled_pause_count` ⚠️ — **thesis candidate**: captures hesitation/uncertainty; could discriminate "thinking" states.

---

## 4. Overlap Features — Two-Layer System (July 2026)

Source: Canonical transcripts `transcripts/final/`, processed by `tools/relabel_overlaps.py`.  
Coverage: **100%** for all overlap features (non-transcript windows = 0, which is valid).  
**All events deduplicated by onset time** — multi-speaker overlaps count as 1 event (107 duplicates removed).

> **Thesis note (methodological contribution):** The two-layer system separates:  
> - **Layer 1 (`subtype`)**: Objective, acoustic-duration-based — 5 categories, no annotator bias  
> - **Layer 2 (`context_label`)**: Communicative intent via LLM-assisted rule-based detection  
> This allows features to capture both *mechanics* (how long overlaps are) and *social function* (whether simultaneous speech is cooperative or conflictual). **81% of T1–T3 windows have zero overlaps** — the zero value itself is meaningful (quiet/focused state).

### Global Overlap Metrics

| Feature | Description | Units | Coverage | HMM | Preprocessing |
|---|---|---|---|---|---|
| `tr_ovl_count` | Deduplicated overlap event count | count | 100% | ⚠️ | log1p |
| `tr_ovl_time_s` | Total overlap duration (deduplicated) | s | 100% | ✅ | log1p |

`tr_ovl_time_s` ✅ — better than count: continuous signal, 0=quiet, >10s=heavy overlap.  
`tr_ovl_count` ⚠️ — highly correlated with time_s; available but not in HMM.

### Timing Subtypes (Layer 1 — pure acoustic duration, no annotator bias)

| Feature | Rule | Coverage | HMM | Preprocessing |
|---|---|---|---|---|
| `tr_ovl_simultaneous` | start_diff < 200ms | 100% | ❌ | log1p |
| `tr_ovl_backchannel` | overlap < 250ms | 100% | ❌ | log1p |
| `tr_ovl_smooth` | overlap 250–500ms | 100% | ⚠️ | log1p |
| `tr_ovl_competitive_timing` | overlap 500–1000ms | 100% | ✅ | log1p |
| `tr_ovl_floor_fight` | overlap ≥ 1000ms | 100% | ✅ | log1p |

`tr_ovl_competitive_timing` ✅ — timing-contested floor (0.5–1s); tension signal.  
`tr_ovl_floor_fight` ✅ — extreme sustained conflict (≥1s); qualitatively distinct severity level.  
`tr_ovl_smooth` ⚠️ — coordinated handoffs; good sensitivity test feature; thesis: turn coordination quality.  
`tr_ovl_simultaneous` / `tr_ovl_backchannel` ❌ — correlated with collaboration_index; captured by global time_s.

### Context Labels (Layer 2 — communicative intent, LLM-assisted)

| Feature | Description | Coverage | HMM | Notes |
|---|---|---|---|---|
| `tr_ovl_ctx_collaborative` | Count: overlaps with support-word context | 100% | ❌ | r=0.57–0.76 with all timing subtypes — redundant |
| `tr_ovl_ctx_completion` | Count: sentence-completion overlaps | 100% | ❌ | Only 26 events in T1–T3; too sparse |
| `tr_ovl_collaboration_index` | ctx_collaborative / total_ovl | 100% | ✅ | Rate normalises for activity level |
| `tr_ovl_completion_index` | ctx_completion / total_ovl | 100% | ⚠️ | Sparse but linguistically interesting |

`tr_ovl_collaboration_index` ✅ — rate (not count) is less correlated with timing subtypes; captures whether overlaps are *used cooperatively* vs. just being acoustically present.  
`tr_ovl_ctx_completion` ⚠️ — **thesis candidate**: sentence completions are rare (26 events) but linguistically significant; report as descriptive statistics.

### Derived Ratios

| Feature | Formula | HMM | Notes |
|---|---|---|---|
| `tr_ovl_competitive_ratio` | (competitive + floor_fight) / total | ❌ | Meaningful only in 19% windows with overlaps; unstable denominator |
| `tr_ovl_floor_fight_ratio` | floor_fight / total | ❌ | Same issue |

❌ All ratio features excluded from HMM — see zero-inflation analysis in `timing_based_overlap_features.ipynb`.

---

## 5. Legacy Overlap Features (OLD system — superseded July 2026)

> ⚠️ **Do not use** for new analyses. Superseded by the two-layer overlap system above.

| Feature | Old meaning | Superseded by |
|---|---|---|
| `tr_overlap_count` | Raw OVL rows (not deduplicated) | `tr_ovl_count` |
| `tr_overlap_time_s` | Total overlap time from SPK rows | `tr_ovl_time_s` |
| `tr_competitive_overlap` | Old mixed competitive/floor_fight count | `tr_ovl_competitive_timing` + `tr_ovl_floor_fight` |
| `tr_cooperative_overlap` | Old mixed collaborative/smooth/backchannel | `tr_ovl_collaboration_index` |
| `tr_collaborative_overlap` | Old semantic collaborative count | `tr_ovl_collaboration_index` |
| `tr_conflict_overlap_ratio` | Old competitive / (competitive + cooperative) | `tr_ovl_competitive_ratio` (also excluded) |

---

## 6. Audio Prosodic Features (DPA close-talk microphones + opensmile GeMAPSv01b)

Source: Per-speaker DPA 4060 close-talk microphone recordings, extracted by `tools/audio_feature_extraction_egemaps.py` using **opensmile GeMAPSv01b** (62-parameter set) and VAD.  
Current availability: **task-level only** (one summary value per participant × task). Not in `collective_window_features.tsv`.

> **Why not in HMM:** The audio pipeline runs on full task recordings, producing one summary statistic per participant per task. The HMM requires **30-second window-level** features. Windowing would require re-running opensmile with a 30-second sliding window configuration — feasible but not yet implemented. Additionally, several audio features overlap conceptually with transcript-based features already in the HMM (`speaking_fraction` ≈ `tr_silence_duration_s`; `overlap_fraction` ≈ `tr_ovl_time_s`).
>
> **For the thesis:** These features capture *how* people speak (vocal quality, pitch, energy) rather than *what* they say or *who* speaks when. Particularly valuable for individual affect expression analysis. The granularity issue (task vs. window) is the primary reason for exclusion — not theoretical irrelevance.

### Acoustic / Prosodic Features (opensmile GeMAPSv01b)

| Feature | Description | Units | Level | HMM | Unique signal not in HMM? |
|---|---|---|---|---|---|
| `audio_energy_mean` | Mean vocal loudness | dBFS | participant × task | ❌ | ✓ Not captured by any other modality |
| `audio_energy_sd` | Loudness variability — prosodic dynamics | dBFS | participant × task | ❌ | ✓ Expressiveness signal |
| `audio_pitch_mean` | Mean F0 on voiced segments (semitone scale) | semitone | participant × task | ❌ | ✓ Emotional tone, not captured elsewhere |
| `audio_pitch_sd` | Pitch variability — intonation range | semitone | participant × task | ❌ | ✓ Arousal / expressiveness |
| `audio_hnr_mean` | Harmonics-to-noise ratio — voice quality | dB | participant × task | ❌ | ✓ Vocal stress indicator |
| `audio_jitter_mean` | F0 period perturbation — voice instability | % | participant × task | ❌ | ✓ Stress / fatigue signal |
| `audio_shimmer_mean` | Amplitude perturbation — voice quality | dB | participant × task | ❌ | ✓ Fatigue / vocal quality |
| `audio_mean_voiced_segment_s` | Mean voiced segment duration — fluency | s | participant × task | ❌ | ✓ Speech fluency proxy |
| `audio_mean_unvoiced_segment_s` | Mean pause / silence duration | s | participant × task | ❌ | ≈ partially captured by `tr_silence_duration_s` |
| `audio_voiced_segments_per_sec` | Speech fragmentation rate | count/s | participant × task | ❌ | ✓ Hesitation / fragmented speech |
| `audio_speech_rate_proxy` | Loudness-peak-based syllable rate proxy | peaks/s | participant × task | ❌ | ✓ Speaking pace, not captured elsewhere |

**Full eGeMAPS feature set:** 62 parameters (18 LLDs × 2 functionals + 16 F0/Loudness functionals + 4 unvoiced means + 6 prosodic descriptors). See `features/audio_feature_definitions.tsv` for complete list.

### Interaction / VAD Features

| Feature | Description | Units | Level | HMM | Notes |
|---|---|---|---|---|---|
| `audio_speaking_fraction` | Fraction of task time spent speaking | [0,1] | participant × task | ❌ | ≈ complement of `tr_silence_duration_s`; task-level not window-level |
| `audio_speaking_time_s` | Total speaking duration | s | participant × task | ❌ | Same as above |
| `audio_overlap_fraction` | Fraction of time speaking simultaneously with others | [0,1] | participant × task | ❌ | ≈ `tr_ovl_time_s / window_duration`; superseded |
| `audio_pause_count` | Number of pauses ≥300ms | count | participant × task | ❌ | ≈ partially captured by silence; task-level only |

### Why windowing would enable HMM use

If the audio pipeline is run with a 30-second sliding window, the following would be most valuable additions to the HMM feature set:

| Would add | Why |
|---|---|
| `audio_energy_mean` (group mean) | Vocal arousal — complementary to HR/EDA |
| `audio_pitch_mean` (group mean) | Emotional tone — unique signal |
| `audio_pitch_sd` (group mean) | Expressiveness — unique signal |
| `audio_hnr_mean` (group mean) | Vocal stress — unique signal |

**Implementation needed:** Re-run `tools/audio_feature_extraction_egemaps.py` with `--window-s 60 --hop-s 60` flag (not currently implemented), then group-average across P1–P4 per window and merge into `collective_window_features.tsv`.


---

## 7. VAD Self-Report (Valence–Arousal–Dominance, in-task probes)

Source: Tablet-delivered SAM-scale probes during each task, timestamped with LSL clock.  
Granularity: **~4–5 probes per participant per task** (T1: mean 4.4, T2: mean 4.7, T3: mean 4.0; range 2–8).  
Scale: 1–9 (SAM scale). Three items per probe: Valence, Arousal, Dominance.  
Inter-probe interval: jittered 90–150 s.

> **Why not in HMM (methodological decision, not a data gap):**
>
> VAD self-report is the **outcome/validation measure** for this analysis. Including it as an HMM input would create a **circular analysis**: the model would simply discover clusters that reflect how participants rated themselves, rather than discovering latent states from objective behavioral and physiological signals.
>
> The intended design is:
> 1. **HMM input** → objective signals only (physio, ET, interaction)
> 2. **Post-hoc validation** → correlate discovered states with VAD ratings
>
> If VAD were included in the input, any found state structure would trivially reflect the VAD covariance, not independent behavioral/physiological dynamics. This would undermine the key claim that the model discovers states from multimodal behavioral signals.
>
> **The data IS available and CAN be windowed**: With ~4 probes per 10-min task, each probe covers ~2 windows (60s windows). Forward-filling from probe LSL timestamp to subsequent windows is feasible. This approach IS appropriate for **validation** (correlating state assignment with forward-filled VAD ratings).

| Feature | Description | Current level | HMM | Use |
|---|---|---|---|---|
| `vad_valence` | Subjective positive/negative affect (1–9 SAM) | probe × participant | ❌ methodological | ✅ Post-hoc state validation |
| `vad_arousal` | Subjective activated/calm (1–9 SAM) | probe × participant | ❌ methodological | ✅ Post-hoc state validation |
| `vad_dominance` | Subjective in-control/submissive (1–9 SAM) | probe × participant | ❌ methodological | ✅ Post-hoc state validation |
| `task_vad_valence_mean` | Mean valence across all probes in task (group mean) | task | ❌ task-level | ✅ Task-level state characterisation |
| `task_vad_arousal_mean` | Mean arousal across all probes in task (group mean) | task | ❌ task-level | ✅ Task-level state characterisation |
| `task_vad_dominance_mean` | Mean dominance across task (group mean) | task | ❌ task-level | ✅ Task-level state characterisation |

**What the pipeline currently does:** Merges task-mean VAD on `group_id × task_id` — every window in a task gets the same value. See `compare_latent_state_runs.py`.

**How to use for validation:**
1. Forward-fill each participant's probe rating to subsequent windows (until next probe)
2. Average across P1–P4 to get group-level VAD per window
3. After HMM state assignment, compute mean VAD per state and test state × VAD association (ANOVA or Kruskal-Wallis)
4. Expected: Competitive-Tense state → high group arousal; Reflective state → moderate valence

---

## 8. Participant-Level Interaction Features

Source: `features/transcript_participant_task.tsv` — one row per group × task × participant.  
> **Thesis note:** Building blocks for individual contribution analysis. Can be aggregated to group level (mean, std, entropy across P1–P4) to create group window features.

| Feature | Description | Thesis application |
|---|---|---|
| `speaking_share` | Fraction of task duration speaking | Floor holding / dominance |
| `turn_count` | Number of speech turns | Turn frequency |
| `mean_turn_duration_s` | Mean turn length | Speaking style |
| `mean_response_gap_ms` | Mean gap before taking floor | Social timing, reactivity |
| `overlap_fraction` | Overlap time / speaking time | Individual conflict tendency |
| `interruption_count` | Times interrupted or interrupted others | Conflict behaviour |
| `backchannel_given_count` | Backchannels given | Affiliation, attentiveness |
| `backchannel_latency_mean_ms` | Mean BCK response latency | Attentiveness speed |
| `filled_pause_count` | Filled pauses (um/uh) | Uncertainty, cognitive load |


---

## 9. Final HMM Feature Set — Selection Methodology & Results

### 9.1 How features were selected

Feature selection used three criteria scored 0-3 each (max 9), documented in `icmi_paper/analysis/timing_based_overlap_features.ipynb`:

| Criterion | Score 3 | Score 2 | Score 1 | Score 0 |
|---|---|---|---|---|
| **Coverage** | >=90% of T1-T3 windows | >=50% | >=30% | <30% |
| **Variance** | log1p std >= 0.50 | >=0.20 | >=0.05 | <0.05 |
| **Theory** | Direct, unique signal for collective affective state | Useful but partly derivative | Low priority | Not applicable |

After scoring, **pairwise correlations** were computed among top-scoring features. Correlations were checked within the 105 windows that have overlaps (not globally) to avoid zero-inflation artefacts — 81% of windows have no overlaps, so two count features that are both zero for the same 440 windows appear spuriously correlated globally.

Features with max|r| > 0.75 with an already-selected higher-priority feature were dropped.

### 9.2 Verified numbers (all confirmed from data)

| Quantity | Value |
|---|---|
| Total T1-T3 windows in matrix | 1521 |
| Windows with transcript data | 545 |
| Windows with >= 1 overlap | 105 (19%) |
| Complete-case (dropna on FINAL_FEATS) | 507 |
| After truncation + isolated imputation | **508** (recommended) |
| Previous HMM used SimpleImputer — why wrong | 3 sequences had 8-16 consecutive sensor-failure windows filled with artificial neutral values |

**Sensor failure details (3 sequences truncated):**
- `grp-12 T3`: EmotiBit off for 100% of task — entire sequence dropped
- `grp-10 T2`: Tobii dropout from window 17 onward — truncated at window 17
- `grp-16 T3`: Tobii dropout windows 11-18 — truncated at window 11
- `grp-10 T3` window 12: single isolated gap — imputed with within-group mean (safe)

### 9.3 Key drop decisions (verified correlations, overlap-windows only, n=105)

| Dropped feature | Kept instead | r | Reason |
|---|---|---|---|
| `tr_ovl_floor_fight` | `tr_ovl_time_s` | r=0.85 | Floor fights (>=1s) dominate total overlap time — near-identical information |
| `tr_ovl_ctx_collaborative` (count) | `tr_ovl_collaboration_index` (rate) | Count r=0.79 with `tr_ovl_count` | Count driven by quantity. Rate measures quality independent of how many overlaps occurred |
| `tr_ovl_count` | `tr_ovl_time_s` | r=0.85 | Duration is a richer continuous signal than count |
| `group_temp_mean_mean` | — | var=0.015 | Near-constant within sessions; contributes no state discrimination |
| `tr_speaking_entropy` | `tr_n_active_speakers` | r>0.7 | Near-equivalent measures of participation spread |
| `group_hr_mean_bpm_std` | `group_hr_mean_bpm_mean` | r=0.65 | Synchrony signal correlated with mean; use in separate synchrony analysis |

### 9.4 FINAL_FEATS (10 features)

```python
FINAL_FEATS = [
    # Physiological arousal (3) — z-score within group
    'group_hr_mean_bpm_mean',          # arousal baseline
    'group_eda_phasic_rate_hz_mean',   # emotional reactivity
    'group_et_pupil_mean_mean',        # attention / cognitive load
    # Speech & floor dynamics (4) — log1p
    'tr_silence_duration_s',           # floor activity: how quiet
    'tr_backchannel_count',            # listener affiliation
    'tr_laughter_count',               # positive affect (unique signal)
    'tr_n_active_speakers',            # participation breadth
    # Overlap (3) — log1p
    'tr_ovl_time_s',                   # total simultaneous speech load
    'tr_ovl_competitive_timing',       # conflict intensity (500-1000ms, timing-based)
    'tr_ovl_collaboration_index',      # cooperative intent fraction (rate, not count)
]
```

**Complete-case windows (after sensor-failure truncation + isolated imputation): 508**  
Groups: all 10 (grp-07 through grp-16) | Tasks: T1, T2, T3

| Dimension | Features |
|---|---|
| Physiological arousal | HR mean, EDA phasic rate, pupil diameter |
| Floor activity | Silence duration, active speakers |
| Social signals | Backchannel count, laughter count |
| Overlap — how much | ovl_time_s |
| Overlap — how contested | ovl_competitive_timing |
| Overlap — how cooperative | ovl_collaboration_index |

### 9.5 Additional candidates — plausible additions for robustness tests

These were not in the primary selection but are theoretically sound and independent:

| Feature | Coverage | max|r| with FINAL_FEATS | Dimension added | Verdict |
|---|---|---|---|---|
| `group_et_blink_rate_per_min_mean` | 95% | 0.23 (near-independent) | Fatigue / disengagement | Reasonable addition — adds dimension missing from current set |
| `tr_ovl_smooth` | 100% | 0.44 | Coordinated turn transitions (250-500ms) | Reasonable — r=0.06 with competitive_timing (truly independent) |
| `group_hrv_rmssd_ms_mean` | 88% | 0.14 (independent) | Stress regulation quality | Reasonable — distinct from HR; 12% missing |
| `group_hr_mean_bpm_std` | 88% | 0.65 (moderate) | Physiological synchrony | Optional — correlated with HR mean; use in focused synchrony analysis |
| `tr_filled_pause_count` | 36% | 0.19 | Cognitive load / uncertainty | Weak — low variance, use in thesis robustness check only |

---

## 10. Thesis Analysis Opportunities

Features not in the HMM but worth exploring in thesis chapters:

| Feature | Chapter suggestion |
|---|---|
| `group_hr_mean_bpm_std` | Physiological synchrony across states — group cohesion marker |
| `group_et_blink_rate_per_min_mean` | Fatigue / disengagement trends across task time |
| `group_et_gaze_dispersion_mean` | Visual attention coordination — shared vs. individual focus |
| `tr_filled_pause_count` | Cognitive load / uncertainty during idea generation |
| `tr_ovl_smooth` | Smooth turn coordination — quality of floor management |
| `tr_ovl_ctx_completion` | Collaborative sentence completions — qualitative analysis |
| `group_hrv_rmssd_ms_mean` | Stress regulation across states |
| eGeMAPS prosodic (windowed) | Individual vocal affect expression — requires audio windowing pipeline |
| Participant-level speaking share | Dominance / floor holding within states |



