# Feature Catalog — Technical Reference

_Last updated: 2026-08-12._

This document is the authoritative technical reference for **every derived feature** produced anywhere in this repository: what it is, exactly how it is computed (grounded in the actual extraction code, not aspirational plans), and what it means. It complements [dataset_description.md](dataset_description.md) (which reports *coverage* — how much data exists) by describing *content* — what each column in every features table actually represents.

## How to read this document

- Every section names the exact extraction script(s) and, where relevant, the exact output column names as they appear on disk.
- **Status legend**, shown per pipeline:
  - ✅ **Generated** — the script has been run and its output `.tsv` exists under `analysis/results/` right now; numbers from these tables can be cited directly.
  - 🧩 **Implemented, not yet generated** — the extraction code exists and is correct/runnable, but no output file has been committed to `analysis/results/` in this checkout. Do not cite feature values from these pipelines until they are actually run; only cite the *methodology* if needed.
- All participant identifiers are `P1`–`P4` (never real names) and all task identifiers are `T0`–`T4`, per [security.instructions.md](../.github/instructions/security.instructions.md) and [bids-conventions.instructions.md](../.github/instructions/bids-conventions.instructions.md).
- "Task-level" tables have one row per `session × task × participant` (or `session × task` for group-level tables). "Window-level" tables additionally roll a 30 s sliding window (15 s step, 50% overlap) across the task, with one row per `session × task × participant × window_index`.

## Contents

1. [Physiology (EmotiBit)](#1-physiology-emotibit) — ✅ Generated
2. [Eye-tracking / pupillometry](#2-eye-tracking--pupillometry) — ✅ Generated (simple participant-level pipeline; rich per-participant gaze+blink extractor generated 2026-08-12; group-level rich gaze/blink also recovered from the paper matrix)
3. [Audio / prosody](#3-audio--prosody) — ✅ Generated (Pipeline B, participant-level) / 🧩 (Pipeline A, full per-task extractor — blocked, raw DPA WAVs not in this workspace)
4. [Transcript / turn-taking](#4-transcript--turn-taking) — ✅ Generated (both participant-level and richer participant+group pipelines)
4b. [Lexical / text content](#4b-lexical--text-content-features) — ✅ Generated (2026-08-12)
5. [Group dynamics / cross-participant synchrony](#5-group-dynamics--cross-participant-synchrony) — ✅ Generated
6. [Semantic biomarker composites](#6-semantic-biomarker-composites) — ✅ Generated
7. [Personality & demographic traits (BFI-44)](#7-personality--demographic-traits-bfi-44) — ✅ Generated
8. [Video / 3D pose / gaze-world / gesture](#8-video--3d-pose--gaze-world--gesture) — 🧩 Raw artifacts only, no aggregated feature table yet

---

## 1. Physiology (EmotiBit)

**Status:** ✅ Generated. **Script:** [tools/features/extract_physio_features.py](../tools/features/extract_physio_features.py). **Outputs:** `analysis/results/physio_features/{physio_participant_task.tsv, physio_window_30s.tsv, physio_qc_summary.tsv, physio_feature_definitions.tsv}` (plus legacy-named aliases `features_physio_participant_task.tsv`, `features_physio_window_30s.tsv`).

### Pipeline

1. **Input**: task-split files under `physio/`, named `*_task-T*_run-01_acq-P*_emotibit.tsv.gz`, one per participant per task, with `lsl_time` plus numeric `value_0..value_21` channels.
2. **Channel mapping** (CLI-overridable indices): PPG green/red/IR (3 candidate channels, best one auto-selected per row — see below), EDA, skin temperature, device-reported HR (`value_6`), thermopile (`value_9`), auxiliary temperature (`value_11`), accelerometer XYZ (`value_13,14,15`), gyroscope XYZ (`value_16,17,18`), magnetometer XYZ (`value_19,20,21`). `channel_map_unconfirmed` is flagged until `--channel-map-confirmed` is explicitly passed.
3. **Sample rate**: $1/\text{median}(\Delta t)$ over `lsl_time`; expected plausible device range 10–60 Hz (`sample_rate_unusual` flag otherwise).
4. **Signal cleaning**: each channel is range-clipped to a physiologically plausible band and any sentinel/out-of-range value is set to NaN (`_clean_signal`/`_clean_matrix`) before any statistic is computed.
5. **PPG → heart rate & HRV** (`_select_ppg_features`, tries all 3 PPG channels — green/red/IR — and keeps the best):
   - Signal is band-pass filtered 0.7–3.0 Hz (`_bandpass_ppg`, Butterworth SOS filter, or FFT band-mask fallback if `scipy` is unavailable) and z-normalized.
   - Both signal polarities (`+1`, `-1`) are tried; peaks are detected (`scipy.signal.find_peaks`, minimum inter-beat distance derived from the expected HR, prominence ≥ 0.35) and inter-beat intervals (IBIs) computed.
   - IBIs outside 300–2000 ms are dropped as implausible; remaining IBIs within 25% (min 250 ms) of the local median are kept as "clean."
   - A **0–1 quality score** is computed as `0.4·clean_fraction + 0.3·agreement_score + 0.2·regularity_score + 0.1·count_score`, where `agreement_score` compares PPG-derived HR to the device's own HR channel. The channel/polarity with the highest quality score wins.
   - `hrv_rmssd_ms` (root-mean-square of successive IBI differences), `hrv_sdnn_ms` (SD of IBIs), `hrv_pnn50_pct` (% of successive IBI differences > 50 ms) are computed from the winning channel's clean IBIs and are **blanked to NaN** if `hrv_quality_score < 0.65` or `rmssd > 300 ms` (i.e. quality-gated — an unreliable estimate is never silently reported).
   - `hr_mean_bpm` prefers the device's own HR channel (when ≥ 50% of samples are in the plausible 35–220 bpm range); otherwise falls back to an FFT-based PPG rate proxy (`_ppg_bpm`, dominant frequency in the 0.6–3.5 Hz band × 60).
6. **EDA → tonic/phasic decomposition** (`_eda_deep_features`):
   - Raw EDA is linearly interpolated across gaps, then low-pass filtered at 0.05 Hz (Butterworth SOS or centered rolling-median fallback) to obtain the **tonic** (slow, baseline skin-conductance level) component; **phasic** = raw − tonic.
   - Skin conductance responses (SCRs) are peak-detected on the phasic signal (`scipy.signal.find_peaks`, minimum inter-peak distance 0.8 s, prominence ≥ `max(0.5·phasic_std, 0.002)`); `eda_scr_count`, `eda_scr_rate_hz` (count/duration), and mean/95th-percentile SCR amplitude (peak prominence) are reported.
   - `eda_phasic_detection_limited` flag fires when `scipy` is unavailable or EDA coverage is too low for reliable peak detection.
7. **Temperature, thermopile, auxiliary temperature**: mean/median/SD/min/p95/max and linear slope-per-second (`linear_slope`, OLS fit of value vs. time) for each channel independently.
8. **IMU (accelerometer/gyroscope/magnetometer)** (`_vector_features`): per-sample Euclidean vector magnitude (e.g. $\sqrt{a_x^2+a_y^2+a_z^2}$ for accel); `_dynamic` = |magnitude − median magnitude| (deviation from resting posture); `_jerk` = |first difference of magnitude| × sample rate (rate of change); `_high_fraction` (accel only) = fraction of samples exceeding a 1.5 g motion threshold, used for the `motion_contaminated` QC flag.
9. **T0 baselining**: `hrv_rmssd_ms_delta_t0` = task-level RMSSD minus the same participant's own `T0` RMSSD (within-participant, not cross-participant, normalization).
10. **QC flagging**: a machine-readable `qc_flag` (semicolon-joined codes, `ok` if none fired) and human-readable `qc_notes` are attached to every row. Flags include (exact strings): `missing_physio`, `short_duration` (< 60 s), `sample_rate_unavailable`, `sample_rate_unusual` (outside 10–60 Hz), `ppg_low_coverage`/`eda_low_coverage`/`temp_low_coverage`/`accel_low_coverage`/`gyro_low_coverage`/`mag_low_coverage` (< 80% finite samples), `hr_low_coverage` (< 50% device-HR samples plausible), `hr_implausible` (mean HR outside 35–220 bpm), `ppg_hr_mismatch` (PPG vs. device HR disagree by > 20 bpm), `hrv_unreliable`, `temp_implausible`/`temp_aux_implausible` (outside 20–45 °C), `motion_contaminated` (> 20% high-acceleration samples), `eda_phasic_detection_limited`, `channel_map_unconfirmed`.
11. Both a whole-task aggregate (`physio_participant_task.tsv`) and a rolling 30 s / 15 s-step window table (`physio_window_30s.tsv`) are produced from the same feature-computation function, applied to the full task span vs. each window slice respectively.

### Feature list

| Feature | Technical definition | Interpretation |
|---|---|---|
| `physio_available` | Whether an EmotiBit file was found and read for this participant-task. | Data-availability flag. |
| `duration_s` | $\max(t)-\min(t)$ over `lsl_time`. | Task/window duration covered by physio samples. |
| `coverage_pct` | $\min$ of finite-sample coverage across PPG, EDA, and temperature. | Overall usable-data fraction. |
| `hr_mean_bpm` / `hr_sd_bpm` | Mean/SD of device-reported heart rate (bpm), falling back to FFT-based PPG rate proxy when device HR is unusable. | Average heart rate and its variability over the interval. |
| `hrv_rmssd_ms` | Quality-gated RMSSD (ms) of the winning PPG channel's beat-to-beat intervals; blank when `hrv_quality_score < 0.65`. | Root-mean-square of successive heartbeat-interval differences — parasympathetic/vagal tone proxy; higher = more relaxed/recovered. |
| `hrv_sdnn_ms` | SD of clean IBIs (ms). | Overall heart-rate variability magnitude. |
| `hrv_pnn50_pct` | % of successive IBI differences exceeding 50 ms. | Another vagal-tone/HRV proxy. |
| `hrv_quality_score` | 0–1 composite of clean-IBI fraction, HR agreement, IBI regularity, and beat count. | Confidence in the HRV estimate for this row; below 0.65 the HRV columns are blanked. |
| `ppg_hr_agreement_bpm` | \|PPG-derived HR − device HR\|. | Cross-check between two independent HR estimation methods. |
| `ppg_channel_idx` | Which of the 3 PPG channels (green=0/red=1/ir=2) won the quality competition. | Diagnostic — which raw PPG wavelength produced the used HRV/HR estimate. |
| `hrv_rmssd_ms_delta_t0` | Task RMSSD − same participant's own `T0` RMSSD. | Within-participant change in vagal tone relative to their own baseline/intro task. |
| `ppg_green_mean` / `ppg_red_mean` / `ppg_ir_mean` (+ `_median`, `_std`, `_min`, `_p95`, `_max`, `_slope`, `_coverage_pct`, `_available`) | Basic distributional statistics of each raw PPG channel. | Raw photoplethysmography signal level/variability per wavelength; mostly diagnostic/QC rather than directly interpretable. |
| `eda_mean` / `eda_tonic_mean` / `eda_tonic_slope` | Mean of raw EDA; mean and per-second linear slope of the 0.05 Hz low-pass "tonic" component. | Baseline skin-conductance level and its drift — higher tonic EDA = higher sustained sympathetic arousal. |
| `eda_phasic_mean` / `eda_phasic_std` | Mean/SD of raw-minus-tonic residual. | Magnitude of moment-to-moment skin-conductance fluctuation. |
| `eda_scr_count` / `eda_scr_rate_hz` / `eda_scr_peak_rate_per_min` | Count/rate of detected phasic peaks (skin conductance responses). | Frequency of discrete sympathetic-arousal events (transient stress responses). |
| `eda_scr_amplitude_mean` / `eda_scr_amplitude_p95` | Mean / 95th-percentile peak prominence of detected SCRs. | Typical/extreme intensity of arousal spikes. |
| `temp_mean` / `temp_slope` (and `temp_skin_*`, `thermopile_*`, `temp_aux_*` variants) | Mean and per-second linear slope of each temperature-like channel. | Skin temperature level and trend — can reflect vasoconstriction/relaxation and, over a session, fatigue/thermal drift. |
| `accel_motion_mean` / `accel_motion_std` / `accel_motion_p95` / `accel_motion_max` | Mean/SD/p95/max of accelerometer vector magnitude. | Overall physical movement intensity. |
| `accel_dynamic_mean` / `_p95` | Mean/p95 deviation of acceleration magnitude from its own median. | Movement variability around a resting posture (fidgeting proxy). |
| `accel_jerk_mean` / `_p95` | Mean/p95 of \|Δ(accel magnitude)\| × sample rate. | Abruptness of movement changes. |
| `accel_high_fraction` | Fraction of samples with acceleration magnitude > 1.5 g. | Proportion of time showing likely large/motion-artefact movement; feeds the `motion_contaminated` flag. |
| `gyro_motion_mean` / `mag_motion_mean` (+ same `_dynamic`/`_jerk`/`_p95` family) | Same vector-magnitude statistics for gyroscope and magnetometer. | Rotational movement / device orientation-change intensity (secondary motion channels). |
| `qc_flag` / `qc_notes` | Semicolon-joined machine flags / human-readable descriptions. | Row-level data-quality status — always check before trusting a value. |

### Output files

| File | Granularity |
|---|---|
| `analysis/results/physio_features/physio_participant_task.tsv` | One row per session × task × participant — canonical paper-ready table. |
| `analysis/results/physio_features/physio_window_30s.tsv` | One row per session × task × participant × 30 s window (15 s step). |
| `analysis/results/physio_features/physio_qc_summary.tsv` | Participant-task QC/missingness summary. |
| `analysis/results/physio_features/physio_feature_definitions.tsv` | Two-column (`feature`, `definition`) dictionary baked into the script — the source of the plain-language definitions above. |

### Caveats

- `channel_map_unconfirmed` is set on every row until the value_* → channel mapping is explicitly confirmed via `--channel-map-confirmed`; treat PPG/EDA/temperature/IMU channel identities as provisional until that flag is cleared for the dataset.
- HRV columns are intentionally blanked (not just flagged) when quality is insufficient — a missing `hrv_rmssd_ms` is a deliberate quality gate, not a bug.
- `eda_phasic_rate_hz` / SCR features are unreliable when `scipy` is unavailable at runtime or EDA coverage is low — check `eda_phasic_detection_limited`.

---

## 2. Eye-tracking / pupillometry

**Status:** ✅ **Generated** for the simple participant-level pupil pipeline, the richer per-participant gaze/blink pipeline (raw ET files located 2026-08-12), and a group-level recovery of the rich gaze/blink features.

Three things exist for Tobii `et/*_acq-P*_tobii.tsv.gz` data:

| Script / source | Status | Output |
|---|---|---|
| [tools/features/extract_pupil_features.py](../tools/features/extract_pupil_features.py) | ✅ Generated | `analysis/results/pupil_features/{features_pupil_participant_task.tsv, features_pupil_window_30s.tsv}` |
| [tools/features/extract_eyetracking_features.py](../tools/features/extract_eyetracking_features.py) | ✅ Generated (2026-08-12) | `analysis/results/et_features_v20260812/{et_participant_task.tsv (196 rows), et_window_30s.tsv (8,248 rows), et_qc_summary.tsv (196 rows), et_feature_definitions.tsv}`. Raw `et/*_acq-P*_tobii.tsv.gz` source files were located at `C:\Users\amodica\Downloads\affectai_et_all\bids_release_no_video` (10 groups, `sub-01/ses-*_grp-NN_run01/et/`) and run against `--data-root` pointed there. |
| [analysis/extract_group_eyetracking_features.py](../analysis/extract_group_eyetracking_features.py) | ✅ Generated (2026-08-12) | `analysis/results/{group_eyetracking_features_window_30s.tsv (1,521 rows), group_eyetracking_features_task.tsv (28 rows)}` — see §2c. |

> **Important correction (2026-08-12):** the richer gaze/blink features (`blink_rate_per_min`, `gaze_dispersion`, `gaze_velocity_mean`, `pupil_slope_per_s`) were **already computed once**, at group level (mean/std across P1-P4, per 30 s window), for the ICMI paper's HMM analysis — they survive in `icmi_paper/results/enhanced_features_final.tsv` as `group_et_*` columns, covering **all 10 groups** (grp-07..grp-16, ~84-86% window coverage), not just a subset. The `tools/features/build_collective_feature_matrix.py` script that originally produced this had a stale docstring/default (`DEFAULT_GROUPS = ["grp-07","grp-10","grp-14","grp-15"]`) implying only 4 groups had manual-transcript coverage — this is **not true**; all 10 groups have manual transcripts in `transcripts/final/` and all 10 appear in the ET coverage. §2c below extracts this already-existing group-level data into a standalone table so it doesn't have to be re-derived from the paper's HMM matrix each time. What genuinely still requires raw ET file access is **per-participant** blink/gaze-dispersion/gaze-velocity (see §2b) — the paper matrix only has group-aggregated values, which cannot be disaggregated back to individuals.

### 2a. Pupil features (✅ generated, currently used for coverage stats and the `pupil_*` columns feeding `dataset_coverage_verification.ipynb`)

**Pipeline:** channels are `value_2` = left pupil diameter, `value_3` = right pupil diameter, `value_4` = gaze validity (1.0 = valid). Left/right are combined into `pupil` via a pairwise NaN-aware mean (`_pairwise_nanmean` — averages whichever eye(s) are finite per sample, not a naive `(l+r)/2`). Computed once per whole task and again per rolling 30 s / 15 s-step window.

| Feature | Technical definition | Interpretation |
|---|---|---|
| `sample_rate_hz` | $1/\text{median}(\Delta t)$ over `lsl_time`. | Estimated Tobii sampling rate. |
| `duration_s` | $\max(t)-\min(t)$. | Task/window duration covered. |
| `pupil_left_mean` / `pupil_right_mean` | Mean per-eye pupil diameter (mm). | Per-eye average dilation. |
| `pupil_mean` | Mean of the pairwise-combined left/right series. | Overall average pupil dilation — a common proxy for cognitive load/arousal. |
| `pupil_std` | SD of the combined series. | Dilation variability over the interval. |
| `pupil_slope_per_s` | OLS linear slope of combined pupil vs. time (mm/s). | Dilation/constriction trend within the window. |
| `pupil_missing_frac` | Fraction of samples where the combined pupil value is non-finite. | Pupil-signal data-loss rate. |
| `gaze_valid_frac` | Mean of the gaze-validity channel. | Fraction of samples with usable gaze tracking. |

### 2b. Richer gaze + blink features (✅ generated 2026-08-12) — `analysis/results/et_features_v20260812/{et_participant_task.tsv, et_window_30s.tsv}`

The more detailed script additionally computes, per participant-task and per window (same channel mapping, plus `value_0`/`value_1` = normalized gaze x/y):

- **Gaze position/dispersion**: `gaze_x_mean`/`gaze_y_mean`/`_std` (position and spread on the 0–1 normalized scene-camera plane), `gaze_dispersion` (mean Euclidean distance of each valid sample from the window centroid), `gaze_out_of_bounds_frac` (fraction of valid samples with x or y outside [0,1] — a calibration-drift/head-rotation proxy).
- **Gaze velocity**: `gaze_velocity_mean`/`_p95` — frame-to-frame $\sqrt{\Delta x^2+\Delta y^2}/\Delta t$ between consecutive valid samples (saccadic-activity proxy).
- **Pupil extras**: `pupil_range` (max−min), `pupil_velocity_mean` (|Δpupil|/Δt, blink-excluded), `pupil_lr_diff_mean` (mean |left−right|, an anisocoria/tracking-failure proxy), per-eye `_missing_frac`.
- **Blink detection** (`_estimate_blinks`): three regimes based on `gaze_valid_frac` — below 0.70 is treated as unreliable; 0.70–0.99 detects invalid-gaze runs of 0.05–0.40 s duration as blinks; ≥0.99 (device tracks through blinks, gaze_valid stuck at 1) falls back to NaN-runs in whichever eye's pupil signal has the best coverage (must be 70–98%) as a blink proxy. Produces `blink_count_est`, `blink_rate_per_min`, `blink_duration_mean_s`, `blink_detectable`.
- **T0 baselining** (participant-task table only): `<feature>_delta_t0` and `<feature>_z_t0` for `pupil_mean`, `pupil_left_mean`, `pupil_right_mean`, `gaze_x_mean`, `gaze_y_mean`.
- **QC flags**: `missing_et`, `short_duration` (< 30 s), `sample_rate_unavailable`/`_unusual` (outside 40–60 Hz), `gaze_low_validity` (< 70%), `pupil_low_coverage`/`pupil_left_low_coverage`/`pupil_right_low_coverage` (> 30% missing), `pupil_implausible` (mean outside 1.5–9.0 mm), `pupil_asymmetry_large` (mean |L−R| > 1.5 mm), `blink_detection_unreliable`, `blink_rate_unusual` (outside 2–45/min), `gaze_out_of_bounds` (> 10% of valid samples out of bounds), `pupil_velocity_spike` (> 10.0 mm/s).

This supersedes §2a's simple pupil script as the richer gaze/blink source at per-participant granularity — cite `analysis/results/et_features_v20260812/et_feature_definitions.tsv` (generated at runtime from the script's `FEATURE_DEFINITIONS` list) for the exact wording.

### 2c. Group-level rich gaze/blink features recovered from the paper matrix (✅ generated) — `analysis/results/group_eyetracking_features_{window_30s,task}.tsv`

**Script:** [analysis/extract_group_eyetracking_features.py](../analysis/extract_group_eyetracking_features.py). **Source:** `icmi_paper/results/enhanced_features_final.tsv` (the paper's HMM input matrix) — no raw ET files needed, since this data was already extracted and group-aggregated upstream.

Pulls the 14 `group_et_*` columns (plus `group_id`, `task_id`, `window_index`, `window_start_s`, `n_participants`) straight out of the paper matrix into a standalone table, then adds a task-level rollup (mean across windows per `group_id × task_id`, plus `n_windows`).

| Feature (exact column) | Technical definition | Interpretation |
|---|---|---|
| `group_et_pupil_mean_mean` / `_std` | Mean/SD across P1-P4 of each participant's window-mean pupil diameter. | Group-average pupil dilation and how much participants diverge (attention/arousal synchrony). |
| `group_et_pupil_std_mean` / `_std` | Mean/SD across P1-P4 of each participant's within-window pupil SD. | Typical within-participant pupil volatility, averaged over the group. |
| `group_et_pupil_slope_per_s_mean` / `_std` | Mean/SD across P1-P4 of each participant's pupil dilation/constriction trend. | Group-average and dispersion of moment-to-moment dilation trend. |
| `group_et_blink_rate_per_min_mean` / `_std` | Mean/SD across P1-P4 of estimated blink rate. | Group fatigue/disengagement level and how uniformly it's distributed across the group. |
| `group_et_gaze_dispersion_mean` / `_std` | Mean/SD across P1-P4 of mean gaze spread from each participant's own window centroid. | Whether the group is collectively scanning widely (materials, each other) vs. fixed-focus, and how much that varies across participants. |
| `group_et_gaze_valid_frac_mean` / `_std` | Mean/SD across P1-P4 of gaze-tracking validity fraction. | Group-average ET data quality for that window — check before trusting other `group_et_*` values in the same row. |
| `group_et_gaze_velocity_mean_mean` / `_std` | Mean/SD across P1-P4 of mean frame-to-frame gaze velocity. | Group-average saccadic/visual-scanning activity level. |

### Output files (§2c)

| File | Granularity |
|---|---|
| `analysis/results/group_eyetracking_features_window_30s.tsv` | One row per group × task × 30 s window (1,521 rows, 10 groups, T1-T3). |
| `analysis/results/group_eyetracking_features_task.tsv` | One row per group × task, mean across windows (28 rows). |

### Caveats (§2c)

- This table is **group-level only** (already averaged across P1-P4 upstream) and is sourced from the original paper matrix (`icmi_paper/results/enhanced_features_final.tsv`, **already 10 groups**, `grp-07`..`grp-16`, 1,521 rows — verified directly 2026-08-12; it is NOT 4 groups, see the correction below) with the pre-fix window-index alignment — it is a historical snapshot, not re-derived here. For true per-participant blink rate/gaze dispersion/gaze velocity with the corrected window alignment, use §2b's `et_window_30s.tsv` directly, or the regenerated collective matrix described in §4's "Fixed" note (`analysis/results/collective_features_v20260812/`).
- Coverage is ~84-86% per column (window rows where the upstream per-participant ET extraction succeeded for enough of P1-P4); check `n_participants`/attendance context alongside these values.
- `enhanced_features_final.tsv` (this table's source) still predates the 2026-08-12 window-index fix — do not treat it as equivalent to the regenerated `analysis/results/collective_features_v20260812/collective_window_features.tsv` (§4). It is **not** missing groups (all 10 are present), only the window-index alignment and (for transcript columns specifically) some overlap-subtype detail is different — see §4's correction.

---

## 3. Audio / prosody

**Status:** ✅ **Generated** for the participant-level aggregator (Pipeline B); 🧩 **implemented, not yet generated** for the full per-task extractor (Pipeline A).

Two independent pipelines exist and do not share a codepath:

| Pipeline | Script | Status | Output |
|---|---|---|---|
| A — full per-task DPA extractor (bleed rejection + GeMAPS + prosody + transcript merge) | [tools/features/extract_audio_features.py](../tools/features/extract_audio_features.py) | 🧩 Blocked in this workspace | Would write `audio_participant_task.tsv`, `audio_qc_summary.tsv`. **Cannot be run from this checkout**: raw DPA mic WAVs (`dpa*.wav`) are not present here (confirmed via workspace search, 2026-08-12) — must be run on the machine/drive holding the raw audio recordings. |
| B — participant-level aggregator of a pre-computed window table | [analysis/extract_participant_audio_features.py](../analysis/extract_participant_audio_features.py) | ✅ Generated | `analysis/results/participant_audio_features.tsv` |

### Pipeline A — full extractor (🧩 not yet generated, described for methodology reference)

1. **Input**: 4 DPA close-talk mic WAVs per session×task, matched via filename regex requiring `dpa[-_]mic\d+` and a `_task-T\d+_` segment. Fixed mic→participant map: `mic9`→P1, `mic10`→P2, `mic11`→P3, `mic12`→P4.
2. **Frame-level RMS**: each WAV is split into non-overlapping 25 ms frames (`FRAME_S`) and per-frame RMS computed.
3. **Bleed rejection**: a frame is attributed to a participant's mic only if that mic's RMS exceeds `SILENCE_FLOOR = 1e-5` (not silence) **and** its RMS is at least `energy_ratio` (default **1.5**, CLI `--energy-ratio`) times every other mic's RMS. Accepted runs shorter than `MIN_SPEECH_S = 0.10 s` (4 frames) are discarded as noise.
4. **Overlap/cross-talk**: among frames where ≥2 mics simultaneously exceed the silence floor, a frame counts as "overlap" if no mic beats the runner-up by the `energy_ratio` factor; `overlap_fraction` is one session×task-level value (identical across that task's 4 participant rows).
5. **Speaking time / pauses**: `speaking_time_s` = accepted-frame count × 25 ms; `speaking_fraction` = `speaking_time_s / duration_s`; `pause_count` = number of inter-segment gaps ≥ `PAUSE_THRESHOLD_S = 0.50 s` within that participant's own accepted speech.
6. **Acoustic features**: accepted frames only are concatenated into a temporary clean WAV and run through **opensmile GeMAPSv01b, Functionals level** (`opensmile.FeatureSet.GeMAPSv01b`, `opensmile.FeatureLevel.Functionals` — 62 parameters total; 16 kHz, 60 ms frame / 10 ms hop, F0 search 50–500 Hz). Only 11 of the 62 are surfaced as named output columns (below); the rest are computed internally but not exported.
7. **Transcript merge (optional)**: if `audio_annot/{task}/master_transcript.tsv` exists, `turn_count` (transitions from a different speaker onto this participant) and `uncertain_fraction` (share of this participant's segments flagged `confidence == "uncertain"`) are added.
8. **QC flagging**: `missing_audio`, `no_clean_speech`, `very_low_speaking_fraction` (0 < speaking_fraction < 0.05), `opensmile_failed`, `opensmile_not_installed`, else `ok`.

**Would-be feature list (Pipeline A):**

| Feature | Technical definition | Interpretation |
|---|---|---|
| `audio_available` | Whether this participant's mic WAV was found for the task. | Data-availability flag. |
| `duration_s` | Raw WAV length (s). | Total recording duration. |
| `speaking_time_s` / `speaking_fraction` | Accepted-frame time / duration_s. | How much of the task this participant spent speaking. |
| `pause_count` | Intra-speaker gaps ≥ 0.5 s. | Number of pauses within this participant's own speech. |
| `turn_count` | Transcript-derived speaker-change-to-this-participant count. | Turn-taking frequency (requires transcript). |
| `overlap_fraction` | Session/task-level cross-talk rate. | How much simultaneous speech occurred in that task. |
| `uncertain_fraction` | Share of transcript segments flagged uncertain. | Transcription/attribution confidence (requires transcript). |
| `energy_mean` / `energy_sd` | GeMAPS `loudness_sma3_amean` / `stddevNorm`. | Mean perceived loudness and its variability. |
| `pitch_mean` / `pitch_sd` | GeMAPS `F0semitoneFrom27.5Hz_sma3nz_amean` / `stddevNorm`. | Mean F0 (pitch) and prosodic pitch variability. |
| `hnr_mean` | GeMAPS `HNRdBACF_sma3nz_amean`. | Harmonics-to-noise ratio — voice clarity/quality. |
| `jitter_mean` / `shimmer_mean` | GeMAPS `jitterLocal_sma3nz_amean` / `shimmerLocaldB_sma3nz_amean`. | Cycle-to-cycle F0/amplitude perturbation — voice-quality/strain proxies. |
| `voiced_segments_per_sec` | GeMAPS `VoicedSegmentsPerSec`. | Speech-rate proxy. |
| `mean_voiced_segment_s` / `mean_unvoiced_segment_s` | GeMAPS `MeanVoicedSegmentLengthSec` / `MeanUnvoicedSegmentLength`. | Typical voiced-speech and unvoiced-gap durations. |
| `speech_rate_proxy` | GeMAPS `loudnessPeaksPerSec`. | Rate of loudness peaks — articulation-rate proxy. |
| `qc_flag` | See flagging list above. | Row-level data-quality status. |

### Pipeline B — participant-level aggregator (✅ generated)

Reads a pre-existing **window-level** GeMAPS table (`icmi_paper/results/_audio_window_partial.tsv`, produced separately by a notebook not audited here), renames 4 `gemap_*` columns, and averages across windows per `(group_id, task_id, participant_id)`.

| Feature (exact column) | Technical definition | Interpretation |
|---|---|---|
| `audio_energy_mean` | Mean of `gemap_loudness_sma3_amean` across windows. | Mean loudness for that participant/task. |
| `audio_pitch_mean` | Mean of `gemap_F0semitoneFrom27.5Hz_sma3nz_amean` across windows. | Mean pitch (F0, semitone scale). |
| `audio_pitch_sd` | Mean of `gemap_F0semitoneFrom27.5Hz_sma3nz_stddevNorm` across windows. | Mean of window-level pitch variability. |
| `audio_hnr_mean` | Mean of `gemap_HNRdBACF_sma3nz_amean` across windows. | Mean harmonics-to-noise ratio. |
| `n_windows` | Count of source window-rows aggregated. | Coverage/reliability indicator for the participant-task mean. |

### Output files

| File | Producer | Status |
|---|---|---|
| `features/audio_participant_task.tsv`, `features/audio_qc_summary.tsv` | Pipeline A | 🧩 not present |
| `analysis/results/participant_audio_features.tsv` | Pipeline B | ✅ present |
| `icmi_paper/results/_audio_window_partial.tsv` | Upstream window-level GeMAPS table consumed by Pipeline B | present, source not audited here |

### Caveats

- Bleed rejection is **energy-only, frame-level (25 ms)** — no spectral/content check; per [docs/audio_annotation_pipeline.md](audio_annotation_pipeline.md), the same logic upstream is described as conservative and known to silently drop backchannels (a second-pass script, `tools/audio_refine_backchannels.py`, recovers these into a separate file that Pipeline A does not currently read).
- `overlap_fraction` is session/task-level, not per-participant (same value written into all 4 rows for a task).
- opensmile is a soft dependency — if not installed, all acoustic columns are blank and a warning is logged.
- Mic→participant mapping is a hardcoded assumption (`mic9`=P1 … `mic12`=P4) with no runtime cross-check against `participant_map.tsv`.
- GeMAPSv01b's F0 search range is opensmile's default 50–500 Hz (not the 55–1000 Hz sometimes assumed in earlier planning docs); changing it requires a custom SMILExtract config.

---

## 4. Transcript / turn-taking

**Status:** ✅ **Generated** — both the participant-level extractor and the richer participant+group extractor have been run.

| Script | Status | Output |
|---|---|---|
| [analysis/extract_participant_transcript_features.py](../analysis/extract_participant_transcript_features.py) | ✅ Generated | `analysis/results/participant_transcript_features.tsv` |
| [tools/features/extract_transcript_features.py](../tools/features/extract_transcript_features.py) | ✅ Generated (2026-08-12) | `analysis/results/transcript_features_full/{transcript_participant_task.tsv, transcript_group_task.tsv, transcript_feature_definitions.tsv}` |

### Annotation scheme (shared input, per [labels_codebook.md](labels_codebook.md))

Manually annotated transcript rows use a `type` vocabulary:

| Code | Represents | Attribution |
|---|---|---|
| `SPK` | Speech segment | `P1`–`P4` or `MODERATOR` |
| `BCK` | Backchannel (e.g. "mm-hmm") | Individual participant |
| `OVL` | Overlap (2+ simultaneous speakers); `subtype` encodes function | `GROUP` |
| `SIL` | Silence/pause | `GROUP` |
| `LAU` | Laughter | Individual or `GROUP` |
| `FP` | Filled pause ("um", "uh") | Individual |
| `BRE` | Audible breath | Individual |

`OVL` **subtypes**, classified by a timing+lexical-cue decision tree: `simultaneous` (start offset < 200 ms), `smooth` (< 300 ms, no cue), `backchannel_ovl` (short + support cue), `collaborative` (support cue, no conflict cue) — all **cooperative**; `competitive` (> 500 ms + conflict cue), `floor_fight` (> 1000 ms sustained + conflict cue) — **conflict**; `needs_review` (ambiguous, should be treated as missing unless manually reviewed). Enrichment columns when present: `gap_prev_ms` (pause before this segment), `trp_silence_ms` (silence at a transition-relevance place before speaking), `interruption_gap_ms` (negative = spoke over someone), `bclatency_ms` (backchannel reaction latency), `overlap_ms` (ms of a `SPK` segment overlapped by another speaker).

### Participant-level features (✅ generated) — `analysis/results/participant_transcript_features.tsv`

Explicitly scoped to metrics "legitimately attributable to an individual speaker" (group-level metrics live elsewhere, e.g. `icmi_paper/results/hmm_input_features_final.tsv`). Always emits all 4 participants per file (zero-filled if a participant has no rows).

| Feature (exact column) | Technical definition | Interpretation |
|---|---|---|
| `tr_speaking_time_s` | Sum of this participant's `SPK` durations. | Total speaking time. |
| `tr_n_turns` | Count of this participant's `SPK` rows. | Number of speaking turns. |
| `tr_backchannel_count` | Count of `BCK` rows. | Supportive interjections given. |
| `tr_laughter_count` | Count of `LAU` rows attributed to this participant. | Laughter frequency. |
| `tr_overlap_count` / `tr_overlap_time_s` | Count / summed duration of this participant's `OVL` rows. | Overlap involvement (count and total duration). |
| `tr_overlap_competitive` | Count of `OVL` rows with `subtype == "competitive"`. | Floor-contest overlaps. |
| `tr_overlap_floor_fight` | Count with `subtype == "floor_fight"`. | Sustained floor-fight overlaps. |
| `tr_overlap_smooth` | Count with `subtype == "smooth"`. | Smooth-transition overlaps. |
| `tr_overlap_backchannel_ovl` | Count with `subtype == "backchannel_ovl"`. | Backchannel-caused overlaps. |

Note: `collaborative`, `simultaneous`, and `needs_review` subtypes are **not** separately tallied in this table (only in the richer 🧩 pipeline below).

### Richer participant + group features (✅ generated) — `analysis/results/transcript_features_full/`

Participant-level (`transcript_participant_task.tsv`) adds, beyond the above: `speaking_share` (speaking_time / task_duration), `mean_turn_duration_s`/`median_turn_duration_s`/`std_turn_duration_s`, `mean_response_gap_ms` (mean `gap_prev_ms` > 0), `mean_trp_silence_ms`, `overlap_fraction` (overlap_time/speaking_time), `interruption_count`/`interruption_rate_per_min` (from `interruption_gap_ms < 0`), `backchannel_given_rate_per_min`, `backchannel_latency_mean_ms`, `filled_pause_count`/`_rate_per_min`, `breath_count`.

Group-level (`transcript_group_task.tsv`): `task_duration_s`, `n_active_speakers`, `total_turns`, `sil_time_s`/`sil_fraction`/`sil_event_count`, `laughter_count`/`laughter_time_s`, `overlap_event_count`, `competitive_overlap_count`, `cooperative_overlap_count`, `collaborative_overlap_count`, `needs_review_overlap_count`, `conflict_overlap_ratio` (competitive / (competitive+cooperative)), `simultaneous_overlap_count`, `backchannel_ovl_count`, `transition_entropy_bits` (Shannon entropy in bits of the inter-speaker turn-transition distribution — higher = floor exchanged more evenly), `gini_speaking_time` (Gini coefficient of total `SPK` duration across the 4 participants — 0 = equal, 1 = one dominates), `engagement_density_per_min` (non-`SIL` rows per minute).

### Output files

| File | Producer | Status |
|---|---|---|
| `analysis/results/participant_transcript_features.tsv` | `analysis/extract_participant_transcript_features.py` | ✅ present |
| `analysis/results/transcript_features_full/transcript_participant_task.tsv` (148 rows), `transcript_group_task.tsv` (37 rows), `transcript_feature_definitions.tsv` | `tools/features/extract_transcript_features.py` | ✅ present (generated 2026-08-12) |

### Caveats

- **Discovery regex mismatch (fixed 2026-08-12)**: `tools/features/extract_transcript_features.py` originally required a dash in `grp-NN` (`transcript_grp-\d+_...`), which silently skipped the no-dash `_reviewed` filenames used for groups 08/09/11/12/13 in `transcripts/final/`. The glob and regex were widened to accept an optional dash (matching `analysis/extract_participant_transcript_features.py`'s more permissive pattern) and a `_normalize_group_id` step was added so `grp8`/`grp-08` both key to `grp-08`. All 37/37 available transcript files are now discovered and processed.
- Transcript scope is `T1`–`T4` only, 40-file universe (10 groups × 4 tasks); 37/40 exist (`grp-09` missing `T1`; `grp-13` missing `T3`/`T4`) — see [dataset_description.md](dataset_description.md) §5/§7 for the full coverage discussion and why the "window-level coverage vs. device recording" figure is not a completeness gap.
- `OVL` subtype classification is heuristic (timing + lexical cue); `needs_review` rows should be treated as missing data unless manually reviewed.
- `grp-10` has inflated `T0`/`T1` durations from the stimuli app running overnight before the session — rate-per-minute transcript features for that session should be treated with caution.

### ✅ Fixed 2026-08-12: cross-modal window-index misalignment (was: known limitation, documented earlier the same day)

Transcript features intentionally cover only the **actual dialogue** span within a task (per [dataset_description.md](dataset_description.md) §5/§7), while physio/pupil/ET window features cover the **full recorded task span** (`tobii_calibration`→`finish` for `T1`–`T4`, per [bids-conventions.instructions.md](../.github/instructions/bids-conventions.instructions.md)), which includes calibration, briefing, and settling-in time *before* the conversation starts. This is fine at task-level granularity (durations/rates from different pipelines are never mixed), but it was a real bug at **window-level** granularity:

- `tools/features/common.py`'s `rolling_windows()` (used for physio/pupil/ET) sets `window_index = 0` at the task file's own `lsl_time.min()` — i.e. the task-start marker, including overhead.
- `tools/features/build_collective_feature_matrix.py`'s `_bin_transcript_events()` **used to** set `window_index = 0` at `grp_df["onset"].min()` — i.e. the first transcribed utterance. Since each transcript is itself sourced from the same task-split WAV as physio/ET (per [audio_annotation_pipeline.md](audio_annotation_pipeline.md) — "recordings are split by task... producing 20 WAV files per session"), `onset == 0` in the transcript is the same task-split-recording start as physio/ET's `window_index == 0`, so rebasing to the first utterance instead of `onset == 0` was the bug: it silently shifted every window_index by however long the pre-conversation overhead lasted.
- **Fix applied**: `_bin_transcript_events()` now bins on the raw `onset` (task-split-recording-relative, unshifted) — `window_index = floor(onset / 30)` — matching the physio/ET anchor. No other pipeline needed changing.
- **`transcript_coverage` + jitter tolerance added**: because the transcribed span's boundary is itself a manually-judged (and sometimes truncated-recording-driven) cut, a window with zero transcript rows is only distinguished from "unknown/out of scope" if it falls within `[dialogue_onset_s − jitter_s, dialogue_offset_s + jitter_s]` (`jitter_s` defaults to **90 s** ≈ 2–3 window steps, tunable via `--transcript-jitter-s`, per user note 2026-08-12 that some recordings were themselves cut off). Windows inside that jittered span with no events are emitted as real, zero-filled, `transcript_coverage="covered"` rows (confirmed silence); windows outside it are omitted entirely, so after the outer-merge in `run()` they read as NaN (unknown), never as a silent 0. Every emitted row also carries `dialogue_onset_s`/`dialogue_offset_s` (first/last transcribed event for that group×task) for transparency.
- **Second bug found + fixed in the same script, same day**: `_load_all_transcripts()`'s file discovery had the identical dash-required regex bug already fixed elsewhere in `extract_transcript_features.py` (see §4's other caveat above) — it silently skipped the no-dash `_reviewed` transcripts for `grp-08/09/11/12/13`. Widened to `transcript_grp*.tsv` + optional-dash regex + `_normalize_group_id`, matching the other script. Also corrected the stale `DEFAULT_GROUPS` (was 4 groups) to all 10 (`grp-07`..`grp-16`) and the module docstring/CLI help text repeating that stale claim.
- **Regenerated (2026-08-12)**, now that raw ET files were located (`C:\Users\amodica\Downloads\affectai_et_all\bids_release_no_video`) and run through `extract_eyetracking_features.py` to produce a real `et_window_30s.tsv` (see §2b): `analysis/results/collective_features_v20260812/{collective_window_features.tsv (1,580 rows, 47 cols), collective_task_selfreport.tsv (30 rows)}`, covering all 10 groups with 717 window rows carrying real `transcript_coverage="covered"` data (up from 355 before the discovery-regex fix). This is a **new standalone file** — the originally-published `icmi_paper/results/enhanced_features_final.tsv` was **not modified**; if the paper pipeline should adopt the corrected version, that swap is a separate, deliberate decision, not made here.
- **CORRECTION (2026-08-12, later): `enhanced_features_final.tsv` was NOT "4 groups"** — verified directly by reading the file: it already has all 10 groups (`grp-07`..`grp-16`, 1,521 rows). The "4 groups" belief (repeated in an earlier version of this doc and in the script's own stale `DEFAULT_GROUPS`) is **not evidence of what actually generated that file** — its generating code could not be located in this repo (per [dataset_description.md](dataset_description.md)). Do not repeat the "4 groups" claim elsewhere.
- **Demonstrated impact of the window-index fix, since transcript-derived features are load-bearing for the HMM analysis**: direct comparison for `grp-07`/`T1`, transcript-covered `window_index` values —
  - OLD (`enhanced_features_final.tsv`): `[0, 1, 2, ..., 8]` (anchored to the first transcribed utterance — the bug)
  - NEW (`collective_features_v20260812/collective_window_features.tsv`): `[2, 3, ..., 17]` (anchored to task-recording start, matching physio/ET — the fix)

  This is a genuine ~2-window (60 s) shift for this group/task alone: physio/ET features that were joined to "window 0" in the published matrix were actually paired with transcript data from a different real-world time slice. Any window-level cross-modal analysis (e.g. the HMM state discovery in `icmi_paper/analysis/hmm_collective_states_updated_overlaps.ipynb`) that joins transcript columns to physio/ET columns by `window_index` from the old file was trained on mis-joined feature vectors. The per-group/task shift size has not been checked beyond `grp-07`/T1 — do not assume it's uniform across groups.
- **✅ RESTORED (2026-08-12, later still): all `tr_ovl_*` columns ported into `build_collective_feature_matrix.py`.** The 12 columns that were missing (`tr_needs_review_overlap`, `tr_ovl_count`, `tr_ovl_time_s`, `tr_ovl_simultaneous`, `tr_ovl_backchannel`, `tr_ovl_smooth`, `tr_ovl_competitive_timing`, `tr_ovl_floor_fight`, `tr_ovl_ctx_collaborative`, `tr_ovl_ctx_completion`, `tr_ovl_collaboration_index`, `tr_ovl_completion_index`) are now computed directly in `_bin_transcript_events()`, ported from `icmi_paper/analysis/timing_based_overlap_features.ipynb`'s pure-timing-subtype + `context_label` logic (including its same-onset multi-speaker `_dedup_ovl` step) — but anchored to the already-fixed task-recording-start window index, **not** that notebook's own `onset.min()` rebase (which has the identical misalignment bug this script's window-index fix addressed; the notebook's version was never corrected). `analysis/results/collective_features_v20260812/collective_window_features.tsv` now has all 59 columns with **zero** `tr_*` column difference vs. `enhanced_features_final.tsv` (verified: `old_tr_cols - new_tr_cols == set()` and vice versa).
- **Fourth bug found + fixed in the same pass: `tr_collaborative_overlap` was silently always zero.** The code checked `subtype == "collaborative"`, but `"collaborative"` is a `context_label` value, never a `subtype` value (`subtype` is always one of the five pure-timing labels per `tools/relabel_overlaps.py`'s `VALID_SUBTYPES`: `simultaneous`, `backchannel_ovl`, `smooth`, `competitive`, `floor_fight`) — so that condition could never match. Confirmed before the fix: regenerated matrix had `tr_collaborative_overlap` summing to 0 across all 1,580 rows, vs. 151 in the original published file. Now sourced from `context_label` instead; post-fix sum is 189 (same order of magnitude as the original 151 — the residual difference is expected, from the window-alignment/coverage changes already described above, not a remaining bug). `tr_cooperative_overlap` (which folds in `collaborative`-context overlaps) shifted accordingly (378→508).
- **Current status: `analysis/results/collective_features_v20260812/collective_window_features.tsv` now has full column parity with `enhanced_features_final.tsv`, plus the window-alignment fix, plus the collaborative-overlap bugfix, plus all 10 groups.** It is still a **new standalone file** — the originally-published `icmi_paper/results/enhanced_features_final.tsv` has **not** been modified. Swapping any downstream paper/thesis notebook to read from the new file is a separate, deliberate decision (out of scope here) — but there is no longer a "missing features" blocker to that swap.

### 🔍 Full audit (2026-08-12): every place overlaps are labeled/classified in this repo

There are **two eras** and **four separate implementations** of overlap classification in this codebase. Confusing one for another is the root cause of every `tr_collaborative_overlap`/`tr_ovl_*`-family bug found this session. Full picture, cross-checked against the actual current transcript files:

**1. Canonical current scheme — `tools/relabel_overlaps.py` (two independent columns, both on every OVL row):**
- `subtype` (timing, objective, always present): `simultaneous` (start-diff < 200 ms) / `backchannel_ovl` (overlap < 250 ms) / `smooth` (250–500 ms) / `competitive` (500–1000 ms) / `floor_fight` (≥ 1000 ms). `classify_timing()` **always** returns one of these 5 values — never `collaborative`, never `needs_review`. `VALID_SUBTYPES` in the same file enforces this set.
- `context_label` (lexical, interpretive, may be empty): `collaborative` (support-word cue, any timing subtype) / `completion` (prior speaker trails off + this speaker continues with content) / `competitive` (conflict-word cue, only on `competitive`/`floor_fight` timing) / `""` (no clear signal). See `classify_context()`, `_is_completion_context()`.
- This is the scheme documented in [icmi_paper/ovl_annotation_methodology.md](../icmi_paper/ovl_annotation_methodology.md) as "Phase 2 — July 2026 reclassification," and it is what `tools/features/build_collective_feature_matrix.py` now correctly reads (after today's fix).

**2. Legacy scheme #1 — `tools/transcript_corrector.html`'s `classifyOVLSubtype()` (browser-based manual correction tool, JS):** a single **7-value mixed timing+lexical vocabulary** written directly to `subtype`: `simultaneous`, `backchannel_ovl`, `smooth`, `floor_fight`, `competitive`, **`collaborative`**, **`needs_review`**. This was the original annotation-time scheme used by the human annotator for groups 07/10/14/15/16 (per `ovl_annotation_methodology.md` "Phase 1"). It writes `collaborative`/`needs_review` as literal `subtype` values — values that Scheme 1 has since made impossible in `subtype` (they moved to `context_label`, or in the case of `needs_review`, got auto-resolved away by duration). **Any file annotated before the July 2026 `relabel_overlaps.py` pass would have had these legacy values in `subtype`; the pass overwrites them all.**

**3. Legacy scheme #2 — `tools/annotation_gui/app.py`'s live real-time annotation dropdown:** offers only **4** subtype choices when logging an OVL event live during a session (`_on_transcript_event_category_changed()`): `competitive`, `floor_fight`, `simultaneous`, `backchannel_ovl` — no `smooth`, no `collaborative`, no `needs_review`. This is a distinct, narrower vocabulary from both of the above, used only at capture time (not post-hoc correction).

**4. Empirically confirmed current state (checked directly against `transcripts/final/transcript_grp*.tsv`, 2026-08-12):**
- `subtype` values across **all** current OVL rows: `floor_fight` 281, `competitive` 204, `backchannel_ovl` 174, `smooth` 157, `simultaneous` 124 — **zero** rows with `collaborative` or `needs_review` in `subtype`. Confirms the July 2026 `relabel_overlaps.py` pass has been fully applied to every current file; no legacy-scheme values survive.
- `context_label` column is present in **35/37** files; entirely missing from 2 (`transcript_grp-07_T4_2026-06-17.tsv`, `transcript_grp-15_T4_2026-06-17.tsv`) — these 2 files' OVL rows have no lexical layer at all (not just empty values — the column itself is absent, so any code that assumes the column exists must guard for it, as `build_collective_feature_matrix.py` and `relabel_overlaps.py` both do).

**5. ⚠️ NEW finding, not yet fixed — `tools/features/extract_transcript_features.py` has the same subtype/context_label mix-up bug found earlier today in `build_collective_feature_matrix.py`, still unpatched:** its `OVL_COLLABORATIVE = frozenset({"collaborative"})` and `OVL_NEEDS_REVIEW = frozenset({"needs_review"})` filter the `subtype` column (`sub.isin(OVL_COLLABORATIVE)` / `sub.isin(OVL_NEEDS_REVIEW)`, lines ~337–349) — but per finding #4 above, `subtype` can **never** hold those values in any current file (they only ever appear, if at all, in `context_label` or not at all). So in `transcript_group_task.tsv`'s output (`analysis/results/transcript_features_full/transcript_group_task.tsv`), **`collaborative_overlap_count` and `needs_review_overlap_count` are silently 0 for every row**, and `cooperative_overlap_count`/`conflict_overlap_ratio` (which fold `OVL_COOPERATIVE = {"collaborative", "smooth", "simultaneous", "backchannel_ovl"}` into their denominator) are systematically undercounted for the same reason. This file's participant-level sibling (`transcript_participant_task.tsv`) does **not** appear to carry an equivalent collaborative-count column (only the group-level table does — see the caveat table above), but `tools/features/export_latent_state_outputs.py` does consume `transcript_participant_task.tsv`, so it's worth checking that pipeline isn't relying on the group-level collaborative counts elsewhere. **This bug has not been fixed — flagging only, since fixing it changes a different output file (`transcript_group_task.tsv`) than the one this session's fixes targeted; confirm before patching.**
- Note also that [icmi_paper/ovl_annotation_methodology.md](../icmi_paper/ovl_annotation_methodology.md) itself concatenates documentation for **both** the legacy 7-value scheme and the current two-layer scheme (it has two separate "Feature Pipeline Implications" sections and two separate "Paper Methods Note" sections, one per era) without a clear "superseded" banner between them — easy to misread the legacy `tr_competitive_overlap`/`tr_collaborative_overlap` definitions there as if they were the current implementation. The doc itself flags `tr_collaborative_overlap` as "superseded by `tr_ovl_collaboration_index`" in its legacy section, which is consistent with what this document recommends.
- **T4 is out of analysis scope (confirmed 2026-08-12)** — the 2 files missing `context_label` entirely (`grp-07_T4`, `grp-15_T4`) are therefore not a real gap; T1–T3 have `context_label` on every file.

### 🔬 In progress (2026-08-12): validating the `context_label` heuristic against human judgment

`subtype` (timing) needs no validation — it's a deterministic function of measured duration. `context_label`
(collaborative/completion/competitive) is a single deterministic lexical-cue regex pass with **no** prior
inter-rater or ground-truth validation, which is a real methodological gap for using it as a scientific
feature (flagged by the user 2026-08-12: "as they are they can't exist... we need more justification").

**Plan agreed with user:**
1. Human-annotate a stratified validation subset (~100–150 OVL events, T1–T3 only, all 10 groups) as ground
   truth, blind to the current automated label.
2. If validation shows the heuristic is weak, replace `context_label` with a multi-model ensemble/majority-vote
   label (exact ensemble composition — cloud API keys vs. local open-weight models via `transformers` — still
   TBD; user has no personal LLM API subscriptions, an unconfirmed company ChatGPT (may or may not include a
   scriptable API, pending IT), and unconfirmed but empirically-verified-working `pip install` on their machine).
3. Re-run the relabeling across all T1–T3 data, all 10 groups, once an approach is validated.

**Tooling built so far (2026-08-12):**
- [tools/sample_overlap_context_validation.py](../tools/sample_overlap_context_validation.py) — draws the
  stratified sample (quota-based: 40 collaborative / 25 completion / 30 competitive / 30 empty by default,
  reduced to what's available for rarer classes) from `transcripts/final/`, T1–T3 only, with ±2s of
  surrounding SPK/BCK text per event. Outputs `context_label_validation_BLIND.tsv` (for the annotator —
  `human_context_label` defaults to the sentinel `"PENDING"`, must be overwritten with `collaborative` /
  `completion` / `competitive` / the literal word `empty`) and `context_label_validation_ANSWER_KEY.tsv`
  (holds the automated label; not to be viewed while annotating, to avoid anchoring bias).
- Already run once: `analysis/results/context_label_validation/{context_label_validation_BLIND.tsv,
  context_label_validation_ANSWER_KEY.tsv}` (125 rows: 40 collaborative / 25 completion / 30 competitive / 30
  empty, seed 42 for reproducibility). Pool was 769 T1–T3 OVL events across all 10 groups (478 empty / 189
  collaborative / 76 competitive-context / 26 completion) — matches the `tr_ovl_ctx_*` sums confirmed earlier
  today, cross-validating both scripts against each other.
- [tools/score_overlap_context_validation.py](../tools/score_overlap_context_validation.py) — once
  `human_context_label` is filled in, computes raw agreement %, Cohen's kappa, per-class precision/recall/F1,
  and a confusion matrix (auto vs. human). Smoke-tested against a synthetic 100%-agreement copy — works
  correctly (95/95, kappa 1.000, per-class P/R/F1 1.000 for the 3 non-empty classes tested).
- **Not yet done**: the actual manual annotation (the 125-row BLIND file is generated but unfilled), and the
  multi-model ensemble infrastructure decision (blocked on IT / provider access, per user 2026-08-12).

---

## 4b. Lexical / text content features

**Status:** ✅ Generated (2026-08-12). **Script:** [analysis/extract_participant_lexical_features.py](../analysis/extract_participant_lexical_features.py). **Output:** `analysis/results/participant_lexical_features.tsv` (148 rows, 32 columns).

> **Full documentation:** See [lexical_features.md](lexical_features.md) for detailed methodology, word lists, and design rationale.

Complements §4 (turn-taking/timing) by analyzing **what** participants say, not just when/how long. Extracts text-based features from the `text` column of manually-annotated SPK rows in `transcripts/final/*.tsv`.

### Feature list

| Feature | Technical definition | Interpretation |
|---|---|---|
| **Basic metrics** | | |
| `lex_word_count` | Total word count (unigrams) across all SPK rows for this participant-task. | Speaking volume (content quantity). |
| `lex_unique_words` | Count of distinct word types (lowercased). | Vocabulary breadth. |
| `lex_ttr` | Type-token ratio: `unique_words / word_count`. | Vocabulary richness/diversity — higher = more varied language. |
| `lex_mean_word_len` | Mean character length per word. | Lexical complexity proxy. |
| **N-gram counts** | | |
| `lex_n_unigrams` | Total unigram count (= `lex_word_count`). | Token count at word level. |
| `lex_n_bigrams` | Total bigram count (`n_words - 1`). | Two-word phrase count. |
| `lex_n_trigrams` | Total trigram count (`n_words - 2`). | Three-word phrase count. |
| `lex_unique_bigrams` | Count of distinct bigram types. | Two-word phrase vocabulary size. |
| `lex_unique_trigrams` | Count of distinct trigram types. | Three-word phrase vocabulary size. |
| **N-gram diversity** | | |
| `lex_unigram_diversity` | `unique_unigrams / n_unigrams` (= `lex_ttr`). | Word-level vocabulary diversity. |
| `lex_bigram_diversity` | `unique_bigrams / n_bigrams`. | Bigram repetition rate — lower = more formulaic/repetitive phrase patterns. |
| `lex_trigram_diversity` | `unique_trigrams / n_trigrams`. | Trigram repetition rate — closer to 1.0 = nearly all trigrams are unique (creative/varied phrasing). |
| **N-gram entropy (bits)** | | |
| `lex_unigram_entropy` | Shannon entropy of unigram distribution: $-\sum p_i \log_2 p_i$. | Linguistic unpredictability at word level — higher = more diverse word choice. |
| `lex_bigram_entropy` | Shannon entropy of bigram distribution. | Phrase-level unpredictability — higher = less predictable word sequences. |
| `lex_trigram_entropy` | Shannon entropy of trigram distribution. | Higher-order linguistic complexity — higher = more varied phrase structures. |
| **Hapax legomena** | | |
| `lex_hapax_count` | Count of words appearing exactly once. | Vocabulary richness indicator — rare/unique word usage. |
| `lex_hapax_ratio` | `hapax_count / word_count`. | Proportion of once-occurring words — higher = more diverse vocabulary with less repetition. |
| **Marker counts** | | |
| `lex_agreement_count` | Count of agreement markers: "yes", "yeah", "right", "exactly", "agree", etc. (24 terms). | Supportive/collaborative speech acts. |
| `lex_hedging_count` | Count of hedging markers: "maybe", "perhaps", "might", "kind of", "think", "guess", etc. (26 terms). | Tentativeness, face-saving, uncertainty in assertions. |
| `lex_certainty_count` | Count of certainty markers: "definitely", "certainly", "clearly", "always", "know", etc. (18 terms). | Confidence/conviction in assertions. |
| `lex_question_count` | Count of question marks or question-word-initiated sentences. | Inquiry frequency — information-seeking, engagement. |
| `lex_suggestion_count` | Count of suggestion phrases: "we could", "what if", "let's", "how about", "I suggest", etc. | Idea generation, proposal-making. |
| `lex_positive_count` | Count of positive valence words: "good", "great", "helpful", "agree", "success", etc. (28 terms). | Positive sentiment expression. |
| `lex_negative_count` | Count of negative valence words: "bad", "problem", "difficult", "disagree", "fail", etc. (27 terms). | Negative sentiment expression. |
| **Derived rates** | | |
| `lex_agreement_rate` | `agreement_count / word_count`. | Normalized agreement density. |
| `lex_question_rate_per_min` | `question_count / speaking_time_min`. | Question frequency normalized by speaking time. |
| `lex_suggestion_rate_per_min` | `suggestion_count / speaking_time_min`. | Proposal frequency normalized by speaking time. |
| `lex_sentiment_ratio` | `(positive_count - negative_count) / (positive_count + negative_count)` or 0 if both zero. | Sentiment polarity: +1 = all positive, −1 = all negative, 0 = balanced or absent. |
| `lex_speaking_time_s` | Sum of SPK row durations (same as `tr_speaking_time_s`). | Included for normalization cross-check. |

### Key statistics (N=148 participant-tasks)

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| `lex_word_count` | 231 | 137 | 9 | 820 |
| `lex_bigram_diversity` | 0.90 | 0.05 | 0.71 | 1.00 |
| `lex_trigram_diversity` | 0.97 | 0.03 | 0.79 | 1.00 |
| `lex_unigram_entropy` | 6.2 bits | 0.7 | 2.9 | 7.3 |
| `lex_bigram_entropy` | 7.3 bits | 1.1 | 3.0 | 9.2 |
| `lex_trigram_entropy` | 7.4 bits | 1.2 | 2.8 | 9.6 |
| `lex_hapax_ratio` | 0.36 | 0.14 | 0.18 | 1.00 |

### Lexical marker word lists (embedded in script)

- **Agreement** (24 terms): yes, yeah, yep, yup, right, exactly, absolutely, definitely, agree, agreed, true, correct, sure, okay, ok, good, great, perfect, fine, indeed, precisely, totally, certainly
- **Hedging** (26 terms): maybe, perhaps, possibly, probably, might, could, would, kind, sort, somewhat, fairly, rather, quite, basically, actually, just, like, guess, think, believe, suppose, seem, seems, apparently, presumably
- **Certainty** (18 terms): definitely, certainly, absolutely, clearly, obviously, surely, undoubtedly, always, never, must, know, confident, certain, positive, convinced, guarantee, fact, proven
- **Suggestion patterns** (11 regex patterns): "we could", "we should", "what if", "how about", "maybe we", "why don't we", "let's", "I suggest", "I propose", "I think we", "what do you think"
- **Positive valence** (28 terms): good, great, excellent, nice, wonderful, fantastic, amazing, awesome, brilliant, perfect, love, like, enjoy, happy, pleased, glad, excited, interesting, helpful, useful, agree, best, better, benefit, success, successful, effective, efficient
- **Negative valence** (27 terms): bad, poor, terrible, awful, horrible, wrong, problem, issue, difficult, hard, hate, dislike, annoying, frustrating, confused, worried, concerned, doubt, fail, failure, worse, worst, disagree, unfortunately, impossible, never, can't, won't

### Caveats

- **Lemmatization enabled** (NLTK WordNet): Words are reduced to base forms ("agreeing" → "agree", "problems" → "problem") for improved marker matching.
- Word lists are English-only and domain-general. For specific tasks (negotiation, brainstorming), domain-specific markers could be added.
- Sentiment lexicon is a simplified subset, not a validated instrument like LIWC or VADER. Useful for relative comparisons within-dataset, not absolute scores.
- Transcripts are already filtered to `type == "SPK"` only — backchannels, laughter, etc. are excluded from text analysis.
- N-gram entropy is sensitive to text length: very short utterances have lower entropy by construction. Normalize or compare within similar word-count ranges.

---

## 5. Group dynamics / cross-participant synchrony

**Status:** ✅ Generated (2026-08-12). **Script:** [tools/features/compute_group_dynamics.py](../tools/features/compute_group_dynamics.py). **Outputs:** `analysis/results/cross_modal/{features_group_dynamics_window_30s.tsv (2,430 rows), features_group_dynamics_task.tsv (745 rows)}`.

> Note: this script requires `features_physio_window_30s.tsv` and `features_pupil_window_30s.tsv` in the **same** `--features-dir`, but those tables live in separate directories (`analysis/results/physio_features/` and `analysis/results/pupil_features/`). They were copied into a shared staging directory, `analysis/results/cross_modal/`, before running; that directory now holds both the staged inputs and this pipeline's outputs.

### Pipeline

- **Inputs**: the already-produced physio and pupil window tables (`physio_window_30s.tsv`, `features_pupil_window_30s.tsv`), outer-joined on `session_id, task, participant_id, window_index, window_start_lsl, window_end_lsl`.
- **Signals compared**: whichever of `eda_mean`, `ppg_rate_proxy_bpm` (physio), and `pupil_mean` (pupil) are present — same-metric, cross-participant synchrony only (no cross-modal comparison, e.g. EDA-vs-pupil).
- **Group-level aggregation**: per `(session_id, task, window_index)`, `n_participants` (attendance) plus NaN-safe `{metric}_group_mean`/`_group_std` across all participants present.
- **Dyadic synchrony**: for every unique participant pair in a session×task (all $\binom{n}{2}$ dyads via `itertools.combinations`), per metric:
  - `corr`: zero-lag Pearson correlation of the two participants' per-window time series (requires ≥ 3 finite paired samples and non-zero SD in both).
  - `best_lag_corr` / `best_lag_windows`: searches lags from −2 to +2 window-steps (±30 s at the upstream 15 s step), shifting one series relative to the other, and keeps the lag with the highest Pearson `r` (sign of `best_lag_windows` indicates offset direction — a leader/follower relationship).

### Feature list

| Feature | Technical definition | Interpretation |
|---|---|---|
| `n_participants` | Unique participant count in the window. | Group attendance/completeness for that window. |
| `{metric}_group_mean` | NaN-safe mean of `eda_mean`/`ppg_rate_proxy_bpm`/`pupil_mean` across participants in the window. | Group-average arousal/dilation level at that moment. |
| `{metric}_group_std` | NaN-safe SD across participants. | Cross-participant dispersion — lower = more physiologically synchronized group. |
| `corr` | Zero-lag Pearson correlation between a dyad's per-window metric series. | Concurrent physiological/pupillary synchrony between two participants. |
| `best_lag_corr` | Pearson `r` at whichever ±2-window lag maximizes it. | Best-achievable synchrony allowing a time offset (leader/follower coupling). |
| `best_lag_windows` | The lag (in 15 s window-steps) that produced `best_lag_corr`. | Temporal offset of peak synchrony; sign indicates which participant leads. |

### Output files

| File | Granularity |
|---|---|
| `analysis/results/cross_modal/features_group_dynamics_window_30s.tsv` | One row per session × task × window (group mean/SD per metric). |
| `analysis/results/cross_modal/features_group_dynamics_task.tsv` | One row per session × task × participant-pair × metric (correlation/lag). |

Both would be written empty (headers only) if no usable metrics/rows were found for a session (not the case here).

---

## 6. Semantic biomarker composites

**Status:** ✅ Generated (2026-08-12). **Script:** [tools/features/build_semantic_biomarkers.py](../tools/features/build_semantic_biomarkers.py). **Outputs:** `analysis/results/cross_modal/{semantic_biomarkers_participant_task.tsv (196 rows), semantic_biomarkers_window_30s.tsv (16,836 rows)}`.

> Generated from the same staged `analysis/results/cross_modal/` directory used for group dynamics (see §5) — this script requires `features_physio_participant_task.tsv`/`features_physio_window_30s.tsv` and `features_pupil_participant_task.tsv`/`features_pupil_window_30s.tsv` together in one `--features-dir`, plus the optional `features_group_dynamics_window_30s.tsv` (present, so `n_participants` is joined into the window-level output).

This script does not parse raw signals — it combines already-extracted physio/pupil features (and `n_participants` from group dynamics) into interpretable composites. **Only 6 of the 10 biomarkers described in the aspirational [semantic_biomarkers_catalog.md](semantic_biomarkers_catalog.md) are actually coded** — `Social Synchrony`, `Conversation Dominance Strain`, `Conflict/Tension Episodes`, and `Engagement/Involvement` are not implemented in this script.

### Pipeline

- **Normalization** (two schemes, not one uniform method):
  - Participant-task-level composites use `_baseline_z`: center on that participant's own `T0` value, divide by their within-session cross-task SD (falls back to a plain within-participant z-score if `T0` is missing).
  - Window-level composites use `_z_within`: plain z-score within `(session_id, participant_id, task)` — i.e. across that task's own 30 s windows only, no `T0` reference.
- **Composite combination**: always an **unweighted arithmetic mean** of the finite input z-scores (`_composite_mean`) — not PCA or a fitted/weighted index. NaNs are excluded per-row, not imputed.
- **Labeling**: `_score_to_label` buckets any composite into `High` (≥ 1.0), `Low` (≤ −1.0), `Moderate` (between), `Unknown` (non-finite).

### Feature list

| Feature (exact column) | Technical definition | Interpretation |
|---|---|---|
| `biomarker_cognitive_load` | `mean(z(pupil_mean), z(eda_mean), −z(hrv_rmssd_ms))` | Cognitive Load Index — higher pupil dilation/EDA plus lower HRV → higher mental effort. |
| `biomarker_arousal_stress` | `mean(z(eda_scr_rate_hz), z(ppg_rate_proxy_bpm), z(pupil_std))` | Arousal/Stress Reactivity — SCR rate, heart-rate proxy, pupil volatility. |
| `biomarker_attention` | `mean(z(gaze_valid_frac), −z(pupil_missing_frac), −z(pupil_std))` | Sustained Attention — stable gaze/pupil signal, low data dropout. |
| `biomarker_decision_pressure` | `mean(biomarker_arousal_stress, z(eda_mean))` | Decision Pressure — a composite built from another composite plus raw EDA. |
| `biomarker_recovery_capacity` | `mean(−z(eda_mean), −z(ppg_rate_proxy_bpm), z(hrv_rmssd_ms))` | Recovery/Regulation Capacity — **participant-task table only**, not computed at window level. |
| `biomarker_fatigue_depletion` | `task_index · (−z(pupil_mean, T0-baseline)) / 4.0` if `task_index ≥ 1` (T0=0…T4=4), else `NaN` | Fatigue/Cognitive Depletion — a linear task-progression-weighted term, **participant-task only**, not a symmetric composite. |
| `state_load_label` / `state_arousal_label` | `_score_to_label` applied to the cognitive-load / arousal-stress composites. | High/Moderate/Low/Unknown bucket for easy reporting. |
| `state_attention_label` | `_score_to_label(biomarker_attention)`. | **Participant-task table only** — not emitted at window level. |

### Output files

| File | Join keys |
|---|---|
| `analysis/results/cross_modal/semantic_biomarkers_participant_task.tsv` | Merges `features_physio_participant_task.tsv` + `features_pupil_participant_task.tsv` on `session_id, task, participant_id`. |
| `analysis/results/cross_modal/semantic_biomarkers_window_30s.tsv` | Merges `features_physio_window_30s.tsv` + `features_pupil_window_30s.tsv` on `session_id, task, window_index, window_start_lsl, window_end_lsl, participant_id`; left-joins `n_participants` from group dynamics. |

Reminder: only 6 of the 10 aspirational biomarkers are coded (see above) — do not cite `Social Synchrony`, `Conversation Dominance Strain`, `Conflict/Tension Episodes`, or `Engagement/Involvement` as available, since they are not implemented in this script.

---

## 7. Personality & demographic traits (BFI-44)

**Status:** ✅ Generated. **Scripts:** [tools/extract_bfi44_participants.py](../tools/extract_bfi44_participants.py) (raw-item scoring) → [analysis/build_participant_traits.py](../analysis/build_participant_traits.py) (re-keys onto group_id/P1–P4).

### Pipeline

1. `tools/extract_bfi44_participants.py` reads the questionnaire export (44-item Excel, item 1 at column index 14), maps each Likert response through `RESPONSE_MAP = {"strongly disagree":1, "disagree":2, "neither disagree nor agree":3, "agree":4, "strongly agree":5}`, applies reverse-scoring (`6 − value`) for the reverse-keyed items in each domain, and computes each domain's mean rounded to 3 decimals.
2. Respondents are fuzzy-matched (`difflib`) to session participants and assigned `sub-NNN` IDs; writes `metadata/participants.tsv`, `metadata/participants.json` (BIDS sidecar with descriptions/levels/ranges), and `metadata/name_matching_review.tsv` (match audit trail).
3. `analysis/build_participant_traits.py` re-keys the same scored columns onto `(group_id, P1..P4)` using the participant-name **order** in `metadata/high_level_session_inventory.csv` (position 1 → P1, etc.) — the script's own docstring flags this **positional mapping as an unverified assumption**, distinct from the verified name-matching already logged in `/memories/repo/metadata_verification_complete.md`.

### Feature list

| Feature (exact column) | Technical definition | Interpretation |
|---|---|---|
| `bfi44_e` | Mean of items `[1,6,11,16,21,26,31,36]`, reverse-scored for `[6,21,31]`. | Extraversion (1.0–5.0 scale). |
| `bfi44_a` | Mean of items `[2,7,12,17,22,27,32,37,42]`, reversed `[2,12,27,37]`. | Agreeableness. |
| `bfi44_c` | Mean of items `[3,8,13,18,23,28,33,38,43]`, reversed `[8,18,23,43]`. | Conscientiousness. |
| `bfi44_n` | Mean of items `[4,9,14,19,24,29,34,39]`, reversed `[9,24,34]`. | Neuroticism. |
| `bfi44_o` | Mean of items `[5,10,15,20,25,30,35,40,41,44]`, reversed `[35,41]`. | Openness to Experience. |
| `age`, `sex`, `handedness`, `english_proficiency`, `education` | Normalized demographic fields (`_parse_age`, `_normalize_sex`, `_normalize_hand`, `_normalize_education`). | Participant demographics, copied through unchanged by the trait-crosswalk step. |

### Output files

| File | Producer | Content |
|---|---|---|
| `metadata/participants.tsv` | `tools/extract_bfi44_participants.py` | `participant_id` (`sub-NNN`), real name (must never leave this file — see [security.instructions.md](../.github/instructions/security.instructions.md)), demographics, `bfi44_e/a/c/n/o`. |
| `metadata/participants.json` | same | BIDS sidecar (`Description`, `Levels`, `Units`, `Range`, `Items`, `ReversedItems`). |
| `metadata/name_matching_review.tsv` | same | Fuzzy-match audit trail (`sub_id`, `session_name`, `questionnaire_name`, `match_score`, `status`). |
| `analysis/results/participant_traits.tsv` | `analysis/build_participant_traits.py` | `group_id`, `participant` (`P1`–`P4`), same trait/demographic columns — no name columns. |

### Caveats

- The `group_id`/`P1–P4` positional re-keying in `build_participant_traits.py` is flagged by its own author as unverified — cross-check against the already-completed seat-assignment verification in repo memory (`/memories/repo/metadata_verification_complete.md`) before citing personality-trait correlations by participant seat.
- Real participant names live only in `metadata/participants.tsv`/`name_matching_review.tsv` (and `.private/registration_ledger.jsonl`) and must never be copied into any other output.

---

## 8. Video / 3D pose / gaze-world / gesture

**Status:** 🧩 Raw artifacts only — **no aggregated numeric feature table exists yet**, unlike every other modality above. There is no script analogous to `extract_physio_features.py` that turns these outputs into per-task means/rates/z-scores.

### Stage 1 — Low-level 2D extraction

**Script:** `tools/extract_video_features.py` (per [video_feature_extraction.md](video_feature_extraction.md)). Decodes each camera video once and writes:

| Artifact | Content |
|---|---|
| `frame_sync.jsonl` | Per-frame `pts_time`, `unix_time_s`, `wall_time_s`, `lsl_time` — clock-alignment scaffold, not a behavioral feature. |
| `marker_detections_2d.jsonl` | Per-detection `marker_id`, `ambiguous_marker_id`, `corners_px` — raw ArUco marker detections for calibration/world alignment. |
| `body_2d.npz` / `face_2d.npz` / `hands_2d.npz` | Dense per-frame landmark arrays (body 33/17/133 pts depending on model, face 478 pts, hands 21 pts × `[x,y,z-or-conf,visibility]`) — raw coordinates, no aggregation. |

### Stage 2 — 3D reconstruction + rule-based gesture events

**Script:** `tools/video_only_3d_pipeline.py` (per [video_only_3d_pipeline.md](video_only_3d_pipeline.md)):

| Artifact | Content |
|---|---|
| `skeleton_3d.npy` / `skeleton_3d_refined.npy` | 4D array `(frames, people, keypoints, ≥4 dims)` — triangulated multi-person 3D pose trajectories. |
| `{glasses_id}_pose.ndjson`, `{glasses_id}_gaze_world.ndjson` | Tobii glasses pose and gaze transformed into the shared calibrated world coordinate frame. |
| `gestures_events.ndjson` | Discrete rule-based gesture episodes: `participant`, `gesture` (one of `left_hand_to_head`, `right_hand_to_head`, `left_hand_to_chest`, `right_hand_to_chest`, `left_arm_extended`, `right_arm_extended`, `both_hands_to_head`), `start_frame`/`end_frame`/`start_time_s`/`end_time_s`/`duration_s`/`n_frames`. Detected via geometric thresholds (`hand_near_head_m`, `hand_near_chest_m`, `arm_extension_ratio`, `min_event_frames`) on BODY_25-style joints. |
| `gestures_summary.json` | `{n_events, events_per_participant, events_per_gesture, thresholds, fps, frames, people}` — raw counts, no rates/z-scores/normalization. |

### Caveats

- These are the closest things to "features" for this modality, but they are **categorical event lists**, not a numeric per-task/window table comparable to physio/pupil/audio/transcript.
- Anyone wanting per-task video-derived metrics (e.g. gesture rate per minute, average gaze-world dispersion) must currently write a new aggregation script — none exists in the repo yet.

---

## Cross-references

- [dataset_description.md](dataset_description.md) — coverage/completeness statistics (how much data exists, not what it contains).
- [semantic_biomarkers_catalog.md](semantic_biomarkers_catalog.md) — the aspirational/interpretive biomarker plan (10 composites); §6 above documents only the 6 that are actually implemented.
- [physio_feature_extraction_plan.md](physio_feature_extraction_plan.md) — the original design plan for the physio/pupil pipelines (superseded in detail by §1–§2 above, which describe the actual shipped code).
- [labels_codebook.md](labels_codebook.md) — full transcript annotation taxonomy referenced in §4.
- [audio_annotation_pipeline.md](audio_annotation_pipeline.md) — upstream manual audio transcription/annotation process feeding §3/§4.
- [bfi44_personality_scores.md](bfi44_personality_scores.md) — narrative description of the BFI-44 scoring referenced in §7.
