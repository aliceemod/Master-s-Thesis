# Dataset Description — AffectAI Multimodal Group Interaction Dataset

_Last updated: 2026-08-09._

_Raw collection scope (§4) is sourced from `docs/data_audit.md` (dated 2026-03-31, not re-verified since — raw data is not accessible from this workspace to re-audit). **Modality coverage (§5) is computed and verified in [`analysis/dataset_coverage_verification.ipynb`](../analysis/dataset_coverage_verification.ipynb)**, directly from the raw per-participant window feature files (`analysis/results/physio_features/physio_window_30s.tsv`, `analysis/results/pupil_features/features_pupil_window_30s.tsv`, `transcripts/final/*.tsv`), not from the pre-aggregated `icmi_paper/results/enhanced_features_final.tsv` (whose generating code could not be located in this repo, and which was found to contain a data-integrity bug — see §7). Scope is consistently `T1`–`T4` across all modalities (`T0` is excluded — see §5). The notebook is the authoritative, re-runnable source for every number in §5; exported CSVs live in `analysis/results/coverage_verification/`._

## 1. Overview

- Multimodal, multi-sensor dataset of **4-person groups** performing structured collaborative/negotiation tasks
- Captures synchronized **video, audio, eye-tracking, physiology, motion capture, and behavioral/self-report** streams
- Designed around emergent **group-level phenomena** (collective engagement, rapport, cohesion, coordination) rather than single-person affect
- Post-collection processing packages raw multi-source recordings into **BIDS-structured** outputs per session

## 2. Study design

- **5 tasks per session**, run in fixed order: `T0` (baseline/intro) → `T1` (Hidden-Profile Decision) → `T2` (Mini-Negotiation) → `T3` (Idea Generation / NGT) → `T4` (Public-Goods Micro-Game)
- Session length: ~75–90 min (main protocol) or ~60–75 min (small/pilot protocol)
- Timed phases within tasks (e.g., T1 discussion 420 s, T2 negotiation 480 s + settlement 60 s, T3 idea-gen 150 s silent + ranking/discussion 420 s, T4 contribution 60 s + reveal 60 s + discussion 180 s)
- Periodic **VAD (valence-arousal-dominance) self-report probes** pushed to tablets at scheduled timepoints per task (≥2 probes/task, jittered ±5 s)
- Post-task questionnaires per block, plus a final wrap-up VAD collection

## 3. Participants

- **49 individuals registered** across all groups (see `metadata/participants.tsv`), forming **4-person groups**
- **42 / 49 (~86%)** have complete demographics + personality data; 7 have no demographic record on file
- Demographic fields collected: age, sex, handedness, English proficiency, education level
- Personality: **BFI-44** Big Five trait scores (extraversion, agreeableness, conscientiousness, neuroticism, openness)
- Anonymised as `sub-001`…`sub-0NN` (metadata roster) and `P1`–`P4` (within-session BIDS/LSL identifiers) — no real names in processed/derived outputs

## 4. Sessions & coverage (raw collection scope — unverified since 2026-03-31)

> These figures come from `docs/data_audit.md` and have **not** been re-checked against raw data (not accessible from this workspace — see `docs/offline_compute_stack.md`, raw sources live on separate GPU/staging workstations + USB archive). Treat as the collection-time record, not a live count. For directly verified numbers, see §5.

| Item | Value |
|------|-------|
| Scheduled/recorded groups | 18 (`grp-01`…`grp-16` + pilot/test groups `grp-A`, `grp-B`, `grp-C`) |
| Total sessions in inventory | 26 (after duplicate-entry correction — see `docs/data_audit.md` §3.3) |
| **Final, analysable sessions** | **13 sessions across 11 groups** |
| Pilot sessions | grp-03 (×2 runs), grp-04 — limited modalities |
| Test sessions | grp-05, grp-A (×7 runs), grp-B, grp-C — limited modalities, not for analysis |
| Recording date range | 2026-03-09 → 2026-03-20 |
| Participants per group | 4 (except grp-06: 3 — one participant absent) |

### Completeness tiers (final sessions)

- **Tier 1 — fully complete** (all sources, all tasks T0–T4, schedule matches): 5 sessions (grp-09, grp-12, grp-13, grp-14, grp-15)
- **Tier 2 — nearly complete** (minor, resolved or isolated gaps): grp-07 (Tobii P1 gap filled), grp-08 (Tobii P4 gap filled), grp-11 (T0 events missing), grp-01 (no AV recordings)
- **Tier 3 — significant gaps**: grp-06 (3 participants only), grp-10 (no Tobii LSL in XDF; inflated T0/T1 durations from overnight stimuli app run), grp-16 (no CurrentStudy XDF)
- Pilot/test sessions retained for reference only — reduced modality set, not intended for primary analysis

## 5. Modalities & devices — coverage verified in the notebook (as of 2026-08-09)

Computed directly from raw per-participant, per-30 s-window feature files in [`analysis/dataset_coverage_verification.ipynb`](../analysis/dataset_coverage_verification.ipynb) (1849 physio windows, 1703 eye-tracking windows), cross-checked against `icmi_paper/results/enhanced_features_final.tsv`. **Scope is consistently `T1`–`T4` for every modality** — `T0` (baseline/intro) is excluded everywhere, since it is never manually transcribed (confirmed: 0 `T0` transcript files exist) and including it for physio/ET but not transcript would make the modalities non-comparable. **Only 10 groups have been carried through processing so far: `grp-07`–`grp-16`** (the earlier/pilot/test groups `grp-01`–`grp-06`, `grp-A/B/C` are in the raw collection but not in this processed set).

| Modality | Device(s) | Overall window coverage (`T1`–`T4`) | Notes |
|----------|-----------|--------------------------|-------|
| Physiology (EmotiBit) | 4× wrist sensors | **95.5%** of windows have ≥1 participant valid; **82.6%** have all 4 valid (participant-window cell validity: 92.3%) | Recomputed from the raw `physio_window_30s.tsv` — the raw source is clean (max 4 participants/window everywhere, including `grp-08`); `grp-09` capped at ~2/4 valid participants for the whole session (real sensor gap); `grp-12` mostly 0–1/4 valid (physio largely unusable) |
| Eye-tracking (Tobii pupil) | 4× Tobii Pro Glasses 3 | **99.6%** of windows have ≥1 participant valid; **96.7%** have all 4 valid (participant-window cell validity: 99.1%) | Recomputed from raw `features_pupil_window_30s.tsv`, `T1`–`T4` only — much higher than the earlier ad-hoc 85.3% figure, which used a narrower/inconsistent aggregate; see the notebook for the full per-group breakdown |
| Transcript / turn-taking annotation | Manual transcription (SPK/SIL/BCK/LAU/FP + OVL overlap taxonomy) | **92.5%** file-level existence (37/40); **95.4%** span contiguity within transcribed files | `T0` is never transcribed by design (no task-relevant conversation to annotate) — the 40-file universe is 10 groups × `T1`–`T4` only. Of those 40, 37 exist (`grp-09` missing `T1`, `grp-13` missing `T3`/`T4`). **Important:** transcription only covers the actual task activity, not the surrounding recording overhead within a task block (glasses fitting, settling in, post-task questionnaires) — those segments were deliberately not transcribed. A third metric, "window-level coverage vs. device recording" (35.8%), compares transcribed windows against the much longer physio/ET recording span and is **not** a completeness gap — it mostly reflects non-task overhead time, not missing annotation work. The 95.4% span-contiguity figure (windows between a file's first and last event that are themselves covered) is the more meaningful completeness signal and confirms transcription proceeds without large internal gaps once it starts |
| Audio (openSMILE/GeMAPS window features) | 5× DPA mics + central mic | 526 windows present across the same 10 groups (`icmi_paper/results/audio_window_features.tsv`) | Informational only — no independent raw-audio source in this repo to recompute coverage against |
| Video | 6× Jabra PanaCast 20 + 1× PanaCast 50 | Not represented in the processed window features | Coverage claims here would need a fresh scan of raw AV folders, which are not accessible from this workspace — see §7 |
| Motion capture (Vicon) | 6× optical cameras | Not represented in the processed window features | Same caveat as video |
| Behavioral (tablets/Big Screen) | Task responses, VAD probes, questionnaires | Not independently re-verified here | Last checked in `docs/data_audit.md` (2026-03-31) |

## 6. BIDS output structure

- `sub-{id}/ses-{id}/` with modality subfolders: `eeg/`, `et/`, `physio/`, `audio/`, `video/`, `mocap/`, `beh/`, `annot/`
- One authoritative `*_events.tsv` per session (timeline spine); never duplicated across files
- `participants.tsv` at study root — anonymised roster only (`P1`–`P4` per session)
- Per-task run outputs: `task-{T0..T4}_run-01` (zero-padded), including run-sliced LSL-derived tables, normalized stimuli answers (`beh/*_stimuli_answers.tsv`), and device-to-participant signal maps (`annot/*_participant_signal_map.tsv`)
- See `.github/instructions/bids-conventions.instructions.md` for full naming rules

## 7. Known limitations / caveats

- Demographic and personality data missing for 7/49 participants (no BFI-44/age/sex on file)
- Raw-collection coverage (§4) is a 2026-03-31 snapshot; only processed-feature coverage (§5) has been directly re-verified as of 2026-08-09, via [`analysis/dataset_coverage_verification.ipynb`](../analysis/dataset_coverage_verification.ipynb)
- Only 10 of 18 collected groups (`grp-07`–`grp-16`) have been carried through to processed window features so far — no verified current coverage exists for video, motion capture, or the remaining 8 groups
- **`icmi_paper/results/enhanced_features_final.tsv` contains a confirmed data-integrity bug for `grp-08`**: it reports `n_physio_valid=8` for every one of `grp-08`'s 182 windows (impossible for a 4-person group). The notebook traced this to the raw source (`physio_window_30s.tsv`), which is clean — exactly 4 participant rows per window for grp-08 throughout. The bug is therefore isolated to the untraceable script that produced `enhanced_features_final.tsv` (most plausibly a duplicate join against grp-08's known Mar-16 re-recording session, per `docs/data_audit.md`), not the underlying data. **Do not use `enhanced_features_final.tsv`'s `n_physio_valid` column for grp-08 in the paper/thesis** — use the notebook's raw recomputation instead, which is unaffected
- Physio and eye-tracking recordings do not always span the same duration within a task (EmotiBit/Tobii start/stop independently) — window counts differ for 11 of 40 group/task combinations, sometimes by dozens of windows; each modality's coverage is computed against its own window grid, not a shared one
- Transcript annotation intentionally covers only the task activity itself, not the recording overhead within a task block (glasses fitting, settling in, post-task questionnaires) — don't cite the notebook's "window-level coverage vs. device recording" (35.8%) as a transcription gap; use file-level existence (92.5%) and span-contiguity (95.4%) instead
- `grp-09` has genuinely reduced physio (~2/4 participants valid all session) and no `T1` transcript; `grp-12` has largely unusable physio (0–1/4 valid); `grp-13` is missing `T3`/`T4` transcripts
- grp-06 has only 3 of 4 participants (genuine absence, not a data-loss artifact)
- grp-10 has inflated T0/T1 durations due to the stimuli application running overnight before the session — treat task-window durations with caution for this session
- Pilot (`grp-03`, `grp-04`) and test (`grp-05`, `grp-A/B/C`) sessions have reduced modality sets and are excluded from primary analysis

## 8. Related documents

- `analysis/dataset_coverage_verification.ipynb` — authoritative, re-runnable notebook computing all §5 coverage numbers from raw source files, with integrity checks and cross-checks; exports to `analysis/results/coverage_verification/`
- `docs/data_audit.md` — full per-session audit trail and fixes applied
- `docs/architecture.md`, `docs/data_flow.md` — system design and pipeline data flow
- `docs/labels_codebook.md` — annotation taxonomy for transcript/overlap labels
- `docs/bfi44_personality_scores.md` — personality scoring methodology
- `docs/feature_catalog.md` — full technical reference for every extracted feature across all modalities (physio, eye-tracking, audio, transcript, group dynamics, semantic biomarkers, personality, video): exact extraction code, formulas, and per-feature interpretation, plus which pipelines are actually generated vs. implemented-but-not-yet-run
- `icmi_paper/feature_inventory.md` — derived feature catalog with per-feature coverage for the ICMI 2026 analysis subset (group-level, 30 s-window aggregates actually used in the HMM analysis) — for feature *selection rationale*, see this file; for feature *computation methodology*, see `docs/feature_catalog.md`
