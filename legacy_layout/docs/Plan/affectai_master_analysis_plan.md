# AffectAI — Multimodal Analysis Master Plan

> **Purpose**: This document provides a high-level overview of the complete
> behavioral analysis and 3D modeling improvement programme for the AffectAI
> dataset. It synthesises four detailed technical plans into a single reference
> that describes what we are building, why, how the pieces connect, and in what
> order to implement them.
>
> **Audience**: Research team, RAs, and any new collaborator needing to
> understand the full scope before diving into a specific sub-system.
>
> **Four detailed plans this document oversees**:
> 1. `3d_gaze_improvement_plan.md` — Gaze ray projection and world-frame alignment
> 2. `3d_roi_gesture_skeleton_plan.md` — ROI detection, skeleton, and gesture
> 3. `multimodal_fusion_3d_enhancement_plan.md` — IMU, audio, and tablet log fusion
> 4. `additional_behavioral_signals_plan.md` — Higher-level behavioral features

---

## Table of Contents

1. [What We Have](#1-what-we-have)
2. [What We Are Building](#2-what-we-are-building)
3. [The Four-Layer Analysis Stack](#3-the-four-layer-analysis-stack)
4. [How the Layers Connect](#4-how-the-layers-connect)
5. [Shared Infrastructure and Dependencies](#5-shared-infrastructure-and-dependencies)
6. [Implementation Phases](#6-implementation-phases)
7. [Quality Control Strategy](#7-quality-control-strategy)
8. [Known Risks and Mitigations](#8-known-risks-and-mitigations)
9. [Research Questions This Enables](#9-research-questions-this-enables)
10. [Quick Reference: Priority Items Across All Plans](#10-quick-reference-priority-items-across-all-plans)

---

## 1. What We Have

### 1.1 The Dataset

Sixteen groups of four participants completed four collaborative tasks (T1–T4)
in a controlled lab environment. All sessions are recorded and stored. Data
collection is complete.

```
16 groups × 4 participants × 4 tasks × ~75 minutes = ~80 hours of synchronized data
```

### 1.2 Sensor Inventory

| Sensor | Count | Raw output | Current status |
|--------|-------|-----------|----------------|
| Tobii Pro Glasses 3 | 4 (one/participant) | Gaze NDJSON, scene video, IMU | Stored in BIDS |
| Jabra PanaCast 20 cameras | 6 | 1080p@30fps MKV video | Stored, calibrated |
| Jabra PanaCast 50 (wide) | 1 | 1080p@30fps MKV video | Stored |
| DPA 4060 microphones | 5 (4 close-talk + 1 room) | WAV 48 kHz | Stored, split by task |
| EmotiBit | 4 (one/participant) | PPG, EDA, temperature, IMU | Stored in BIDS |
| Vicon optical mocap | 1 system (6 cameras) | 3D marker trajectories | Stored |
| Android tablets | 4 | Interaction logs, responses | JSONL in BIDS beh/ |
| Big screen (HDMI) | 1 | Stimuli events | LSL/TSV logs |

### 1.3 Synchronisation

All streams share a common LSL clock. Timestamps in Tobii gaze NDJSON are
LSL seconds (`ticks_per_second=1`). DPA audio, Vicon, and tablet logs are
aligned via the four-tier synchronisation pipeline (`ffmpeg_multicap` frame
logs → LSL → progress TSV → events JSONL). Clock drift across sessions is
< 100 ms.

### 1.4 Physical Setup

```
World frame: ChArUco 7×5 board center (desk center)
  x = right, y = back (toward big screen), z = up

Seating:  P1 = back-right   P2 = front-right
          P4 = back-left    P3 = front-left

Desk: 1.8 m × 0.8 m × 0.75 m tall
Camera mount height: ~0.88 m above desk (cam1–cam4, cam6)
```

### 1.5 What Is Already Working

- BIDS packaging with per-task media splits (`multisource_to_bids_runs.py`)
- 7-camera spatial calibration (ChArUco + anipose, `calibrate_charuco.py`)
- Multi-camera 3D body skeleton (MediaPipe + triangulation, `multicam_pose3d.py`)
- Face and hand 2D detection (`face_hand_pipeline.py`)
- Scene-video PnP gaze alignment for sessions with scene video (`tobii_multi_glasses_world_align.py`)
- Glasses marker tracker (Approach A, `tobii_multicam_glasses_tracker.py`)
- LSL event logging with dual-write (TSV + LSL outlets, `event_logger.py`)
- Tablet response collection with per-participant JSONL files

### 1.6 What Is Not Yet Working Well

- ArUco marker ID collisions between board, desk, and glasses markers corrupt
  detection in all three tools simultaneously
- Tobii scene-camera intrinsics are hardcoded approximations (fx=1300) rather
  than per-device factory values
- Gaze rays are not yet corrected for the `T_glasses_sceneCamera` extrinsic
- Camera zones for skeleton triangulation use same-side cameras (poor baseline)
- 3D hand keypoints are not triangulated
- Gesture detection is not yet implemented
- Tablet logs are parsed for submitted values only — timing, latency, and
  trajectory features are unused
- VAD ratings are used as raw values only — trajectory features are not extracted
- Post-block dominance sociometric matrices are not constructed
- Trust and familiarity networks are not constructed

---

## 2. What We Are Building

The goal is a **unified per-frame multimodal feature dataset** covering all
16 sessions, all 4 participants per session, and all 4 tasks per session, at
30 fps resolution, with lower-frequency features (VAD, questionnaire) aligned
to the same time axis.

```
Input (already collected)           Output (to be built)
────────────────────────────        ─────────────────────────────────────
Tobii gaze NDJSON            →      3D gaze ray in world frame
Tobii IMU stream             →      Head orientation + dead-reckoning pose
Scene video (MP4)            →      Glasses pose via PnP (Approach B)
Fixed camera video (MKV)     →      Skeleton (25 KP, 4 persons) +
                                    Glasses pose via markers (Approach A) +
                                    3D hand keypoints (21 KP × 2 × 4 persons)
DPA audio (WAV)              →      Per-participant VAD + speaker ID +
                                    turn structure + gesture timing
Tablet logs (JSONL)          →      Confirmed gaze windows + ROI labels +
                                    response latency + form dwell + skip rate
VAD responses (JSONL)        →      Trajectory features + group synchrony
Post-block questionnaires    →      Dominance matrices + trust/familiarity graphs
Task decision logs (LSL/TSV) →      Influence indices + decision timing +
                                    hidden-profile outcome + negotiation quality
────────────────────────────        ─────────────────────────────────────
                                    → unified multimodal_features_T{N}.ndjson
                                       per session, per task, per frame
```

---

## 3. The Four-Layer Analysis Stack

The improvement programme is organised into four layers. Each layer feeds the
next. Skipping a layer produces unreliable outputs in all layers above it.

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — BEHAVIORAL FEATURES                                       │
│  VAD trajectories, dominance matrices, trust networks,              │
│  decision quality, turn-taking, influence indices                    │
│  Source: additional_behavioral_signals_plan.md                       │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — MULTIMODAL FUSION                                         │
│  IMU dead-reckoning, audio VAD gating, tablet log overrides,        │
│  Bayesian ROI posterior, weak supervision labels                     │
│  Source: multimodal_fusion_3d_enhancement_plan.md                    │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — 3D SCENE UNDERSTANDING                                    │
│  Static ROI registry, skeleton per-seat assignment,                 │
│  tablet ROI tracking, gesture detection, hand keypoints             │
│  Source: 3d_roi_gesture_skeleton_plan.md                             │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — 3D GAZE AND POSE (FOUNDATION)                             │
│  ArUco ID collision fix, factory intrinsics, T_glasses_sceneCamera, │
│  Approach A+B+C pose fusion, gaze ray projection                    │
│  Source: 3d_gaze_improvement_plan.md                                 │
└─────────────────────────────────────────────────────────────────────┘
          ↑ must be correct before anything above is reliable
```

### Layer 1 — 3D Gaze and Pose (Foundation)

**What it does**: Transforms Tobii gaze data from each glasses unit's scene-camera
frame into the shared world coordinate system. This requires knowing where each
glasses unit is in the world at every moment (6-DoF pose), and the factory
transform between the scene camera and the glasses frame.

**Why it is the foundation**: Every ROI label, gesture event, joint attention
event, and social behavior measure in layers 2–4 depends on knowing where each
participant is looking. A 10 cm gaze offset would mis-classify desk gaze as
screen gaze, or one participant's personal space as another's.

**Key dependencies**:
- ArUco ID collision fix (shared with Layer 2)
- Factory intrinsics from Tobii recording metadata
- `T_glasses_sceneCamera` from Tobii recording metadata
- Calibrated camera TOML (already exists but needs validation)

### Layer 2 — 3D Scene Understanding

**What it does**: Maintains a 3D model of the scene — where each participant is
sitting, where their body parts are, where the tablets and screen are, and what
gestures they are performing.

**Key outputs**:
- `configs/scene_rois.yaml` — static 3D bounding boxes for all ROIs
- Per-frame skeleton `(4, 25, 7)` with seat-locked identity
- Per-frame hand keypoints `(4, 21, 3)` per hand
- Gesture event NDJSON with VAD-gated communicative gestures
- Body-part ROI assignments per frame

### Layer 3 — Multimodal Fusion

**What it does**: Compensates for the gaps and ambiguities in the vision pipeline
using auxiliary sensors. When a glasses marker is occluded, IMU integration
fills the gap. When gaze is ambiguous between desk and tablet, the tablet touch
log confirms it. When gesture detection fires during silence, audio VAD suppresses
the false positive.

**Key outputs**:
- Fused glasses pose with `pose_source` label (vision / IMU / blend)
- Bayesian ROI posterior per participant per frame
- Confirmed gaze windows from tablet touch logs
- VAD-gated gesture events

### Layer 4 — Behavioral Features

**What it does**: Derives higher-level behavioral and social constructs from all
layers below. These are the features that enable the research questions about
group affect, influence, cooperation, and emotional dynamics.

**Key outputs**:
- VAD trajectory features (slope, delta, variability) per participant per task
- Dominance sociometric matrices (4×4 per task)
- Trust and familiarity directed graphs
- Turn-taking structure (floor-holding, interruptions, backchannels)
- Emotional synchrony between participant pairs
- Influence index combining decision, speech, and gaze-received
- Decision quality labels (hidden-profile success, negotiation integrativness)

---

## 4. How the Layers Connect

### 4.1 Data Flow

```
Raw data                         Intermediate                    Final output
────────────                     ────────────                    ────────────

Tobii NDJSON  ──→  [Layer 1]  →  gaze_world_T{N}.ndjson   ──┐
Tobii IMU     ──→  [Layer 3]  →  pose_fused_T{N}.ndjson    ──┤
Scene video   ──→  [Layer 1]  →  glasses_pose_T{N}.ndjson  ──┤
Fixed cameras ──→  [Layer 2]  →  skeleton_3d_T{N}.npy      ──┤  multimodal_
Fixed cameras ──→  [Layer 1]  →  glasses_pose_T{N}.ndjson  ──┤  features_
DPA audio     ──→  [Layer 3]  →  vad_T{N}.npy              ──┤  T{N}.ndjson
Tablet logs   ──→  [Layer 3]  →  confirmed_gaze_T{N}.json  ──┤  (per session,
VAD responses ──→  [Layer 4]  →  vad_trajectory_T{N}.json  ──┤   per task)
Postblock     ──→  [Layer 4]  →  dominance_matrix_T{N}.csv ──┘
Decision logs ──→  [Layer 4]  →  influence_T{N}.json       ──┘
```

### 4.2 Shared Objects

These objects are created once and consumed by all layers. Getting them right
is therefore the highest-leverage investment:

| Shared object | Created by | Consumed by |
|---------------|-----------|-------------|
| `calibration_charuco_merged.toml` | `calibrate_charuco.py` | L1 PnP, L2 triangulation |
| `configs/scene_rois.yaml` | manual (one-time) | L2 ROI lookup, L3 ROI posterior |
| ArUco allowlists per tool | ID collision fix (L1) | L1 tracker, L1 aligner, L2 validator |
| Tobii factory intrinsics | parsed from recording metadata | L1 Approach B PnP |
| `T_glasses_sceneCamera` | parsed from recording metadata | L1 gaze-to-world |
| LSL master clock | existing infrastructure | all layers (timestamp alignment) |
| `session_rois.yaml` ROI bounding boxes | L2 setup | L2 gestures, L3 posterior, L4 gaze-on-tablet |

### 4.3 Key Cross-Layer Connections

**Tablet log → Layer 1 (gaze validation)**
Confirmed touch windows from Layer 3 provide ground truth for evaluating
Layer 1 gaze accuracy. During `touch_start–touch_end`, gaze must be on the
tablet ROI. This is the primary QC metric for the entire gaze pipeline.

**Skeleton head position → Layer 3 (IMU fusion)**
The last known 3D head position from the Layer 2 skeleton gives the translation
anchor for Layer 3 IMU dead-reckoning. When the skeleton fails (cam occlusion),
the last valid skeleton position is held constant while IMU integrates orientation.

**Audio VAD → Layer 2 (gesture gating)**
Layer 3 per-participant VAD (speaking/silent) is applied as a gate in Layer 2
gesture detection. Beat and iconic gestures are only emitted during speech frames,
reducing false positives by an estimated 40–60%.

**Dominance matrix → Layer 1 (gaze validation)**
The Layer 4 dominance matrix provides a behavioral prior for gaze direction.
High-dominance participants receive more gaze from others. This can be used to
validate the Layer 1 gaze classifier (dominant participants should appear as
gaze targets more frequently).

---

## 5. Shared Infrastructure and Dependencies

### 5.1 The ArUco Marker ID Collision — Affects All Layers

This is the single most critical issue in the entire pipeline. It must be fixed
before any other improvement is attempted.

**Problem**: Three marker families share overlapping ID ranges across two
dictionaries that are not fully disjoint (DICT_4X4_50 ⊂ DICT_4X4_250):

```
ChArUco 7×5 board   DICT_4X4_250   IDs 0–23
Desk edge markers   DICT_4X4_50    IDs 0–5   ← alias ChArUco IDs 0–5
Glasses P1          DICT_4X4_50    IDs 10–11  ← alias ChArUco IDs 10–11
Glasses P2          DICT_4X4_50    IDs 12–13  ← alias ChArUco IDs 12–13
Glasses P3          DICT_4X4_50    IDs 14–15  ← alias ChArUco IDs 14–15
Glasses P4          DICT_4X4_50    IDs 16–17  ← alias ChArUco IDs 16–17
```

**Fix** (software, no re-printing): Add ID allowlists in every detector call.
Each tool only trusts the IDs belonging to its own family:

```python
BOARD_IDS_EXCLUSIVE = set(range(17)) - {0,1,2,3,4,5, 10,11,12,13,14,15,16,17}
DESK_IDS            = frozenset({0, 1, 2, 3, 4, 5})
GLASSES_IDS         = frozenset({10, 11, 12, 13, 14, 15, 16, 17})
TABLET_IDS          = frozenset({20, 21, 22, 23})   # to be added (new markers)
```

**Hardware fix** (for future studies): Assign glasses markers to DICT_5X5_50.
This dictionary is physically incompatible with DICT_4X4_250 — no cross-detection.

### 5.2 Temporal Synchronisation Reminder

All tools must use the **same timestamp reference**. The existing system:

```
Tobii gaze NDJSON:   timestamp_ticks = LSL seconds  (--ticks-per-second 1)
DPA audio:           sample_index / 48000 + audio_lsl_start
Camera frames:       frame_log unix_time - pts_time  (tier-1 sync)
Tablet events:       server_received_lsl             (direct LSL clock)
IMU:                 timestamp_ticks = LSL seconds   (same as gaze)
```

**Do not use**: wall-clock `time.time()` for cross-modal alignment. Use LSL
clock for all cross-modal operations.

### 5.3 BIDS Output Schema

All outputs follow BIDS conventions. New files should be placed at:

```
sub-01/ses-{date}_{group}_run01/
  gaze/
    {pid}_task-T{N}_gaze_world.ndjson      (Layer 1 output)
    {pid}_task-T{N}_pose_fused.ndjson      (Layer 3 output)
  mocap/
    skeleton_3d_T{N}.npy                   (Layer 2 output)
    skeleton_3d_T{N}.json                  (metadata)
    gestures_events_T{N}.ndjson            (Layer 2 output)
    {pid}_hand_L_T{N}.npy                  (Layer 2 output)
    {pid}_hand_R_T{N}.npy
  annot/
    multimodal_features_T{N}.ndjson        (unified per-frame, Layers 1–3)
    gaze_roi_fused_T{N}.tsv                (ROI labels with source column)
    vad_trajectory_T{N}.json              (Layer 4 output)
    dominance_matrix_T{N}.csv             (Layer 4 output)
    trust_network_T{N}.json               (Layer 4 output)
    influence_T{N}.json                   (Layer 4 output)
    turn_structure_T{N}.json              (Layer 4 output)
```

---

## 6. Implementation Phases

Implementation is divided into three phases. Each phase is independently
useful — earlier phases do not need to be perfect before later phases begin,
but the critical fixes must precede all other work.

### Phase 0 — Critical Fixes (1–2 weeks)
*Prerequisite for everything. Do these first.*

| Item | Tool to modify | Estimated effort |
|------|---------------|-----------------|
| ArUco ID allowlists in `calibrate_charuco.py` | `_board_marker_id_set_exclusive()` | 1 hour |
| ArUco ID allowlists in `tobii_multicam_glasses_tracker.py` | Post-detection filter | 1 hour |
| ArUco ID allowlists in `tobii_multi_glasses_world_align.py` | Post-detection filter | 1 hour |
| ArUco ID allowlists in `validate_calibration_robust.py` | `_detect_markers_with_multiple_dicts()` | 1 hour |
| Flip-consistency check for cam1–4 (rotate_180) in calibration validator | New check in `validate_calibration_robust.py` | 2 hours |
| Fix camera zones: cam1+cam3, cam2+cam4 (not cam1+cam4, cam2+cam3) | `configs/offline_compute_stack.yaml` | 5 minutes |
| Create `configs/scene_rois.yaml` with static 3D ROI definitions | New file (desk, screen, zones, tablets) | 2 hours |

### Phase 1 — Layer 1 Core (2–4 weeks)
*Produces reliable 3D gaze. Needed before ROI analysis is meaningful.*

| Item | Source | Effort |
|------|--------|--------|
| Load Tobii factory intrinsics from recording metadata | Gaze plan §2 | Medium |
| Apply `T_glasses_sceneCamera` extrinsic from metadata | Gaze plan §3 | Medium |
| Apply 180° frame rotation before ArUco detection for cam1–4 | Gaze plan §4.4 | Low |
| Two-marker rigid-body joint PnP solve | Gaze plan §4.1 | Medium |
| RANSAC PnP in scene-video aligner | Gaze plan §5.3 | Low |
| Add ChArUco board as additional anchor in Approach B | Gaze plan §5.1 | Medium |
| Implement Approach C pose fusion (confidence-weighted) | Gaze plan §6 | Medium |
| Tobii calibration event as per-session gaze validation | Fusion plan §5.1, Behavioral §5.1 | Low |

### Phase 2 — Layers 2 and 3 Core (4–8 weeks)
*Adds ROI awareness, gestures, and auxiliary modality fusion.*

| Item | Source | Effort |
|------|--------|--------|
| Seat-based skeleton identity lock | ROI plan §2.2 | Low |
| Body-part ROI assignment per frame | ROI plan §2.4 | Low |
| IMU dead-reckoning for pose gap-filling | Fusion plan §3.2 | Medium |
| Tablet touch log → confirmed gaze windows | Fusion plan §2.2 | Low |
| Per-mic VAD from DPA audio energy | Fusion plan §5.2 | Medium |
| VAD gate on gesture events | Fusion plan §5.3 | Low |
| Bayesian ROI posterior (combining all modalities) | Fusion plan §8.1 | Medium |
| 3D hand keypoint triangulation | ROI plan §7.2 | Medium |
| Hand state classifier | ROI plan §7.3 | Medium |
| Sliding-window gesture detector | ROI plan §8.3 | Medium |
| Tablet ArUco markers (IDs 20–23, hardware) | ROI plan §4.2 | Low (hardware) |
| Temporal pose smoothing | Gaze plan §4.3 | Low |

### Phase 3 — Layer 4 and Integration (4–8 weeks)
*Derives behavioral constructs. Can partly run in parallel with Phase 2.*

| Item | Source | Effort |
|------|--------|--------|
| Dominance sociometric matrix from postblock | Behavioral §3.1 | Low |
| Familiarity network from T1 postblock | Behavioral §3.2 | Low |
| Trust network from T2 and T4 postblock | Behavioral §3.3 | Low |
| VAD trajectory features (slope, delta, std) | Behavioral §2.1 | Low |
| Response latency per probe per participant | Behavioral §1.2 | Low |
| Form dwell time per task form | Behavioral §1.4 | Low |
| T4 contribution profiling | Behavioral §1.5 | Low |
| T3 idea authorship → influence index | Behavioral §4.2 | Low |
| Floor-holding and speech proportion | Behavioral §6.2 | Medium |
| Interruption detection | Behavioral §6.1 | Medium |
| Group emotional convergence | Behavioral §2.2 | Low |
| Unified multimodal feature frame builder | Fusion plan §10 | High |
| Weakly supervised gaze classifier training | Fusion plan §9 | High |

---

## 7. Quality Control Strategy

### 7.1 QC Hierarchy

Each layer has a primary QC metric that validates the layer's output before
downstream layers consume it:

| Layer | Primary QC metric | Target | Tool |
|-------|------------------|--------|------|
| L1 Gaze | Gaze-on-tablet accuracy during confirmed touch windows | > 80% | `qc_tobii_world_gaze.py` |
| L1 Pose | Glasses marker reprojection error (Approach A) | < 3 px | `tobii_multicam_glasses_tracker.py` |
| L2 Skeleton | Bone-length violation rate | < 5% | `refine_skeleton_3d.py` |
| L2 Identity | Seat assignment stability across frames | > 98% | Manual check |
| L2 Gestures | Gesture rate plausibility (< 30/min/participant) | < 30/min | `validate_gesture_events()` |
| L3 IMU | Pose source = "imu_dr_stale" rate | < 5% | `monitor_imu_drift()` |
| L3 Fusion | Fused ROI agrees with confirmed touch windows | > 80% | `validate_fusion_consistency()` |
| L4 VAD | Coverage: > 2 valid measurements per participant per task | > 2 | `vad_coverage_check()` |
| L4 Postblock | Dominance matrix completeness (all 4 raters for each task) | > 75% | `dominance_matrix_qc()` |

### 7.2 Per-Session QC Report

A unified QC report should be generated for every session before analysis:

```bash
python tools/qc/qc_session_report.py \
    --session-dir <session> \
    --scene-rois configs/scene_rois.yaml \
    --calibration calibration_charuco_merged.toml \
    --output <session>/annot/qc_report.json
```

Sections in the report:
1. Sensor coverage (which modalities present, any gaps)
2. Synchronisation drift (LSL vs. video, max offset)
3. Gaze quality (per-participant validity rate, calibration event accuracy)
4. Skeleton quality (per-participant valid frame rate, bone violations)
5. Fusion quality (pose source breakdown, IMU drift events)
6. Behavioral completeness (VAD responses, postblock completeness)

### 7.3 Session Tier Classification

Based on QC report, each session is classified into one of three tiers
for downstream analysis:

| Tier | Criteria | Use in analysis |
|------|----------|----------------|
| **Gold** | Gaze accuracy > 80%, skeleton coverage > 85%, all modalities present | Full multimodal analysis |
| **Silver** | Gaze accuracy 60–80% or skeleton coverage 60–85%, most modalities | Analysis with caveats; vision-only ROI |
| **Bronze** | Gaze accuracy < 60% or major modality missing | Audio/questionnaire analysis only |

From the data audit (`docs/data_audit.md`), initial tier estimates:
- Gold: grp-09, grp-12, grp-13, grp-14, grp-15 (~5 sessions)
- Silver: grp-07, grp-08, grp-11 (~3 sessions, Tobii gaps resolved)
- Bronze or review: grp-06, grp-10, grp-16 (significant gaps)

---

## 8. Known Risks and Mitigations

### 8.1 Tobii Scene Video Availability

Some sessions are missing scene video for P1 and P3 (e.g., grp-12). Approach B
(scene-video PnP) is unavailable for those participants.

**Mitigation**: Approach A (glasses markers via fixed cameras) is the primary
method and does not require scene video. Where Approach A also fails (marker
occlusion), Layer 3 IMU dead-reckoning fills short gaps.

### 8.2 cam6 Calibration — Untransplanted Intrinsics

cam6 had zero ChArUco board frames in the March 2026 calibration run. Its
intrinsics are transplanted from a prior session.

**Mitigation**: Exclude cam6 from skeleton triangulation for sessions where
its intrinsics are unverified. Use it only for overview/qualitative checks.
Plan a dedicated cam6 calibration pass.

### 8.3 Glasses Marker Occlusion Rate

Participants lean forward, placing tablets between themselves and the cameras.
This regularly occludes the ArUco markers on the glasses frame.

**Mitigation**: IMU dead-reckoning (Layer 3) bridges up to ~5 seconds of
occlusion before drift becomes significant. Approach B (scene-video PnP)
provides an independent estimate when the desk/board markers are visible in
the scene video. The confidence-weighted fusion (Approach C) automatically
selects the best available estimate per frame.

### 8.4 Small N for Statistical Inference

Eleven analysable groups × 4 participants = 44 participants. This limits
statistical power for between-group and within-session analyses.

**Mitigation**: Focus on within-session, within-task, and within-participant
repeated-measures designs. The 30 fps temporal resolution and ~75 minutes per
session give >100,000 data points per participant per session even at this
group count.

### 8.5 DPA Crosstalk

Target: < −30 dB crosstalk between microphones. If not achieved, per-mic VAD
will misattribute speech to the wrong participant.

**Mitigation**: Crosstalk detection implemented in `multimodal_fusion_3d_enhancement_plan.md §7.2`.
Frames with suspected crosstalk are flagged and excluded from speaker diarisation.

---

## 9. Research Questions This Enables

Once all four layers are operational, the dataset supports the following
research questions. These are ordered by the layers they require.

### Requiring Layer 1 Only (Gaze)
- Does gaze coordination (mutual gaze, joint attention) predict decision quality
  in the hidden-profile task (T1)?
- Do participants with higher perceived dominance receive more gaze from others?
- How does screen gaze vs. desk gaze vs. face gaze distribute across task phases?

### Requiring Layers 1–2 (Gaze + Skeleton + ROI)
- Does tablet gaze predict VAD submission latency (a person looks at the tablet
  before responding)?
- Are hand gestures (pointing, beat, iconic) more frequent for high-dominance
  participants during T2 negotiation?
- Does lean-forward posture predict speaking floor acquisition?

### Requiring Layers 1–3 (Full 3D + Fusion)
- Can a multimodal classifier (gaze + IMU + audio energy) predict VAD arousal
  better than any single modality?
- Is T4 contribution (cooperation) predictable from pre-decision behavioral
  signals (gaze toward other participants, lean posture, audio energy)?
- Does emotional contagion (VAD synchrony across participants) predict group
  decision satisfaction?

### Requiring All Four Layers (Full Analysis)
- Does the trust network (from postblock questionnaires) predict the pattern
  of gaze exchange during T2 negotiation?
- Is the influence index (speaking + gaze received + idea selected) consistent
  across all four tasks, suggesting a stable group hierarchy?
- Does hidden-profile success (T1) predict subsequent cooperation (T4)?
- Can dominance perception (postblock matrix) be predicted from automated
  behavioral features, enabling scalable automated dominance estimation?

---

## 10. Quick Reference: Priority Items Across All Plans

The table below consolidates the top-priority items from all four plans into a
single decision list. Items marked 🔴 must be done before any session analysis
begins. Items marked 🟠 should be done before the paper submission deadline
(April 20 paper submission is past, but camera-ready is July 23).

| # | Priority | Item | Plan | Section | Effort |
|---|----------|------|------|---------|--------|
| 1 | 🔴 | ArUco ID allowlists (4 tools) | Gaze | §1 | Low |
| 2 | 🔴 | Fix camera zones cam1+cam3, cam2+cam4 | ROI | §6.2 | Trivial |
| 3 | 🔴 | Create `configs/scene_rois.yaml` | ROI | §1.2 | Low |
| 4 | 🔴 | Flip-consistency check cam1–4 in validator | Gaze | §9.1 | Low |
| 5 | 🔴 | Seat-based skeleton identity lock | ROI | §2.2 | Low |
| 6 | 🟠 | Load Tobii factory intrinsics from metadata | Gaze | §2 | Medium |
| 7 | 🟠 | Apply T_glasses_sceneCamera extrinsic | Gaze | §3 | Medium |
| 8 | 🟠 | 180° frame rotation before ArUco in tracker | Gaze | §4.4 | Low |
| 9 | 🟠 | Tablet touch log → confirmed gaze windows | Fusion | §2.2 | Low |
| 10 | 🟠 | Dominance sociometric matrix | Behavioral | §3.1 | Low |
| 11 | 🟠 | Familiarity + trust networks from postblock | Behavioral | §3.2–3.3 | Low |
| 12 | 🟠 | VAD trajectory features | Behavioral | §2.1 | Low |
| 13 | 🟠 | Response latency extraction | Behavioral | §1.2 | Low |
| 14 | 🟠 | T4 contribution profiling | Behavioral | §1.5 | Low |
| 15 | 🟠 | Tobii calibration event as gaze QC baseline | Behavioral | §5.1 | Low |
| 16 | 🟡 | IMU dead-reckoning for pose gap-filling | Fusion | §3.2 | Medium |
| 17 | 🟡 | Per-mic VAD from DPA audio | Fusion | §5.2 | Medium |
| 18 | 🟡 | Approach C confidence-weighted pose fusion | Gaze | §6 | Medium |
| 19 | 🟡 | 3D hand keypoint triangulation | ROI | §7.2 | Medium |
| 20 | 🟡 | Gesture detector + NDJSON output | ROI | §8.3 | Medium |
| 21 | 🟡 | Floor-holding and turn structure | Behavioral | §6.2 | Medium |
| 22 | 🟡 | Interruption detection | Behavioral | §6.1 | Medium |
| 23 | 🟢 | Weakly supervised gaze classifier | Fusion | §9 | High |
| 24 | 🟢 | Unified multimodal feature frame builder | Fusion | §10 | High |
| 25 | 🟢 | Motion entrainment analysis | Behavioral | §9.2 | High |

---

## Appendix: File Map

```
configs/
  calibration_charuco_merged.toml    Camera calibration (shared L1+L2)
  scene_rois.yaml                    Static 3D ROI definitions (shared L2+L3+L4)
  desk_markers_large.yaml            World frame + desk markers
  tobii_multicam_glasses_tracker.yaml
  offline_compute_stack.yaml         Camera zones (fix: cam1+cam3, cam2+cam4)

tools/
  calibrate_charuco.py               L1: ChArUco calibration
  tobii_multicam_glasses_tracker.py  L1: Approach A glasses pose
  tobii_multi_glasses_world_align.py L1: Approach B scene-video PnP
  multicam_pose3d.py                 L2: Skeleton triangulation
  refine_skeleton_3d.py              L2: Skeleton refinement
  face_hand_pipeline.py              L2: Hand keypoints (2D)
  [new] imu_dead_reckoning.py        L3: IMU pose gap-filling
  [new] audio_vad.py                 L3: Per-participant VAD
  [new] roi_fusion.py                L3: Bayesian ROI posterior
  [new] parse_tablet_logs.py         L3: Confirmed gaze windows
  [new] gesture_extractor.py         L2+L3: Gesture detection + VAD gate
  [new] multimodal_feature_builder.py L4: Unified per-frame feature frame
  [new] behavioral_features.py       L4: VAD trajectories, dominance, trust

qc/
  qc_tobii_world_gaze.py            L1 QC: gaze scatter + accuracy
  qc_skeleton_roi.py                L2 QC: skeleton coverage
  [new] qc_session_report.py        Unified per-session QC report
```

---

*Master plan generated 2026-04-19.*
*Sub-plans: `3d_gaze_improvement_plan.md`, `3d_roi_gesture_skeleton_plan.md`,*
*`multimodal_fusion_3d_enhancement_plan.md`, `additional_behavioral_signals_plan.md`.*
*Review this document at the start of each work sprint to reprioritise.*
