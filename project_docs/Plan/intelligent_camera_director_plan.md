# AffectAI — Intelligent Camera Director: Automated Cinematography for Group Conversation

> **Purpose**: This document describes the design and implementation of an
> automated camera director system that uses the multimodal behavioral signals
> from the AffectAI analysis pipeline to select the optimal camera view, zoom
> level, and shot composition at each moment in a recorded group conversation.
>
> **Input**: The unified multimodal feature frames produced by Layers 1–4 of
> the master analysis plan, plus raw video from all 7 fixed cameras and 4
> Tobii scene cameras.
>
> **Output**: A dynamically directed video that presents the group conversation
> in the most cinematographically informative way — automatically cutting to the
> speaker, zooming to emphasize social dynamics, and composing multi-person
> shots when interaction structure requires it.
>
> **This is not real-time**: The director operates in post-processing over
> already-recorded sessions. The output is a produced video file, not a live
> feed.

---

## Table of Contents

1. [Motivation and Scope](#1-motivation-and-scope)
2. [Available Camera Inventory](#2-available-camera-inventory)
3. [Shot Grammar — Types of Shots](#3-shot-grammar--types-of-shots)
4. [Behavioral Signals That Drive Shot Selection](#4-behavioral-signals-that-drive-shot-selection)
5. [The Shot Selection Engine](#5-the-shot-selection-engine)
6. [Virtual Camera Control — Zoom and Crop](#6-virtual-camera-control--zoom-and-crop)
7. [Cut Timing and Pacing](#7-cut-timing-and-pacing)
8. [Multi-Person Composition Rules](#8-multi-person-composition-rules)
9. [Scene-Specific Rules by Task Phase](#9-scene-specific-rules-by-task-phase)
10. [Director Output Format](#10-director-output-format)
11. [Rendering Pipeline](#11-rendering-pipeline)
12. [Quality Control and Manual Override](#12-quality-control-and-manual-override)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. Motivation and Scope

### 1.1 The Problem with Static Multi-Camera Output

The AffectAI setup records seven simultaneous fixed-camera views. Raw footage
from any single camera is adequate for sensor-level analysis but poor for human
viewing and annotation. Problems include:

- The speaker is often a small figure in a wide-angle group shot
- Facial expressions — essential for affect coding — are unreadable at full-room scale
- Fixed cameras mounted for technical coverage (not cinematography) have awkward
  compositions: participants at frame edges, backs of heads, upside-down orientations
- Switching between cameras manually for a 75-minute session is labour-intensive
  and introduces annotator bias

### 1.2 What the Director Does

The director is a post-processing pipeline that:

1. **Reads** the multimodal feature stream (who is speaking, where everyone is
   looking, what gestures are occurring, which task phase is active)
2. **Decides** which camera, crop, and zoom level best represents each moment
3. **Plans** smooth transitions between shots (cuts, zooms, pans within a crop)
4. **Renders** a final directed video at broadcast quality

The result is a video that feels as if a human camera operator was present,
directing attention to the most behaviorally informative content at each moment.

### 1.3 What the Director Does Not Do

- It does not alter the actual recorded footage (no artificial camera movement
  beyond digital crop/zoom within a single frame)
- It does not perform aesthetic colour grading
- It does not add any artificial audio — the directed video uses the original
  room microphone or close-talk audio mix
- It is not real-time; it is a post-processing step over stored data

### 1.4 Dual Use

The directed video serves two purposes:

**Research use**: Annotation of social behaviors (dominance, turn-taking, affect)
is substantially more accurate and faster when annotators see close-up,
well-composed shots of the relevant participants rather than fixed-camera wide shots.

**Dissemination use**: For papers, presentations, and ethics review, a short
directed excerpt demonstrates the dataset and the behaviors of interest far more
compellingly than raw footage.

---

## 2. Available Camera Inventory

### 2.1 Fixed Cameras — Physical Reality

The lab layout must be understood precisely before assigning cameras to shot
types. The seating arrangement and camera positions from
`docs/camera_layout_and_positions.md`:

```
                    ┌─────────────────────────────┐
                    │        BIG SCREEN            │  ← y = +back
                    └──────────────┬──────────────┘
                                   │
         cam6 (back-center, 0.88m) ─────────────────────── cam6
         ELEVATED rear overview → sees ALL 4 faces from behind-ish

              cam4 ┐                         ┌ cam3
              (left │    P4(BL)   P1(BR)     │ (right
               back)│                        │  back)
                    │       [D E S K]        │
                    │                        │
              cam1 ┘    P3(FL)   P2(FR)     └ cam2
              (left       ↑                   (right
               front)     │                   front)
                           │
         cam7 (P50, front-center, 0.90m) ────────────── cam7
         TABLE-LEVEL front overview → sees ALL 4 faces from front

         cam5 (front-left-low, 0.0m) ─── table surface / hand area

Seating: P1=back-right  P2=front-right  P3=front-left  P4=back-left
         BR=back-right  FR=front-right  FL=front-left  BL=back-left
```

From `configs/desk_markers_large.yaml`:

| ID | Label | Position (m) | Orientation | Directing role |
|----|-------|-------------|-------------|---------------|
| cam1 | left_front_middle | [−0.9, −0.2, 0.88] | **Upside-down** | Side view: P3(FL) face + P4(BL) partial |
| cam2 | right_front_middle | [+0.9, −0.2, 0.88] | **Upside-down** | Side view: P4(BL) face + P3(FL) partial |
| cam3 | right_back_middle | [+0.9, +0.2, 0.88] | **Upside-down** | Side view: P1(BR) face + P2(FR) partial |
| cam4 | left_back_middle | [−0.9, +0.2, 0.88] | **Upside-down** | Side view: P2(FR) face + P1(BR) partial |
| cam5 | front_left_low | [−0.9, −0.4, 0.0] | Upright | **Table-surface level: hands, tablets, desk items** |
| cam6 | back_center | [0.0, +0.4, 0.88] | Upright | **Elevated rear overview: all 4 participants, faces visible** |
| cam7 | front_center_p50 | [0.0, −0.4, 0.90] | Upright (PanaCast 50) | **Table-level front: all 4 faces, wide-angle safety shot** |

All cameras: 1920×1080 @ 30 fps. cam1–cam4 must be rotated 180° in post-processing.

### 2.2 Understanding the Three Camera Tiers

**Tier 1 — Side cameras (cam1–cam4): Individual close-ups**

These four upside-down cameras are mounted at ~88 cm height on each side of the
desk. Each camera sees two participants — one whose face is clear and close
(primary subject), and one whose shoulder/profile is visible (secondary subject).
The side angle makes them excellent for tight individual shots but not for
group composition.

```
cam1 (left-front):  primary=P3(FL) face visible, secondary=P4(BL) side profile
cam2 (right-front): primary=P4(BL) face visible, secondary=P3(FL) side profile
cam3 (right-back):  primary=P1(BR) face visible, secondary=P2(FR) side profile
cam4 (left-back):   primary=P2(FR) face visible, secondary=P1(BR) side profile

Key insight: cam3 and cam4 see the BACK ROW faces (P1, P2).
             cam1 and cam2 see the FRONT ROW faces (P3, P4).
             Within each row, the two side cameras give opposite angles.
```

**Tier 2 — Overview cameras (cam6, cam7): Safe group shots**

These are the two most important cameras for the director. Both provide
reliable views of all four participants simultaneously and are always safe
to cut to.

- **cam6** (back-center, elevated at 0.88 m): mounted at the rear of the desk,
  looking toward the front wall and screen. From this position it sees all four
  participants from a slightly elevated angle. Front-row participants (P3, P4)
  are seen nearly face-on; back-row participants (P1, P2) are seen from a
  3/4 angle. Good for establishing shots and group dynamics.

- **cam7** (front-center, PanaCast 50, table-level at 0.90 m): mounted at the
  front of the room looking toward the back (toward the big screen). It sees all
  four participants' faces directly. Its wide 133° HFOV means all participants
  are always in frame. This is the **primary safety camera** — when in doubt,
  cut to cam7. The table-level position gives a natural conversation-level
  perspective, similar to how a camera placed at the center of the table would look.

**Tier 3 — Surface camera (cam5): Hands and tablets only**

cam5 is at desk level (0.0 m height), providing a table-surface perspective.
It does not show faces reliably but is excellent for INSERT shots showing
hands, tablets, written materials, and contribution forms.

### 2.3 Revised Camera Quality Ratings for Directing

| Camera | Tier | All-participant view | Best shot type | Face quality | Directing priority |
|--------|------|---------------------|----------------|-------------|-------------------|
| **cam6** | Overview | ✓✓ Elevated rear, all faces | GS, 2S (front pair), REACT | Good–Excellent | **Primary group** |
| **cam7** | Overview | ✓✓ Table-level front, all faces | GS, 2S, establishing | Good (wide) | **Safety/default** |
| cam3 | Side | P1 face (excellent), P2 partial | CU, MCU, OTS for P1 | Excellent for P1 | Individual P1 |
| cam4 | Side | P2 face (excellent), P1 partial | CU, MCU, OTS for P2 | Excellent for P2 | Individual P2 |
| cam1 | Side | P3 face (excellent), P4 partial | CU, MCU, OTS for P3 | Excellent for P3 | Individual P3 |
| cam2 | Side | P4 face (excellent), P3 partial | CU, MCU, OTS for P4 | Excellent for P4 | Individual P4 |
| cam5 | Surface | None (table-level) | INSERT only | None | Hands/tablets |
| tobii_pN | POV | First-person scene | POV, insert variety | Natural/POV | Variety shots |

### 2.4 Camera-to-Participant Face Mapping (Corrected)

This is the authoritative mapping used by the shot selection engine. A camera
is rated **primary** if the participant's face fills ≥30% of the frame at
natural crop, **secondary** if the face is visible but angled or smaller:

```
           cam1    cam2    cam3    cam4    cam5    cam6    cam7
P1 (BR)    ✗       ✗       ✓✓      ✓       ~hands  ✓       ✓
P2 (FR)    ✗       ✗       ✓       ✓✓      ~hands  ✓       ✓
P3 (FL)    ✓✓      ✓       ✗       ✗       ~hands  ✓       ✓
P4 (BL)    ✓       ✓✓      ✗       ✗       ~hands  ✓       ✓

✓✓ = primary face shot (clean front/3-quarter view, expressible)
✓  = secondary face shot (usable, angled)
✗  = face not visible or back-of-head
~  = hands/arms visible, no face
```

**Key corrections from previous version:**
- cam5 is NOT "table overview showing all participants" — it is desk-surface
  level and primarily shows hands and tablet screens
- cam6 is NOT "rear view with only backs of heads" — it is elevated and
  sees all faces from a 3/4 or frontal angle
- cam7 is NOT "too wide for individual shots" — it is at table-level and
  shows all faces; it works as the reliable group shot even when cropped to
  show two participants
- cam1 and cam2 see FRONT ROW faces (P3, P4), not back row
- cam3 and cam4 see BACK ROW faces (P1, P2), not front row

### 2.5 Egocentric Cameras (Tobii Scene Cameras)

Each Tobii Pro Glasses 3 has a scene camera providing a first-person view from
the participant's head position. This is an unconventional but powerful shot
type — a first-person view showing exactly what the participant sees.

| ID | Participant | Seat | Typical scene content | Resolution |
|----|------------|------|----------------------|-----------|
| tobii_p1 | P1 | back-right | Desk, P3 and P4 faces, big screen | 1920×1080 @ 25 fps |
| tobii_p2 | P2 | front-right | Desk, P4 and P1 faces, big screen | 1920×1080 @ 25 fps |
| tobii_p3 | P3 | front-left | Desk, P1 and P2 faces, big screen | 1920×1080 @ 25 fps |
| tobii_p4 | P4 | back-left | Desk, P2 and P3 faces, big screen | 1920×1080 @ 25 fps |

The scene camera's gaze overlay (from Tobii's SDK) can optionally be rendered
as a dot on the POV shot, showing exactly where the participant's eyes are
directed at that moment.

---

## 3. Shot Grammar — Types of Shots

The director uses a constrained vocabulary of shot types drawn from documentary
and social-science video conventions. Each shot type is defined by its **subject**
(who), **camera** (which physical camera or virtual crop), and **framing**
(what part of the frame).

### 3.1 Shot Type Definitions

```
SHOT TYPES
──────────────────────────────────────────────────────────────────────

CU   Close-Up        Single participant, face fills 60–80% of frame height.
                     Best for high-affect moments, final decisions, reactions.
                     Source: cam1–4 (cropped), or Tobii scene camera.

MCU  Medium Close-Up Single participant, head and shoulders.
                     Standard "talking head" shot during speech.
                     Source: cam1–4 (light crop).

MS   Medium Shot     Single participant, head to waist. Shows hand gestures.
                     Source: cam1–4 (wide crop) or cam5 (if participant visible).

OTS  Over-the-Shoulder Two participants: listener in foreground (partial),
                     speaker in background (full face).
                     Best for showing listener reaction during argument or disclosure.
                     Source: cam6 (sees all faces, crop to a pair) or cam1–cam4
                     (side cameras when two participants face each other).

2S   Two-Shot        Two participants visible simultaneously, both faces readable.
                     Best for dyadic exchange in T2 negotiation or mutual gaze.
                     Source: cam6 (crop to front or back pair) or cam7 (crop to
                     left or right pair) — both cameras always show all faces.

GS   Group Shot      All four participants visible simultaneously.
                     Used for task transitions, decision moments, silent phases.
                     Source: cam7 (full frame — primary group shot, table-level
                     perspective shows all faces naturally) or cam6 (slightly
                     elevated rear view, also shows all faces).

POV  Point of View   Participant's own Tobii scene camera (first-person view).
                     Shows what the participant sees, optionally with gaze dot overlay.
                     Use sparingly — at most once per 30 seconds.
                     Source: tobii_p1 – tobii_p4.

INSERT Table insert  Close-up of desk surface: tablet screen, hands, written
                     notes, contribution form, or physical materials.
                     Source: cam5 (dedicated table-level camera — only camera
                     optimised for this angle; shows tablet screens and hands clearly).

REACT Reaction shot  Cut to a listener's face while speaker audio continues.
                     Source: cam6 or cam7 (crop to listener) for any participant;
                     cam1–cam4 (individual side cameras) for primary-face participants.
```

### 3.2 Shot Size Reference

```
Frame height occupied by subject head:
  CU:  > 60%   face clearly readable, individual pores visible
  MCU: 30–60%  face readable, shoulder line visible
  MS:  15–30%  full upper body, hand gestures visible
  GS:  < 15%   all persons visible, no individual face readable
```

### 3.3 Shot Priority Hierarchy

When multiple shot types could apply at the same moment, the director resolves
conflicts using this priority order (highest to lowest):

```
1. CU  — active speaker with high arousal (VAD arousal > 7) or
          decision moment (final form submission, LSL decision marker)
2. MCU — active speaker, normal conditions
3. OTS — listener showing strong reaction (head nod, lean-forward, frown)
4. REACT — secondary speaker or active listener during long speaker turn
5. 2S  — dyadic exchange (same two people alternating turns)
6. MS  — speaker using prominent hand gesture (pointing, iconic)
7. INSERT — hand on tablet / form being filled
8. POV — speaker's point of view (used for variety, max 1 per 30s)
9. GS  — silence, phase transition, no clear speaker
```

---

## 4. Behavioral Signals That Drive Shot Selection

The director reads these signals from the multimodal feature stream
(produced by Layers 1–4 of the master plan). Each signal maps to
one or more shot selection decisions.

### 4.1 Signal → Shot Mapping

| Signal | Source | Shot decision |
|--------|--------|---------------|
| `speaking[pid]` = True | Audio VAD | Cut to MCU/CU of `pid` |
| `arousal[pid]` > 7 | VAD probe | Upgrade current shot to CU |
| `valence_delta[pid]` > 2 | VAD trajectory | Hold on this participant (affective shift) |
| `head_nod[pid]` | IMU detector | Cut to REACT of `pid` |
| `lean_forward[pid]` | Skeleton | Upgrade to MCU (interest signal) |
| `pointing_gesture[pid]` | Gesture detector | Cut to MS to show gesture |
| `tablet_interaction[pid]` | Tablet log | INSERT shot of `pid` tablet zone |
| `gaze_target[pid]` = `zone_Pj` | Gaze ROI | OTS from `pid`'s side showing `Pj` |
| `mutual_gaze(Pi, Pj)` | Gaze | 2S of Pi and Pj |
| `joint_attention(roi, n≥3)` | Gaze | GS showing all looking at same target |
| `n_speakers` > 1 (overlap) | VAD | Stay on previous speaker; show both if 2S available |
| `phase_transition` | LSL event | GS establishing shot (2–4 seconds) |
| `task_decision` | LSL marker | CU of person filling in form / moderator log |
| `silence` > 3s | VAD | GS or slow zoom-out |
| `tobii_calibration` | LSL marker | GS (all looking at screen center) |

### 4.2 Speaker Detection Confidence Levels

The director maintains a confidence level for the current speaker identity:

```python
class SpeakerConfidence(Enum):
    HIGH    = "high"    # single mic active, no overlap, VAD clear
    MEDIUM  = "medium"  # mild overlap or brief cross-talk
    LOW     = "low"     # heavy overlap or VAD ambiguous
    NONE    = "none"    # silence (> 500ms all mics below threshold)
```

Shot selection degrades gracefully with confidence:

```
HIGH   → MCU or CU of identified speaker
MEDIUM → MCU of most likely speaker, smaller crop (safety margin)
LOW    → 2S of the overlapping participants
NONE   → Hold current shot; if silence > 3s, fade to GS
```

### 4.3 Reaction Priority Score

Not all listeners are equally worth cutting to for a reaction shot. Compute
a **reaction priority score** for each non-speaking participant:

```python
def reaction_priority(
    pid: str,
    feature_frame: dict,
    speaker_pid: str,
) -> float:
    """
    Score in [0, 1]. Higher = more worth cutting to for a reaction shot.
    """
    data = feature_frame["participants"][pid]
    score = 0.0

    # Is this participant looking at the speaker?
    if data["gaze_roi_fused"] == f"zone_{speaker_pid}":
        score += 0.30    # engaged listener

    # Is this participant nodding?
    if data.get("head_nod_active", False):
        score += 0.25    # strong agreement signal

    # High arousal (may be emotional reaction)
    arousal = data.get("mean_vad_arousal", 5.0)
    score += 0.15 * max(0, (arousal - 5.0) / 4.0)

    # Leaning forward (physical engagement)
    if data.get("lean_forward", False):
        score += 0.15

    # Is this participant about to speak? (VAD rising)
    if data.get("vad_rising", False):
        score += 0.15    # pre-turn engagement

    return min(score, 1.0)
```

---

## 5. The Shot Selection Engine

### 5.1 Architecture

The shot selection engine runs as a frame-by-frame state machine over the
multimodal feature stream. It maintains a **current shot state** and evaluates
whether to cut, hold, or transition at each frame.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

@dataclass
class ShotState:
    shot_type: str              # CU, MCU, MS, OTS, 2S, GS, POV, INSERT, REACT
    camera_id: str              # cam1–cam7 or tobii_p1–tobii_p4
    subject_pids: list[str]     # which participants are the subject
    crop_box: tuple             # (x, y, w, h) in pixel coordinates, or None for full frame
    zoom_level: float           # 1.0 = full frame, 2.0 = 2x zoom crop
    onset_frame: int            # frame index when this shot began
    duration_frames: int        # how long this shot has lasted
    confidence: float           # 0–1, how certain the director is this is the right shot
    reason: str                 # human-readable reason for this shot (for QC)


class Director:
    def __init__(
        self,
        feature_frames: list[dict],
        camera_specs: dict,
        scene_rois: dict,
        fps: float = 30.0,
    ):
        self.frames = feature_frames
        self.cameras = camera_specs
        self.rois = scene_rois
        self.fps = fps
        self.current_shot: Optional[ShotState] = None
        self.shot_history: list[ShotState] = []
        self.pov_last_used_frame: int = -999   # throttle POV shots

    def run(self) -> list[ShotState]:
        """Process all frames and return shot list."""
        shots = []
        for fi, frame in enumerate(self.frames):
            shot = self._select_shot(fi, frame)
            if self._should_cut(shot, fi):
                self._finalize_current_shot(fi)
                self.current_shot = shot
                self.current_shot.onset_frame = fi
            else:
                # Continue current shot, update confidence
                self.current_shot.duration_frames += 1
            shots.append(self.current_shot)
        return shots

    def _select_shot(self, frame_idx: int, frame: dict) -> ShotState:
        """Core shot selection logic."""
        speakers = self._get_speakers(frame)
        n_speakers = len(speakers)
        task_phase = frame.get("task_phase", "unknown")

        # Phase transition → establishing group shot
        if frame.get("phase_just_changed", False):
            return self._make_group_shot("phase_transition")

        # Task decision moment → CU of decision-maker
        if frame.get("task_decision_active", False):
            decision_pid = frame.get("decision_participant")
            if decision_pid:
                return self._make_close_up(decision_pid, "task_decision")

        # High arousal speaker → CU
        if n_speakers == 1:
            spk = speakers[0]
            arousal = frame["participants"][spk].get("mean_vad_arousal", 5.0)
            if arousal > 7.0:
                return self._make_close_up(spk, "high_arousal_speaker")

            # Prominent gesture → MS
            if frame["participants"][spk].get("rh_state") in ("point", "iconic"):
                return self._make_medium_shot(spk, "gesture")

            # Tablet interaction → INSERT
            if frame["participants"][spk].get("tablet_confirmed", False):
                return self._make_insert(spk, "tablet_interaction")

            # Check for high-priority reaction
            best_react_pid, react_score = self._best_reaction(spk, frame)
            if react_score > 0.55 and self.current_shot.duration_frames > int(3 * self.fps):
                return self._make_reaction(best_react_pid, spk, "strong_reaction")

            # Standard speaker shot
            return self._make_mcu(spk, "active_speaker")

        # Two speakers (overlap) → 2S
        if n_speakers == 2:
            return self._make_two_shot(speakers[0], speakers[1], "speaker_overlap")

        # Mutual gaze between two participants → 2S
        mutual = frame.get("mutual_gaze_pairs", [])
        if mutual:
            pi, pj = mutual[0]
            return self._make_two_shot(pi, pj, "mutual_gaze")

        # Joint attention (3+ looking at same ROI) → GS
        ja = frame.get("joint_attention", [])
        if ja and ja[0]["n_participants"] >= 3:
            return self._make_group_shot("joint_attention")

        # Silence → hold or fade to GS
        return self._handle_silence(frame_idx, frame)

    def _should_cut(self, proposed: ShotState, frame_idx: int) -> bool:
        """Determine whether to cut to the proposed shot."""
        if self.current_shot is None:
            return True

        # Never cut before minimum hold duration
        min_hold = self._min_hold_frames(self.current_shot.shot_type)
        if self.current_shot.duration_frames < min_hold:
            return False

        # Always cut on phase transition
        if proposed.reason == "phase_transition":
            return True

        # Cut if shot type changes meaningfully
        if proposed.shot_type != self.current_shot.shot_type:
            return True

        # Cut if subject changes
        if set(proposed.subject_pids) != set(self.current_shot.subject_pids):
            return True

        # Cut if proposed confidence significantly higher
        if proposed.confidence > self.current_shot.confidence + 0.3:
            return True

        return False

    def _min_hold_frames(self, shot_type: str) -> int:
        """Minimum frames before a shot can be cut away."""
        MIN_HOLD = {
            "CU":     int(2.0 * self.fps),   # 2 seconds
            "MCU":    int(2.5 * self.fps),   # 2.5 seconds
            "MS":     int(2.0 * self.fps),
            "OTS":    int(3.0 * self.fps),
            "2S":     int(3.0 * self.fps),
            "GS":     int(4.0 * self.fps),   # group shots need time
            "POV":    int(1.5 * self.fps),
            "INSERT": int(1.5 * self.fps),
            "REACT":  int(1.5 * self.fps),
        }
        return MIN_HOLD.get(shot_type, int(2.0 * self.fps))
```

### 5.2 Camera Selection per Shot Type

Given a target shot type and subject participants, select the best physical
camera:

```python
def select_camera_for_shot(
    shot_type: str,
    subject_pids: list[str],
    camera_coverage: dict,    # {cam_id: [pids visible from this camera]}
    skeleton_3d: np.ndarray,  # (4, 25, 7) current frame
    camera_specs: dict,
) -> tuple[str, tuple]:
    """
    Returns (camera_id, crop_box).
    crop_box = (x, y, w, h) in pixels, or None for full frame.
    """
    # Camera preference tables per shot type and primary subject.
    # Based on corrected camera layout:
    #   cam3+cam4 see BACK ROW (P1=BR, P2=FR)
    #   cam1+cam2 see FRONT ROW (P3=FL, P4=BL)
    #   cam6+cam7 see ALL participants and are always safe choices
    #   cam5 = desk surface / hands only
    CAMERA_PREFERENCE = {
        "P1": {   # back-right seat
            "CU":    ["cam3", "cam4"],     # cam3 is primary face, cam4 secondary
            "MCU":   ["cam3", "cam4"],
            "REACT": ["cam3", "cam4"],
            "OTS":   ["cam6", "cam7"],     # overview cameras crop to P1 + interlocutor
            "MS":    ["cam3", "cam6"],
        },
        "P2": {   # front-right seat
            "CU":    ["cam4", "cam3"],     # cam4 is primary face, cam3 secondary
            "MCU":   ["cam4", "cam3"],
            "REACT": ["cam4", "cam3"],
            "OTS":   ["cam6", "cam7"],
            "MS":    ["cam4", "cam6"],
        },
        "P3": {   # front-left seat
            "CU":    ["cam1", "cam2"],     # cam1 is primary face, cam2 secondary
            "MCU":   ["cam1", "cam2"],
            "REACT": ["cam1", "cam2"],
            "OTS":   ["cam6", "cam7"],
            "MS":    ["cam1", "cam6"],
        },
        "P4": {   # back-left seat
            "CU":    ["cam2", "cam1"],     # cam2 is primary face, cam1 secondary
            "MCU":   ["cam2", "cam1"],
            "REACT": ["cam2", "cam1"],
            "OTS":   ["cam6", "cam7"],
            "MS":    ["cam2", "cam6"],
        },
    }

    if shot_type == "GS":
        # cam7 (table-level front, PanaCast 50) is the primary group shot —
        # natural conversation-level perspective, all faces visible.
        # cam6 (elevated rear) is the secondary option for variety.
        return "cam7", None

    if shot_type == "INSERT":
        # cam5 is the only camera optimised for desk-surface content
        return "cam5", None

    if len(subject_pids) == 1:
        pid = subject_pids[0]
        prefs = CAMERA_PREFERENCE.get(pid, {}).get(shot_type, ["cam7"])
        for cam_id in prefs:
            if pid in camera_coverage.get(cam_id, []):
                crop = compute_tight_crop(cam_id, pid, shot_type, skeleton_3d)
                return cam_id, crop
        # Fallback: best available camera
        return "cam7", None

    if len(subject_pids) == 2:
        # Two-shot: cam6 and cam7 both always see all participants and can
        # be cropped to any pair. Prefer cam6 for back-row pairs (P1+P2),
        # cam7 for front-row or cross-table pairs. Both are valid.
        pid_set = set(subject_pids)
        if pid_set == {"P1", "P2"}:
            # Back row pair: cam6 (elevated rear) gives best 2-person composition
            crop = compute_two_person_crop("cam6", subject_pids[0], subject_pids[1],
                                           skeleton_3d, camera_calibration)
            return "cam6", crop
        elif pid_set == {"P3", "P4"}:
            # Front row pair: cam6 or cam7, cam7 at table level is natural
            crop = compute_two_person_crop("cam7", subject_pids[0], subject_pids[1],
                                           skeleton_3d, camera_calibration)
            return "cam7", crop
        else:
            # Cross-table pair (e.g. P1+P3, P2+P4): cam7 full width crop works best
            crop = compute_two_person_crop("cam7", subject_pids[0], subject_pids[1],
                                           skeleton_3d, camera_calibration)
            return "cam7", crop
    return "cam7", None
```

---

## 6. Virtual Camera Control — Zoom and Crop

Since all cameras are physically fixed, "zooming" and "panning" are achieved
by cropping a sub-region of the 1920×1080 frame and scaling it up to the output
resolution. The skeleton and gaze data provide the 2D coordinates needed to
place the crop precisely.

### 6.1 Crop Calculation from 3D Skeleton

Project the participant's 3D keypoints into 2D camera coordinates to find the
bounding box:

```python
import cv2
import numpy as np

def project_keypoints_to_camera(
    skeleton_kps_world: np.ndarray,   # (25, 7) in world frame
    camera_matrix: np.ndarray,         # 3×3 K
    rvec: np.ndarray,                  # Rodrigues rotation
    tvec: np.ndarray,                  # translation
    dist_coeffs: np.ndarray,
    valid_kps: set = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18}),
) -> np.ndarray:
    """
    Project upper-body keypoints to pixel coordinates.
    Returns (N, 2) array of valid pixel coordinates.
    """
    pts = []
    for idx in valid_kps:
        kp = skeleton_kps_world[idx]
        if np.any(np.isnan(kp[:3])) or kp[3] < 0.3:
            continue
        pts.append(kp[:3])
    if not pts:
        return np.empty((0, 2))

    obj_pts = np.array(pts, dtype=np.float64)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    return proj.reshape(-1, 2)


def compute_tight_crop(
    cam_id: str,
    pid: str,
    shot_type: str,
    skeleton_3d: np.ndarray,    # (4, 25, 7)
    camera_calibration: dict,
    output_res: tuple = (1920, 1080),
    padding_fraction: float = 0.25,
) -> tuple[int, int, int, int]:
    """
    Compute (x, y, w, h) crop in the source camera frame for the given
    participant and shot type.

    padding_fraction: extra space around the keypoint bounding box.
    """
    PERSON_IDX = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    p_idx = PERSON_IDX[pid]
    kps = skeleton_3d[p_idx]

    # Shot-type-specific keypoint sets
    KPS_BY_SHOT = {
        "CU":    {0, 15, 16, 17, 18},           # face only
        "MCU":   {0, 1, 2, 5, 15, 16, 17, 18},  # face + shoulders
        "MS":    {0, 1, 2, 3, 4, 5, 6, 7},       # head + arms
        "REACT": {0, 1, 15, 16, 17, 18},         # face + neck
    }
    valid_kps = KPS_BY_SHOT.get(shot_type, {0, 1, 2, 5})

    cam = camera_calibration[cam_id]
    projected = project_keypoints_to_camera(
        kps, cam["K"], cam["rvec"], cam["tvec"], cam["dist"],
        valid_kps=valid_kps,
    )

    if len(projected) < 2:
        # Fallback: center of frame
        return (0, 0, 1920, 1080)

    x_min, y_min = projected.min(axis=0)
    x_max, y_max = projected.max(axis=0)

    # Add padding
    pw = (x_max - x_min) * padding_fraction
    ph = (y_max - y_min) * padding_fraction
    x_min = max(0,    int(x_min - pw))
    y_min = max(0,    int(y_min - ph))
    x_max = min(1920, int(x_max + pw))
    y_max = min(1080, int(y_max + ph))

    # Enforce output aspect ratio (16:9)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    target_ar = output_res[0] / output_res[1]
    current_ar = crop_w / max(crop_h, 1)

    if current_ar > target_ar:
        # Too wide: increase height
        new_h = int(crop_w / target_ar)
        pad = (new_h - crop_h) // 2
        y_min = max(0, y_min - pad)
        crop_h = min(1080 - y_min, new_h)
    else:
        # Too tall: increase width
        new_w = int(crop_h * target_ar)
        pad = (new_w - crop_w) // 2
        x_min = max(0, x_min - pad)
        crop_w = min(1920 - x_min, new_w)

    return (x_min, y_min, crop_w, crop_h)
```

### 6.2 Smooth Crop Interpolation

When the crop target changes (subject moves, shot type transitions gradually),
interpolate the crop box smoothly to avoid jarring digital pans:

```python
def interpolate_crop(
    crop_start: tuple,
    crop_end: tuple,
    n_frames: int,
    ease: str = "ease_in_out",
) -> list[tuple]:
    """
    Smooth crop interpolation over n_frames.
    Returns list of (x, y, w, h) for each frame.
    """
    def ease_in_out(t):
        return t * t * (3 - 2 * t)

    def linear(t):
        return t

    ease_fn = ease_in_out if ease == "ease_in_out" else linear
    crops = []
    for i in range(n_frames):
        t = ease_fn(i / max(n_frames - 1, 1))
        x = int(crop_start[0] + t * (crop_end[0] - crop_start[0]))
        y = int(crop_start[1] + t * (crop_end[1] - crop_start[1]))
        w = int(crop_start[2] + t * (crop_end[2] - crop_start[2]))
        h = int(crop_start[3] + t * (crop_end[3] - crop_start[3]))
        crops.append((x, y, w, h))
    return crops
```

### 6.3 Within-Shot Subject Tracking

If the subject moves within the frame during a shot (head turns, leans),
the crop center should follow them smoothly. Apply a **low-pass filter**
on the crop center to avoid twitchy camera movement:

```python
class CropTracker:
    """
    Maintains a smoothed crop box that follows a subject
    across frames within a single shot.
    """
    def __init__(self, initial_crop: tuple, alpha: float = 0.08):
        """alpha: smoothing factor. Lower = smoother but more lag."""
        self.x, self.y, self.w, self.h = map(float, initial_crop)
        self.alpha = alpha

    def update(self, target_crop: tuple) -> tuple[int, int, int, int]:
        """Exponential moving average toward target crop."""
        tx, ty, tw, th = map(float, target_crop)
        self.x += self.alpha * (tx - self.x)
        self.y += self.alpha * (ty - self.y)
        self.w += self.alpha * (tw - self.w)
        self.h += self.alpha * (th - self.h)
        return (int(self.x), int(self.y), int(self.w), int(self.h))
```

---

## 7. Cut Timing and Pacing

### 7.1 Minimum and Maximum Hold Durations

Professional documentary editing conventions, adapted for social science footage:

| Shot type | Min hold | Max hold before forced variety |
|-----------|----------|-------------------------------|
| CU | 2.0 s | 8.0 s (then cut to REACT or 2S) |
| MCU | 2.5 s | 12.0 s |
| MS | 2.0 s | 10.0 s |
| OTS | 3.0 s | 15.0 s |
| 2S | 3.0 s | 20.0 s |
| GS | 4.0 s | 6.0 s (establishing shots are brief) |
| POV | 1.5 s | 4.0 s (disorienting if too long) |
| INSERT | 1.5 s | 5.0 s |
| REACT | 1.5 s | 4.0 s |

### 7.2 Cut on Natural Boundaries

Prefer to cut at natural speech boundaries rather than mid-word:

```python
def find_next_speech_boundary(
    vad_signal: np.ndarray,    # binary VAD for the active speaker
    current_frame: int,
    max_wait_frames: int,      # don't wait longer than this
    fps: float = 30.0,
) -> int:
    """
    Find the next frame where the speaker pauses (VAD drops to 0)
    within max_wait_frames. Returns frame index for the cut.
    If no pause found, return current_frame + max_wait_frames.
    """
    for fi in range(current_frame, current_frame + max_wait_frames):
        if fi >= len(vad_signal):
            break
        if vad_signal[fi] == 0:    # speaker paused
            return fi
    return current_frame + max_wait_frames


def find_next_breath_pause(
    vad_signal: np.ndarray,
    current_frame: int,
    max_wait_s: float = 1.0,
    min_pause_frames: int = 3,
    fps: float = 30.0,
) -> int:
    """
    Find the next natural breath pause (VAD=0 for >= min_pause_frames).
    More refined than find_next_speech_boundary.
    """
    max_wait = int(max_wait_s * fps)
    consecutive_silence = 0
    for fi in range(current_frame, current_frame + max_wait):
        if fi >= len(vad_signal):
            break
        if vad_signal[fi] == 0:
            consecutive_silence += 1
            if consecutive_silence >= min_pause_frames:
                return fi - min_pause_frames   # start of pause
        else:
            consecutive_silence = 0
    return current_frame + max_wait
```

### 7.3 Pace Modulation by Task Phase

Different task phases have different natural energy levels. The director
adjusts its cutting pace to match:

```python
PHASE_PACING = {
    # Task phase         : (min_hold_multiplier, max_cut_rate_per_min)
    "T1_information_distribution": (1.5,  8),   # slow, analytical
    "T1_discussion":               (1.0, 15),   # medium
    "T1_decision":                 (0.8, 18),   # faster, tension
    "T2_brief":                    (1.5,  6),   # slow setup
    "T2_negotiation":              (0.9, 20),   # active negotiation
    "T2_settlement":               (0.7, 22),   # high stakes
    "T3_silent_generation":        (2.0,  4),   # very slow (no speech)
    "T3_round_robin":              (0.9, 18),
    "T3_group_selection":          (0.8, 20),
    "T4_contribution":             (2.0,  5),   # private, slow
    "T4_outcome_reveal":           (0.6, 25),   # emotional peak
    "T4_discussion":               (1.0, 15),
}
```

---

## 8. Multi-Person Composition Rules

### 8.1 Rule of Thirds for Subject Placement

When the director selects a crop, the subject's eye-line should align with
the upper rule-of-thirds line. Adjust crop vertically to achieve this:

```python
def apply_rule_of_thirds(
    crop: tuple,                     # (x, y, w, h)
    eye_midpoint_px: tuple,          # (x, y) in crop-relative coords
    target_vertical_fraction: float = 0.33,   # upper third
) -> tuple[int, int, int, int]:
    """
    Shift crop so that eye_midpoint_px sits at target_vertical_fraction
    of the crop height.
    """
    x, y, w, h = crop
    target_y_in_crop = int(h * target_vertical_fraction)
    delta_y = eye_midpoint_px[1] - target_y_in_crop
    new_y = max(0, y + delta_y)
    return (x, new_y, w, h)
```

### 8.2 Look-Room — Subject Faces Direction of Conversation

When a participant faces another person, leave space on the side they're
facing (look-room). This is the most common composition error in fixed-camera
setups:

```python
def apply_look_room(
    crop: tuple,
    head_direction_2d: tuple,    # (dx, dy) normalised head facing direction in camera frame
    look_room_fraction: float = 0.15,
) -> tuple[int, int, int, int]:
    """
    Shift crop horizontally so subject has look-room in the direction they face.
    """
    x, y, w, h = crop
    dx = head_direction_2d[0]   # positive = facing right in camera

    # Shift crop center toward the direction of facing
    shift_px = int(w * look_room_fraction * dx)
    new_x = max(0, min(1920 - w, x + shift_px))
    return (new_x, y, w, h)
```

### 8.3 Two-Person Shot Composition

For a 2S shot, both subjects should be comfortably within frame with
appropriate negative space between them:

```python
def compute_two_person_crop(
    cam_id: str,
    pid_a: str,
    pid_b: str,
    skeleton_3d: np.ndarray,
    camera_calibration: dict,
    min_padding_fraction: float = 0.20,
) -> tuple[int, int, int, int]:
    """
    Compute a crop that frames both participants with head and shoulders visible.
    The crop centers between the two subjects.
    """
    PERSON_IDX = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    cam = camera_calibration[cam_id]

    # Project both subjects' upper-body keypoints
    kps_a = skeleton_3d[PERSON_IDX[pid_a]]
    kps_b = skeleton_3d[PERSON_IDX[pid_b]]

    proj_a = project_keypoints_to_camera(kps_a, cam["K"], cam["rvec"], cam["tvec"],
                                          cam["dist"], valid_kps={0, 1, 2, 5, 15, 16})
    proj_b = project_keypoints_to_camera(kps_b, cam["K"], cam["rvec"], cam["tvec"],
                                          cam["dist"], valid_kps={0, 1, 2, 5, 15, 16})

    all_pts = np.vstack([proj_a, proj_b]) if len(proj_a) and len(proj_b) else \
              (proj_a if len(proj_a) else proj_b)

    if not len(all_pts):
        return (0, 0, 1920, 1080)

    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    # Add padding around both subjects
    pw = (x_max - x_min) * min_padding_fraction
    ph = (y_max - y_min) * min_padding_fraction
    x_min = max(0,    int(x_min - pw))
    y_min = max(0,    int(y_min - ph * 1.5))   # more headroom above
    x_max = min(1920, int(x_max + pw))
    y_max = min(1080, int(y_max + ph))

    return (x_min, y_min, x_max - x_min, y_max - y_min)
```

### 8.4 Group Shot Framing

For GS shots (cam7 full frame), the PanaCast 50 wide-angle view is used
directly. No crop adjustment is applied — the established wide shot should
show all four participants and the desk surface.

---

## 9. Scene-Specific Rules by Task Phase

### 9.1 T1 — Hidden-Profile Decision Task

```
Phase: information_distribution (silent reading)
  → GS or INSERT shots. No individual faces needed.
    All participants looking at tablets → INSERT shot of each tablet in turn.

Phase: discussion
  → Standard speaker-tracking (MCU/CU of active speaker).
    Watch for the moment a participant shares information from their card:
    VAD audio energy rises → cut to CU (important disclosure moment).
    Cut to 2S when two participants engage directly on a specific candidate.

Phase: decision
  → Hold on the participant filling in the selection form (INSERT → MCU).
    After form submitted: GS for group reaction beat.
```

### 9.2 T2 — Negotiation

```
Phase: brief
  → GS establishing shot, then individual MCU as each person reads their role card.

Phase: negotiation
  → Most cinematically rich phase. Prioritize:
    - OTS when one person is making a strong argument
    - 2S when two participants are negotiating directly
    - REACT when a concession is made (VAD valence drops then rises)
    - CU when arousal is very high (heated moment)

Phase: settlement_form
  → INSERT of the form being filled. Cut to GS when submitted.
```

### 9.3 T3 — Idea Generation (NGT)

```
Phase: silent_generation
  → GS or INSERT only. No speech → no speaker tracking.
    Show the desk surface, tablets, participants writing.
    Occasional slow zoom-in on the most active tablet (wrist movement from skeleton).

Phase: round_robin_sharing
  → Strict turn-taking: MCU of the current sharer, REACT of others.
    The moderator enforces the order (P1→P2→P3→P4); the director knows
    who should speak next and can anticipate the cut.

Phase: group_selection
  → 2S between participants who are debating different ideas.
    CU for the participant whose idea is being discussed.
    INSERT of group selection form when submitted.
```

### 9.4 T4 — Public Goods Micro-Game

```
Phase: contribution (private, silent)
  → INSERT shots of each tablet (contribution slider).
    GS for atmosphere. No individual face shots (respects privacy of choice).

Phase: outcome_reveal
  → GS for the reveal moment (all seeing the same screen simultaneously).
    Immediately cut to CU of each participant in turn (2-3s each) to capture
    emotional reactions to the outcome.
    Joint attention event (all looking at screen) → GS.

Phase: discussion
  → Return to standard speaker-tracking.
    High priority: CU of any participant who regrets their choice
    (VAD valence drop after outcome reveal).
    2S between participants debating fairness (trust_front/next ratings in postblock).
```

---

## 10. Director Output Format

### 10.1 Shot List (Edit Decision List)

The director produces a shot list in a standard format compatible with FFmpeg
and video editing tools:

```python
from dataclasses import dataclass, asdict
import json

@dataclass
class Shot:
    shot_id:       int        # sequential shot number
    onset_frame:   int        # start frame in the output video
    offset_frame:  int        # end frame in the output video
    onset_s:       float      # start time (LSL seconds)
    offset_s:      float      # end time (LSL seconds)
    camera_id:     str        # source camera: cam1–cam7 or tobii_p1–tobii_p4
    shot_type:     str        # CU, MCU, MS, OTS, 2S, GS, POV, INSERT, REACT
    subjects:      list[str]  # participant IDs: ["P1"], ["P2", "P3"]
    crop_box:      list[int]  # [x, y, w, h] or null for full frame
    zoom_level:    float      # effective zoom (for metadata only)
    transition_in: str        # "cut", "dissolve_12f", "fade_from_black"
    reason:        str        # why this shot was chosen (for QC and review)
    confidence:    float      # 0–1, director confidence
    task:          str        # T1–T4
    phase:         str        # task sub-phase

def save_shot_list(shots: list[Shot], output_path: str) -> None:
    """Save shot list as NDJSON (one shot per line)."""
    with open(output_path, "w") as f:
        for shot in shots:
            f.write(json.dumps(asdict(shot)) + "\n")
```

Example shot list (first 6 shots of a T2 session):

```json
{"shot_id": 1, "onset_s": 0.0, "offset_s": 4.2, "camera_id": "cam7",
 "shot_type": "GS", "subjects": ["P1","P2","P3","P4"], "crop_box": null,
 "transition_in": "fade_from_black", "reason": "phase_transition",
 "task": "T2", "phase": "shared_brief"}

{"shot_id": 2, "onset_s": 4.2, "offset_s": 9.8, "camera_id": "cam1",
 "shot_type": "MCU", "subjects": ["P1"], "crop_box": [240, 80, 780, 440],
 "transition_in": "cut", "reason": "active_speaker",
 "task": "T2", "phase": "negotiation"}

{"shot_id": 3, "onset_s": 9.8, "offset_s": 12.1, "camera_id": "cam3",
 "shot_type": "REACT", "subjects": ["P4"], "crop_box": [180, 100, 740, 420],
 "transition_in": "cut", "reason": "strong_reaction",
 "task": "T2", "phase": "negotiation"}
```

### 10.2 Audio Mix Plan

The director also specifies an audio mix for each shot segment:

```python
@dataclass
class AudioSegment:
    onset_s:   float
    offset_s:  float
    primary_mic:   str     # "mic_P1", "mic_P2", "mic_P3", "mic_P4", "room"
    mix:           dict    # {mic_id: level_db} for all mics in this segment
    noise_gate:    float   # dB threshold below which mics are silenced

# Example: during a P2 close-up, boost P2's mic, attenuate others
AUDIO_MIX_RULES = {
    "speaker": {
        "primary": +0.0,    # speaker mic at unity
        "others":  -18.0,   # other close-talks attenuated heavily
        "room":    -12.0,   # room mic at background level
    },
    "group_shot": {
        "primary": -3.0,    # all mics blended
        "others":  -3.0,
        "room":    -6.0,
    },
    "silence": {
        "primary": -inf,    # gate all mics
        "others":  -inf,
        "room":    -20.0,   # keep room mic at very low level
    },
}
```

---

## 11. Rendering Pipeline

### 11.1 FFmpeg-Based Rendering

The director produces an FFmpeg command sequence that reads from multiple
source files, applies crops, and concatenates shots:

```python
import subprocess
from pathlib import Path

def render_directed_video(
    shots: list[Shot],
    camera_video_paths: dict[str, str],   # {cam_id: video_file_path}
    audio_segments: list[AudioSegment],
    output_path: str,
    output_fps: float = 30.0,
    output_res: tuple = (1920, 1080),
    encode_preset: str = "fast",
    crf: int = 18,
) -> None:
    """
    Render directed video using FFmpeg concat + filter_complex.
    Each shot is extracted, cropped, scaled, and concatenated.
    """
    # Step 1: Generate per-shot clips
    clip_paths = []
    for shot in shots:
        clip_path = f"/tmp/shot_{shot.shot_id:05d}.mp4"
        src_video = camera_video_paths[shot.camera_id]

        # Build crop filter
        if shot.crop_box:
            x, y, w, h = shot.crop_box
            video_filter = (
                f"crop={w}:{h}:{x}:{y},"
                f"scale={output_res[0]}:{output_res[1]}:flags=lanczos"
            )
        else:
            video_filter = f"scale={output_res[0]}:{output_res[1]}:flags=lanczos"

        # Handle upside-down cameras (cam1–cam4)
        if shot.camera_id in {"cam1", "cam2", "cam3", "cam4"}:
            video_filter = "transpose=2,transpose=2," + video_filter

        duration = shot.offset_s - shot.onset_s

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(shot.onset_s),
            "-t",  str(duration),
            "-i",  src_video,
            "-vf", video_filter,
            "-r",  str(output_fps),
            "-c:v", "libx264",
            "-preset", encode_preset,
            "-crf", str(crf),
            "-an",   # audio handled separately
            clip_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        clip_paths.append(clip_path)

    # Step 2: Concatenate all clips
    concat_list = Path("/tmp/concat_list.txt")
    concat_list.write_text("\n".join(f"file '{p}'" for p in clip_paths))

    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-preset", encode_preset,
        "-crf", str(crf),
        output_path,
    ]
    subprocess.run(cmd_concat, check=True)

    # Clean up temporary clips
    for p in clip_paths:
        Path(p).unlink(missing_ok=True)
```

### 11.2 Transition Effects

```python
TRANSITION_EFFECTS = {
    "cut":            None,                    # hard cut, no transition frames
    "dissolve_12f":   ("blend", 12),           # 12-frame (400ms) cross-dissolve
    "dissolve_6f":    ("blend", 6),            # 6-frame (200ms) dissolve
    "fade_from_black":("fade_in", 15),         # 15-frame fade up
    "fade_to_black":  ("fade_out", 15),        # 15-frame fade down
}

# When to use each transition:
TRANSITION_RULES = {
    "phase_transition":    "dissolve_12f",     # smooth major phase changes
    "task_decision":       "dissolve_6f",      # emphasise decision moments
    "silence_to_speaker":  "cut",              # abrupt entry of new speaker
    "session_start":       "fade_from_black",
    "session_end":         "fade_to_black",
    "default":             "cut",              # everything else: hard cut
}
```

### 11.3 Overlay Options (Optional)

For annotation and research use (not for dissemination):

```python
OVERLAYS = {
    "participant_id":    True,   # small label (P1–P4) in corner
    "task_phase":        True,   # current task and phase in footer
    "speaker_indicator": True,   # microphone icon when participant speaking
    "gaze_dot":          False,  # optional: gaze point projected to video
    "shot_type":         False,  # debug: current shot type
}
```

---

## 12. Quality Control and Manual Override

### 12.1 Director QC Report

After each session is directed, generate a QC report:

```python
def director_qc_report(shots: list[Shot], fps: float = 30.0) -> dict:
    """
    Check for common directing errors.
    """
    total_duration_s = (shots[-1].offset_s - shots[0].onset_s) if shots else 0

    # Cut rate per minute
    n_cuts = sum(1 for s in shots if s.transition_in == "cut")
    cut_rate = n_cuts / (total_duration_s / 60) if total_duration_s > 0 else 0

    # Shot type distribution
    from collections import Counter
    shot_dist = Counter(s.shot_type for s in shots)

    # Very short shots (< 1.5s) — likely errors
    short_shots = [s for s in shots if (s.offset_s - s.onset_s) < 1.5]

    # Shots with low confidence
    low_conf = [s for s in shots if s.confidence < 0.40]

    # Same camera held for > 20 consecutive shots (variety problem)
    camera_runs = []
    current_cam, run = shots[0].camera_id if shots else None, 0
    for s in shots:
        if s.camera_id == current_cam:
            run += 1
        else:
            if run > 20:
                camera_runs.append((current_cam, run))
            current_cam, run = s.camera_id, 1

    return {
        "total_shots":      len(shots),
        "total_duration_s": total_duration_s,
        "cut_rate_per_min": cut_rate,
        "shot_distribution": dict(shot_dist),
        "n_short_shots":    len(short_shots),
        "n_low_conf_shots": len(low_conf),
        "camera_monotony_warnings": camera_runs,
        "warnings": [
            f"{len(short_shots)} shots shorter than 1.5s" if short_shots else None,
            f"Cut rate {cut_rate:.1f}/min — {'too fast' if cut_rate > 30 else 'OK'}" ,
            f"{len(low_conf)} low-confidence shots" if low_conf else None,
        ],
    }
```

### 12.2 Manual Override System

Provide a simple YAML override file that a human reviewer can edit to
correct specific moments:

```yaml
# director_overrides_grp13_T2.yaml
# Each entry replaces the auto-directed shot for the specified time range.

overrides:
  - onset_s:    245.3
    offset_s:   252.1
    camera_id:  cam3
    shot_type:  CU
    subjects:   [P4]
    crop_box:   [150, 90, 760, 430]
    reason:     "Manually selected: key emotional reaction not captured by auto-director"

  - onset_s:    312.0
    offset_s:   315.5
    camera_id:  cam7
    shot_type:  GS
    subjects:   [P1, P2, P3, P4]
    crop_box:   null
    reason:     "Group reaction to settlement — auto-director missed, cut to group shot"
```

Load and apply overrides before rendering:

```python
def apply_overrides(
    shots: list[Shot],
    override_file: str,
) -> list[Shot]:
    """Replace auto-directed shots with manual overrides where specified."""
    import yaml
    with open(override_file) as f:
        cfg = yaml.safe_load(f)

    for override in cfg.get("overrides", []):
        onset_s  = override["onset_s"]
        offset_s = override["offset_s"]
        # Find shots in this range and replace them
        # (simplified: replace shots whose onset falls in the window)
        shots = [
            Shot(**{**asdict(s), **{k: v for k, v in override.items() if k != "onset_s" and k != "offset_s"}})
            if onset_s <= s.onset_s < offset_s else s
            for s in shots
        ]
    return shots
```

### 12.3 Annotator Feedback Loop

Annotators who use the directed video for behavioral coding should be able
to flag moments where the director failed:

```python
ANNOTATOR_FLAGS = {
    "missed_speaker":       "Director was on wrong person during speech",
    "too_long_on_one_cam":  "Monotonous — same camera held too long",
    "missed_reaction":      "Interesting reaction not captured",
    "disorienting_cut":     "Cut was jarring or disorienting",
    "wrong_shot_size":      "Shot was too wide / too close for this moment",
    "missed_gesture":       "Gesture was not visible in shot",
}
# Flags stored in: <session>/annot/director_feedback_T{N}.tsv
```

---

## 13. Implementation Roadmap

### Phase 0 — Prerequisites (from master plan)
The director requires reliable signals from all four analysis layers. At minimum:
- Layer 1: 3D gaze working for at least Approach A or B (glasses pose)
- Layer 2: Skeleton 3D with seat-locked identity
- Layer 3: Per-participant audio VAD (speaker detection)
- Layer 4: Not required for basic directing

### Phase 1 — Basic Speaker-Tracking Director (2–3 weeks)

Build the minimum viable director using only audio VAD for shot selection.
This requires no gaze or skeleton data.

```
Input:  Per-participant VAD (binary, 30 fps)
        Task phase events (from LSL)
        Raw video files (7 cameras)

Logic:  If speaker identified → MCU of speaker (full camera frame, no crop)
        If silence → GS (cam7)
        If phase transition → GS (cam7, 4s)

Output: Shot list NDJSON
        Rendered video (no dynamic crop, just camera switching)
```

**Expected quality**: Useful for annotation. Not cinematographically refined.
**Estimated directed cut rate**: 8–12 cuts/minute.

### Phase 2 — Skeleton-Guided Crop and Composition (3–4 weeks)

Add dynamic cropping using 3D skeleton projection. This makes the shots
significantly tighter and more watchable.

```
Add:    compute_tight_crop() using skeleton + camera calibration
        CropTracker for smooth within-shot following
        Rule-of-thirds and look-room adjustments
        Two-person shots for dialogic exchanges

Expected quality: Broadcast-comparable for research purposes.
Estimated cut rate: 10–18 cuts/minute.
```

### Phase 3 — Gaze and Gesture Enrichment (4–6 weeks)

Add gaze-driven shot selection (OTS, REACT, POV) and gesture-motivated
cuts (MS for pointing, INSERT for tablet interaction).

```
Add:    reaction_priority() using gaze and head gesture signals
        OTS shot selection when gaze target is confirmed
        INSERT shots triggered by tablet_confirmed events
        POV shots (Tobii scene camera) as variety shots
        Phase-specific pacing rules

Expected quality: Documentarian-quality social science video.
Estimated cut rate: 15–25 cuts/minute.
```

### Phase 4 — ML-Based Shot Quality Scoring (research extension)

Train a shot quality classifier on human-annotated ratings of the directed
output. Use this to refine the rule-based engine:

```
Training data:  Annotator flags (§12.3) + manual overrides (§12.2)
Features:       Shot confidence score, behavioral signal values at cut point,
                shot duration, camera angle relative to speaker
Target:         Annotator rating of shot quality (1–5)

Use:            Replace or augment rule-based confidence scores with
                learned quality predictions
```

---

## Appendix A: Camera Coverage Map (Corrected)

Which cameras provide clear face visibility for each participant:

```
           cam1    cam2    cam3    cam4    cam5    cam6    cam7
P1 (BR)    ✗       ✗       ✓✓      ✓       ~hands  ✓       ✓
P2 (FR)    ✗       ✗       ✓       ✓✓      ~hands  ✓       ✓
P3 (FL)    ✓✓      ✓       ✗       ✗       ~hands  ✓       ✓
P4 (BL)    ✓       ✓✓      ✗       ✗       ~hands  ✓       ✓

✓✓ = primary face shot (frontal or 3-quarter, face fully expressible)
✓  = secondary face shot (angled but usable)
✗  = face not clearly visible (back of head or wrong side)
~  = hands and tablet area visible but no face

Key facts:
  cam6 (back-center, elevated):  ALL participants visible, good faces — primary group camera
  cam7 (front-center, P50, table-level):  ALL participants visible — safety/default group camera
  cam5 (front-left, table surface):  NO faces — INSERT shots only (hands, tablets)
  cam1+cam2 (left/right front):  front-row participants P3 and P4
  cam3+cam4 (left/right back):   back-row participants P1 and P2
```

**Shot source summary by use case:**

| Use case | Best camera(s) | Notes |
|----------|---------------|-------|
| Establishing / group wide | cam7 (primary), cam6 (secondary) | Both always safe |
| All 4 faces simultaneously | cam7 or cam6 | Alternate for visual variety |
| P1 close-up | cam3 (primary face), cam4 | cam3 right-back sees P1 directly |
| P2 close-up | cam4 (primary face), cam3 | cam4 left-back sees P2 directly |
| P3 close-up | cam1 (primary face), cam2 | cam1 left-front sees P3 directly |
| P4 close-up | cam2 (primary face), cam1 | cam2 right-front sees P4 directly |
| Front row pair (P3+P4) | cam7 crop | Both faces at table level |
| Back row pair (P1+P2) | cam6 crop | Both faces from elevated rear |
| Cross-table pair (P1+P3, P2+P4) | cam7 or cam6 crop | Wide enough to frame both |
| Hands / tablet / desk items | cam5 | Only camera at surface level |
| First-person POV | tobii_pN | What participant sees, gaze dot optional |

## Appendix B: Shot Selection Decision Tree

```
START
  │
  ├─ Phase just changed?       → GS (4s establishing)
  │
  ├─ Task decision active?     → CU of decision-maker
  │
  ├─ n_speakers == 1?
  │     │
  │     ├─ arousal > 7?        → CU of speaker
  │     │
  │     ├─ gesture == point?   → MS of speaker
  │     │
  │     ├─ tablet_confirmed?   → INSERT
  │     │
  │     ├─ reaction > 0.55 AND duration > 3s?
  │     │                      → REACT of best reactor
  │     │
  │     └─ default             → MCU of speaker
  │
  ├─ n_speakers == 2?          → 2S of both speakers
  │
  ├─ mutual_gaze pair?         → 2S of gazing pair
  │
  ├─ joint_attention ≥ 3?      → GS
  │
  └─ silence > 3s?             → GS (or hold and zoom out slowly)
```

## Appendix C: File Naming Convention

```
<session_dir>/
  directed/
    shot_list_T{N}.ndjson           Shot decision list
    audio_mix_T{N}.ndjson           Audio segment plan
    director_overrides_T{N}.yaml    Manual corrections (if any)
    director_qc_T{N}.json           QC report
    directed_T{N}.mp4               Final rendered video
    directed_T{N}_annotated.mp4     With overlays (for annotation use)
```

---

*Document generated 2026-04-19.*
*Depends on: `affectai_master_analysis_plan.md` (all four layers).*
*Minimum viable version requires only Layer 3 audio VAD and raw video files.*
*Full version requires Layers 1–3 outputs (gaze, skeleton, fusion).*
