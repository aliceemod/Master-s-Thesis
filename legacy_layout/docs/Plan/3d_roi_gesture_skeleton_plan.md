# AffectAI — 3D ROI Detection, Skeleton & Gesture: Improvement Plan

> **Context**: Four participants (P1–P4) sit around a 1.8 m × 0.8 m desk in fixed seats.
> A moderator may be present off-camera. Physical objects in the scene include four tablets
> (one per participant), a large screen on the rear wall, the desk surface, and the ChArUco
> calibration board at desk center. Seven PanaCast cameras (cam1–cam7) provide exocentric
> views; four Tobii Pro Glasses 3 provide egocentric views. The 3D world frame is the
> ChArUco board center (x=right, y=back, z=up).

---

## Table of Contents

1. [3D ROI Definitions — Static Scene Objects](#1-3d-roi-definitions--static-scene-objects)
2. [Participant Detection and Tracking](#2-participant-detection-and-tracking)
3. [Moderator Detection](#3-moderator-detection)
4. [Tablet ROI Tracking](#4-tablet-roi-tracking)
5. [Big Screen ROI](#5-big-screen-roi)
6. [3D Skeleton Pipeline Improvements](#6-3d-skeleton-pipeline-improvements)
7. [Hand and Gesture Detection](#7-hand-and-gesture-detection)
8. [Gesture Event Extraction](#8-gesture-event-extraction)
9. [Cross-Modality Integration — Gaze + Skeleton + ROI](#9-cross-modality-integration--gaze--skeleton--roi)
10. [Quality Control for ROI and Skeleton](#10-quality-control-for-roi-and-skeleton)
11. [Priority Order](#11-priority-order)

---

## 1. 3D ROI Definitions — Static Scene Objects

### 1.1 Coordinate Anchors

All ROIs are expressed in the world frame (origin = ChArUco board center). Known
physical dimensions from `configs/desk_markers_large.yaml` and
`configs/desk_markers_large.yaml`:

```
World frame:
  x: right (positive = right when facing screen)
  y: back  (positive = toward big screen / rear wall)
  z: up    (positive = above desk surface)

Desk surface:      z = 0.000 m  (by definition — world origin is on desk)
Desk width:        x ∈ [-0.900, +0.900] m
Desk depth:        y ∈ [-0.400, +0.400] m
Desk height:       0.750 m above floor
Camera mount:      z ≈ 0.880 m above desk (cam1–cam4, cam6)
```

### 1.2 Static ROI Registry

Define all static ROIs in a single YAML config so they can be loaded by any tool:

```yaml
# configs/scene_rois.yaml
world_frame:
  origin: fixed_7x5_board_center
  axes: {x: right, y: back, z: up}

rois:

  desk_surface:
    type: box
    description: "Full desk surface (z=0 plane)"
    center_m: [0.000, 0.000, 0.000]
    half_extents_m: [0.900, 0.400, 0.020]   # 20mm thick slab

  desk_center:
    type: box
    description: "Center of desk — ChArUco board + shared items"
    center_m: [0.000, 0.000, 0.000]
    half_extents_m: [0.250, 0.200, 0.050]

  big_screen:
    type: box
    description: "Rear wall display, facing participants"
    center_m: [0.000, 0.800, 1.000]          # estimate — measure in lab
    half_extents_m: [0.600, 0.050, 0.400]    # 1.2m wide, 0.8m tall

  # Participant personal space zones (one per seat)
  zone_P1:                                    # back-right
    type: box
    center_m: [0.550, 0.200, 0.400]
    half_extents_m: [0.300, 0.180, 0.500]

  zone_P2:                                    # front-right
    type: box
    center_m: [0.550, -0.200, 0.400]
    half_extents_m: [0.300, 0.180, 0.500]

  zone_P3:                                    # front-left
    type: box
    center_m: [-0.550, -0.200, 0.400]
    half_extents_m: [0.300, 0.180, 0.500]

  zone_P4:                                    # back-left
    type: box
    center_m: [-0.550, 0.200, 0.400]
    half_extents_m: [0.300, 0.180, 0.500]

  # Tablet ROIs — updated dynamically (see §4)
  tablet_P1:
    type: box
    center_m: [0.600, -0.250, 0.010]
    half_extents_m: [0.090, 0.120, 0.015]
    dynamic: true                              # updated per-frame from skeleton

  tablet_P2:
    type: box
    center_m: [0.600,  0.250, 0.010]
    half_extents_m: [0.090, 0.120, 0.015]
    dynamic: true

  tablet_P3:
    type: box
    center_m: [-0.600,  0.250, 0.010]
    half_extents_m: [0.090, 0.120, 0.015]
    dynamic: true

  tablet_P4:
    type: box
    center_m: [-0.600, -0.250, 0.010]
    half_extents_m: [0.090, 0.120, 0.015]
    dynamic: true
```

### 1.3 ROI Lookup — Point in Box Test

```python
import numpy as np
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class BoxROI:
    name: str
    center: np.ndarray       # (3,) world metres
    half_extents: np.ndarray # (3,) world metres

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(np.abs(point - self.center) <= self.half_extents))

    def distance_to(self, point: np.ndarray) -> float:
        """Signed distance: negative = inside, positive = outside."""
        d = np.abs(point - self.center) - self.half_extents
        return float(np.linalg.norm(np.maximum(d, 0)))


def load_rois(config_path: str) -> dict[str, BoxROI]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    rois = {}
    for name, spec in cfg["rois"].items():
        rois[name] = BoxROI(
            name=name,
            center=np.array(spec["center_m"]),
            half_extents=np.array(spec["half_extents_m"]),
        )
    return rois


def classify_point(point_3d: np.ndarray,
                   rois: dict[str, BoxROI],
                   priority_order: list[str]) -> str:
    """Return the name of the first (highest-priority) ROI containing point."""
    for roi_name in priority_order:
        if roi_name in rois and rois[roi_name].contains(point_3d):
            return roi_name
    return "other"


# Recommended lookup priority (most specific first)
ROI_PRIORITY = [
    "tablet_P1", "tablet_P2", "tablet_P3", "tablet_P4",
    "big_screen",
    "zone_P1", "zone_P2", "zone_P3", "zone_P4",
    "desk_center", "desk_surface",
]
```

---

## 2. Participant Detection and Tracking

### 2.1 Current State

`multicam_pose3d.py` uses zone-aware epipolar matching to assign skeleton detections
to global person IDs (0=P1, 1=P2, 2=P3, 3=P4). The output shape is
`(n_frames, 4, 25, 7)` with columns `[x, y, z, conf, reproj_err, n_cams, group_id]`.

### 2.2 Seat-Based Identity Lock

Participants sit in fixed seats for the entire session. Use seat position priors to
enforce consistent identity even when MediaPipe briefly swaps person IDs:

```python
# Known seat centroids in world frame (from configs/desk_markers_large.yaml)
SEAT_CENTROIDS = {
    "P1": np.array([ 0.55,  0.25, 0.85]),   # back-right,  head height
    "P2": np.array([ 0.55, -0.25, 0.85]),   # front-right
    "P3": np.array([-0.55, -0.25, 0.85]),   # front-left
    "P4": np.array([-0.55,  0.25, 0.85]),   # back-left
}
SEAT_ORDER = ["P1", "P2", "P3", "P4"]

def assign_skeletons_to_seats(
    skeletons: np.ndarray,               # (n_people, 25, 7) for one frame
    seat_centroids: dict[str, np.ndarray],
    nose_kp_idx: int = 0,
    max_dist_m: float = 0.40,
) -> dict[str, Optional[int]]:
    """
    Match each detected skeleton to the nearest seat by nose keypoint position.
    Returns {person_label: skeleton_index} or {person_label: None} if undetected.
    """
    assignment = {label: None for label in seat_centroids}
    used = set()

    for label, centroid in seat_centroids.items():
        best_idx, best_dist = None, np.inf
        for i in range(skeletons.shape[0]):
            if i in used:
                continue
            nose = skeletons[i, nose_kp_idx, :3]
            if np.any(np.isnan(nose)):
                continue
            dist = np.linalg.norm(nose - centroid)
            if dist < best_dist and dist < max_dist_m:
                best_dist, best_idx = dist, i
        if best_idx is not None:
            assignment[label] = best_idx
            used.add(best_idx)

    return assignment
```

Apply this per-frame after triangulation and write the assignment to column 6
(`group_id`) of the skeleton array.

### 2.3 Head Pose Estimation from 3D Skeleton

MediaPipe BODY_25 provides nose (0), eyes (15, 16), and ears (17, 18). Use these
five head keypoints to estimate a head orientation vector:

```python
def head_orientation(skeleton_kps: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate head forward-facing direction from face keypoints.
    Returns unit vector in world frame, or None if keypoints are missing.

    BODY_25 indices: 0=Nose, 15=REye, 16=LEye, 17=REar, 18=LEar
    """
    nose  = skeleton_kps[0,  :3]
    reye  = skeleton_kps[15, :3]
    leye  = skeleton_kps[16, :3]
    rear  = skeleton_kps[17, :3]
    lear  = skeleton_kps[18, :3]

    if np.any(np.isnan([nose, reye, leye])):
        return None

    # Eye midpoint → nose gives approximate forward direction
    eye_mid = (reye + leye) / 2
    forward = nose - eye_mid
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return None
    return forward / norm
```

### 2.4 Body-Part ROI Assignment per Frame

For each participant and each body part, record which ROI the keypoint falls in.
This enables downstream queries like "how long was P2's right hand on the tablet?":

```python
BODY_PARTS = {
    "head":        [0, 15, 16, 17, 18],    # nose + eyes + ears
    "right_hand":  [4],                    # RWrist
    "left_hand":   [7],                    # LWrist
    "right_elbow": [3],
    "left_elbow":  [6],
    "torso":       [1, 2, 5, 8],           # neck + shoulders + mid-hip
}

def bodypart_roi_frame(skeleton_kps: np.ndarray,
                       rois: dict[str, BoxROI],
                       priority_order: list[str]) -> dict[str, str]:
    """
    Returns {body_part_name: roi_name} for one person in one frame.
    """
    result = {}
    for part_name, kp_indices in BODY_PARTS.items():
        pts = [skeleton_kps[idx, :3] for idx in kp_indices
               if not np.any(np.isnan(skeleton_kps[idx, :3]))]
        if not pts:
            result[part_name] = "missing"
            continue
        centroid = np.mean(pts, axis=0)
        result[part_name] = classify_point(centroid, rois, priority_order)
    return result
```

---

## 3. Moderator Detection

### 3.1 Problem

The moderator is not a fixed participant and may be standing, walking, or partially
visible. They are not assigned a permanent seat and will not be matched by the
zone-aware matcher in `multicam_pose3d.py`.

### 3.2 Moderator Zone Definition

Define a moderator zone outside the participant area. From the lab layout in
`docs/camera_layout_and_positions.md`, the moderator typically stands at the front
of the room (y < −0.5 m, outside the desk depth range):

```yaml
# Add to configs/scene_rois.yaml
  zone_moderator:
    type: box
    center_m: [0.000, -0.650, 0.900]
    half_extents_m: [0.600, 0.200, 0.900]
```

### 3.3 Moderator Skeleton Detection

Add a `moderator` slot to the pipeline by allowing an additional (5th) person
detection outside the participant zones:

```python
def detect_moderator(
    all_detections: dict[str, list[dict]],   # cam_key → people dicts
    cameras: dict,
    participant_skeletons: np.ndarray,        # (4, 25, 7) already assigned
    moderator_zone: BoxROI,
    max_reproj_px: float = 30.0,
) -> Optional[np.ndarray]:
    """
    Find a 5th person skeleton that is not one of the 4 participants
    and whose head keypoint falls in the moderator zone.
    Returns (25, 7) skeleton or None.
    """
    # Build a set of pixel regions already claimed by participants
    # Then look for remaining detections in moderator zone
    ...
```

### 3.4 Moderator Activity Flags

Even without a full skeleton, flag moderator presence per task segment using
motion detection in cam7 (front-center PanaCast 50) which has the widest
field of view:

```python
def detect_moderator_motion(video_path: str,
                             task_windows: list[tuple[float, float]],
                             roi_pixels: tuple[int, int, int, int],
                             fps: float = 30.0,
                             threshold: float = 25.0) -> list[float]:
    """
    Frame-level motion energy in the moderator zone pixel ROI.
    Returns list of motion scores (one per frame) for flagging moderator activity.
    roi_pixels: (x, y, w, h) in cam7 image coordinates.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    scores = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = roi_pixels
        crop = gray[y:y+h, x:x+w]
        if prev_gray is not None:
            diff = cv2.absdiff(crop, prev_gray)
            scores.append(float(diff.mean()))
        else:
            scores.append(0.0)
        prev_gray = crop.copy()
    cap.release()
    return scores
```

---

## 4. Tablet ROI Tracking

### 4.1 Problem

Tablets are placed on the desk but can be moved by participants. Their exact 3D
position changes during the session. Static ROIs are insufficient for gaze-on-tablet
analysis.

### 4.2 Approach A — ArUco Markers on Tablets (preferred)

Place one ArUco marker (DICT_4X4_50, IDs 20–23, ~40 mm) on the back of each tablet.
Fixed cameras detect the marker in each frame, giving a precise 6-DoF tablet pose.
The screen plane is then offset from the marker by a known rigid transform
(measured once per tablet model):

```python
# Rigid transform from marker (back of tablet) to screen plane (front)
# Measure once with calipers: tablet thickness + marker position
T_screen_marker = np.array([
    [1, 0, 0,  0.000],   # x: no lateral offset if marker is centered
    [0, 1, 0,  0.000],   # y: no lateral offset
    [0, 0, 1, -0.010],   # z: 10mm from marker to screen plane (tablet thickness)
    [0, 0, 0,  1.000],
])

def tablet_screen_pose(marker_rvec, marker_tvec, T_screen_marker):
    R, _ = cv2.Rodrigues(marker_rvec)
    T_world_marker = np.eye(4)
    T_world_marker[:3, :3] = R
    T_world_marker[:3,  3] = marker_tvec.flatten()
    T_world_screen = T_world_marker @ T_screen_marker
    return T_world_screen   # 4×4: screen plane center in world frame
```

Assign ArUco IDs 20–23 to tablets P1–P4 respectively (no collision with existing IDs).

### 4.3 Approach B — Wrist-Anchored Dynamic ROI (fallback)

When no tablet marker is visible, estimate tablet position from the participant's
wrist keypoint (from the 3D skeleton). Participants hold or rest their hand near
the tablet. Offset the wrist position by a participant-specific prior:

```python
TABLET_WRIST_OFFSET = {
    "P1": np.array([ 0.05,  0.05, -0.05]),   # tablet is ~5cm forward/right of wrist
    "P2": np.array([ 0.05, -0.05, -0.05]),
    "P3": np.array([-0.05, -0.05, -0.05]),
    "P4": np.array([-0.05,  0.05, -0.05]),
}

def estimate_tablet_pos(wrist_3d, participant_id):
    offset = TABLET_WRIST_OFFSET.get(participant_id, np.zeros(3))
    return wrist_3d + offset
```

This gives ~5–10 cm accuracy — sufficient for gaze-on-tablet classification but
not for precise AOI subdivision of the tablet screen.

### 4.4 Tablet Screen Subdivisions

Once the tablet plane is known, define sub-regions for fine-grained gaze analysis
(e.g., "looking at the submit button" vs. "looking at the rating scale"):

```python
# Normalized coordinates within tablet screen [0,1] × [0,1]
# origin: top-left, x: right, y: down
TABLET_ZONES = {
    "header":       (0.00, 0.00, 1.00, 0.15),   # (x0, y0, x1, y1) normalised
    "rating_scale": (0.00, 0.15, 1.00, 0.60),
    "submit_button":(0.20, 0.75, 0.80, 1.00),
}

def map_gaze_to_tablet_zone(gaze_hit_uv, tablet_zones):
    """
    gaze_hit_uv: (u, v) normalised coordinates on tablet screen [0,1]²
    Returns zone name or 'none'.
    """
    for zone_name, (x0, y0, x1, y1) in tablet_zones.items():
        if x0 <= gaze_hit_uv[0] <= x1 and y0 <= gaze_hit_uv[1] <= y1:
            return zone_name
    return "none"
```

---

## 5. Big Screen ROI

### 5.1 3D Position Estimation

The big screen is on the rear wall. Its exact position must be measured once in the
lab (tape measure from ChArUco board center to screen corners) and stored in
`configs/scene_rois.yaml`. From the lab layout doc, the screen is behind P1/P4
(back row) at approximately y ≈ +0.8 m (rear wall), z ≈ 0.6–1.4 m.

Refine by placing a temporary ArUco marker (or ChArUco board) on the screen during
setup, detecting it with the fixed cameras, and recording the 3D corner positions:

```bash
# One-time measurement during lab setup:
python tools/tobii_multicam_glasses_tracker.py \
    --calibration calibration_charuco.toml \
    --videos-dir data/setup/screen_calibration/video \
    --marker-map configs/screen_marker.yaml \
    --output-dir data/setup/screen_3d_corners.json
```

### 5.2 Screen Content Tracking

The big screen displays task stimuli via `stimuli/display_server.py`. The content
changes are logged to LSL as `AffectAI_BigScreen` events. Combine the gaze-on-screen
classification with the content event log to determine which stimulus element the
participant was looking at:

```python
def gaze_screen_content(gaze_ts: float,
                         screen_events: pd.DataFrame,
                         screen_roi: BoxROI,
                         gaze_point_3d: np.ndarray) -> dict:
    """
    Returns the screen content being displayed at gaze_ts,
    given that gaze_point_3d is already classified as 'big_screen'.
    """
    # Find the most recent screen event before this timestamp
    prior = screen_events[screen_events["onset"] <= gaze_ts]
    if prior.empty:
        return {"content": "unknown", "task": "none"}
    latest = prior.iloc[-1]
    return {
        "content": latest.get("value", "unknown"),
        "task":    latest.get("task_label", "none"),
        "onset_s": float(latest["onset"]),
    }
```

### 5.3 Screen Gaze UV Coordinates

Project gaze hit point from 3D world frame to normalised screen coordinates:

```python
def world_to_screen_uv(gaze_point_3d: np.ndarray,
                        screen_center: np.ndarray,
                        screen_right: np.ndarray,    # unit vector
                        screen_up: np.ndarray,       # unit vector
                        screen_half_w: float,
                        screen_half_h: float) -> Optional[tuple[float, float]]:
    """
    Returns (u, v) in [0,1]² where (0,0)=top-left, (1,1)=bottom-right.
    Returns None if point is outside screen bounds.
    """
    delta = gaze_point_3d - screen_center
    u_raw = np.dot(delta, screen_right)   / screen_half_w
    v_raw = -np.dot(delta, screen_up)     / screen_half_h   # flip y

    if abs(u_raw) > 1.0 or abs(v_raw) > 1.0:
        return None

    return ((u_raw + 1.0) / 2.0, (v_raw + 1.0) / 2.0)
```

---

## 6. 3D Skeleton Pipeline Improvements

### 6.1 Current Pipeline

`video_only_3d_pipeline.py` chains:
1. ChArUco calibration
2. Glasses pose estimation (Approach A/B)
3. MediaPipe 2D pose → triangulation via `multicam_pose3d.py`
4. `refine_skeleton_3d.py` — smoothing + interpolation
5. `face_hand_pipeline.py` — face mesh + blendshapes + hands

Output: `(n_frames, 4, 25, 7)` skeleton array.

### 6.2 Camera Zone Optimisation for 4-Person Seated Layout

Your `offline_compute_stack.yaml` already defines:

```
Zone A: cam1 + cam4 → P1 (back-right) + P2 (front-right)
Zone B: cam2 + cam3 → P3 (front-left) + P4 (back-left)
Shared: cam5, cam6, cam7
```

**Problem**: cam1 and cam4 are on the *same side* (both left-side cameras). They
share a nearly identical viewpoint for P1 and P2, giving a poor triangulation
baseline. Recommend reassigning zones to maximise baseline:

```
Recommended Zone A: cam1 + cam3 → P1 + P2  (opposite corners of desk)
Recommended Zone B: cam2 + cam4 → P3 + P4  (opposite corners)
```

cam1 (left-front) and cam3 (right-back) are ~1.8 m apart in x and have
a 0.4 m baseline in y — much better than cam1+cam4 which are only 0.4 m
apart in y. Adjust `offline_compute_stack.yaml`:

```yaml
pose3d:
  camera_zones:
    - "cam1+cam3:0,1"    # was cam1+cam4
    - "cam2+cam4:2,3"    # was cam2+cam3
```

Validate by checking that `mean_reproj_accepted_px` decreases in the QC summary.

### 6.3 Upper-Body-Only Mode for Seated Participants

All participants are seated. Lower body keypoints (hips downward: IDs 9–14) are
rarely visible and add noise. Add an `--upper-body-only` flag to
`multicam_pose3d.py` that:

1. Skips triangulation for keypoints 9–14
2. Uses a tighter reprojection threshold for upper body (15 px vs. 30 px)
3. Weights face and shoulder keypoints more heavily in the zone matching step

```python
UPPER_BODY_KPS = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18})
LOWER_BODY_KPS = frozenset({9, 10, 11, 12, 13, 14})

# In process_frame():
if upper_body_only and ji in LOWER_BODY_KPS:
    continue   # skip lower body triangulation
```

### 6.4 Skeleton Refinement Improvements

Current `refine_skeleton_3d.py` applies smoothing and interpolation. Add:

**Anatomical constraint enforcement** — enforce bone length consistency using
known average adult proportions. Keypoints that violate bone length by > 20%
are flagged as outliers and interpolated:

```python
BONE_PAIRS = [
    (1, 0,  0.25),   # neck → nose,       ~25 cm
    (1, 2,  0.35),   # neck → R shoulder, ~35 cm
    (2, 3,  0.30),   # R shoulder → R elbow
    (3, 4,  0.28),   # R elbow → R wrist
    (1, 5,  0.35),   # neck → L shoulder
    (5, 6,  0.30),
    (6, 7,  0.28),
    (1, 8,  0.50),   # neck → mid-hip
]

def enforce_bone_lengths(skeleton_3d, bone_pairs,
                          tolerance=0.20):
    """
    For each bone, if the observed length deviates from expected by
    more than `tolerance` fraction, mark the distal keypoint as NaN
    so it gets interpolated in the next pass.
    """
    refined = skeleton_3d.copy()
    for prox_idx, dist_idx, expected_len in bone_pairs:
        prox = skeleton_3d[prox_idx, :3]
        dist = skeleton_3d[dist_idx, :3]
        if np.any(np.isnan(prox)) or np.any(np.isnan(dist)):
            continue
        actual_len = np.linalg.norm(dist - prox)
        if abs(actual_len - expected_len) / expected_len > tolerance:
            refined[dist_idx, :3] = np.nan   # flag for interpolation
    return refined
```

**Velocity-based outlier rejection** — reject keypoints whose frame-to-frame
velocity exceeds a biomechanical limit (seated participant cannot move a wrist
faster than ~3 m/s):

```python
MAX_VELOCITY_MS = {
    "default":     3.0,
    "head":        1.5,    # slower for head
    "hand":        3.0,
    "elbow":       2.5,
}

def velocity_outlier_reject(positions, fps, max_v):
    """positions: (n_frames, 3). Returns cleaned positions."""
    cleaned = positions.copy()
    dt = 1.0 / fps
    for i in range(1, len(positions)):
        if np.any(np.isnan(positions[i-1])) or np.any(np.isnan(positions[i])):
            continue
        v = np.linalg.norm(positions[i] - positions[i-1]) / dt
        if v > max_v:
            cleaned[i] = np.nan
    return cleaned
```

### 6.5 Per-Frame Skeleton Confidence Score

Add an overall per-person confidence score per frame, useful for downstream
filtering:

```python
def skeleton_confidence(skeleton_kps: np.ndarray,
                         required_kps: list[int] = [0, 1, 2, 5]) -> float:
    """
    Returns fraction of required keypoints that are valid (non-NaN, conf > 0.3).
    required_kps: minimum set needed for the frame to be useful.
    """
    valid = sum(
        1 for idx in required_kps
        if not np.any(np.isnan(skeleton_kps[idx, :3]))
        and skeleton_kps[idx, 3] > 0.3
    )
    return valid / len(required_kps)
```

---

## 7. Hand and Gesture Detection

### 7.1 Current State

`face_hand_pipeline.py` runs MediaPipe Holistic to extract hand keypoints
(21 per hand) in 2D from each camera view. These are not yet triangulated to 3D.

### 7.2 3D Hand Keypoint Triangulation

Extend `multicam_pose3d.py` to handle the 21-keypoint hand model alongside
BODY_25. Use the wrist position from the body skeleton as a spatial prior to
disambiguate left vs. right hand across cameras:

```python
HAND_KP_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

def triangulate_hand(
    hand_detections: dict,    # cam_key → (21, 3) keypoints [x, y, conf]
    cameras: dict,
    body_wrist_3d: np.ndarray,   # (3,) from BODY_25 wrist as spatial prior
    max_wrist_dist_m: float = 0.15,
) -> Optional[np.ndarray]:
    """
    Triangulate 21 hand keypoints to 3D.
    Only use camera detections whose wrist position is within
    max_wrist_dist_m of the body skeleton wrist — filters wrong-hand detections.
    Returns (21, 3) or None.
    """
    hand_3d = np.full((21, 3), np.nan)
    for kp_idx in range(21):
        observations = []
        for cam_key, kps in hand_detections.items():
            if kps[kp_idx, 2] > 0.3:   # confidence gate
                observations.append((cameras[cam_key], kps[kp_idx]))
        if len(observations) >= 2:
            pt3d, reproj, n_c = triangulate_point(observations)
            if not np.any(np.isnan(pt3d)):
                hand_3d[kp_idx] = pt3d
    return hand_3d if not np.all(np.isnan(hand_3d)) else None
```

### 7.3 Hand State Classification

Classify each hand frame into a discrete state for gesture segmentation:

```python
from enum import Enum

class HandState(Enum):
    OPEN      = "open"        # all fingers extended
    FIST      = "fist"        # all fingers curled
    POINT     = "point"       # index extended, others curled
    PINCH     = "pinch"       # thumb + index close
    THUMBS_UP = "thumbs_up"
    UNKNOWN   = "unknown"

def classify_hand_state(hand_3d: np.ndarray) -> HandState:
    """
    hand_3d: (21, 3) hand keypoints in any frame.
    Uses fingertip-to-palm distances to classify state.
    """
    if np.any(np.isnan(hand_3d[0])):
        return HandState.UNKNOWN

    wrist = hand_3d[0]

    # Fingertip indices in MediaPipe 21-keypoint hand model
    tips  = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    mcps  = {"thumb": 2, "index": 5, "middle": 9,  "ring": 13, "pinky": 17}

    extended = {}
    for finger, tip_idx in tips.items():
        mcp_idx = mcps[finger]
        if np.any(np.isnan(hand_3d[tip_idx])) or np.any(np.isnan(hand_3d[mcp_idx])):
            extended[finger] = None
            continue
        tip_dist  = np.linalg.norm(hand_3d[tip_idx] - wrist)
        mcp_dist  = np.linalg.norm(hand_3d[mcp_idx] - wrist)
        extended[finger] = tip_dist > mcp_dist * 1.3

    if all(v for v in extended.values() if v is not None):
        return HandState.OPEN
    if not any(v for v in extended.values() if v is not None):
        return HandState.FIST
    if extended.get("index") and not extended.get("middle"):
        return HandState.POINT
    if extended.get("thumb") and not extended.get("index"):
        return HandState.THUMBS_UP

    # Pinch: thumb tip close to index tip
    if not np.any(np.isnan(hand_3d[4])) and not np.any(np.isnan(hand_3d[8])):
        if np.linalg.norm(hand_3d[4] - hand_3d[8]) < 0.030:   # 3 cm
            return HandState.PINCH

    return HandState.UNKNOWN
```

---

## 8. Gesture Event Extraction

### 8.1 Gesture Taxonomy for AffectAI

Define a hierarchical gesture vocabulary appropriate for the collaborative task context:

```yaml
# configs/gesture_taxonomy.yaml
gestures:

  # Deictic (pointing) gestures
  pointing_desk:
    description: "Pointing at desk surface / document"
    required: ["index_extended", "hand_near_desk"]
    min_duration_s: 0.3

  pointing_screen:
    description: "Pointing at big screen"
    required: ["index_extended", "hand_facing_screen"]
    min_duration_s: 0.3

  pointing_participant:
    description: "Pointing at another person"
    required: ["index_extended", "hand_facing_participant_zone"]
    min_duration_s: 0.3

  # Manipulation gestures
  tablet_interaction:
    description: "Hand touching / hovering tablet"
    required: ["hand_in_tablet_roi"]
    min_duration_s: 0.2

  writing_drawing:
    description: "Stylus or finger tracing on desk surface"
    required: ["hand_near_desk", "high_velocity_fine_motor"]
    min_duration_s: 0.5

  # Representational gestures
  beat_gesture:
    description: "Rhythmic hand movement during speech"
    required: ["hand_oscillating", "wrist_velocity_rhythmic"]
    min_duration_s: 0.4

  iconic_gesture:
    description: "Shape-tracing gesture (hands outline object)"
    required: ["both_hands_active", "symmetric_movement"]
    min_duration_s: 0.5

  # Social / regulatory
  head_nod:
    description: "Vertical head oscillation"
    required: ["nose_oscillating_vertical", "frequency_1_4hz"]
    min_duration_s: 0.4

  head_shake:
    description: "Horizontal head oscillation"
    required: ["nose_oscillating_horizontal", "frequency_1_4hz"]
    min_duration_s: 0.4

  lean_forward:
    description: "Upper body lean toward desk/screen"
    required: ["torso_angle_change", "head_forward_displacement"]
    min_duration_s: 0.8

  lean_back:
    description: "Upper body lean away from desk"
    required: ["torso_angle_change_back"]
    min_duration_s: 0.8
```

### 8.2 Feature Extraction for Gesture Detection

```python
def extract_gesture_features(
    skeleton_window: np.ndarray,    # (n_frames, 25, 7) window around current frame
    hand_l_window: np.ndarray,      # (n_frames, 21, 3) left hand
    hand_r_window: np.ndarray,      # (n_frames, 21, 3) right hand
    rois: dict[str, BoxROI],
    fps: float = 30.0,
) -> dict:
    """
    Compute gesture-relevant features for a temporal window.
    Returns a feature dict used by gesture classifiers.
    """
    dt = 1.0 / fps
    n = skeleton_window.shape[0]
    mid = n // 2   # current frame

    # Wrist positions and velocities
    rw = skeleton_window[:, 4, :3]   # right wrist trajectory
    lw = skeleton_window[:, 7, :3]   # left wrist trajectory

    rw_vel = np.linalg.norm(np.diff(rw, axis=0), axis=1) / dt
    lw_vel = np.linalg.norm(np.diff(lw, axis=0), axis=1) / dt

    # Head position and oscillation
    nose = skeleton_window[:, 0, :3]
    nose_vel_y = np.diff(nose[:, 1]) / dt   # vertical velocity
    nose_vel_x = np.diff(nose[:, 0]) / dt   # horizontal velocity

    # Torso angle (neck → mid-hip vector)
    neck    = skeleton_window[:, 1, :3]
    mid_hip = skeleton_window[:, 8, :3]
    torso_v = mid_hip - neck
    torso_angle = np.arctan2(torso_v[:, 1], torso_v[:, 2])   # sagittal tilt

    # ROI membership at current frame
    rw_roi = classify_point(rw[mid], rois, ROI_PRIORITY) if not np.any(np.isnan(rw[mid])) else "missing"
    lw_roi = classify_point(lw[mid], rois, ROI_PRIORITY) if not np.any(np.isnan(lw[mid])) else "missing"

    # Hand states (current frame)
    rh_state = classify_hand_state(hand_r_window[mid]) if hand_r_window is not None else HandState.UNKNOWN
    lh_state = classify_hand_state(hand_l_window[mid]) if hand_l_window is not None else HandState.UNKNOWN

    return {
        "rw_mean_velocity":     float(np.nanmean(rw_vel)),
        "lw_mean_velocity":     float(np.nanmean(lw_vel)),
        "rw_roi":               rw_roi,
        "lw_roi":               lw_roi,
        "rh_state":             rh_state.value,
        "lh_state":             lh_state.value,
        "nose_osc_vertical":    float(np.std(nose_vel_y)),
        "nose_osc_horizontal":  float(np.std(nose_vel_x)),
        "torso_tilt_mean":      float(np.nanmean(torso_angle)),
        "torso_tilt_change":    float(np.nanmax(torso_angle) - np.nanmin(torso_angle)),
    }
```

### 8.3 Gesture Segmentation — Sliding Window Detector

```python
def detect_gestures(
    skeleton: np.ndarray,       # (n_frames, 4, 25, 7) full session
    hand_l: np.ndarray,         # (n_frames, 4, 21, 3)
    hand_r: np.ndarray,
    rois: dict,
    fps: float = 30.0,
    window_frames: int = 45,    # 1.5 second window
    step_frames: int = 5,       # 5-frame step = 167 ms
) -> list[dict]:
    """
    Slide a temporal window over the session and emit gesture events.
    Returns list of gesture event dicts compatible with BIDS events.tsv schema.
    """
    events = []
    n_frames = skeleton.shape[0]
    half = window_frames // 2

    for person_idx in range(4):
        participant_id = f"P{person_idx + 1}"
        for fi in range(half, n_frames - half, step_frames):
            window = skeleton[fi-half:fi+half, person_idx]
            hl_win = hand_l[fi-half:fi+half, person_idx] if hand_l is not None else None
            hr_win = hand_r[fi-half:fi+half, person_idx] if hand_r is not None else None

            feats = extract_gesture_features(window, hl_win, hr_win, rois, fps)

            # Rule-based gesture detection (replace with classifier if labelled data available)
            gesture = _rule_based_gesture(feats)
            if gesture:
                events.append({
                    "onset":        fi / fps,
                    "duration":     window_frames / fps,
                    "trial_type":   "gesture",
                    "value":        gesture,
                    "participant":  participant_id,
                    "rw_roi":       feats["rw_roi"],
                    "lw_roi":       feats["lw_roi"],
                })

    return sorted(events, key=lambda e: e["onset"])


def _rule_based_gesture(feats: dict) -> Optional[str]:
    """Simple rule-based gesture classifier from feature dict."""
    # Pointing
    if (feats["rh_state"] == "point" and
            feats["rw_roi"] in ("big_screen", "zone_P1", "zone_P2", "zone_P3", "zone_P4")):
        return f"pointing_{feats['rw_roi']}"

    # Tablet interaction
    if feats["rw_roi"].startswith("tablet_") or feats["lw_roi"].startswith("tablet_"):
        return "tablet_interaction"

    # Head nod (vertical oscillation 1–4 Hz, window ~1.5s → 1–6 full cycles)
    if feats["nose_osc_vertical"] > 0.015 and feats["nose_osc_horizontal"] < 0.010:
        return "head_nod"

    # Head shake
    if feats["nose_osc_horizontal"] > 0.015 and feats["nose_osc_vertical"] < 0.010:
        return "head_shake"

    # Lean forward
    if feats["torso_tilt_change"] > 0.12 and feats["torso_tilt_mean"] > 0.05:
        return "lean_forward"

    # Beat gesture (high wrist velocity, no specific target)
    if feats["rw_mean_velocity"] > 0.40 and feats["rw_roi"] == "other":
        return "beat_gesture"

    return None
```

### 8.4 Output Schema — Gesture Events NDJSON

```json
{
  "onset_s": 45.23,
  "duration_s": 1.50,
  "gesture_type": "pointing_big_screen",
  "participant": "P2",
  "confidence": 0.82,
  "right_wrist_roi": "big_screen",
  "left_wrist_roi": "desk_surface",
  "right_hand_state": "point",
  "left_hand_state": "open",
  "right_wrist_pos_world": [0.42, -0.18, 0.75],
  "head_orientation_world": [0.05, 0.85, -0.53],
  "task": "T2"
}
```

Write to `<session>/mocap/gestures_events.ndjson` (already defined as a pipeline
output in `data_flow.md`).

---

## 9. Cross-Modality Integration — Gaze + Skeleton + ROI

### 9.1 Unified Per-Frame Feature Frame

Merge gaze, skeleton, and ROI data into a single per-frame feature record for
downstream ML and annotation:

```python
def build_feature_frame(
    frame_idx: int,
    skeleton: np.ndarray,         # (4, 25, 7)
    gaze_world: dict,             # {participant: (origin, direction, point_3d)}
    hand_states: dict,            # {participant: {left, right: HandState}}
    rois: dict,
    screen_events: pd.DataFrame,
    fps: float = 30.0,
) -> dict:
    ts = frame_idx / fps
    record = {"timestamp_s": ts, "frame_idx": frame_idx, "participants": {}}

    for p_idx, label in enumerate(["P1", "P2", "P3", "P4"]):
        sk = skeleton[p_idx]
        gaze = gaze_world.get(label, {})
        hands = hand_states.get(label, {})

        # Head position and orientation
        nose = sk[0, :3] if not np.any(np.isnan(sk[0, :3])) else None
        head_dir = head_orientation(sk)

        # Gaze target
        gaze_pt = gaze.get("point_3d")
        gaze_roi = classify_point(gaze_pt, rois, ROI_PRIORITY) if gaze_pt is not None else "missing"

        # Hand ROIs
        rw_pos  = sk[4, :3] if not np.any(np.isnan(sk[4, :3])) else None
        lw_pos  = sk[7, :3] if not np.any(np.isnan(sk[7, :3])) else None
        rw_roi  = classify_point(rw_pos, rois, ROI_PRIORITY) if rw_pos is not None else "missing"
        lw_roi  = classify_point(lw_pos, rois, ROI_PRIORITY) if lw_pos is not None else "missing"

        # Screen content if looking at screen
        screen_content = None
        if gaze_roi == "big_screen" and gaze_pt is not None:
            screen_content = gaze_screen_content(ts, screen_events, rois["big_screen"], gaze_pt)

        record["participants"][label] = {
            "head_pos_world":      nose.tolist() if nose is not None else None,
            "head_dir_world":      head_dir.tolist() if head_dir is not None else None,
            "gaze_roi":            gaze_roi,
            "gaze_point_world":    gaze_pt.tolist() if gaze_pt is not None else None,
            "gaze_validity":       gaze.get("validity", "unknown"),
            "rw_roi":              rw_roi,
            "lw_roi":              lw_roi,
            "rh_state":            hands.get("right", HandState.UNKNOWN).value,
            "lh_state":            hands.get("left",  HandState.UNKNOWN).value,
            "screen_content":      screen_content,
            "skeleton_conf":       skeleton_confidence(sk),
        }

    return record
```

### 9.2 Mutual Gaze Detection

Two participants are in mutual gaze when each is looking toward the other's head ROI:

```python
def detect_mutual_gaze(
    feature_frame: dict,
    head_radius_m: float = 0.15,
) -> list[tuple[str, str]]:
    """
    Returns list of (P_i, P_j) pairs in mutual gaze at this frame.
    """
    participants = feature_frame["participants"]
    pairs = []

    labels = list(participants.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pi, pj = labels[i], labels[j]
            gi = participants[pi].get("gaze_roi", "")
            gj = participants[pj].get("gaze_roi", "")
            # Check if Pi's gaze ROI is Pj's zone and vice versa
            if (f"zone_{pj}" in gi and f"zone_{pi}" in gj):
                pairs.append((pi, pj))

    return pairs
```

### 9.3 Joint Attention Detection

Detect when two or more participants look at the same ROI simultaneously:

```python
def detect_joint_attention(
    feature_frame: dict,
    min_participants: int = 2,
) -> list[dict]:
    """
    Returns list of joint attention events: {roi, participants, timestamp}.
    """
    from collections import defaultdict
    roi_to_participants = defaultdict(list)

    for label, data in feature_frame["participants"].items():
        roi = data.get("gaze_roi", "missing")
        if roi not in ("missing", "other"):
            roi_to_participants[roi].append(label)

    events = []
    for roi, ps in roi_to_participants.items():
        if len(ps) >= min_participants:
            events.append({
                "roi": roi,
                "participants": ps,
                "n_participants": len(ps),
                "timestamp_s": feature_frame["timestamp_s"],
            })
    return events
```

---

## 10. Quality Control for ROI and Skeleton

### 10.1 Skeleton Coverage Report

Add to `qc_tobii_world_gaze.py` or create `qc_skeleton_roi.py`:

| Metric | Good | Warn | Fail |
|--------|------|------|------|
| Skeleton valid frames per person | > 85% | 60–85% | < 60% |
| Head keypoint (nose) valid | > 90% | 70–90% | < 70% |
| Both wrists valid simultaneously | > 75% | 50–75% | < 50% |
| Mean bone-length violation rate | < 5% | 5–15% | > 15% |
| Mean reproj error (accepted joints) | < 10 px | 10–20 px | > 20 px |
| Seat assignment stability | > 98% | 95–98% | < 95% |

### 10.2 ROI Sanity Checks

```python
def validate_roi_assignments(feature_frames: list[dict]) -> dict:
    """
    Check for implausible ROI assignments that indicate calibration issues.
    e.g., participant head classified as 'big_screen' or 'tablet' ROI.
    """
    warnings = []
    for ff in feature_frames:
        ts = ff["timestamp_s"]
        for label, data in ff["participants"].items():
            head_pos = data.get("head_pos_world")
            if head_pos is not None:
                head_roi = classify_point(np.array(head_pos), rois, ROI_PRIORITY)
                # Head should always be in its own zone or 'other'
                if head_roi.startswith("tablet_") or head_roi == "big_screen":
                    warnings.append({
                        "timestamp_s": ts,
                        "participant": label,
                        "issue": f"Head classified as {head_roi} — check skeleton calibration",
                    })
    return {"n_warnings": len(warnings), "warnings": warnings[:20]}  # cap at 20
```

### 10.3 Gesture Plausibility Check

```python
def validate_gesture_events(events: list[dict], fps: float = 30.0) -> dict:
    issues = []
    for ev in events:
        # Gestures shorter than 200ms or longer than 10s are suspect
        if ev["duration_s"] < 0.2:
            issues.append(f"Too-short gesture ({ev['duration_s']*1000:.0f}ms): {ev['gesture_type']}")
        if ev["duration_s"] > 10.0:
            issues.append(f"Too-long gesture ({ev['duration_s']:.1f}s): {ev['gesture_type']}")

    # Check gesture rate is plausible (< 30 gestures/min per person)
    by_person = {}
    for ev in events:
        by_person.setdefault(ev["participant"], []).append(ev)

    for label, person_events in by_person.items():
        if not person_events:
            continue
        session_min = (person_events[-1]["onset_s"] - person_events[0]["onset_s"]) / 60.0
        rate = len(person_events) / max(session_min, 0.1)
        if rate > 30:
            issues.append(f"{label}: {rate:.0f} gestures/min — likely false positives")

    return {"n_issues": len(issues), "issues": issues}
```

---

## 11. Priority Order

| Priority | Section | Task | Effort | Impact |
|----------|---------|------|--------|--------|
| 🔴 Critical | **§1.2** | Create `configs/scene_rois.yaml` with static ROI definitions | Low | Enables all downstream ROI analysis |
| 🔴 Critical | **§2.2** | Seat-based identity lock — prevent skeleton ID swaps | Low | Consistent person tracking across all sessions |
| 🔴 Critical | **§6.2** | Fix camera zone assignment (cam1+cam3, cam2+cam4) | Low | Improves triangulation baseline by ~4× |
| 🟠 High | **§4.2** | Add ArUco markers (IDs 20–23) to tablets | Low (hardware) | Precise gaze-on-tablet detection |
| 🟠 High | **§6.4** | Bone-length enforcement + velocity outlier rejection | Medium | Cleaner skeleton, fewer gross errors |
| 🟠 High | **§7.2** | 3D hand keypoint triangulation from face_hand_pipeline | Medium | Enables all hand/gesture features |
| 🟠 High | **§8.3** | Sliding-window gesture detector + NDJSON output | Medium | Completes gesture extraction pipeline |
| 🟡 Medium | **§2.3** | Head pose from 3D skeleton keypoints | Low | Mutual gaze without eye tracker |
| 🟡 Medium | **§2.4** | Body-part ROI assignment per frame | Low | Fine-grained interaction analysis |
| 🟡 Medium | **§3.2** | Moderator zone + motion detector | Medium | Removes moderator as noise source |
| 🟡 Medium | **§5.2** | Screen content ↔ gaze integration | Medium | Task-aligned gaze AOI analysis |
| 🟡 Medium | **§9.2** | Mutual gaze detection | Low | Key social behavior metric |
| 🟡 Medium | **§9.3** | Joint attention detection | Low | Key group behavior metric |
| 🟡 Medium | **§6.3** | Upper-body-only mode for seated participants | Low | Faster pipeline, less noise |
| 🟢 Lower | **§6.5** | Per-frame skeleton confidence score | Low | QC and downstream filtering |
| 🟢 Lower | **§7.3** | Hand state classifier (open/fist/point/pinch) | Medium | Richer gesture vocabulary |
| 🟢 Lower | **§10.1** | Skeleton + ROI QC report script | Medium | Catch data quality issues early |
| 🟢 Lower | **§9.1** | Unified per-frame feature frame builder | High | Enables ML-ready feature dataset |

---

*Document generated 2026-04-19 — based on AffectAI repository state and lab physical setup.*
*Coordinate with gaze improvement plan (`3d_gaze_improvement_plan.md`) for shared dependencies*
*(ArUco ID collision fix, factory intrinsics, calibration quality).*
