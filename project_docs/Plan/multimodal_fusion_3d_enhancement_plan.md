# AffectAI — Multi-Modal Fusion for Enhanced 3D Gaze, Gesture & ROI Detection

> **Context**: This document describes how auxiliary sensor streams — DPA close-talk
> audio (5 channels), Tobii IMU (accelerometer + gyroscope per glasses unit), and
> tablet interaction logs (touch events, response timestamps) — can be fused with the
> 3D vision pipeline to improve gaze classification, gesture detection, and ROI
> attribution. All streams share a common LSL timebase (timestamps in seconds).
>
> **Prerequisites**: Read `3d_gaze_improvement_plan.md` and
> `3d_roi_gesture_skeleton_plan.md` first. This document builds on top of those
> pipelines, not in isolation.

---

## Table of Contents

1. [Fusion Philosophy — Why Each Modality Helps](#1-fusion-philosophy--why-each-modality-helps)
2. [Tablet Interaction Logs as Ground Truth](#2-tablet-interaction-logs-as-ground-truth)
3. [IMU-Aided Glasses Pose Estimation](#3-imu-aided-glasses-pose-estimation)
4. [IMU for Head Gesture Detection](#4-imu-for-head-gesture-detection)
5. [Audio for Speech-Gesture Coupling](#5-audio-for-speech-gesture-coupling)
6. [Audio for Speaker Diarisation and Gaze Grounding](#6-audio-for-speaker-diarisation-and-gaze-grounding)
7. [Audio Energy as ROI Attention Prior](#7-audio-energy-as-roi-attention-prior)
8. [Cross-Modal Confidence Weighting](#8-cross-modal-confidence-weighting)
9. [Weak Supervision — Using Logs to Train Better Detectors](#9-weak-supervision--using-logs-to-train-better-detectors)
10. [Unified Multi-Modal Feature Frame](#10-unified-multi-modal-feature-frame)
11. [Quality Control and Failure Mode Analysis](#11-quality-control-and-failure-mode-analysis)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Fusion Philosophy — Why Each Modality Helps

Each auxiliary stream provides a different type of information that compensates
for specific failure modes in the 3D vision pipeline:

```
Failure mode                      Best compensating modality
──────────────────────────────────────────────────────────────────────
Glasses markers occluded          IMU integration → dead-reckoning pose
Scene video markers occluded      IMU + tablet logs → gaze prior
Low gaze validity (blinks, etc.)  IMU head orientation as fallback
Ambiguous gaze target (desk vs.   Tablet touch log → confirmed tablet gaze
  tablet vs. hand)
Gesture detector misfire during   Audio VAD → suppress gesture events
  stillness                         during silence
Speaker identity unknown          DPA per-mic energy → assign speech acts
Skeleton ID swap                  IMU trajectory continuity → re-link
Tablet AOI subdivision unknown    Touch coordinates from log → exact hit
Camera occluded by participant    Audio + IMU confirm activity when
  body                              vision is blind
```

### 1.1 Data Availability Summary

| Stream | Source | Rate | LSL stream name | Latency |
|--------|--------|------|-----------------|---------|
| Gaze (2D+3D) | Tobii SDK | 50 Hz | `Tobii_P#_Gaze` | real-time |
| IMU | Tobii glasses | 100 Hz | `Tobii_P#_Imu` | real-time |
| Scene video | Tobii | 25 fps | file only | offline |
| DPA close-talk | RME Fireface | 48 kHz | `EmotiBit` (via LSL) | real-time |
| EmotiBit EDA/PPG | EmotiBit | 15–25 Hz | `EmotiBit_P#_*` | real-time |
| Tablet events | display_server | event-driven | `AffectAI_Tablet#` | real-time |
| Fixed cameras | ffmpeg_multicap | 30 fps | frame logs | offline |

### 1.2 Temporal Alignment Reminder

All fusion operations require timestamps to be on the **same clock**. Your LSL
master clock already aligns these streams. When loading NDJSON or TSV files:

```python
# Tobii gaze NDJSON: timestamp_ticks = LSL seconds (ticks_per_second=1)
gaze_ts = record["timestamp_ticks"]          # already LSL seconds

# IMU from same NDJSON or separate stream
imu_ts  = imu_record["timestamp_ticks"]      # LSL seconds

# Tablet events from AffectAI_Tablet LSL stream (BIDS TSV)
tablet_ts = tablet_event["onset"]            # seconds from session start

# Audio: derive frame timestamps from sample index
audio_ts  = sample_idx / 48000.0 + audio_lsl_start
```

---

## 2. Tablet Interaction Logs as Ground Truth

### 2.1 What the Logs Contain

The tablet display server (`stimuli/display_server.py`) logs every user interaction
to `AffectAI_Tablet{1-4}` LSL streams and to BIDS `beh/` TSV files:

```
onset       duration  event_type         value         participant
45.230      0.000     probe_displayed    va_grid        P2
47.891      0.000     touch_start        va_grid        P2
48.340      0.000     touch_end          va_grid        P2
48.340      0.449     response_logged    {v:0.6,a:0.4}  P2
```

Key events for gaze/ROI fusion:

| Event type | What it tells us |
|------------|-----------------|
| `probe_displayed` | Tablet screen turned on — participant will look at it soon |
| `touch_start` | **Confirmed**: participant is looking at tablet right now |
| `touch_end` | End of confirmed tablet gaze window |
| `response_logged` | Touch coordinates available if logged (add to display_server) |
| `task_start` / `task_end` | Task phase — constrains which screen content is shown |

### 2.2 Confirmed Gaze Windows

Between `touch_start` and `touch_end` (plus a ±500 ms buffer for saccade/fixation),
the participant's gaze is **confirmed to be on their tablet**. Use this as a hard
constraint to:

1. Override ambiguous gaze classifications during this window
2. Evaluate and calibrate the 3D gaze pipeline accuracy
3. Provide training labels for the weak supervision approach (§9)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ConfirmedGazeWindow:
    participant: str          # "P1" – "P4"
    start_s: float            # touch_start - buffer
    end_s: float              # touch_end   + buffer
    roi: str                  # e.g. "tablet_P2"
    touch_xy_norm: tuple      # normalised touch coords if available
    confidence: float = 1.0  # 1.0 = hard constraint


def extract_confirmed_windows(
    tablet_events: list[dict],
    buffer_s: float = 0.500,
) -> list[ConfirmedGazeWindow]:
    """
    Parse tablet event log into confirmed gaze windows.
    Each touch_start/touch_end pair becomes one window.
    """
    windows = []
    pending = {}   # participant -> touch_start event

    for ev in sorted(tablet_events, key=lambda e: e["onset"]):
        pid = ev.get("participant", "unknown")
        etype = ev.get("event_type", "")

        if etype == "touch_start":
            pending[pid] = ev

        elif etype == "touch_end" and pid in pending:
            start_ev = pending.pop(pid)
            windows.append(ConfirmedGazeWindow(
                participant=pid,
                start_s=start_ev["onset"] - buffer_s,
                end_s=ev["onset"] + buffer_s,
                roi=f"tablet_{pid}",
                touch_xy_norm=ev.get("touch_xy_norm", (0.5, 0.5)),
            ))

    return windows


def override_gaze_with_confirmed(
    gaze_records: list[dict],
    confirmed_windows: list[ConfirmedGazeWindow],
) -> list[dict]:
    """
    For gaze samples inside a confirmed window, override gaze_roi
    and boost confidence. Mark source as 'tablet_log'.
    """
    for rec in gaze_records:
        ts  = rec["timestamp_s"]
        pid = rec["participant"]
        for win in confirmed_windows:
            if win.participant == pid and win.start_s <= ts <= win.end_s:
                rec["gaze_roi"]       = win.roi
                rec["gaze_roi_conf"]  = win.confidence
                rec["gaze_roi_source"] = "tablet_log"
                if win.touch_xy_norm:
                    rec["tablet_uv"]  = win.touch_xy_norm
                break
    return gaze_records
```

### 2.3 Probe-Displayed as Predictive Prior

When a probe is displayed (`probe_displayed` event), the participant will very likely
look at the tablet within the next 0–3 seconds. Use this as a **soft prior** that
increases the tablet ROI probability for ambiguous gaze samples in this window:

```python
def apply_probe_prior(
    gaze_record: dict,
    probe_events: list[dict],
    prior_window_s: tuple = (0.0, 3.0),  # seconds after probe display
    prior_boost: float = 0.30,           # add to tablet ROI probability
) -> dict:
    ts  = gaze_record["timestamp_s"]
    pid = gaze_record["participant"]

    for ev in probe_events:
        if ev["participant"] != pid:
            continue
        dt = ts - ev["onset"]
        if prior_window_s[0] <= dt <= prior_window_s[1]:
            # Scale boost linearly: full boost at dt=0, zero at dt=window_end
            scale = 1.0 - dt / prior_window_s[1]
            gaze_record.setdefault("roi_priors", {})[f"tablet_{pid}"] = \
                prior_boost * scale
            break

    return gaze_record
```

### 2.4 Touch Coordinates for Sub-Tablet AOI

If `display_server.py` logs the normalised touch coordinate `(u, v)` within the
tablet screen, you get **exact screen AOI labels for free** during interaction windows:

```python
# Add to display_server.py touch handler:
touch_payload = {
    "event_type": "touch_end",
    "participant": participant_id,
    "onset": lsl_clock(),
    "touch_xy_norm": [event.x / SCREEN_WIDTH, event.y / SCREEN_HEIGHT],
    "screen_content": current_stimulus_id,
}
```

Map to tablet AOI zones defined in `3d_roi_gesture_skeleton_plan.md §4.4`:

```python
def touch_to_tablet_aoi(touch_uv: tuple, tablet_zones: dict) -> str:
    u, v = touch_uv
    for zone_name, (x0, y0, x1, y1) in tablet_zones.items():
        if x0 <= u <= x1 and y0 <= v <= y1:
            return zone_name
    return "outside"
```

---

## 3. IMU-Aided Glasses Pose Estimation

### 3.1 What Tobii IMU Provides

The Tobii Pro Glasses 3 IMU stream (`Tobii_P#_Imu`, 100 Hz) contains:

```
channels (9):
  accel_x, accel_y, accel_z   [m/s²]  — linear acceleration in glasses frame
  gyro_x,  gyro_y,  gyro_z    [rad/s] — angular velocity in glasses frame
  mag_x,   mag_y,   mag_z     [µT]    — magnetometer (heading reference)
```

The glasses-frame axes are defined by Tobii's factory calibration (same
`T_glasses_sceneCamera` transform discussed in the gaze plan §3).

### 3.2 IMU Integration for Dead-Reckoning

When both Approach A (glasses markers) and Approach B (scene video PnP) fail —
because the participant leans forward, occludes markers, and the scene video shows
only the desk — IMU integration provides a short-term pose estimate:

```python
import numpy as np
from scipy.spatial.transform import Rotation

class IMUDeadReckoning:
    """
    Propagate glasses orientation using gyroscope integration.
    Position is NOT reliably recoverable from accelerometer alone
    (double-integration drift), but orientation is usable for ~2–5 seconds.
    """

    def __init__(self, R_init: np.ndarray, t_init: np.ndarray,
                 bias_window_s: float = 2.0, fps_imu: float = 100.0):
        self.R = R_init.copy()          # 3×3 rotation matrix (world ← glasses)
        self.t = t_init.copy()          # (3,) position — held constant during DR
        self.gyro_bias = np.zeros(3)    # estimated from still period before task
        self.dt = 1.0 / fps_imu
        self.drift_age_s = 0.0          # seconds since last vision fix

    def calibrate_bias(self, gyro_samples: np.ndarray):
        """
        Estimate gyroscope bias from a stationary period.
        gyro_samples: (N, 3) during known-still period (e.g., calibration phase T0).
        """
        self.gyro_bias = np.mean(gyro_samples, axis=0)

    def update(self, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Integrate one IMU sample.
        Returns (R_current, t_current, drift_confidence).
        drift_confidence decays from 1.0 at vision fix to 0.0 after 5 seconds.
        """
        omega = gyro - self.gyro_bias           # bias-corrected angular velocity
        angle = np.linalg.norm(omega) * self.dt
        if angle > 1e-8:
            axis = omega / np.linalg.norm(omega)
            dR   = Rotation.from_rotvec(axis * angle).as_matrix()
            self.R = self.R @ dR

        self.drift_age_s += self.dt
        confidence = max(0.0, 1.0 - self.drift_age_s / 5.0)
        return self.R.copy(), self.t.copy(), confidence

    def reset_from_vision(self, R_vision: np.ndarray, t_vision: np.ndarray):
        """Called when a vision-based pose estimate is available."""
        self.R = R_vision.copy()
        self.t = t_vision.copy()
        self.drift_age_s = 0.0
```

### 3.3 Fusing Vision Pose + IMU Dead-Reckoning

Extend the Approach C fusion from `3d_gaze_improvement_plan.md §6` with an IMU tier:

```python
def fuse_pose_with_imu(
    pose_vision: tuple,           # (R, t) from A/B fusion, or None
    conf_vision: float,           # reprojection error (lower = better)
    imu_dr: IMUDeadReckoning,     # running dead-reckoning state
    gyro_sample: np.ndarray,      # current IMU gyro reading
    vision_threshold_px: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (R_fused, t_fused, source_label).
    source_label: 'vision_A', 'vision_B', 'vision_fused', 'imu_dr', 'imu_dr_stale'
    """
    R_imu, t_imu, imu_conf = imu_dr.update(gyro_sample)

    if pose_vision is not None and conf_vision < vision_threshold_px:
        # Good vision fix — reset dead-reckoning and use vision
        imu_dr.reset_from_vision(*pose_vision)
        return pose_vision[0], pose_vision[1], "vision"

    elif imu_conf > 0.3:
        # Vision failed but IMU is recent enough
        if pose_vision is not None:
            # Blend: weight vision by (1-imu_conf), IMU by imu_conf
            # (If vision is poor but exists, blend with IMU)
            w_v = (1.0 - imu_conf) * (1.0 / (conf_vision + 1e-6))
            w_i = imu_conf
            w_total = w_v + w_i
            t_fused = (w_v * pose_vision[1] + w_i * t_imu) / w_total
            key_rots = Rotation.from_matrix([pose_vision[0], R_imu])
            R_fused  = Slerp([0, 1], key_rots)(w_i / w_total).as_matrix()
            return R_fused, t_fused, "vision_imu_blend"
        else:
            return R_imu, t_imu, "imu_dr"

    else:
        # Both failed — interpolate between last known poses
        return R_imu, t_imu, "imu_dr_stale"
```

### 3.4 IMU Magnitude as Occlusion Predictor

High IMU acceleration predicts that the participant is moving — and moving often
causes marker occlusion. Use this as a **pre-emptive confidence penalty** on the
vision pipeline outputs:

```python
def imu_motion_score(accel: np.ndarray, gyro: np.ndarray,
                      gravity_g: float = 9.81) -> float:
    """
    Returns motion intensity in [0, 1].
    0 = completely still, 1 = maximum expected motion.
    Gravity is subtracted from accelerometer norm.
    """
    accel_norm = np.linalg.norm(accel)
    dynamic_accel = abs(accel_norm - gravity_g)   # remove gravity component
    gyro_norm  = np.linalg.norm(gyro)

    # Normalise to expected ranges for seated participant:
    # dynamic accel: 0–5 m/s², gyro: 0–3 rad/s
    accel_score = min(dynamic_accel / 5.0, 1.0)
    gyro_score  = min(gyro_norm    / 3.0, 1.0)

    return float(0.5 * accel_score + 0.5 * gyro_score)


def penalise_vision_confidence(conf_px: float,
                                motion_score: float,
                                penalty_scale: float = 5.0) -> float:
    """
    Increase effective reprojection error threshold penalty
    when the participant is moving (marker corners will be motion-blurred).
    """
    return conf_px + motion_score * penalty_scale
```

---

## 4. IMU for Head Gesture Detection

### 4.1 Why IMU Beats Skeleton for Head Gestures

The skeleton pipeline (MediaPipe, 30 fps) detects head nods and shakes from nose
keypoint oscillation. This has two problems:
1. **Low temporal resolution**: 30 fps misses fast gestures (nod frequency up to 4 Hz
   requires at least 8 fps — barely met at 30 fps)
2. **Occlusion**: When a participant leans forward, the face disappears from overhead
   cameras

The Tobii IMU runs at **100 Hz** and is always attached to the participant's head.
It is the ideal sensor for head gesture detection.

### 4.2 Head Nod Detector

```python
from scipy.signal import butter, filtfilt, find_peaks

def detect_head_nods(
    gyro_stream: np.ndarray,    # (N, 3) angular velocity in glasses frame
    timestamps: np.ndarray,     # (N,) LSL seconds
    axis: int = 0,              # 0=pitch (nod), 1=roll, 2=yaw (shake)
    freq_band_hz: tuple = (0.8, 4.0),
    min_amplitude_rads: float = 0.08,   # ~5 degrees peak
    min_duration_s: float = 0.25,
) -> list[dict]:
    """
    Detect rhythmic head oscillation events from gyroscope stream.
    Returns list of {onset_s, duration_s, n_cycles, amplitude_rads, gesture_type}.
    """
    fps_imu = 1.0 / np.mean(np.diff(timestamps))

    # Band-pass filter around expected nod frequency
    b, a = butter(2, [freq_band_hz[0] / (fps_imu / 2),
                      freq_band_hz[1] / (fps_imu / 2)], btype='band')
    signal = filtfilt(b, a, gyro_stream[:, axis])

    # Find positive peaks (half-cycles)
    min_distance = int(fps_imu / freq_band_hz[1])   # min samples between peaks
    peaks, props = find_peaks(
        np.abs(signal),
        height=min_amplitude_rads,
        distance=min_distance,
    )

    if len(peaks) < 2:
        return []

    # Group consecutive peaks into gesture events
    events = []
    group_start = 0
    for i in range(1, len(peaks)):
        gap_s = timestamps[peaks[i]] - timestamps[peaks[i-1]]
        if gap_s > 1.0 / freq_band_hz[0]:   # gap too large → new gesture
            if i - group_start >= 2:          # at least 2 peaks = 1 cycle
                onset = timestamps[peaks[group_start]]
                end   = timestamps[peaks[i-1]]
                if end - onset >= min_duration_s:
                    events.append({
                        "onset_s":        float(onset),
                        "duration_s":     float(end - onset),
                        "n_cycles":       (i - group_start) // 2,
                        "amplitude_rads": float(np.mean(np.abs(signal[peaks[group_start:i]]))),
                        "gesture_type":   "head_nod" if axis == 0 else "head_shake",
                    })
            group_start = i

    return events
```

### 4.3 IMU-Derived Head Orientation for Gaze Grounding

Even without the vision pipeline, the IMU provides head orientation in the glasses
frame. Combined with the last known world pose (from vision), this gives a
**continuous head orientation estimate** useful for determining which direction the
participant is facing:

```python
def head_facing_direction(
    R_world_glasses: np.ndarray,   # 3×3 rotation: glasses → world
    glasses_forward: np.ndarray = np.array([0, 0, 1]),  # Tobii scene camera forward axis
) -> np.ndarray:
    """
    Returns unit vector in world frame representing head forward direction.
    Used for coarse gaze target estimation when eye tracking is unavailable.
    """
    forward_world = R_world_glasses @ glasses_forward
    return forward_world / np.linalg.norm(forward_world)


def estimate_coarse_gaze_target(
    head_pos_world: np.ndarray,
    head_forward_world: np.ndarray,
    rois: dict,
    max_angle_deg: float = 30.0,
) -> str:
    """
    Estimate which ROI the participant is facing using head orientation.
    Used as fallback when Tobii gaze validity is low.
    Returns ROI name or 'unknown'.
    """
    max_cos = np.cos(np.radians(max_angle_deg))

    for roi_name, roi in rois.items():
        direction_to_roi = roi.center - head_pos_world
        dist = np.linalg.norm(direction_to_roi)
        if dist < 1e-6:
            continue
        direction_to_roi /= dist
        cos_angle = np.dot(head_forward_world, direction_to_roi)
        if cos_angle >= max_cos:
            return roi_name

    return "unknown"
```

---

## 5. Audio for Speech-Gesture Coupling

### 5.1 The Speech-Gesture Link

Decades of gesture research show that **iconic and beat gestures are temporally
coupled with speech**: gestures peak at or slightly before the stressed syllable
they accompany. This means:

- If audio shows a participant is speaking, gesture detector sensitivity should
  increase (gestures are more likely during speech)
- If audio shows silence, hand movements are less likely to be communicative
  gestures (but may be self-touch or task manipulation)
- Beat gestures should only be labelled during speech — silence voids beat gesture
  detections

### 5.2 Per-Participant Voice Activity Detection

Each participant has a dedicated DPA close-talk microphone. Use per-mic energy to
build a frame-level VAD:

```python
import numpy as np
from scipy.signal import stft

def compute_vad(
    audio_signal: np.ndarray,      # (N,) mono audio at 48 kHz
    sample_rate: int = 48000,
    frame_duration_s: float = 0.033,  # 33ms = ~1 video frame at 30fps
    energy_threshold_db: float = -40.0,
    smoothing_frames: int = 5,
) -> np.ndarray:
    """
    Returns binary VAD array at video frame rate (approx. 30 fps).
    1 = speech active, 0 = silence.
    """
    frame_samples = int(frame_duration_s * sample_rate)
    n_frames = len(audio_signal) // frame_samples

    energy_db = np.zeros(n_frames)
    for i in range(n_frames):
        chunk = audio_signal[i * frame_samples: (i + 1) * frame_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        energy_db[i] = 20 * np.log10(rms + 1e-10)

    # Threshold + smooth to avoid rapid switching
    vad_raw = (energy_db > energy_threshold_db).astype(float)
    kernel = np.ones(smoothing_frames) / smoothing_frames
    vad_smooth = np.convolve(vad_raw, kernel, mode='same')
    return (vad_smooth > 0.5).astype(int)


# Map participant to DPA microphone channel
MIC_CHANNEL = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "room": 4}
```

### 5.3 Gesture Suppression During Silence

Apply VAD as a gate on gesture event emission:

```python
def gate_gestures_by_vad(
    gesture_events: list[dict],
    vad_per_participant: dict[str, np.ndarray],  # {pid: vad_array at fps}
    fps: float = 30.0,
    communicative_gestures: set = frozenset({
        "beat_gesture", "iconic_gesture", "pointing_participant",
        "head_nod", "head_shake",
    }),
    silence_buffer_s: float = 0.300,   # allow gestures up to 300ms before speech ends
) -> list[dict]:
    """
    Remove communicative gesture events that occur entirely during silence.
    Non-communicative gestures (tablet_interaction, pointing_desk) are kept.
    """
    gated = []
    for ev in gesture_events:
        if ev["gesture_type"] not in communicative_gestures:
            gated.append(ev)
            continue

        pid = ev["participant"]
        vad = vad_per_participant.get(pid)
        if vad is None:
            gated.append(ev)   # no VAD data → keep
            continue

        # Check if any speech in event window + buffer
        start_f = max(0, int((ev["onset_s"] - silence_buffer_s) * fps))
        end_f   = min(len(vad) - 1, int((ev["onset_s"] + ev["duration_s"]) * fps))
        if np.any(vad[start_f:end_f + 1]):
            gated.append(ev)   # speech present → keep gesture
        # else: drop — no speech during this gesture window

    return gated
```

### 5.4 Turn-Taking Features from Audio

Turn-taking patterns (who speaks when, overlaps, gaps) can be derived from
per-participant VAD and used to contextualise gesture and gaze:

```python
def extract_turn_features(
    vad_dict: dict[str, np.ndarray],   # {pid: vad_array}
    fps: float = 30.0,
) -> np.ndarray:
    """
    Per-frame turn-taking feature vector:
    [n_speakers, is_overlap, gap_since_last_speech_s, current_speaker_P1..P4]
    Shape: (n_frames, 4 + n_participants)
    """
    participants = sorted(vad_dict.keys())
    n_frames = max(len(v) for v in vad_dict.values())
    n_p = len(participants)

    features = np.zeros((n_frames, 2 + 1 + n_p))

    for fi in range(n_frames):
        active = [vad_dict[pid][fi] if fi < len(vad_dict[pid]) else 0
                  for pid in participants]
        n_speakers = sum(active)
        is_overlap = int(n_speakers > 1)

        # Gap since last speech (any speaker)
        gap = 0
        for back in range(fi - 1, max(0, fi - int(5 * fps)), -1):
            if any(vad_dict[pid][back] for pid in participants
                   if back < len(vad_dict[pid])):
                break
            gap += 1
        gap_s = gap / fps

        features[fi, 0] = n_speakers
        features[fi, 1] = is_overlap
        features[fi, 2] = gap_s
        features[fi, 3:3 + n_p] = active

    return features
```

---

## 6. Audio for Speaker Diarisation and Gaze Grounding

### 6.1 Speaker Identity → Addressee Prediction

When a participant is speaking, their listeners tend to look at the speaker. This
creates a predictable gaze pattern that can be used to:

1. **Validate** gaze classifications (if P1 is speaking, P2/P3/P4 should gaze
   toward P1's zone)
2. **Impute** missing gaze data (if Tobii validity is 0 but P1 is speaking,
   assume P2 is gazing at zone_P1)
3. **Detect anomalies** (if P1 is speaking but nobody looks at them — potential
   disengagement event)

```python
def predict_listener_gaze(
    speaker_id: str,
    listener_ids: list[str],
    speaker_zone: str,
    gaze_records: list[dict],
    imputation_conf: float = 0.40,   # lower than hard evidence
) -> list[dict]:
    """
    For listeners with missing gaze during speaker_id's speech,
    impute gaze_roi = speaker's zone with imputation_conf confidence.
    """
    for rec in gaze_records:
        pid = rec["participant"]
        if pid not in listener_ids:
            continue
        if rec.get("gaze_validity") == "valid":
            continue   # have real data — don't impute
        if rec.get("gaze_roi_source") == "tablet_log":
            continue   # confirmed tablet gaze — don't override

        rec["gaze_roi"]        = speaker_zone
        rec["gaze_roi_conf"]   = imputation_conf
        rec["gaze_roi_source"] = "speaker_diarisation_imputed"

    return gaze_records
```

### 6.2 Acoustic Features for Gesture Coupling

Extract acoustic features time-aligned to video frames for downstream ML:

```python
def extract_acoustic_features(
    audio_signal: np.ndarray,
    sample_rate: int = 48000,
    target_fps: float = 30.0,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Returns per-video-frame acoustic feature matrix.
    Features: [energy_db, zero_crossing_rate, mfcc_1..13, pitch_hz, voiced]
    Shape: (n_frames, 16 + n_mfcc)
    """
    import librosa

    frame_length = int(sample_rate / target_fps)
    hop_length   = frame_length

    # Energy
    rms = librosa.feature.rms(y=audio_signal,
                               frame_length=frame_length,
                               hop_length=hop_length)[0]
    energy_db = 20 * np.log10(rms + 1e-10)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_signal,
                                              frame_length=frame_length,
                                              hop_length=hop_length)[0]

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate,
                                  n_mfcc=n_mfcc,
                                  hop_length=hop_length)

    # Pitch (fundamental frequency)
    f0, voiced_flag, _ = librosa.pyin(audio_signal,
                                       fmin=80, fmax=400,
                                       sr=sample_rate,
                                       hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)

    n_frames = min(len(energy_db), mfcc.shape[1], len(f0))
    return np.column_stack([
        energy_db[:n_frames],
        zcr[:n_frames],
        mfcc[:, :n_frames].T,
        f0[:n_frames],
        voiced_flag[:n_frames].astype(float),
    ])
```

---

## 7. Audio Energy as ROI Attention Prior

### 7.1 Acoustic Salience and Attention

Sudden loud sounds draw attention. If the big screen emits a sound cue (task
instruction, timer beep), participants will orient toward it. Similarly, a
participant speaking loudly will attract more gaze from others.

Use per-frame acoustic salience to boost ROI priors:

```python
def acoustic_salience_priors(
    audio_energy_db: dict[str, np.ndarray],  # {source: energy array at fps}
    roi_to_source: dict[str, str],            # {"big_screen": "screen_audio",
                                              #  "zone_P1": "mic_P1", ...}
    baseline_db: float = -50.0,
    max_boost: float = 0.25,
) -> np.ndarray:
    """
    Returns per-frame ROI prior boost matrix.
    Shape: (n_frames, n_rois) — values in [0, max_boost].

    When a ROI's associated audio source is loud relative to baseline,
    the prior probability of gaze to that ROI increases.
    """
    n_frames = max(len(v) for v in audio_energy_db.values())
    roi_names = list(roi_to_source.keys())
    priors = np.zeros((n_frames, len(roi_names)))

    for ri, roi_name in enumerate(roi_names):
        source = roi_to_source.get(roi_name)
        if source not in audio_energy_db:
            continue
        energy = audio_energy_db[source]
        # Normalise: 0 at baseline, 1 at baseline + 20 dB
        boost_raw = np.clip((energy - baseline_db) / 20.0, 0.0, 1.0)
        priors[:len(boost_raw), ri] = boost_raw * max_boost

    return priors
```

### 7.2 Crosstalk Detection and Suppression

DPA microphones have < −30 dB crosstalk target (from `configs/lab_small.yaml`).
Verify this before using individual mic energy for speaker attribution, and flag
frames where crosstalk may corrupt the diarisation:

```python
def detect_crosstalk_frames(
    mic_energies: np.ndarray,    # (n_frames, n_mics) energy in dB
    crosstalk_threshold_db: float = -30.0,
    speech_threshold_db: float = -40.0,
) -> np.ndarray:
    """
    Returns boolean array (n_frames,): True where crosstalk is suspected.
    Crosstalk suspected when: one mic is loud (speech) AND
    another mic shows energy > speech_threshold + crosstalk_threshold.
    """
    n_frames, n_mics = mic_energies.shape
    crosstalk_flag = np.zeros(n_frames, dtype=bool)

    for fi in range(n_frames):
        levels = mic_energies[fi]
        loud_mics = np.where(levels > speech_threshold_db)[0]
        if len(loud_mics) == 0:
            continue
        for loud_mic in loud_mics:
            for other_mic in range(n_mics):
                if other_mic == loud_mic:
                    continue
                expected_crosstalk = levels[loud_mic] + crosstalk_threshold_db
                if levels[other_mic] > expected_crosstalk:
                    crosstalk_flag[fi] = True
                    break

    return crosstalk_flag
```

---

## 8. Cross-Modal Confidence Weighting

### 8.1 Bayesian ROI Posterior

Combine evidence from all modalities into a per-frame ROI posterior for each
participant. Each modality contributes a likelihood or prior update:

```python
def compute_roi_posterior(
    roi_names: list[str],
    # Likelihoods from each modality (probabilities, sum to 1):
    vision_likelihood: dict[str, float],     # from 3D gaze ray
    imu_likelihood: dict[str, float],        # from head orientation
    tablet_log_likelihood: dict[str, float], # from confirmed windows / probes
    audio_likelihood: dict[str, float],      # from acoustic salience
    uniform_prior: bool = True,
) -> dict[str, float]:
    """
    Computes P(ROI | all evidence) via naive Bayes product of likelihoods.
    Returns normalised posterior dict {roi_name: probability}.
    """
    # Uniform prior (can be replaced with spatial/task-based prior)
    prior = {roi: 1.0 / len(roi_names) for roi in roi_names}

    posterior = {}
    for roi in roi_names:
        p = prior[roi]
        p *= vision_likelihood.get(roi, 0.1)      # 0.1 = uniform fallback
        p *= imu_likelihood.get(roi, 1.0)         # 1.0 = no evidence
        p *= tablet_log_likelihood.get(roi, 1.0)
        p *= audio_likelihood.get(roi, 1.0)
        posterior[roi] = p

    # Normalise
    total = sum(posterior.values()) + 1e-12
    return {roi: p / total for roi, p in posterior.items()}


def posterior_to_roi_label(
    posterior: dict[str, float],
    min_confidence: float = 0.30,
) -> tuple[str, float]:
    """
    Returns (best_roi_name, confidence).
    If max probability < min_confidence, returns ('uncertain', max_prob).
    """
    best_roi  = max(posterior, key=posterior.get)
    best_conf = posterior[best_roi]
    if best_conf < min_confidence:
        return "uncertain", best_conf
    return best_roi, best_conf
```

### 8.2 Modality-Specific Likelihood Functions

```python
def vision_gaze_likelihood(
    gaze_roi_vision: str,
    gaze_conf: float,
    all_rois: list[str],
    base_conf: float = 0.80,
) -> dict[str, float]:
    """
    If vision says 'tablet_P2' with conf=0.9, give high probability to that ROI.
    Spread remaining probability to other ROIs uniformly.
    """
    lh = {roi: (1 - base_conf * gaze_conf) / (len(all_rois) - 1)
          for roi in all_rois}
    lh[gaze_roi_vision] = base_conf * gaze_conf
    return lh


def tablet_log_likelihood(
    confirmed_window: ConfirmedGazeWindow,
    all_rois: list[str],
    in_window: bool,
) -> dict[str, float]:
    """
    If in a confirmed touch window: 0.99 → tablet ROI, 0.01 spread elsewhere.
    If in probe window (soft prior): 0.60 → tablet, rest spread.
    If no tablet evidence: uniform (1.0 each, normalised later).
    """
    if in_window:
        lh = {roi: 0.01 / (len(all_rois) - 1) for roi in all_rois}
        lh[confirmed_window.roi] = 0.99
    else:
        lh = {roi: 1.0 for roi in all_rois}
    return lh


def imu_head_orientation_likelihood(
    head_forward_world: np.ndarray,
    head_pos_world: np.ndarray,
    roi_centers: dict[str, np.ndarray],
    sigma_deg: float = 25.0,
) -> dict[str, float]:
    """
    Gaussian likelihood over angle between head forward direction and ROI center.
    Wider sigma = more uncertain head orientation → softer prior.
    """
    sigma_rad = np.radians(sigma_deg)
    lh = {}
    for roi_name, center in roi_centers.items():
        direction = center - head_pos_world
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            lh[roi_name] = 1.0
            continue
        direction /= dist
        cos_a = np.clip(np.dot(head_forward_world, direction), -1.0, 1.0)
        angle_rad = np.arccos(cos_a)
        lh[roi_name] = float(np.exp(-0.5 * (angle_rad / sigma_rad) ** 2))
    return lh
```

---

## 9. Weak Supervision — Using Logs to Train Better Detectors

### 9.1 Labelled Data from Tablet Logs

Tablet touch events provide **free, precise temporal labels** for a specific
behaviour: "participant is looking at and interacting with their tablet." These
labels can train or fine-tune gaze classifiers without any manual annotation:

```
Label source            → Training target
──────────────────────────────────────────────────────────────
touch_start/touch_end   → gaze_roi = "tablet_P{i}"  (positive)
long silence (>5s) after
  task_end              → gaze_roi ≠ tablet           (negative)
head_nod from IMU       → gesture = "head_nod"        (positive)
silence VAD             → gesture ≠ beat/iconic        (negative)
```

### 9.2 Gaze Classifier Training Pipeline

```python
def build_weakly_supervised_dataset(
    sessions: list[str],
    data_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training pairs from tablet logs and vision features.

    X: per-sample feature vector:
       [gaze_x_norm, gaze_y_norm, gaze_validity,
        imu_accel_norm, imu_gyro_norm, imu_head_angle_to_tablet,
        audio_energy_db, vad, time_since_probe_s,
        skeleton_wrist_in_tablet_roi]

    y: binary label — 1 if gaze is on tablet, 0 otherwise
    """
    X_list, y_list = [], []

    for session_id in sessions:
        gaze_df    = load_gaze_ndjson(f"{data_root}/{session_id}/gaze_world/")
        imu_df     = load_imu_stream(f"{data_root}/{session_id}/et/")
        tablet_ev  = load_tablet_events(f"{data_root}/{session_id}/beh/")
        audio_feat = load_acoustic_features(f"{data_root}/{session_id}/audio/")

        confirmed = extract_confirmed_windows(tablet_ev)

        for _, row in gaze_df.iterrows():
            ts  = row["timestamp_s"]
            pid = row["participant"]

            # Label: in confirmed window → 1, confirmed other ROI → 0, else skip
            label = None
            for win in confirmed:
                if win.participant == pid and win.start_s <= ts <= win.end_s:
                    label = 1
                    break
            if label is None:
                # Use silence + no probe as negative examples
                if is_confirmed_non_tablet(ts, pid, tablet_ev, audio_feat):
                    label = 0

            if label is None:
                continue   # ambiguous — skip

            # Build feature vector
            imu_row    = interpolate_to_ts(imu_df, ts, pid)
            audio_row  = interpolate_to_ts(audio_feat, ts, pid)
            skel_row   = interpolate_to_ts(skeleton_df, ts, pid)

            features = np.array([
                row.get("gaze_x_norm", 0.5),
                row.get("gaze_y_norm", 0.5),
                float(row.get("gaze_validity") == "valid"),
                imu_row["accel_norm"],
                imu_row["gyro_norm"],
                imu_row["head_angle_to_tablet_rad"],
                audio_row["energy_db"],
                audio_row["vad"],
                min(ts - last_probe_time(ts, pid, tablet_ev), 10.0),
                float(skel_row["rw_roi"] == f"tablet_{pid}"),
            ])

            X_list.append(features)
            y_list.append(label)

    return np.array(X_list), np.array(y_list)
```

### 9.3 IMU-Based Head Gesture Label Generation

Use the IMU detector (§4.2) to generate head gesture labels automatically, then
cross-validate with the skeleton-based detector. Agreement between the two
independent sensors on the same event increases label confidence:

```python
def cross_validate_head_gestures(
    imu_events: list[dict],      # from detect_head_nods() on IMU
    skel_events: list[dict],     # from skeleton nose-oscillation detector
    iou_threshold: float = 0.50,
) -> list[dict]:
    """
    Return only gesture events confirmed by BOTH IMU and skeleton.
    Intersection-over-Union on time intervals ≥ iou_threshold.
    High-confidence labels for weak supervision.
    """
    confirmed = []
    for imu_ev in imu_events:
        for sk_ev in skel_events:
            if imu_ev["gesture_type"] != sk_ev["gesture_type"]:
                continue
            # Compute temporal IoU
            inter_start = max(imu_ev["onset_s"], sk_ev["onset_s"])
            inter_end   = min(imu_ev["onset_s"] + imu_ev["duration_s"],
                              sk_ev["onset_s"]  + sk_ev["duration_s"])
            if inter_end <= inter_start:
                continue
            inter = inter_end - inter_start
            union = (imu_ev["duration_s"] + sk_ev["duration_s"] - inter)
            if inter / union >= iou_threshold:
                confirmed.append({
                    **imu_ev,
                    "confidence": "high",
                    "source": "imu_skeleton_agreement",
                })
                break
    return confirmed
```

---

## 10. Unified Multi-Modal Feature Frame

### 10.1 Extended Feature Frame Schema

Extend the `build_feature_frame()` function from `3d_roi_gesture_skeleton_plan.md §9.1`
with the auxiliary modalities:

```python
def build_multimodal_feature_frame(
    frame_idx: int,
    skeleton: np.ndarray,
    gaze_world: dict,
    hand_states: dict,
    imu_data: dict,                  # {pid: {accel, gyro, R_world, conf}}
    vad_frame: dict,                 # {pid: 0/1}
    audio_features: dict,            # {pid: acoustic_feature_vector}
    tablet_events_active: dict,      # {pid: ConfirmedGazeWindow or None}
    speaker_ids: list[str],          # active speakers this frame
    rois: dict,
    fps: float = 30.0,
) -> dict:
    """
    Full multi-modal feature frame merging vision, IMU, audio, and tablet logs.
    """
    ts = frame_idx / fps
    record = {
        "timestamp_s": ts,
        "frame_idx":   frame_idx,
        "speakers":    speaker_ids,
        "participants": {},
    }

    for p_idx, label in enumerate(["P1", "P2", "P3", "P4"]):
        sk    = skeleton[p_idx]
        gaze  = gaze_world.get(label, {})
        imu   = imu_data.get(label, {})
        audio = audio_features.get(label, {})
        tab   = tablet_events_active.get(label)

        # --- Vision-based ROI ---
        gaze_pt = gaze.get("point_3d")
        vision_lh = vision_gaze_likelihood(
            gaze.get("gaze_roi", "unknown"),
            gaze.get("gaze_roi_conf", 0.1),
            list(rois.keys()),
        )

        # --- IMU-based ROI prior ---
        head_pos = imu.get("head_pos_world")
        head_fwd = imu.get("head_forward_world")
        imu_lh = (imu_head_orientation_likelihood(
                      head_fwd, head_pos,
                      {n: r.center for n, r in rois.items()})
                  if head_pos is not None and head_fwd is not None
                  else {n: 1.0 for n in rois})

        # --- Tablet log ---
        tab_lh = tablet_log_likelihood(tab, list(rois.keys()),
                                        in_window=(tab is not None))

        # --- Audio salience ---
        energy = audio.get("energy_db", -60.0)
        audio_lh = {n: 1.0 for n in rois}   # neutral unless source-specific

        # --- Posterior ---
        posterior = compute_roi_posterior(
            list(rois.keys()),
            vision_lh, imu_lh, tab_lh, audio_lh,
        )
        best_roi, roi_conf = posterior_to_roi_label(posterior)

        record["participants"][label] = {
            # Vision
            "gaze_roi_vision":      gaze.get("gaze_roi"),
            "gaze_validity":        gaze.get("validity", "unknown"),
            "gaze_point_world":     gaze_pt.tolist() if gaze_pt is not None else None,
            # IMU
            "head_forward_world":   head_fwd.tolist() if head_fwd is not None else None,
            "imu_motion_score":     imu.get("motion_score", 0.0),
            "imu_pose_source":      imu.get("pose_source", "unknown"),
            # Audio
            "speaking":             bool(vad_frame.get(label, 0)),
            "audio_energy_db":      float(energy),
            "pitch_hz":             audio.get("pitch_hz", 0.0),
            # Tablet
            "tablet_confirmed":     tab is not None,
            "tablet_touch_uv":      tab.touch_xy_norm if tab is not None else None,
            # Fused ROI
            "gaze_roi_fused":       best_roi,
            "gaze_roi_conf_fused":  float(roi_conf),
            "gaze_roi_posterior":   {k: float(v) for k, v in posterior.items()},
            # Skeleton (from §9.1 of ROI plan)
            "rw_roi":               None,   # filled from skeleton
            "lw_roi":               None,
            "skeleton_conf":        float(skeleton_confidence(sk)),
        }

    # Social signals
    record["mutual_gaze_pairs"] = detect_mutual_gaze(record)
    record["joint_attention"]   = detect_joint_attention(record)
    record["n_speakers"]        = len(speaker_ids)

    return record
```

### 10.2 Output Files

Write the unified feature frame stream to two formats:

```
<session>/annot/
  multimodal_features_T{task}.ndjson   ← full per-frame records (one JSON per line)
  multimodal_features_T{task}.parquet  ← columnar format for fast ML loading

<session>/beh/
  gestures_events_T{task}.tsv          ← BIDS events format, gated by VAD
  gaze_roi_fused_T{task}.tsv           ← per-frame ROI labels with source column
```

---

## 11. Quality Control and Failure Mode Analysis

### 11.1 Per-Modality Coverage Report

```
Metric                              Good     Warn      Fail
────────────────────────────────────────────────────────────────
Tobii gaze validity (%)             > 80     60–80     < 60
IMU data gaps (% frames)            < 2      2–10      > 10
IMU pose source = "vision" (%)      > 70     50–70     < 50
IMU pose source = "imu_dr_stale"    < 5      5–20      > 20
Tablet log coverage (% of expected
  probe interactions logged)         > 95     85–95     < 85
VAD false positive rate*             < 10%   10–25%    > 25%
Gaze ROI = "uncertain" (%)          < 10     10–25     > 25
Crosstalk flag rate (%)             < 5      5–15      > 15
Confirmed window → gaze ROI
  agreement with vision (%)         > 75     50–75     < 50

* Estimate from cross-mic energy comparison
```

### 11.2 Fusion Sanity Checks

```python
def validate_fusion_consistency(
    feature_frames: list[dict],
    confirmed_windows: list[ConfirmedGazeWindow],
) -> dict:
    """
    Check that fused ROI agrees with confirmed tablet windows.
    This is the ground-truth validation: during a confirmed touch,
    fused ROI MUST be the tablet.
    """
    n_confirmed = 0
    n_correct   = 0

    for ff in feature_frames:
        ts = ff["timestamp_s"]
        for label, data in ff["participants"].items():
            for win in confirmed_windows:
                if win.participant != label:
                    continue
                if not (win.start_s <= ts <= win.end_s):
                    continue
                n_confirmed += 1
                if data["gaze_roi_fused"] == win.roi:
                    n_correct += 1

    accuracy = n_correct / max(n_confirmed, 1)
    return {
        "n_confirmed_frames": n_confirmed,
        "n_correct_fused":    n_correct,
        "tablet_gaze_accuracy": accuracy,
        "pass": accuracy > 0.80,
    }
```

### 11.3 IMU Drift Monitoring

```python
def monitor_imu_drift(
    imu_dr_states: list[dict],    # {timestamp_s, pose_source, drift_age_s}
    max_drift_s: float = 3.0,
) -> dict:
    stale_frames = [s for s in imu_dr_states
                    if s.get("drift_age_s", 0) > max_drift_s]
    return {
        "n_stale_frames":     len(stale_frames),
        "pct_stale":          100 * len(stale_frames) / max(len(imu_dr_states), 1),
        "max_drift_age_s":    max((s.get("drift_age_s", 0) for s in imu_dr_states),
                                  default=0),
        "recommendation":     (
            "Check marker visibility — IMU dead-reckoning exceeds safe window"
            if len(stale_frames) > 0.10 * len(imu_dr_states)
            else "OK"
        ),
    }
```

---

## 12. Implementation Roadmap

### Phase 1 — Low Effort, High Value (1–2 weeks)

| Task | Code touchpoint | Expected gain |
|------|----------------|---------------|
| Parse tablet `touch_start`/`touch_end` into confirmed windows | New `tools/parse_tablet_logs.py` | Free ground-truth labels, immediate ROI override |
| Apply VAD gate to gesture events | Add to `tools/gesture_extractor.py` | Reduces beat-gesture false positives by ~50% |
| IMU motion score → vision confidence penalty | Add to `tobii_multicam_glasses_tracker.py` | Fewer wrong pose estimates during head movement |
| Apply probe-displayed prior | Add to `tobii_multi_glasses_world_align.py` | Improves tablet ROI classification before touch |

### Phase 2 — Medium Effort (2–4 weeks)

| Task | Code touchpoint | Expected gain |
|------|----------------|---------------|
| IMU gyroscope dead-reckoning | New `tools/imu_dead_reckoning.py` | Fills pose gaps during marker occlusion |
| IMU head nod/shake detector | Add to `tools/qc/qc_tobii_world_gaze.py` | 100 Hz head gesture ground truth |
| Per-mic VAD via DPA energy | New `tools/audio_vad.py` | Speaker diarisation for gaze grounding |
| Bayesian ROI posterior | New `tools/roi_fusion.py` | Unified confidence-weighted ROI label |

### Phase 3 — Higher Effort (1–2 months)

| Task | Code touchpoint | Expected gain |
|------|----------------|---------------|
| Weakly supervised gaze classifier | New `tools/train_gaze_classifier.py` | Data-driven ROI model replacing rule-based |
| Acoustic feature extraction (librosa) | New `tools/audio_features.py` | Full gesture-speech coupling features |
| IMU-skeleton cross-validated gesture labels | Add to `tools/gesture_extractor.py` | High-confidence training corpus |
| Unified multi-modal feature frame export | New `tools/multimodal_feature_builder.py` | ML-ready dataset for affect modelling |

### Phase 4 — Research Extensions

| Task | Notes |
|------|-------|
| EmotiBit EDA + gaze ROI coupling | Arousal spikes during mutual gaze / screen events |
| Speaker diarisation fine-tuning | Use confirmed speaking turns to fine-tune pyannote |
| Attention estimation model | Train on confirmed-window labels across all sessions |
| Real-time fusion for future studies | Port Phase 1–2 to LSL online pipeline |

---

*Document generated 2026-04-19. Complements `3d_gaze_improvement_plan.md` and*
*`3d_roi_gesture_skeleton_plan.md`. All three documents share the same world frame,*
*LSL timebase, and BIDS output schema.*
