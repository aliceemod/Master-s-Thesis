#!/usr/bin/env python3
"""
calibration_pipeline.py — Flexible multi-camera calibration pipeline.

Calibration strategies tried in priority order (or as specified):
  1. transfer        — Reuse a calibration TOML from a compatible previous session
                       (camera rig is nearly fixed across sessions; only extrinsics
                       change if a camera was physically moved)
  2. dynamic_board   — ChArUco 5×3 moved in front of all 7 cameras
                       NOTE: only available in dedicated calibration recordings
                       (2-3 short videos), NOT in 1-hour session videos
  3. fixed_board     — ChArUco 7×5 fixed at desk centre (mainly P20 cameras;
                       P50 can see the board but from a very skewed/close angle)
                       NOTE: only available in dedicated calibration recordings
  4. desk_markers    — 6 ArUco 50 mm edge markers with known world coords
                       Works on session videos. Handles partial visibility:
                       cam5 and cam7(P50) cannot see all 6 markers.
  5. glasses_markers — ArUco markers on Tobii glasses (auxiliary last-resort)
                       Works on session videos. Group-aware marker size.
  fusion             — always runs last; merges any succeeded strategies

Key design decisions
--------------------
- Session videos are ~1 hour long. Strategies 4 & 5 use a fast pre-scan to
  identify 30-second windows with the best marker visibility before dense
  sampling, avoiding full-video processing.
- Not all desk markers are visible to every camera (cam5 is low/front,
  P50 is close/skewed). Per-camera visibility constraints are read from
  the YAML config and enforced during PnP.
- The camera rig is nearly fixed across sessions (same group era). A
  calibration from session N can seed or fully replace calibration for
  session N+1. The rig changed between grp-8 and grp-9 (one camera
  shifted), so transfer is only allowed within the same era.

Sync offsets are resolved before every calibration step:
  Tier 1  frame_logs/{label}_frames.jsonl   ~0.5 ms
  Tier 2  lsl/ffmpeg_progress_{label}.jsonl ~1 ms
  Tier 3  sourcedata/sync/{label}_ffmpeg_progress.tsv ~1 ms
  Tier 4  video/ffmpeg_multicap_events.jsonl ~100 ms

Usage
-----
  # Full pipeline on a session with session video only
  python tools/calibration_pipeline.py run \\
      --session-dir  data/sessions/grp09_ses01 \\
      --config       configs/desk_markers_large.yaml \\
      --group        9 \\
      --calib-store  data/calibrations/ \\
      [--strategies  transfer,desk_markers,glasses_markers] \\
      [--cameras     all] \\
      [--out         calibration_pipeline.toml]

  # Run only on dedicated calibration videos (dynamic/fixed board)
  python tools/calibration_pipeline.py run \\
      --session-dir  data/calibrations/grp09_calib01 \\
      --config       configs/desk_markers_large.yaml \\
      --group        9 \\
      --video-type   calibration \\
      --strategies   dynamic_board,fixed_board

  python tools/calibration_pipeline.py validate \\
      --calib calibration_pipeline.toml \\
      --config configs/desk_markers_large.yaml

  python tools/calibration_pipeline.py report \\
      --calib calibration_pipeline.toml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

try:
    import tomllib  # Python ≥ 3.11
    def _toml_loads(s: str) -> dict:
        return tomllib.loads(s)
except ImportError:
    try:
        import toml  # pip install toml
        def _toml_loads(s: str) -> dict:
            return toml.loads(s)
    except ImportError:
        tomllib = None
        toml = None
        def _toml_loads(s: str) -> dict:
            raise RuntimeError("Install 'toml' (pip install toml) or use Python ≥ 3.11")

try:
    import tomli_w  # pip install tomli-w
    def _toml_dumps(d: dict) -> str:
        return tomli_w.dumps(d)
except ImportError:
    try:
        import toml  # noqa: F811 – re-import for the write side
        def _toml_dumps(d: dict) -> str:
            return toml.dumps(d)
    except ImportError:
        def _toml_dumps(d: dict) -> str:
            raise RuntimeError("Install 'toml' or 'tomli-w' to write TOML output")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("calibration_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DYNAMIC_BOARD_SQUARE_MM  = 69.0
FIXED_BOARD_SQUARE_MM    = 69.0
DESK_ARUCO_DICT          = cv2.aruco.DICT_4X4_50
BOARD_ARUCO_DICT         = cv2.aruco.DICT_4X4_250
MIN_PNP_POINTS           = 8    # minimum 2D-3D point correspondences (2 markers × 4 corners)
MIN_PNP_MARKERS          = 2    # minimum distinct desk markers needed (handles partial visibility)
MIN_PNP_FRAMES           = 5    # minimum frames with successful PnP per camera
FRAME_SAMPLE_STEP        = 10   # dense sample step (every Nth frame) inside selected windows
PRESCAN_STEP             = 300  # 1 frame every ~10 s @ 30 fps for initial visibility scan
PRESCAN_WINDOW_FRAMES    = 150  # half-width of a pre-scan window (5 s @ 30 fps)
PRESCAN_TOP_K_WINDOWS    = 4    # number of best windows to densely sample
CALIB_VIDEO_MAX_S        = 300  # videos shorter than this are treated as calibration recordings
GRP_RIG_CHANGE           = 9    # camera rig changed between grp-8 and grp-9 (one cam shifted)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    matrix: np.ndarray           # 3×3 float64
    dist_coeffs: np.ndarray      # (5,) or (8,) float64
    resolution: tuple[int, int]  # (width, height)
    source: str = "unknown"      # "toml", "seeded", "desk_markers", …

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "resolution": list(self.resolution),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        return cls(
            matrix=np.array(d["matrix"], dtype=np.float64),
            dist_coeffs=np.array(d["dist_coeffs"], dtype=np.float64),
            resolution=tuple(d["resolution"]),
            source=d.get("source", "unknown"),
        )


@dataclass
class CameraExtrinsics:
    rvec: np.ndarray   # (3,1) or (3,) rotation vector (Rodrigues)
    tvec: np.ndarray   # (3,1) or (3,) translation vector (mm or m — keep consistent)
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "rvec": self.rvec.flatten().tolist(),
            "tvec": self.tvec.flatten().tolist(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraExtrinsics":
        return cls(
            rvec=np.array(d["rvec"], dtype=np.float64).reshape(3, 1),
            tvec=np.array(d["tvec"], dtype=np.float64).reshape(3, 1),
            source=d.get("source", "unknown"),
        )


@dataclass
class CameraCalibResult:
    camera_id: str
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
    reprojection_error: float = float("inf")
    strategy: str = "unknown"
    notes: str = ""


@dataclass
class PipelineResult:
    cameras: dict[str, CameraCalibResult] = field(default_factory=dict)
    strategy: str = "unknown"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)

    def succeeded(self) -> bool:
        return bool(self.cameras)

    def camera_ids(self) -> list[str]:
        return list(self.cameras.keys())

# ---------------------------------------------------------------------------
# Sync loading — Tier 1 → 4 fallback
# ---------------------------------------------------------------------------

def load_sync_offsets(session_dir: Path) -> dict[str, float]:
    """Return {camera_label: offset_seconds} where offset = unix_time - pts_time.

    Positive offset means the camera's PTS clock runs behind wall clock.
    Apply as:  wall_time = pts_time + offset
    """
    offsets: dict[str, float] = {}

    # Tier 1 — frame_logs/{label}_frames.jsonl
    frame_log_dir = session_dir / "frame_logs"
    if frame_log_dir.is_dir():
        for jsonl_path in sorted(frame_log_dir.glob("*_frames.jsonl")):
            label = jsonl_path.stem.replace("_frames", "")
            diffs: list[float] = []
            try:
                with jsonl_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        row = json.loads(line)
                        unix = row.get("unix_time_s") or row.get("unix_time") or row.get("host_time")
                        pts  = row.get("pts_time")
                        if unix is not None and pts is not None:
                            diffs.append(float(unix) - float(pts))
            except Exception as exc:
                log.warning("Frame log read error %s: %s", jsonl_path, exc)
                continue
            if diffs:
                offsets[label] = statistics.median(diffs)
                # offset is unix_epoch - pts → a large absolute value is expected;
                # spread (stdev) is what indicates sync jitter
                spread = statistics.stdev(diffs) * 1000 if len(diffs) > 1 else 0.0
                log.info("Sync tier-1 %s: epoch_offset=%.3f s  jitter=%.3f ms  (%d frames)",
                         label, offsets[label], spread, len(diffs))
        if offsets:
            return offsets

    # Tier 2 — lsl/ffmpeg_progress_{label}.jsonl
    lsl_dir = session_dir / "lsl"
    if lsl_dir.is_dir():
        for jsonl_path in sorted(lsl_dir.glob("ffmpeg_progress_*.jsonl")):
            label = jsonl_path.stem.replace("ffmpeg_progress_", "")
            diffs: list[float] = []
            try:
                with jsonl_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        row = json.loads(line)
                        rcv  = row.get("received_time") or row.get("stream_time")
                        out  = row.get("out_time_sec")
                        if rcv is not None and out is not None:
                            diffs.append(float(rcv) - float(out))
            except Exception as exc:
                log.warning("LSL JSONL read error %s: %s", jsonl_path, exc)
                continue
            if diffs:
                offsets[label] = statistics.median(diffs)
                log.info("Sync tier-2 %s: offset %.3f ms", label,
                         offsets[label] * 1000)
        if offsets:
            return offsets

    # Tier 3 — sourcedata/sync/{label}_ffmpeg_progress.tsv
    sync_dir = session_dir / "sourcedata" / "sync"
    if sync_dir.is_dir():
        for tsv_path in sorted(sync_dir.glob("*_ffmpeg_progress.tsv")):
            label = tsv_path.stem.replace("_ffmpeg_progress", "")
            diffs: list[float] = []
            try:
                import csv
                with tsv_path.open(newline="") as fh:
                    reader = csv.DictReader(fh, delimiter="\t")
                    for row in reader:
                        host = row.get("host_time_sec")
                        out  = row.get("out_time_sec")
                        if host and out:
                            diffs.append(float(host) - float(out))
            except Exception as exc:
                log.warning("Sync TSV read error %s: %s", tsv_path, exc)
                continue
            if diffs:
                offsets[label] = statistics.median(diffs)
                log.info("Sync tier-3 %s: offset %.3f ms", label,
                         offsets[label] * 1000)
        if offsets:
            return offsets

    # Tier 4 — video/ffmpeg_multicap_events.jsonl (coarse, ~100 ms)
    events_path = session_dir / "video" / "ffmpeg_multicap_events.jsonl"
    if events_path.is_file():
        try:
            t0: Optional[float] = None
            label_times: dict[str, float] = {}
            with events_path.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("event") == "capture_started":
                        ts = row.get("unix_time") or row.get("timestamp")
                        lbl = row.get("label") or row.get("camera")
                        if ts and lbl:
                            label_times[lbl] = float(ts)
            if label_times:
                ref = min(label_times.values())
                for lbl, ts in label_times.items():
                    offsets[lbl] = ts - ref
                log.info("Sync tier-4 (coarse ±100 ms): %d cameras", len(offsets))
        except Exception as exc:
            log.warning("Events JSONL read error: %s", exc)

    if not offsets:
        log.warning("No sync data found in %s; assuming zero offsets", session_dir)
    return offsets

# ---------------------------------------------------------------------------
# Config & intrinsics helpers
# ---------------------------------------------------------------------------

def load_lab_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_camera_specs(specs_path: Path) -> dict:
    with specs_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _match_camera_model(camera_id: str, specs: dict) -> Optional[dict]:
    patterns = specs.get("camera_name_patterns", {})
    models   = specs.get("models", {})
    for pat, model_key in patterns.items():
        if pat.startswith("_"):
            continue
        if re.search(pat, camera_id, re.IGNORECASE):
            return models.get(model_key)
    return None


def seed_intrinsics(camera_id: str, specs: dict,
                    resolution: tuple[int, int] = (1920, 1080)) -> CameraIntrinsics:
    """Build a seeded intrinsics matrix from factory specs."""
    model = _match_camera_model(camera_id, specs)
    if model:
        fx = fy = float(model["expected_fx_1080p"])
        # Scale if resolution differs from 1080p
        w, h = resolution
        scale_x = w / 1920.0
        scale_y = h / 1080.0
        fx *= scale_x
        fy *= scale_y
    else:
        # Generic fallback: assume ~70° HFOV
        w, h = resolution
        fx = fy = (w / 2.0) / math.tan(math.radians(70) / 2)
        log.warning("No spec match for %s; using generic focal length %.1f px", camera_id, fx)

    w, h = resolution
    K = np.array([
        [fx,  0.0, w / 2.0],
        [0.0, fy,  h / 2.0],
        [0.0, 0.0, 1.0    ],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return CameraIntrinsics(matrix=K, dist_coeffs=dist,
                            resolution=resolution, source="seeded")


def load_intrinsics_from_toml(toml_path: Path) -> dict[str, CameraIntrinsics]:
    """Load intrinsics from an existing calibration TOML."""
    data = _toml_loads(toml_path.read_text(encoding="utf-8"))
    cams = data.get("cameras", {})
    result: dict[str, CameraIntrinsics] = {}
    for cam_id, cam_data in cams.items():
        mat = cam_data.get("matrix")
        dist = cam_data.get("dist")
        if mat and dist:
            result[cam_id] = CameraIntrinsics(
                matrix=np.array(mat, dtype=np.float64),
                dist_coeffs=np.array(dist, dtype=np.float64),
                resolution=(1920, 1080),
                source="toml",
            )
    return result

# ---------------------------------------------------------------------------
# Desk marker world geometry
# ---------------------------------------------------------------------------

def load_desk_marker_world_corners(config: dict) -> dict[int, np.ndarray]:
    """Return {marker_id: corners_3d (4×3 float32)} from YAML config."""
    result: dict[int, np.ndarray] = {}
    marker_map = config.get("world", {}).get("marker_map", [])
    for entry in marker_map:
        mid = int(entry["id"])
        corners = np.array(entry["corners_m"], dtype=np.float32)  # (4,3)
        result[mid] = corners
    return result


def load_camera_rotations(config: dict) -> dict[str, bool]:
    """Return {camera_id_pattern: rotate_180} from the cameras list in YAML config."""
    result: dict[str, bool] = {}
    for cam in config.get("cameras", []):
        cam_id = cam.get("id", "")
        if cam.get("rotate_180", False):
            result[cam_id] = True
    return result


def load_camera_visibility(config: dict) -> dict[str, set[int]]:
    """Return {camera_id: set_of_visible_marker_ids} from YAML config.

    Cameras absent from the section can see all markers (no restriction).
    """
    result: dict[str, set[int]] = {}
    for cam_id, info in config.get("camera_marker_visibility", {}).items():
        ids = info.get("visible_marker_ids")
        if ids is not None:
            result[cam_id] = set(int(i) for i in ids)
    return result


def _get_video_duration_s(video_path: Path) -> float:
    """Return video duration in seconds via OpenCV (fast, no ffprobe needed)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frames / fps


def detect_video_type(video_path: Path) -> str:
    """Return 'calibration' for short recordings (≤5 min), else 'session'."""
    dur = _get_video_duration_s(video_path)
    return "calibration" if dur <= CALIB_VIDEO_MAX_S else "session"


def _glasses_world_geometry(glasses_cfg: dict) -> dict[str, np.ndarray]:
    """Return left/right marker 3D corners in glasses local frame (z=0 plane).

    Origin is the midpoint between the two marker centres.
    x = right (from left marker to right marker)
    y = forward (camera pointing direction, away from wearer)
    z = up
    """
    size  = float(glasses_cfg["marker_size_m"])
    half  = size / 2.0
    l_off = np.array(glasses_cfg["left_marker_offset_mm"],  dtype=np.float32) / 1000.0
    r_off = np.array(glasses_cfg["right_marker_offset_mm"], dtype=np.float32) / 1000.0

    def _square_corners(centre: np.ndarray) -> np.ndarray:
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left
        return np.array([
            centre + np.array([-half,  half, 0], dtype=np.float32),
            centre + np.array([ half,  half, 0], dtype=np.float32),
            centre + np.array([ half, -half, 0], dtype=np.float32),
            centre + np.array([-half, -half, 0], dtype=np.float32),
        ], dtype=np.float32)

    return {
        "left":  _square_corners(l_off),
        "right": _square_corners(r_off),
    }

# ---------------------------------------------------------------------------
# Video frame iterator
# ---------------------------------------------------------------------------

def _iter_video_frames(video_path: Path, step: int = FRAME_SAMPLE_STEP):
    """Yield (frame_index, frame_bgr) every `step` frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()

# ---------------------------------------------------------------------------
# Frame pre-selector  — avoid processing entire 1-hour session videos
# ---------------------------------------------------------------------------

class FramePreSelector:
    """Fast two-pass frame sampler for long videos.

    Pass 1 (coarse): sample 1 frame every ~10 s, count ArUco detections.
    Pass 2 (dense):  sample every FRAME_SAMPLE_STEP frames inside the top-K
                     windows identified in pass 1.

    Returns a sorted list of frame indices to process, typically covering
    ~2–4 minutes of the best-visibility portions of the video.
    """

    def select(
        self,
        video_path: Path,
        detector: cv2.aruco.ArucoDetector,
        *,
        allowed_ids: Optional[set[int]] = None,
        rotate_180: bool    = False,
        prescan_step: int   = PRESCAN_STEP,
        window_frames: int  = PRESCAN_WINDOW_FRAMES,
        top_k: int          = PRESCAN_TOP_K_WINDOWS,
        min_detections: int = 1,
    ) -> list[int]:
        """Return list of frame indices worth dense-sampling."""
        dur_s = _get_video_duration_s(video_path)
        if dur_s <= CALIB_VIDEO_MAX_S:
            # Short calibration video — just sample everything
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            return list(range(0, max(1, total), FRAME_SAMPLE_STEP))

        log.info("[prescan] %s (%.0f min) — coarse pass every %d frames",
                 video_path.name, dur_s / 60, prescan_step)
        self._allowed_ids = allowed_ids  # used inside the loop below

        # Pass 1: coarse scan — only count marker IDs that are actually useful
        # (avoids glasses markers IDs 10-17 inflating scores for desk calibration)
        scores: list[tuple[int, int]] = []   # (frame_idx, n_detected_markers)
        for frame_idx, frame in _iter_video_frames(video_path, step=prescan_step):
            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                n = 0
            elif self._allowed_ids is not None:
                n = sum(1 for mid in ids.flatten() if int(mid) in self._allowed_ids)
            else:
                n = len(ids)
            scores.append((frame_idx, n))

        if not scores:
            return []

        # Pick top-K non-overlapping windows
        scores_sorted = sorted(scores, key=lambda x: -x[1])
        selected_centres: list[int] = []
        for frame_idx, n in scores_sorted:
            if n < min_detections:
                break
            if not any(abs(frame_idx - c) < window_frames for c in selected_centres):
                selected_centres.append(frame_idx)
                if len(selected_centres) >= top_k:
                    break

        if not selected_centres:
            # Nothing detected at all — fall back to uniform sparse sample
            log.warning("[prescan] No markers found in coarse scan of %s", video_path.name)
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            return list(range(0, max(1, total), prescan_step * 2))

        total_n = sum(n for _, n in scores_sorted[:top_k])
        log.info("[prescan] %d windows selected (total coarse detections: %d)",
                 len(selected_centres), total_n)

        # Pass 2: dense frame indices inside each window
        result: set[int] = set()
        for centre in selected_centres:
            start = max(0, centre - window_frames)
            end   = centre + window_frames
            result.update(range(start, end + 1, FRAME_SAMPLE_STEP))

        return sorted(result)


# ---------------------------------------------------------------------------
# Session calibration transfer — reuse calibration across sessions
# ---------------------------------------------------------------------------

class SessionCalibrationTransfer:
    """Locate and load a compatible calibration from a previous session.

    The camera rig is physically stable across sessions within the same
    group era:
      grp-1  – grp-8  : original rig positions
      grp-9  – grp-16 : one camera was adjusted; new positions apply

    Transfer is only allowed within the same era.  Within an era the
    intrinsics are reused as-is; extrinsics are used as a warm start
    (they may drift <1 cm between sessions due to vibration/handling).
    """

    def find_best_calibration(
        self,
        group: int,
        calib_store: Path,
    ) -> Optional[Path]:
        if not calib_store.is_dir():
            log.warning("[transfer] Calibration store not found: %s", calib_store)
            return None

        candidates = (
            list(calib_store.glob("**/*calibration*.toml"))
            + list(calib_store.glob("**/*.toml"))
        )
        if not candidates:
            log.info("[transfer] No TOML files found in %s", calib_store)
            return None

        def _era(g: int) -> str:
            return "new" if g >= GRP_RIG_CHANGE else "original"

        def _compatible(path: Path) -> bool:
            m = re.search(r"grp[_-]?(\d+)", str(path), re.IGNORECASE)
            if not m:
                return True  # can't tell — optimistically include it
            other_grp = int(m.group(1))
            return _era(group) == _era(other_grp)

        compatible = [c for c in candidates if _compatible(c)]
        if not compatible:
            log.warning("[transfer] No era-compatible calibrations in store; using any")
            compatible = candidates

        # Prefer highest reprojection quality (read metadata), else most recent mtime
        def _sort_key(path: Path) -> tuple:
            try:
                data = _toml_loads(path.read_text(encoding="utf-8"))
                err  = data.get("metadata", {}).get("reprojection_error", 999.0)
            except Exception:
                err = 999.0
            return (err, -path.stat().st_mtime)

        compatible.sort(key=_sort_key)
        chosen = compatible[0]
        log.info("[transfer] Best calibration for grp-%d: %s", group, chosen)
        return chosen

    def load(
        self,
        group: int,
        calib_store: Path,
        camera_specs: dict,
    ) -> Optional[PipelineResult]:
        best = self.find_best_calibration(group, calib_store)
        if best is None:
            return None
        try:
            pr = load_pipeline_result_from_toml(best)
        except Exception as exc:
            log.error("[transfer] Failed to load %s: %s", best, exc)
            return None

        # Mark all cameras as sourced from transfer
        for cam in pr.cameras.values():
            cam.strategy = "transfer"
            cam.notes    = f"transferred from {best.name}"

        pr.strategy = "transfer"
        pr.metadata["transfer_source"] = str(best)
        log.info("[transfer] Loaded %d cameras from %s", len(pr.cameras), best.name)
        return pr


# ---------------------------------------------------------------------------
# Strategy base
# ---------------------------------------------------------------------------

class CalibrationStrategy:
    name: str = "base"

    def run(
        self,
        session_dir: Path,
        config: dict,
        camera_specs: dict,
        sync_offsets: dict[str, float],
        group: int,
        cameras: Optional[list[str]],
        prior_intrinsics: Optional[dict[str, CameraIntrinsics]],
        tools_dir: Path,
    ) -> Optional[PipelineResult]:
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Strategy 1 — Dynamic ChArUco board (all 7 cameras)
# ---------------------------------------------------------------------------

class DynamicBoardStrategy(CalibrationStrategy):
    """Delegates to calibrate_charuco.py: 5×3 board, all cameras."""
    name = "dynamic_board"

    def run(self, session_dir, config, camera_specs, sync_offsets,
            group, cameras, prior_intrinsics, tools_dir) -> Optional[PipelineResult]:
        log.info("[dynamic_board] Starting: 5×3 ChArUco, all cameras")
        video_dir = session_dir / "video"
        if not video_dir.is_dir():
            log.warning("[dynamic_board] No video dir at %s", video_dir)
            return None

        out_toml = session_dir / "calibration_dynamic_board.toml"
        cmd = [
            sys.executable,
            str(tools_dir / "calibrate_charuco.py"),
            "calibrate",
            "--videos-dir", str(video_dir),
            "--square-size", str(int(DYNAMIC_BOARD_SQUARE_MM)),
            "--board-type", "5x3",
            "--init-focal",
            "--camera-specs", str(tools_dir.parent / "configs" / "camera_specs.json"),
            "--out", str(out_toml),
        ]
        if cameras and cameras != ["all"]:
            cmd += ["--cameras", ",".join(cameras)]

        log.info("[dynamic_board] Running: %s", " ".join(cmd))
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
        ret = subprocess.run(cmd, capture_output=False, env=env)
        if ret.returncode != 0 or not out_toml.is_file():
            log.error("[dynamic_board] calibrate_charuco.py failed (rc=%d)", ret.returncode)
            return None

        return self._toml_to_pipeline_result(out_toml)

    def _toml_to_pipeline_result(self, toml_path: Path) -> Optional[PipelineResult]:
        try:
            data = _toml_loads(toml_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("[dynamic_board] TOML parse error: %s", exc)
            return None

        pr = PipelineResult(strategy=self.name)
        pr.metadata["source_toml"] = str(toml_path)
        pr.metadata["reprojection_error"] = data.get("metadata", {}).get(
            "reprojection_error", float("inf")
        )
        for cam_id, cam_data in data.get("cameras", {}).items():
            mat  = cam_data.get("matrix")
            dist = cam_data.get("dist")
            rot  = cam_data.get("rotation")
            trans = cam_data.get("translation")
            if not (mat and dist):
                continue
            intr = CameraIntrinsics(
                matrix=np.array(mat, dtype=np.float64),
                dist_coeffs=np.array(dist, dtype=np.float64),
                resolution=(1920, 1080),
                source=self.name,
            )
            extr = None
            if rot is not None and trans is not None:
                extr = CameraExtrinsics(
                    rvec=np.array(rot,   dtype=np.float64).reshape(3, 1),
                    tvec=np.array(trans, dtype=np.float64).reshape(3, 1),
                    source=self.name,
                )
            pr.cameras[cam_id] = CameraCalibResult(
                camera_id=cam_id,
                intrinsics=intr,
                extrinsics=extr,
                reprojection_error=pr.metadata.get("reprojection_error", float("inf")),
                strategy=self.name,
            )
        if not pr.cameras:
            log.warning("[dynamic_board] TOML loaded but no camera data found")
            return None
        log.info("[dynamic_board] Succeeded: %d cameras, reproj=%.2f px",
                 len(pr.cameras), pr.metadata.get("reprojection_error", float("inf")))
        return pr

# ---------------------------------------------------------------------------
# Strategy 2 — Fixed ChArUco board (7×5, P20 cameras only)
# ---------------------------------------------------------------------------

class FixedBoardStrategy(CalibrationStrategy):
    """Delegates to calibrate_charuco.py: 7×5 board, mainly P20 cameras.

    P50 (cam7) sees the fixed board but from a very close and skewed angle.
    It is included with a warning flag; downstream validation should check
    its reprojection error carefully and fall back to desk_markers if poor.

    This strategy only makes sense on dedicated calibration recordings
    (short videos ≤5 min with the fixed board in place).  It will refuse
    to run on 1-hour session videos where the fixed board is absent.
    """
    name = "fixed_board"

    # Cameras that can give reliable fixed-board calibration
    _P20_PATTERN  = re.compile(r"cam[1-6]|panacast_20", re.IGNORECASE)
    # P50 is included but flagged as skewed
    _P50_PATTERN  = re.compile(r"cam7|panacast_50", re.IGNORECASE)

    def run(self, session_dir, config, camera_specs, sync_offsets,
            group, cameras, prior_intrinsics, tools_dir) -> Optional[PipelineResult]:
        log.info("[fixed_board] Starting: 7×5 ChArUco, P20 cameras + P50 (skewed)")
        video_dir = session_dir / "video"
        if not video_dir.is_dir():
            log.warning("[fixed_board] No video dir at %s", video_dir)
            return None

        all_videos = sorted(video_dir.glob("*.mkv")) + sorted(video_dir.glob("*.mp4"))

        # Refuse to run on session-length videos (fixed board is not present)
        for v in all_videos[:1]:
            if detect_video_type(v) == "session":
                log.warning(
                    "[fixed_board] Video %s looks like a session recording (~%.0f min). "
                    "Fixed board is only present in dedicated calibration recordings. Skipping.",
                    v.name, _get_video_duration_s(v) / 60,
                )
                return None

        # Identify P20 camera video files (P50 included with caveat)
        p20_videos = [
            str(p) for p in all_videos
            if self._P20_PATTERN.search(p.stem)
        ]
        p50_videos = [
            str(p) for p in all_videos
            if self._P50_PATTERN.search(p.stem)
        ]
        if p50_videos:
            log.warning(
                "[fixed_board] P50 (%s) sees the fixed board but at a skewed/close angle. "
                "Calibration quality may be poor; check reprojection error before using.",
                ", ".join(Path(v).stem for v in p50_videos),
            )

        include_videos = p20_videos + p50_videos
        if not include_videos:
            log.warning("[fixed_board] No camera videos found in %s", video_dir)
            return None

        out_toml  = session_dir / "calibration_fixed_board.toml"

        # calibrate_charuco.py does not support --cameras filtering;
        # it auto-discovers all videos in --videos-dir.
        # For fixed_board we run from a temp dir containing only the P20+P50 videos
        # (symlinks or the real dir — here we just pass the video_dir directly
        #  since include_videos already covers the right cameras).
        cmd = [
            sys.executable,
            str(tools_dir / "calibrate_charuco.py"),
            "calibrate",
            "--videos-dir", str(video_dir),
            "--square-size", str(int(FIXED_BOARD_SQUARE_MM)),
            "--board-type", "7x5",
            "--init-focal",
            "--camera-specs", str(tools_dir.parent / "configs" / "camera_specs.json"),
            "--out", str(out_toml),
        ]

        log.info("[fixed_board] Running (all cameras in video_dir, P50 included with warning)")
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
        ret = subprocess.run(cmd, capture_output=False, env=env)
        if ret.returncode != 0 or not out_toml.is_file():
            log.error("[fixed_board] calibrate_charuco.py failed (rc=%d)", ret.returncode)
            return None

        result = DynamicBoardStrategy()._toml_to_pipeline_result(out_toml)
        if result is None:
            return None

        # Flag P50 calibration quality from the fixed board
        for cam_id, cam_res in result.cameras.items():
            if self._P50_PATTERN.search(cam_id):
                cam_res.notes = "P50: fixed board seen from skewed/close angle — verify reproj error"
        return result

# ---------------------------------------------------------------------------
# Strategy 3 — Desk markers (ArUco PnP)
# ---------------------------------------------------------------------------

class DeskMarkersStrategy(CalibrationStrategy):
    """Calibrate extrinsics from fixed ArUco desk-edge markers.

    Handles partial visibility per camera:
    - cam5 (low/front): typically only sees front & left markers
    - cam7/P50 (close/skewed): sees corner markers but not all centre markers
    - Per-camera constraints are loaded from the YAML config's
      'camera_marker_visibility' section; other cameras accept all markers.

    For 1-hour session videos a FramePreSelector identifies the best
    ~2-minute window(s) where markers are most consistently visible,
    avoiding full-video processing.

    Minimum requirement: ≥2 distinct markers (8 corner points) per frame,
    ≥5 successful PnP frames per camera.
    """
    name = "desk_markers"

    def run(self, session_dir, config, camera_specs, sync_offsets,
            group, cameras, prior_intrinsics, tools_dir) -> Optional[PipelineResult]:
        log.info("[desk_markers] Starting: ArUco PnP from desk edge markers")

        marker_world = load_desk_marker_world_corners(config)
        if not marker_world:
            log.error("[desk_markers] No desk markers found in config")
            return None

        # Per-camera visibility and rotation constraints
        cam_visibility = load_camera_visibility(config)
        cam_rotations  = load_camera_rotations(config)
        log.info("[desk_markers] Per-camera visibility overrides: %s",
                 {k: sorted(v) for k, v in cam_visibility.items()} or "none")
        log.info("[desk_markers] Ceiling-mounted (rotate_180) cameras: %s",
                 [k for k, v in cam_rotations.items() if v] or "none")

        aruco_dict   = cv2.aruco.getPredefinedDictionary(DESK_ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        video_dir = session_dir / "video"
        if not video_dir.is_dir():
            log.warning("[desk_markers] No video dir at %s", video_dir)
            return None

        videos = sorted(video_dir.glob("*.mkv")) + sorted(video_dir.glob("*.mp4"))
        if cameras and cameras != ["all"]:
            videos = [v for v in videos if v.stem in cameras]

        preselector = FramePreSelector()
        pr = PipelineResult(strategy=self.name)

        for video_path in videos:
            cam_id = video_path.stem

            # Restrict world geometry to this camera's visible markers
            allowed_ids = cam_visibility.get(cam_id)  # None → all markers allowed
            if allowed_ids is not None:
                cam_marker_world = {k: v for k, v in marker_world.items()
                                    if k in allowed_ids}
                log.info("[desk_markers] %s: restricted to markers %s",
                         cam_id, sorted(allowed_ids))
            else:
                cam_marker_world = marker_world

            if not cam_marker_world:
                log.warning("[desk_markers] %s: no markers allowed by visibility config", cam_id)
                continue

            # Determine rotation before prescan (ceiling cams need rotate_180)
            # Match video stem (e.g. jabra_panacast_20_cam1_vid_video) against
            # config camera ids (cam1, cam2, …) by substring.
            rotate = any(
                v for k, v in cam_rotations.items()
                if k in cam_id or cam_id in k
            )

            # Select best frames — prescan filters to allowed desk marker IDs only,
            # preventing glasses markers (IDs 10-17) from inflating scores
            frame_indices = preselector.select(
                video_path, detector,
                allowed_ids=set(cam_marker_world.keys()),
                rotate_180=rotate,
            )
            log.info("[desk_markers] %s: will sample %d frames (rotate_180=%s)",
                     cam_id, len(frame_indices), rotate)

            intr = (prior_intrinsics or {}).get(cam_id) or seed_intrinsics(cam_id, camera_specs)
            result = self._calibrate_camera(
                video_path, cam_id, intr, cam_marker_world, detector, frame_indices,
                rotate_180=rotate,
            )
            if result:
                pr.cameras[cam_id] = result
            else:
                log.warning("[desk_markers] %s: insufficient detections, skipped", cam_id)

        if not pr.cameras:
            log.warning("[desk_markers] No cameras calibrated from desk markers")
            return None

        log.info("[desk_markers] Succeeded: %d cameras", len(pr.cameras))
        return pr

    def _calibrate_camera(
        self,
        video_path: Path,
        cam_id: str,
        intr: CameraIntrinsics,
        marker_world: dict[int, np.ndarray],
        detector: cv2.aruco.ArucoDetector,
        frame_indices: list[int],
        rotate_180: bool = False,
    ) -> Optional[CameraCalibResult]:
        # Reprojection threshold is loose (20px) because intrinsics are seeded
        # approximations from factory specs, not calibrated values.
        PNP_REPROJ_THRESH = 20.0
        rvecs, tvecs, reproj_errors = [], [], []
        n_attempted = 0   # frames with ≥MIN_PNP_MARKERS markers after dedup
        n_pnp_ok    = 0   # frames where solvePnPRansac succeeded

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error("[desk_markers] Cannot open %s", video_path)
            return None

        frame_set = set(frame_indices)
        current   = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current not in frame_set:
                    current += 1
                    continue

                if rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                current += 1

                if ids is None or len(ids) == 0:
                    continue

                # Deduplicate: ArUco sometimes detects the same ID twice
                # (reflections, ambiguous perspective). Keep the detection with
                # the largest corner area for each ID — same approach as
                # _filter_duplicate_marker_ids in calibrate_charuco.py.
                best: dict[int, tuple[float, np.ndarray]] = {}  # mid → (area, corners)
                for det_corners, mid in zip(corners, ids.flatten()):
                    mid = int(mid)
                    if mid not in marker_world:
                        continue
                    c = det_corners[0]  # (4,2)
                    area = float(cv2.contourArea(c))
                    if mid not in best or area > best[mid][0]:
                        best[mid] = (area, c)

                obj_pts: list[np.ndarray] = []
                img_pts: list[np.ndarray] = []
                for mid, (_, c) in best.items():
                    obj_pts.append(marker_world[mid])
                    img_pts.append(c)

                seen_marker_ids = set(best.keys())
                if len(seen_marker_ids) < MIN_PNP_MARKERS:
                    continue  # need at least 2 distinct markers
                n_attempted += 1

                obj_arr = np.concatenate(obj_pts, axis=0).astype(np.float32)
                img_arr = np.concatenate(img_pts, axis=0).astype(np.float32)

                if len(obj_arr) < MIN_PNP_POINTS:
                    continue

                ret_pnp, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_arr, img_arr,
                    intr.matrix, intr.dist_coeffs,
                    iterationsCount=500,
                    reprojectionError=PNP_REPROJ_THRESH,
                    confidence=0.99,
                )
                if not ret_pnp or inliers is None or len(inliers) < MIN_PNP_POINTS:
                    continue
                n_pnp_ok += 1

                proj, _ = cv2.projectPoints(
                    obj_arr[inliers.flatten()],
                    rvec, tvec,
                    intr.matrix, intr.dist_coeffs,
                )
                err = float(np.sqrt(np.mean(
                    (img_arr[inliers.flatten()] - proj.reshape(-1, 2)) ** 2
                )))
                rvecs.append(rvec.flatten())
                tvecs.append(tvec.flatten())
                reproj_errors.append(err)
        finally:
            cap.release()

        log.info("[desk_markers] %s: sampled=%d  pnp_attempted=%d  pnp_ok=%d  kept=%d",
                 cam_id, len(frame_indices), n_attempted, n_pnp_ok, len(rvecs))
        if len(rvecs) < MIN_PNP_FRAMES:
            return None

        # Robust average: discard frames with error > 2× median
        err_arr = np.array(reproj_errors)
        med_err = float(np.median(err_arr))
        keep    = err_arr <= 2.0 * med_err
        if keep.sum() == 0:
            keep = np.ones(len(rvecs), dtype=bool)

        rvec_mean = np.mean(np.array(rvecs)[keep], axis=0).reshape(3, 1)
        tvec_mean = np.mean(np.array(tvecs)[keep], axis=0).reshape(3, 1)
        mean_err  = float(np.mean(err_arr[keep]))

        log.info("[desk_markers] %s: %d/%d frames kept, reproj=%.2f px",
                 cam_id, int(keep.sum()), len(rvecs), mean_err)

        return CameraCalibResult(
            camera_id=cam_id,
            intrinsics=intr,
            extrinsics=CameraExtrinsics(rvec=rvec_mean, tvec=tvec_mean, source=self.name),
            reprojection_error=mean_err,
            strategy=self.name,
        )

# ---------------------------------------------------------------------------
# Strategy 4 — Glasses markers (auxiliary / last resort)
# ---------------------------------------------------------------------------

class GlassesMarkersStrategy(CalibrationStrategy):
    """Auxiliary calibration from Tobii glasses ArUco markers.

    Each glasses pair is a rigid body with known inter-marker geometry.
    When 2+ cameras observe the same glasses in the same (synced) frame,
    their relative pose can be estimated.

    This strategy only produces *extrinsics* (no intrinsic refinement).
    It is a last resort and accuracy is lower than board-based methods.
    """
    name = "glasses_markers"

    def run(self, session_dir, config, camera_specs, sync_offsets,
            group, cameras, prior_intrinsics, tools_dir) -> Optional[PipelineResult]:
        log.info("[glasses_markers] Starting: auxiliary relative-pose from glasses ArUco")

        glasses_list = config.get("glasses", [])
        if not glasses_list:
            log.warning("[glasses_markers] No glasses config found")
            return None

        # Determine marker size based on group
        marker_size_by_group = config.get("glasses_marker_size_by_group", {})
        if group >= 9:
            marker_size_m = float(marker_size_by_group.get("grp_9_to_16", 0.038))
            log.info("[glasses_markers] Group %d → 38 mm markers", group)
        else:
            marker_size_m = float(marker_size_by_group.get("grp_1_to_8", 0.025))
            log.info("[glasses_markers] Group %d → 25 mm markers", group)

        aruco_dict   = cv2.aruco.getPredefinedDictionary(DESK_ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        video_dir = session_dir / "video"
        if not video_dir.is_dir():
            return None

        videos = sorted(video_dir.glob("*.mkv")) + sorted(video_dir.glob("*.mp4"))
        if cameras and cameras != ["all"]:
            videos = [v for v in videos if v.stem in cameras]
        if len(videos) < 2:
            log.warning("[glasses_markers] Need ≥2 camera videos, found %d", len(videos))
            return None

        # Build per-camera detection index keyed by wall-clock time (seconds).
        # Frame indices are NOT shared across cameras because each camera starts
        # at a slightly different moment. We convert using:
        #   wall_time = frame_idx / fps + sync_offset[cam_id]
        # where sync_offset = median(unix_time - pts_time) from frame logs.
        FPS = 30.0
        SYNC_TOL_S = 0.05   # 50 ms = 1.5 frames tolerance for co-visibility match

        cam_detections: dict[str, dict[float, dict[int, np.ndarray]]] = {}
        # {cam_id → {wall_time_s → {marker_id → corners(4,2)}}}

        for video_path in videos:
            cam_id = video_path.stem
            # Use stem to look up sync offset (strip _video suffix if present)
            sync_key = cam_id.replace("_video", "")
            cam_offset = sync_offsets.get(sync_key, sync_offsets.get(cam_id, 0.0))

            time_dets: dict[float, dict[int, np.ndarray]] = {}
            for frame_idx, frame in _iter_video_frames(video_path, step=5):
                wall_t = round(frame_idx / FPS + cam_offset, 3)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is None:
                    continue
                # Deduplicate IDs within this frame
                best: dict[int, tuple[float, np.ndarray]] = {}
                for det_corners, mid in zip(corners, ids.flatten()):
                    mid = int(mid)
                    area = float(cv2.contourArea(det_corners[0]))
                    if mid not in best or area > best[mid][0]:
                        best[mid] = (area, det_corners[0])
                if best:
                    time_dets[wall_t] = {mid: c for mid, (_, c) in best.items()}
            cam_detections[cam_id] = time_dets
            log.info("[glasses_markers] %s: %d frames with detections", cam_id, len(time_dets))

        # For each glasses pair, find frames where ≥2 cameras see both markers
        relative_poses: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
        glasses_ids_to_glasses: dict[tuple[int, int], dict] = {}
        for g in glasses_list:
            # Override marker size based on group
            g_copy = dict(g)
            g_copy["marker_size_m"] = marker_size_m
            lid = int(g_copy["left_marker_id"])
            rid = int(g_copy["right_marker_id"])
            glasses_ids_to_glasses[(lid, rid)] = g_copy

        for (lid, rid), glasses_cfg in glasses_ids_to_glasses.items():
            geom = _glasses_world_geometry(glasses_cfg)
            # Object points in glasses local frame (known rigid geometry)
            obj_left  = geom["left"]   # (4,3)
            obj_right = geom["right"]  # (4,3)
            obj_pts   = np.concatenate([obj_left, obj_right], axis=0)  # (8,3)

            # Find camera pairs that co-observe this glasses at the same wall-clock time.
            # Use sorted time keys and match within SYNC_TOL_S (50 ms).
            cam_list = list(cam_detections.keys())
            for i, cam_a in enumerate(cam_list):
                for cam_b in cam_list[i + 1:]:
                    pair = (cam_a, cam_b)
                    times_a = sorted(cam_detections[cam_a].keys())
                    times_b = sorted(cam_detections[cam_b].keys())

                    intr_a = (prior_intrinsics or {}).get(cam_a) or seed_intrinsics(cam_a, camera_specs)
                    intr_b = (prior_intrinsics or {}).get(cam_b) or seed_intrinsics(cam_b, camera_specs)

                    # Two-pointer merge: find time-aligned frames within SYNC_TOL_S
                    ia, ib = 0, 0
                    while ia < len(times_a) and ib < len(times_b):
                        ta = times_a[ia]
                        tb = times_b[ib]
                        dt = abs(ta - tb)
                        if dt > SYNC_TOL_S:
                            if ta < tb:
                                ia += 1
                            else:
                                ib += 1
                            continue

                        # Frames are time-aligned — check for this glasses pair
                        dets_a = cam_detections[cam_a][ta]
                        dets_b = cam_detections[cam_b][tb]
                        if (lid in dets_a and rid in dets_a and
                                lid in dets_b and rid in dets_b):
                            img_a = np.concatenate([dets_a[lid], dets_a[rid]], axis=0)
                            img_b = np.concatenate([dets_b[lid], dets_b[rid]], axis=0)

                            ok_a, rvec_a, tvec_a, in_a = cv2.solvePnPRansac(
                                obj_pts.astype(np.float32), img_a.astype(np.float32),
                                intr_a.matrix, intr_a.dist_coeffs,
                                iterationsCount=100, reprojectionError=10.0,
                            )
                            ok_b, rvec_b, tvec_b, in_b = cv2.solvePnPRansac(
                                obj_pts.astype(np.float32), img_b.astype(np.float32),
                                intr_b.matrix, intr_b.dist_coeffs,
                                iterationsCount=100, reprojectionError=10.0,
                            )
                            if (ok_a and ok_b and
                                    in_a is not None and in_b is not None and
                                    len(in_a) >= 4 and len(in_b) >= 4):
                                R_a, _ = cv2.Rodrigues(rvec_a)
                                R_b, _ = cv2.Rodrigues(rvec_b)
                                R_rel = R_a @ R_b.T
                                t_rel = tvec_a.flatten() - R_rel @ tvec_b.flatten()
                                rvec_rel, _ = cv2.Rodrigues(R_rel)
                                if pair not in relative_poses:
                                    relative_poses[pair] = []
                                relative_poses[pair].append(
                                    (rvec_rel.flatten(), t_rel.flatten())
                                )
                        ia += 1
                        ib += 1

        if not relative_poses:
            log.warning("[glasses_markers] No co-visible glasses frames found")
            return None

        # Average relative poses per pair, then build camera extrinsics
        # Use the camera with most observations as reference (pose = identity)
        cam_obs_count: dict[str, int] = {}
        for (ca, cb), poses in relative_poses.items():
            cam_obs_count[ca] = cam_obs_count.get(ca, 0) + len(poses)
            cam_obs_count[cb] = cam_obs_count.get(cb, 0) + len(poses)

        if not cam_obs_count:
            return None

        ref_cam = max(cam_obs_count, key=cam_obs_count.__getitem__)
        log.info("[glasses_markers] Reference camera: %s (%d observations)", ref_cam, cam_obs_count[ref_cam])

        cam_extrinsics: dict[str, CameraExtrinsics] = {
            ref_cam: CameraExtrinsics(
                rvec=np.zeros((3, 1), dtype=np.float64),
                tvec=np.zeros((3, 1), dtype=np.float64),
                source=self.name,
            )
        }

        # BFS/greedy: propagate poses from reference outward
        for (ca, cb), poses in relative_poses.items():
            if len(poses) < 3:
                continue
            rvec_mean = np.mean([p[0] for p in poses], axis=0).reshape(3, 1)
            tvec_mean = np.mean([p[1] for p in poses], axis=0).reshape(3, 1)
            if ca == ref_cam and cb not in cam_extrinsics:
                cam_extrinsics[cb] = CameraExtrinsics(rvec=rvec_mean, tvec=tvec_mean, source=self.name)
            elif cb == ref_cam and ca not in cam_extrinsics:
                # Invert the relative pose
                R, _ = cv2.Rodrigues(rvec_mean)
                R_inv = R.T
                t_inv = (-R_inv @ tvec_mean).reshape(3, 1)
                rvec_inv, _ = cv2.Rodrigues(R_inv)
                cam_extrinsics[ca] = CameraExtrinsics(rvec=rvec_inv, tvec=t_inv, source=self.name)

        pr = PipelineResult(strategy=self.name)
        for cam_id, extr in cam_extrinsics.items():
            intr = (prior_intrinsics or {}).get(cam_id) or seed_intrinsics(cam_id, camera_specs)
            pr.cameras[cam_id] = CameraCalibResult(
                camera_id=cam_id,
                intrinsics=intr,
                extrinsics=extr,
                reprojection_error=float("inf"),  # no triangulation check here
                strategy=self.name,
            )

        log.info("[glasses_markers] Succeeded: %d cameras", len(pr.cameras))
        return pr

# ---------------------------------------------------------------------------
# Strategy 5 — Fusion (merge any combination of results)
# ---------------------------------------------------------------------------

class FusionStrategy(CalibrationStrategy):
    """Merge results from multiple strategies.

    Per camera, prefer the result with the lowest reprojection error.
    Intrinsics and extrinsics may come from different strategies (e.g.
    intrinsics from dynamic_board, extrinsics from desk_markers).
    """
    name = "fusion"

    def fuse(self, results: list[PipelineResult]) -> Optional[PipelineResult]:
        if not results:
            return None

        # Collect all camera IDs across all results
        all_cam_ids: set[str] = set()
        for r in results:
            all_cam_ids.update(r.camera_ids())

        pr = PipelineResult(strategy=self.name)
        pr.metadata["fused_from"] = [r.strategy for r in results]

        for cam_id in sorted(all_cam_ids):
            candidates = [
                r.cameras[cam_id] for r in results if cam_id in r.cameras
            ]
            if not candidates:
                continue

            # Best intrinsics: lowest reprojection error, then source priority
            SOURCE_PRIORITY = ["dynamic_board", "fixed_board", "desk_markers",
                                "glasses_markers", "seeded"]
            def _intr_score(c: CameraCalibResult) -> tuple:
                prio = SOURCE_PRIORITY.index(c.strategy) if c.strategy in SOURCE_PRIORITY else 99
                return (prio, c.reprojection_error)

            best_intr = min(
                [c for c in candidates if c.intrinsics is not None],
                key=_intr_score,
                default=None,
            )
            best_extr = min(
                [c for c in candidates if c.extrinsics is not None],
                key=lambda c: (c.reprojection_error, 0),
                default=None,
            )
            best_overall = min(candidates, key=lambda c: c.reprojection_error)

            fused = CameraCalibResult(
                camera_id=cam_id,
                intrinsics=best_intr.intrinsics if best_intr else None,
                extrinsics=best_extr.extrinsics if best_extr else None,
                reprojection_error=best_overall.reprojection_error,
                strategy=self.name,
                notes=(
                    f"intr={best_intr.strategy if best_intr else 'none'}, "
                    f"extr={best_extr.strategy if best_extr else 'none'}"
                ),
            )
            pr.cameras[cam_id] = fused

        log.info("[fusion] Merged %d cameras from %d strategies",
                 len(pr.cameras), len(results))
        return pr

# ---------------------------------------------------------------------------
# TOML serialisation (matches existing calibration_charuco.toml format)
# ---------------------------------------------------------------------------

def pipeline_result_to_toml(result: PipelineResult, sync_offsets: dict[str, float]) -> str:
    """Serialise a PipelineResult to a TOML string compatible with downstream tools."""
    doc: dict[str, Any] = {}

    doc["metadata"] = {
        "calibration_date":    result.timestamp,
        "method":              result.strategy,
        "pipeline_version":    "calibration_pipeline/1.0",
        "reprojection_error":  min(
            (c.reprojection_error for c in result.cameras.values()
             if c.reprojection_error < float("inf")),
            default=float("nan"),
        ),
    }
    doc["metadata"].update(result.metadata)

    doc["sync"] = {
        lbl: {"offset_s": float(off)}
        for lbl, off in sync_offsets.items()
    }

    doc["cameras"] = {}
    for cam_id, cam in result.cameras.items():
        cam_doc: dict[str, Any] = {
            "strategy":           cam.strategy,
            "reprojection_error": cam.reprojection_error,
        }
        if cam.notes:
            cam_doc["notes"] = cam.notes
        if cam.intrinsics is not None:
            cam_doc["matrix"] = cam.intrinsics.matrix.tolist()
            cam_doc["dist"]   = cam.intrinsics.dist_coeffs.tolist()
        if cam.extrinsics is not None:
            cam_doc["rotation"]    = cam.extrinsics.rvec.flatten().tolist()
            cam_doc["translation"] = cam.extrinsics.tvec.flatten().tolist()
        doc["cameras"][cam_id] = cam_doc

    return _toml_dumps(doc)


def load_pipeline_result_from_toml(toml_path: Path) -> PipelineResult:
    data = _toml_loads(toml_path.read_text(encoding="utf-8"))
    pr = PipelineResult(
        strategy=data.get("metadata", {}).get("method", "unknown"),
        timestamp=data.get("metadata", {}).get("calibration_date", ""),
        metadata=dict(data.get("metadata", {})),
    )
    for cam_id, cam_data in data.get("cameras", {}).items():
        mat  = cam_data.get("matrix")
        dist = cam_data.get("dist")
        rot  = cam_data.get("rotation")
        trans = cam_data.get("translation")
        intr = CameraIntrinsics(
            matrix=np.array(mat, dtype=np.float64),
            dist_coeffs=np.array(dist, dtype=np.float64),
            resolution=(1920, 1080),
            source=cam_data.get("strategy", "unknown"),
        ) if mat and dist else None
        extr = CameraExtrinsics(
            rvec=np.array(rot,   dtype=np.float64).reshape(3, 1),
            tvec=np.array(trans, dtype=np.float64).reshape(3, 1),
            source=cam_data.get("strategy", "unknown"),
        ) if rot and trans else None
        pr.cameras[cam_id] = CameraCalibResult(
            camera_id=cam_id,
            intrinsics=intr,
            extrinsics=extr,
            reprojection_error=cam_data.get("reprojection_error", float("inf")),
            strategy=cam_data.get("strategy", "unknown"),
            notes=cam_data.get("notes", ""),
        )
    return pr

# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

# "transfer" is handled specially in the pipeline — not a regular strategy
_STRATEGY_ORDER = ["transfer", "dynamic_board", "fixed_board", "desk_markers", "glasses_markers"]
_STRATEGY_MAP: dict[str, CalibrationStrategy] = {
    "dynamic_board":   DynamicBoardStrategy(),
    "fixed_board":     FixedBoardStrategy(),
    "desk_markers":    DeskMarkersStrategy(),
    "glasses_markers": GlassesMarkersStrategy(),
}


class CalibrationPipeline:
    """Orchestrates calibration strategies with fallback and fusion.

    Execution order
    ---------------
    1. Sync offsets loaded (Tier 1-4 fallback).
    2. Transfer calibration: if a compatible previous calibration exists in
       --calib-store it is loaded as a warm start.  Used as-is if no other
       strategy succeeds, or merged in fusion otherwise.
    3. Board strategies (dynamic_board, fixed_board): only run on short
       calibration recordings; auto-skipped on session videos.
    4. Desk-marker + glasses strategies: run on session videos with smart
       frame pre-selection.
    5. Fusion: best result per camera merged from all succeeded strategies.
    """

    def run(
        self,
        session_dir: Path,
        config_path: Path,
        camera_specs_path: Path,
        group: int,
        strategies: list[str],
        cameras: list[str],
        out_path: Path,
        calib_store: Optional[Path] = None,
        video_type: str = "auto",     # "auto" | "calibration" | "session"
    ) -> PipelineResult:
        config       = load_lab_config(config_path)
        camera_specs = load_camera_specs(camera_specs_path)

        # calibration_pipeline.py lives in affectai-data-processing-seed/tools/
        # calibrate_charuco.py lives in affectai-data-processing/tools/  (one level up)
        _this_tools = Path(__file__).parent.resolve()
        tools_dir = _this_tools
        _charuco_candidate = _this_tools.parent.parent / "tools" / "calibrate_charuco.py"
        if _charuco_candidate.is_file():
            tools_dir = _charuco_candidate.parent
            log.info("calibrate_charuco.py found at %s", tools_dir)

        log.info("=== Calibration Pipeline ===")
        log.info("Session   : %s", session_dir)
        log.info("Group     : %d  (rig era: %s)",
                 group, "new (grp-9+)" if group >= GRP_RIG_CHANGE else "original (grp-1–8)")
        log.info("Strategies: %s", strategies)
        log.info("Video type: %s", video_type)

        # ── Step 1: Sync offsets ──────────────────────────────────────────
        sync_offsets = load_sync_offsets(session_dir)
        log.info("Sync offsets loaded: %d cameras", len(sync_offsets))

        # ── Step 2: Detect actual video type if auto ──────────────────────
        if video_type == "auto":
            video_dir = session_dir / "video"
            sample_videos = (
                sorted(video_dir.glob("*.mkv")) + sorted(video_dir.glob("*.mp4"))
                if video_dir.is_dir() else []
            )
            if sample_videos:
                video_type = detect_video_type(sample_videos[0])
                log.info("Auto-detected video type: %s (%.0f min, %s)",
                         video_type, _get_video_duration_s(sample_videos[0]) / 60,
                         sample_videos[0].name)
            else:
                video_type = "session"
                log.warning("No videos found; assuming session type")

        # ── Step 3: Seed intrinsics from existing or transferred calib ────
        prior_intrinsics: Optional[dict[str, CameraIntrinsics]] = None

        # Check for an existing TOML in the session directory
        for existing_name in ("calibration_charuco.toml", "calibration_pipeline.toml"):
            existing_toml = session_dir / existing_name
            if existing_toml.is_file():
                try:
                    prior_intrinsics = load_intrinsics_from_toml(existing_toml)
                    log.info("Seeded intrinsics from %s (%d cameras)",
                             existing_name, len(prior_intrinsics))
                    break
                except Exception as exc:
                    log.warning("Could not load prior intrinsics from %s: %s", existing_name, exc)

        # ── Step 4: Transfer calibration (load from store) ────────────────
        transfer_result: Optional[PipelineResult] = None
        if "transfer" in strategies and calib_store:
            transfer_result = SessionCalibrationTransfer().load(group, calib_store, camera_specs)
            if transfer_result and transfer_result.succeeded():
                log.info("[transfer] Loaded %d cameras as warm start", len(transfer_result.cameras))
                # Seed intrinsics from transfer for downstream strategies
                if prior_intrinsics is None:
                    prior_intrinsics = {}
                for cam_id, cam_res in transfer_result.cameras.items():
                    if cam_res.intrinsics and cam_id not in prior_intrinsics:
                        prior_intrinsics[cam_id] = cam_res.intrinsics
            else:
                log.info("[transfer] No compatible calibration found in store")
                transfer_result = None

        # ── Step 5: Run active strategies ────────────────────────────────
        succeeded: list[PipelineResult] = []
        failed:    list[str]            = []

        # Include transfer result as a baseline for fusion
        if transfer_result and transfer_result.succeeded():
            succeeded.append(transfer_result)

        # Board strategies only make sense on calibration videos
        board_strategies    = {"dynamic_board", "fixed_board"}
        non_board_strategies = {"desk_markers", "glasses_markers"}

        for strat_name in strategies:
            if strat_name == "transfer":
                continue  # already handled above

            # Skip board strategies on session-length videos
            if strat_name in board_strategies and video_type == "session":
                log.info("[%s] Skipped: session video (dynamic board not present)", strat_name)
                failed.append(f"{strat_name}(skipped:session_video)")
                continue

            # Skip desk/glasses strategies on calibration videos if board strategies are available
            # (they are less accurate; only run if board strategies failed)
            if (strat_name in non_board_strategies
                    and video_type == "calibration"
                    and any(s.strategy in board_strategies for s in succeeded)):
                log.info("[%s] Skipped: calibration video and board strategy already succeeded",
                         strat_name)
                continue

            strategy = _STRATEGY_MAP.get(strat_name)
            if strategy is None:
                log.warning("Unknown strategy: %s", strat_name)
                continue

            try:
                result = strategy.run(
                    session_dir=session_dir,
                    config=config,
                    camera_specs=camera_specs,
                    sync_offsets=sync_offsets,
                    group=group,
                    cameras=cameras if cameras != ["all"] else None,
                    prior_intrinsics=prior_intrinsics,
                    tools_dir=tools_dir,
                )
            except Exception as exc:
                log.error("Strategy %s raised: %s", strat_name, exc, exc_info=True)
                result = None

            if result and result.succeeded():
                succeeded.append(result)
                log.info("Strategy %s SUCCEEDED (%d cameras)", strat_name, len(result.cameras))
                # Propagate intrinsics to subsequent strategies
                for cam_id, cam_res in result.cameras.items():
                    if cam_res.intrinsics:
                        if prior_intrinsics is None:
                            prior_intrinsics = {}
                        prior_intrinsics[cam_id] = cam_res.intrinsics
            else:
                failed.append(strat_name)
                log.warning("Strategy %s FAILED or produced no results", strat_name)

        # ── Step 6: Fusion ────────────────────────────────────────────────
        if len(succeeded) > 1:
            fusion_result = FusionStrategy().fuse(succeeded)
        elif len(succeeded) == 1:
            fusion_result = succeeded[0]
        else:
            log.error("All calibration strategies failed for %s", session_dir)
            return PipelineResult(
                strategy="failed",
                metadata={"failed_strategies": failed},
            )

        fusion_result.metadata["failed_strategies"] = failed
        fusion_result.metadata["video_type"]        = video_type
        fusion_result.metadata["group"]             = group

        # ── Step 7: Write output ──────────────────────────────────────────
        toml_str = pipeline_result_to_toml(fusion_result, sync_offsets)
        out_path.write_text(toml_str, encoding="utf-8")
        log.info("Calibration written to %s", out_path)

        return fusion_result

# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def validate(calib_path: Path, config_path: Optional[Path] = None) -> None:
    pr = load_pipeline_result_from_toml(calib_path)
    print(f"\n{'='*60}")
    print(f"Calibration: {calib_path}")
    print(f"Strategy:    {pr.strategy}")
    print(f"Timestamp:   {pr.timestamp}")
    print(f"Cameras:     {len(pr.cameras)}")
    print(f"{'='*60}")
    for cam_id, cam in sorted(pr.cameras.items()):
        status = "OK"
        err = cam.reprojection_error
        if err < 5.0:
            quality = "GOOD"
        elif err < 10.0:
            quality = "FAIR"
        else:
            quality = "POOR"
        has_intr = cam.intrinsics is not None
        has_extr = cam.extrinsics is not None
        fx = cam.intrinsics.matrix[0, 0] if has_intr else float("nan")
        print(
            f"  {cam_id:<35} reproj={err:6.2f}px [{quality}]"
            f"  intr={'Y' if has_intr else 'N'}  extr={'Y' if has_extr else 'N'}"
            f"  fx={fx:.1f}  src={cam.strategy}"
        )
        if cam.notes:
            print(f"    └─ {cam.notes}")
    print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AffectAI flexible multi-camera calibration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run calibration pipeline")
    run_p.add_argument("--session-dir",  required=True, type=Path,
                       help="Session directory containing video/ and frame_logs/")
    run_p.add_argument("--config",       required=True, type=Path,
                       help="Path to desk_markers_large.yaml (or equivalent)")
    run_p.add_argument("--camera-specs", type=Path,
                       default=Path(__file__).parent.parent / "configs" / "camera_specs.json",
                       help="Path to camera_specs.json")
    run_p.add_argument("--group",        type=int, default=1,
                       help="Collection group number (affects glasses marker size: 1–8→25mm, 9–16→38mm)")
    run_p.add_argument("--strategies",   default=",".join(_STRATEGY_ORDER),
                       help=f"Comma-separated strategy list (default: {','.join(_STRATEGY_ORDER)})")
    run_p.add_argument("--cameras",      default="all",
                       help="Comma-separated camera IDs to calibrate, or 'all'")
    run_p.add_argument("--out",          type=Path,
                       help="Output TOML path (default: <session-dir>/calibration_pipeline.toml)")
    run_p.add_argument("--calib-store",  type=Path, default=None,
                       help="Directory of previous calibration TOMLs for transfer strategy")
    run_p.add_argument("--video-type",   default="auto",
                       choices=["auto", "calibration", "session"],
                       help="Override video type detection: "
                            "'calibration' = short dedicated recording with board, "
                            "'session' = 1-hour experiment video (default: auto-detect)")
    run_p.add_argument("--verbose", "-v", action="store_true")

    # --- validate ---
    val_p = sub.add_parser("validate", help="Print calibration quality report")
    val_p.add_argument("--calib",   required=True, type=Path)
    val_p.add_argument("--config",  type=Path)

    # --- report ---
    rep_p = sub.add_parser("report", help="Alias for validate")
    rep_p.add_argument("--calib", required=True, type=Path)
    rep_p.add_argument("--config", type=Path)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "run":
        session_dir  = args.session_dir.resolve()
        out_path     = args.out or (session_dir / "calibration_pipeline.toml")
        strategies   = [s.strip() for s in args.strategies.split(",") if s.strip()]
        cameras      = [c.strip() for c in args.cameras.split(",") if c.strip()]

        pipeline = CalibrationPipeline()
        result = pipeline.run(
            session_dir=session_dir,
            config_path=args.config.resolve(),
            camera_specs_path=args.camera_specs.resolve(),
            group=args.group,
            strategies=strategies,
            cameras=cameras,
            out_path=out_path,
            calib_store=args.calib_store.resolve() if args.calib_store else None,
            video_type=args.video_type,
        )

        if result.strategy == "failed":
            sys.exit(1)

        # Print brief report
        validate(out_path)

    elif args.command in ("validate", "report"):
        validate(args.calib.resolve(), getattr(args, "config", None))


if __name__ == "__main__":
    main()
