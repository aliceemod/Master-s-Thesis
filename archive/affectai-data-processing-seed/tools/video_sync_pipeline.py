#!/usr/bin/env python3
"""Video synchronization & task-splitting pipeline.

Splits raw multicam MKV recordings into per-task, per-camera MP4 clips
aligned to the same XDF/LSL clock used by ``xdf_sync_pipeline.py``.

Synchronization strategy (mirrors audio pipeline)
--------------------------------------------------
1. **Tier 1 — Frame logs** (~0.5 ms accuracy):
   ``frame_logs/{label}_frames.jsonl`` with per-frame ``pts_time`` and
   ``unix_time``.  A linear regression of ``unix_time − pts_time`` over
   time captures clock drift and provides the most precise anchor.

2. **Tier 2 — XDF ffmpeg_progress streams** (~1 ms):
   ``ffmpeg_progress_{label}`` streams in the session XDF, with per-sample
   pyxdf clock-corrected timestamps.

3. **Tier 3 — JSONL progress logs** (~1 ms):
   ``lsl/ffmpeg_progress_{label}.jsonl`` files from the LSL recorder JSONL
   output.  Uses ``received_time`` (wall clock ISO) converted to XDF time.

4. **Tier 4 — Progress TSV** (~1 ms):
   ``sourcedata/sync/{label}_ffmpeg_progress.tsv`` combined with the
   ``ffmpeg_clock`` bridge from XDF.

For each tier the pipeline fits a **linear regression**::

    anchor(t) = slope * t + intercept
    media_time = xdf_time − anchor(xdf_time)

This compensates for the small (~0.04 ms/s) clock drift between the
capture PC's USB frame clock and the LSL clock.

Output quality
--------------
Videos are transcoded to MP4 (H.264) with ``-crf 18 -preset slow`` which
is **visually lossless** while being broadly compatible with FreeMoCap,
OpenPose, MediaPipe, and other downstream tools.  The original resolution
(1920×1080) and frame rate (30 fps) are preserved.

For truly lossless output, pass ``--crf 0``.

Output structure
----------------
::

    <output_root>/sub-01/ses-{id}/video/
        sub-01_ses-{id}_task-T0_run-01_acq-cam1.mp4
        sub-01_ses-{id}_task-T0_run-01_acq-cam2.mp4
        …
        sub-01_ses-{id}_task-T0_run-01_acq-cam7.mp4
        sub-01_ses-{id}_task-T1_run-01_acq-cam1.mp4
        …
        sub-01_ses-{id}_video_sync_metadata.json

Usage
-----
::

    python tools/video_sync_pipeline.py \\
        --data-root affectai-data-processing-seed/data \\
        --output-dir E:/processed_data \\
        [--sessions ses-20260312_grp-07_run01 ...] \\
        [--groups grp-07 grp-12] \\
        [--crf 18] \\
        [--dry-run]

    # Use existing task windows from xdf_sync_pipeline output:
    python tools/video_sync_pipeline.py \\
        --data-root affectai-data-processing-seed/data \\
        --output-dir E:/processed_data \\
        --sync-root E:/processed_data \\
        --groups grp-12
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("video_sync_pipeline")


# ---------------------------------------------------------------------------
# Camera label helpers
# ---------------------------------------------------------------------------

# Matches video device labels produced by ffmpeg_multicap.py
_RE_VIDEO_DEVICE = re.compile(
    r"jabra_panacast_(?:20_cam\d+|50)_vid", re.IGNORECASE,
)

# Short camera name from full label
_CAMERA_SHORT_RE = re.compile(
    r"jabra_panacast_20_(cam\d+)_vid|jabra_panacast_(50)_vid", re.IGNORECASE,
)


def _short_camera_name(label: str) -> str:
    """Convert ``jabra_panacast_20_cam1_vid`` → ``cam1``, ``…_50_vid`` → ``cam7``."""
    m = _CAMERA_SHORT_RE.search(label)
    if not m:
        return label
    if m.group(1):
        return m.group(1)
    return "cam7"  # PanaCast 50 → cam7


# ---------------------------------------------------------------------------
# Linear regression (same as xdf_sync_pipeline)
# ---------------------------------------------------------------------------

def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if abs(den) < 1e-12:
        return 0.0, y_mean
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _regression_rmse(
    xs: list[float], ys: list[float], slope: float, intercept: float,
) -> float:
    mse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys)) / len(xs)
    return mse ** 0.5


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Tier 1: Frame-log anchors
# ---------------------------------------------------------------------------

def _compute_frame_log_anchors(
    frame_log_dir: Path,
) -> dict[str, tuple[float, float, float]]:
    """Compute anchor regressions from per-camera frame logs.

    Each JSONL line has ``{"pts_time": <float>, "unix_time": <float>, ...}``.
    The anchor is ``unix_time − pts_time`` over frames, fit via linear
    regression to capture USB-clock drift.

    Returns dict mapping device label → ``(slope, intercept, rmse)``.
    """
    if not frame_log_dir.exists():
        return {}

    result: dict[str, tuple[float, float, float]] = {}
    for fl in sorted(frame_log_dir.glob("*_frames.jsonl")):
        label = fl.stem.replace("_frames", "")
        if not _RE_VIDEO_DEVICE.match(label):
            continue

        xs: list[float] = []  # unix_time values
        ys: list[float] = []  # unix_time - pts_time (anchor diff)
        with fl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    pts = float(rec["pts_time"])
                    unix_t = float(rec["unix_time"])
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
                xs.append(unix_t)
                ys.append(unix_t - pts)

        if not xs:
            continue
        if len(xs) < 20:
            med = statistics.median(ys)
            rmse = statistics.stdev(ys) if len(ys) > 1 else 0.0
            result[label] = (0.0, med, rmse)
        else:
            slope, intercept = _linreg(xs, ys)
            rmse = _regression_rmse(xs, ys, slope, intercept)
            result[label] = (slope, intercept, rmse)
            logger.info(
                "    %s (frame_log): slope=%.9f intercept=%.6f "
                "(drift=%.3f ms/s, rmse=%.6f s, n=%d)",
                label, slope, intercept, slope * 1000, rmse, len(xs),
            )
    return result


# ---------------------------------------------------------------------------
# Tier 2: XDF ffmpeg_progress anchors
# ---------------------------------------------------------------------------

_RE_VID_PROGRESS_XDF = re.compile(
    r"^ffmpeg_progress_jabra_panacast_(?:20_cam\d+|50)_vid$", re.IGNORECASE,
)


def _compute_video_anchors_from_xdf(
    raw_streams: list[dict],
) -> dict[str, tuple[float, float, float]]:
    """Compute per-camera anchor regression from XDF video progress streams.

    Same approach as ``_compute_dpa_anchors_from_xdf`` in
    ``xdf_sync_pipeline.py`` but filtering for video streams.
    """
    result: dict[str, tuple[float, float, float]] = {}
    for s in raw_streams:
        info = s.get("info", {})
        name = (
            info.get("name", [""])[0]
            if isinstance(info.get("name"), list)
            else info.get("name", "")
        )
        if not _RE_VID_PROGRESS_XDF.match(str(name).strip()):
            continue
        ts = s.get("time_stamps")
        series = s.get("time_series")
        if ts is None or not hasattr(ts, "__len__") or len(ts) < 10:
            continue
        label = str(name).strip().replace("ffmpeg_progress_", "")
        diffs = [float(ts[i]) - float(series[i][0]) for i in range(len(ts))]
        xdf_times = [float(ts[i]) for i in range(len(ts))]

        if len(diffs) < 20:
            med = statistics.median(diffs)
            rmse = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            result[label] = (0.0, med, rmse)
        else:
            slope, intercept = _linreg(xdf_times, diffs)
            rmse = _regression_rmse(xdf_times, diffs, slope, intercept)
            result[label] = (slope, intercept, rmse)
            logger.info(
                "    %s (XDF): slope=%.9f intercept=%.6f "
                "(drift=%.3f ms/s, rmse=%.6f s, n=%d)",
                label, slope, intercept, slope * 1000, rmse, len(xdf_times),
            )
    return result


# ---------------------------------------------------------------------------
# Tier 3: JSONL progress anchors
# ---------------------------------------------------------------------------

def _compute_video_anchors_from_jsonl(
    capture_dir: Path,
    wall_minus_xdf_lsl: float,
) -> dict[str, tuple[float, float, float]]:
    """Compute per-camera anchor from LSL recorder JSONL files.

    Uses ``received_time`` (Lab Recorder wall clock) converted to XDF time
    via ``wall_minus_xdf_lsl``.
    """
    lsl_dir = capture_dir / "lsl"
    if not lsl_dir.exists():
        return {}

    result: dict[str, tuple[float, float, float]] = {}
    for jsonl_path in sorted(lsl_dir.glob("ffmpeg_progress_*.jsonl")):
        label = jsonl_path.stem.replace("ffmpeg_progress_", "")
        if not _RE_VIDEO_DEVICE.match(label):
            continue
        xs: list[float] = []
        ys: list[float] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    received_str = rec["received_time"]
                    out_t = float(rec["values"][0])
                    dt = datetime.fromisoformat(received_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    wall_epoch = dt.timestamp()
                    xdf_t = wall_epoch - wall_minus_xdf_lsl
                except (KeyError, IndexError, ValueError,
                        json.JSONDecodeError, AttributeError):
                    continue
                xs.append(xdf_t)
                ys.append(xdf_t - out_t)
        if not xs:
            continue
        if len(xs) < 20:
            med = statistics.median(ys)
            rmse = statistics.stdev(ys) if len(ys) > 1 else 0.0
            result[label] = (0.0, med, rmse)
        else:
            slope, intercept = _linreg(xs, ys)
            rmse = _regression_rmse(xs, ys, slope, intercept)
            result[label] = (slope, intercept, rmse)
            logger.info(
                "    %s (JSONL): slope=%.9f intercept=%.6f "
                "(drift=%.3f ms/s, rmse=%.6f s, n=%d)",
                label, slope, intercept, slope * 1000, rmse, len(xs),
            )
    return result


# ---------------------------------------------------------------------------
# Tier 4: Progress TSV anchors
# ---------------------------------------------------------------------------

def _compute_video_anchors_from_tsv(
    capture_dir: Path,
    av_to_xdf_offset: float,
) -> dict[str, tuple[float, float, float]]:
    """Fallback: progress TSV + ffmpeg_clock bridge."""
    sync_dir = capture_dir / "sourcedata" / "sync"
    if not sync_dir.exists():
        return {}

    result: dict[str, tuple[float, float, float]] = {}
    for tsv in sorted(sync_dir.glob("*_ffmpeg_progress.tsv")):
        label = tsv.stem.replace("_ffmpeg_progress", "")
        if not _RE_VIDEO_DEVICE.match(label):
            continue
        xs: list[float] = []
        ys: list[float] = []
        with tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    host_t = float(row["host_time_sec"])
                    out_t = float(row["out_time_sec"])
                except (ValueError, TypeError, KeyError):
                    continue
                xdf_t = host_t + av_to_xdf_offset
                xs.append(xdf_t)
                ys.append(xdf_t - out_t)
        if not xs:
            continue
        if len(xs) < 20:
            med = statistics.median(ys)
            rmse = statistics.stdev(ys) if len(ys) > 1 else 0.0
            result[label] = (0.0, med, rmse)
        else:
            slope, intercept = _linreg(xs, ys)
            rmse = _regression_rmse(xs, ys, slope, intercept)
            result[label] = (slope, intercept, rmse)
            logger.info(
                "    %s (TSV): slope=%.9f intercept=%.6f "
                "(drift=%.3f ms/s, rmse=%.6f s, n=%d)",
                label, slope, intercept, slope * 1000, rmse, len(xs),
            )
    return result


# ---------------------------------------------------------------------------
# Anchor selection (pick best per camera across tiers)
# ---------------------------------------------------------------------------

def _select_best_anchors(
    *anchor_dicts: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, str]]:
    """Per-camera: pick the anchor with the lowest RMSE.

    Returns mapping label → ``(slope, intercept, tier_name)``.
    """
    tier_names = ["frame_log", "xdf", "jsonl", "tsv"]
    all_labels: set[str] = set()
    for d in anchor_dicts:
        all_labels.update(d.keys())

    result: dict[str, tuple[float, float, str]] = {}
    for label in sorted(all_labels):
        best_rmse = float("inf")
        best_slope = 0.0
        best_intercept = 0.0
        best_tier = "unknown"
        for d, tier in zip(anchor_dicts, tier_names):
            if label not in d:
                continue
            slope, intercept, rmse = d[label]
            if rmse < best_rmse:
                best_rmse = rmse
                best_slope = slope
                best_intercept = intercept
                best_tier = tier
        result[label] = (best_slope, best_intercept, best_tier)
        logger.info(
            "    %s anchor: %s wins (rmse=%.6f s)",
            label, best_tier, best_rmse,
        )
    return result


# ---------------------------------------------------------------------------
# ffmpeg_clock bridge (AV PC → XDF offset)
# ---------------------------------------------------------------------------

def _compute_av_to_xdf_offset(streams: list[dict]) -> float | None:
    """Compute offset: ``xdf_lsl = av_local_clock + offset``."""
    for s in streams:
        info = s.get("info", {})
        name = (
            info.get("name", [""])[0]
            if isinstance(info.get("name"), list)
            else info.get("name", "")
        )
        if str(name).strip() != "ffmpeg_clock":
            continue
        ts = s.get("time_stamps")
        series = s.get("time_series")
        if ts is None or not hasattr(ts, "__len__") or len(ts) < 2:
            continue
        offsets = [float(ts[i]) - float(series[i][0]) for i in range(len(ts))]
        return statistics.median(offsets)
    return None


# ---------------------------------------------------------------------------
# AV capture directory discovery (reused from xdf_sync_pipeline)
# ---------------------------------------------------------------------------

def _find_av_capture_dir(av_session_dir: Path) -> Path | None:
    """Find the main capture subdirectory with video/ and audio/."""
    sd = av_session_dir / "sourcedata"
    if not sd.exists():
        return None
    best: Path | None = None
    best_size = 0
    for child in sd.iterdir():
        if not child.is_dir():
            continue
        if "calibration" in child.name.lower() or child.name == "av":
            continue
        video_dir = child / "video"
        if not video_dir.exists():
            continue
        total = sum(f.stat().st_size for f in video_dir.glob("*.mkv"))
        if total > best_size:
            best_size = total
            best = child
    return best


def _find_av_session_dir(data_root: Path, session_info: dict) -> Path | None:
    """Find AV session directory."""
    session_name = session_info.get("session", "")
    av_splits = session_info.get("av_splits", [])

    search_roots: list[Path] = []
    for phase in av_splits or ["final", "pilot", "test"]:
        search_roots.append(data_root / "AV" / phase / "sub-01")
        search_roots.append(
            data_root / "affectai-capture-av" / "sessions" / phase / "sub-01"
        )

    for av_root in search_roots:
        if not av_root.exists():
            continue
        for d in av_root.iterdir():
            if d.is_dir() and session_name.replace("ses-", "") in d.name:
                return d
        exact = av_root / session_name
        if exact.exists():
            return exact
    return None


# ---------------------------------------------------------------------------
# Video probing
# ---------------------------------------------------------------------------

def _probe_video(path: Path) -> dict[str, Any]:
    """Probe video file for duration, resolution, codec, fps."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format", {})
        fps_str = stream.get("r_frame_rate", "30/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        except (ValueError, ZeroDivisionError):
            fps = 30.0
        duration = float(
            stream.get("duration") or fmt.get("duration") or 0
        )
        return {
            "codec": stream.get("codec_name", ""),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": fps,
            "duration_s": duration,
        }
    except Exception as exc:
        logger.warning("    ffprobe failed for %s: %s", path.name, exc)
        return {}


# ---------------------------------------------------------------------------
# Video splitting via ffmpeg
# ---------------------------------------------------------------------------

def split_av_video_by_windows(
    capture_dir: Path,
    output_dir: Path,
    sub_label: str,
    ses_label: str,
    windows: list[dict[str, Any]],
    video_anchors: dict[str, tuple[float, float, str]],
    wall_minus_xdf_lsl: float,
    crf: int = 18,
    preset: str = "slow",
) -> list[Path]:
    """Crop multicam MKV videos into per-task MP4 clips using ffmpeg.

    Uses **time-varying anchors** (linear regression) to compensate for
    video-clock drift relative to the XDF/LSL clock, identical to the
    audio pipeline approach.

    ffmpeg is invoked with ``-ss`` **after** ``-i`` for frame-accurate
    seeking, then transcodes to H.264 MP4.  The ``-ss`` after ``-i``
    approach decodes from the nearest preceding keyframe, ensuring no
    frames are skipped at the expense of slightly longer processing.

    Parameters
    ----------
    capture_dir :
        AV capture subdirectory containing ``video/*.mkv``.
    output_dir :
        BIDS session output directory (files go under ``video/``).
    sub_label, ses_label :
        BIDS subject / session labels.
    windows :
        Task windows with ``start_wall_clock`` and ``end_wall_clock``.
    video_anchors :
        Device label → ``(slope, intercept, tier_name)`` from
        ``_select_best_anchors()``.
    wall_minus_xdf_lsl :
        Offset: ``stimuli_wall_clock − xdf_lsl``.
    crf :
        H.264 CRF value.  0 = lossless, 18 = visually lossless (default),
        23 = ffmpeg default.
    preset :
        H.264 encoding preset (``slow`` for quality, ``medium`` for speed).

    Returns
    -------
    list[Path]
        Written MP4 file paths.
    """
    video_dir = capture_dir / "video"
    if not video_dir.exists():
        logger.warning("  No video/ directory in %s", capture_dir)
        return []

    out_video = output_dir / "video"
    out_video.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Discover MKV files and match to anchor labels
    active_cameras: list[tuple[str, Path, str]] = []  # (label, mkv_path, short_name)
    for mkv in sorted(video_dir.glob("*.mkv")):
        # Derive label: strip _video suffix → matches anchor label
        # e.g.  jabra_panacast_20_cam1_vid_video.mkv → jabra_panacast_20_cam1_vid
        label = mkv.stem
        if label.endswith("_video"):
            label = label[: -len("_video")]
        short = _short_camera_name(label)
        if label in video_anchors:
            active_cameras.append((label, mkv, short))
        else:
            logger.warning("    No sync anchor for %s — skipping", mkv.name)

    if not active_cameras:
        logger.warning("  No anchored cameras found in %s", video_dir)
        return []

    # Probe first video for resolution/fps
    probe = _probe_video(active_cameras[0][1])
    src_fps = probe.get("fps", 30.0)
    logger.info(
        "  Found %d cameras, source fps=%.2f, crf=%d, preset=%s",
        len(active_cameras), src_fps, crf, preset,
    )

    def _anchor_at(label: str, xdf_t: float) -> float:
        slope, intercept, _ = video_anchors[label]
        return slope * xdf_t + intercept

    def _run_ffmpeg(
        mkv_path: Path, out_path: Path,
        ss: float, duration: float,
        label: str, task: str,
    ) -> bool:
        """Transcode MKV clip to MP4 with frame-accurate seeking."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(mkv_path),
            "-ss", f"{ss:.6f}",
            "-t", f"{duration:.6f}",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-r", f"{src_fps:.6f}",
            "-an",  # strip audio (separate audio pipeline handles it)
            "-movflags", "+faststart",
            str(out_path),
        ]
        logger.info(
            "    %s %s: -ss %.6f -t %.6f → %s",
            label, task, ss, duration, out_path.name,
        )
        try:
            proc = subprocess.run(
                cmd, check=True, capture_output=True, timeout=1800,
            )
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(
                "    ffmpeg failed for %s %s: %s",
                label, task,
                exc.stderr.decode(errors="replace")[-300:],
            )
        except FileNotFoundError:
            logger.error("    ffmpeg not found on PATH")
        except subprocess.TimeoutExpired:
            logger.error("    ffmpeg timed out for %s %s", label, task)
        return False

    for w in windows:
        task = w["task"]
        xdf_start = w["start_wall_clock"] - wall_minus_xdf_lsl
        xdf_end = w["end_wall_clock"] - wall_minus_xdf_lsl
        task_duration = xdf_end - xdf_start

        for label, mkv_path, short_name in active_cameras:
            anchor = _anchor_at(label, xdf_start)
            media_start = xdf_start - anchor

            # Clamp negative (task starts before recording)
            if media_start < 0:
                clip_duration = task_duration + media_start
                media_start = 0.0
            else:
                clip_duration = task_duration

            if clip_duration <= 0:
                logger.warning(
                    "    %s %s: clip duration ≤ 0, skipping", short_name, task,
                )
                continue

            out_name = (
                f"sub-{sub_label}_ses-{ses_label}"
                f"_task-{task}_run-01_acq-{short_name}.mp4"
            )
            out_path = out_video / out_name
            if _run_ffmpeg(mkv_path, out_path, media_start, clip_duration,
                           short_name, task):
                outputs.append(out_path)

    return outputs


# ---------------------------------------------------------------------------
# Load task windows from xdf_sync_pipeline output
# ---------------------------------------------------------------------------

def _load_task_windows_from_sync(
    sync_session_dir: Path,
) -> tuple[list[dict[str, Any]], float | None]:
    """Load pre-computed task windows and wall_minus_lsl offset.

    Reads ``annot/*_task_run_windows.tsv`` and
    ``annot/*_sync_metadata.json`` from an existing
    ``xdf_sync_pipeline`` output.
    """
    # Find task run windows TSV
    annot = sync_session_dir / "annot"
    if not annot.exists():
        return [], None

    window_files = sorted(annot.glob("*_task_run_windows.tsv"))
    if not window_files:
        return [], None

    windows: list[dict[str, Any]] = []
    with window_files[0].open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                windows.append({
                    "task": row["task"],
                    "run": row.get("run", "01"),
                    "start_wall_clock": float(row["start_wall_clock"]),
                    "end_wall_clock": float(row["end_wall_clock"]),
                    "duration_s": float(row["duration_s"]),
                    "start_lsl": _parse_float(row.get("start_lsl", ""), -1),
                    "end_lsl": _parse_float(row.get("end_lsl", ""), -1),
                })
            except (KeyError, ValueError) as exc:
                logger.warning("  Skipping window row: %s", exc)

    # Load offset from sync metadata
    offset: float | None = None
    meta_files = sorted(annot.glob("*_sync_metadata.json"))
    if meta_files:
        try:
            meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
            offset = meta.get("wall_minus_lsl_offset")
        except Exception:
            pass

    return windows, offset


# ---------------------------------------------------------------------------
# Session discovery (reused from xdf_sync_pipeline)
# ---------------------------------------------------------------------------

def _load_inventory(data_root: Path) -> dict[str, Any]:
    inv_path = data_root / "high_level_data_inventory.json"
    if not inv_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inv_path}")
    return json.loads(inv_path.read_text(encoding="utf-8"))


def _find_xdf_files(data_root: Path, group_id: str, session_id: str = "") -> list[Path]:
    found: list[Path] = []
    cs_root = data_root / "CurrentStudy"
    if cs_root.exists():
        grp_num = group_id.replace("grp-", "")
        candidates = [
            cs_root / f"sub-{group_id}",
            cs_root / f"sub-grp-{grp_num}",
        ]
        for cand in candidates:
            if cand.exists():
                found.extend(sorted(cand.rglob("*.xdf")))
        if not found:
            for d in cs_root.iterdir():
                if d.is_dir() and grp_num in d.name:
                    found.extend(sorted(d.rglob("*.xdf")))
        if not found and session_id:
            m = re.search(r"ses-(\d{8})", session_id)
            if m:
                date_str = m.group(1)
                for d in cs_root.iterdir():
                    if not d.is_dir():
                        continue
                    for xdf in d.rglob("*.xdf"):
                        if date_str in xdf.name or date_str in xdf.parent.name:
                            found.append(xdf)

    rec_root = data_root / "affectai-capture-recording" / "sessions"
    if rec_root.exists():
        for xdf in rec_root.rglob("*.xdf"):
            if group_id in str(xdf) or group_id.replace("grp-", "") in xdf.parent.name:
                found.append(xdf)

    unique: list[Path] = []
    seen_names: set[str] = set()
    for p in found:
        if "_old" in p.stem:
            continue
        if p.name not in seen_names:
            seen_names.add(p.name)
            unique.append(p)
    return sorted(unique)


def _session_entities(session_id: str) -> tuple[str, str, str]:
    ses_label = session_id[4:] if session_id.startswith("ses-") else session_id
    sub_label = "01"
    base = f"sub-{sub_label}_ses-{ses_label}"
    return sub_label, ses_label, base


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def process_session(
    session_info: dict,
    data_root: Path,
    output_root: Path,
    dry_run: bool = False,
    sync_root: Path | None = None,
    crf: int = 18,
    preset: str = "slow",
) -> dict[str, Any]:
    """Process a single session: synchronize + split multicam video."""
    session_id = session_info["session"]
    group_id = session_info.get("group_id", "")
    sub_label, ses_label, base = _session_entities(session_id)

    result: dict[str, Any] = {
        "session": session_id,
        "group_id": group_id,
        "success": False,
        "error": None,
        "cameras": [],
        "task_windows": [],
        "output_files": [],
        "anchor_tiers": {},
    }

    logger.info("=" * 70)
    logger.info("Video processing: %s (group=%s)", session_id, group_id)
    logger.info("=" * 70)

    # 1. Find AV session directory
    av_session_dir = _find_av_session_dir(data_root, session_info)
    if not av_session_dir:
        msg = f"No AV session directory found for {session_id}"
        logger.warning("  SKIP: %s", msg)
        result["error"] = msg
        return result

    capture_dir = _find_av_capture_dir(av_session_dir)
    if not capture_dir:
        msg = f"No capture subdirectory found in {av_session_dir}"
        logger.warning("  SKIP: %s", msg)
        result["error"] = msg
        return result

    logger.info("  AV session: %s", av_session_dir.name)
    logger.info("  Capture dir: %s", capture_dir.name)

    # List available video files
    video_dir = capture_dir / "video"
    if not video_dir.exists() or not list(video_dir.glob("*.mkv")):
        msg = f"No MKV files in {video_dir}"
        logger.warning("  SKIP: %s", msg)
        result["error"] = msg
        return result

    mkv_files = sorted(video_dir.glob("*.mkv"))
    result["cameras"] = [f.name for f in mkv_files]
    logger.info("  MKV files: %s", [f.name for f in mkv_files])

    # 2. Load task windows — prefer pre-computed from xdf_sync_pipeline
    windows: list[dict[str, Any]] = []
    offset: float | None = None

    if sync_root:
        sync_session_dir = sync_root / f"sub-{sub_label}" / f"ses-{ses_label}"
        windows, offset = _load_task_windows_from_sync(sync_session_dir)
        if windows:
            logger.info(
                "  Loaded %d task windows from sync output (offset=%.6f)",
                len(windows), offset or 0,
            )

    if not windows:
        # Derive windows from XDF + stimuli (inline, like xdf_sync_pipeline)
        logger.info("  No pre-computed windows — deriving from XDF + stimuli")
        windows, offset = _derive_task_windows(
            data_root, session_info, group_id, session_id,
        )

    if not windows:
        msg = "No task windows could be derived"
        logger.warning("  SKIP: %s", msg)
        result["error"] = msg
        return result

    result["task_windows"] = [
        {"task": w["task"], "duration_s": round(w["duration_s"], 1)}
        for w in windows
    ]
    for w in windows:
        logger.info(
            "    %s: %.1fs (wall %.0f–%.0f)",
            w["task"], w["duration_s"],
            w["start_wall_clock"], w["end_wall_clock"],
        )

    if dry_run:
        logger.info("  [DRY-RUN] Would split %d cameras × %d tasks",
                     len(mkv_files), len(windows))
        result["success"] = True
        return result

    # 3. Compute video sync anchors (tiered)
    wall_minus_xdf = offset if offset is not None else 0.0

    # Tier 1: frame logs
    frame_log_dir = capture_dir / "frame_logs"
    tier1 = _compute_frame_log_anchors(frame_log_dir)
    if tier1:
        logger.info("  Tier 1 (frame_log): %d cameras", len(tier1))

    # Tier 2: XDF progress streams
    tier2: dict[str, tuple[float, float, float]] = {}
    xdf_files = _find_xdf_files(data_root, group_id, session_id)
    raw_streams: list[dict] = []
    if xdf_files:
        try:
            import importlib.util
            if importlib.util.find_spec("pyxdf"):
                import pyxdf
                for xdf_path in xdf_files:
                    streams, _ = pyxdf.load_xdf(str(xdf_path))
                    raw_streams.extend(streams)
                tier2 = _compute_video_anchors_from_xdf(raw_streams)
                if tier2:
                    logger.info("  Tier 2 (XDF): %d cameras", len(tier2))
        except Exception as exc:
            logger.warning("  XDF loading failed: %s", exc)

    # Tier 3: JSONL logs
    tier3: dict[str, tuple[float, float, float]] = {}
    if wall_minus_xdf != 0.0:
        tier3 = _compute_video_anchors_from_jsonl(capture_dir, wall_minus_xdf)
        if tier3:
            logger.info("  Tier 3 (JSONL): %d cameras", len(tier3))

    # Tier 4: Progress TSV + ffmpeg_clock bridge
    tier4: dict[str, tuple[float, float, float]] = {}
    av_to_xdf = _compute_av_to_xdf_offset(raw_streams) if raw_streams else None
    if av_to_xdf is not None:
        tier4 = _compute_video_anchors_from_tsv(capture_dir, av_to_xdf)
        if tier4:
            logger.info("  Tier 4 (TSV): %d cameras", len(tier4))

    # Select best anchor per camera
    video_anchors = _select_best_anchors(tier1, tier2, tier3, tier4)
    if not video_anchors:
        msg = "No video sync anchors found from any tier"
        logger.error("  %s", msg)
        result["error"] = msg
        return result

    result["anchor_tiers"] = {
        label: tier for label, (_, _, tier) in video_anchors.items()
    }

    # 4. Create output directory
    session_dir = output_root / f"sub-{sub_label}" / f"ses-{ses_label}"

    # 5. Split videos
    output_files = split_av_video_by_windows(
        capture_dir, session_dir,
        sub_label, ses_label,
        windows, video_anchors, wall_minus_xdf,
        crf=crf, preset=preset,
    )
    result["output_files"] = [str(p) for p in output_files]

    # 6. Write video sync metadata
    meta = {
        "session_id": session_id,
        "group_id": group_id,
        "pipeline": "video_sync_pipeline",
        "pipeline_version": "1.0",
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "av_session_dir": str(av_session_dir),
        "capture_dir": str(capture_dir),
        "wall_minus_lsl_offset": offset,
        "crf": crf,
        "preset": preset,
        "cameras": {
            label: {
                "anchor_tier": tier,
                "slope": slope,
                "intercept": intercept,
                "short_name": _short_camera_name(label),
            }
            for label, (slope, intercept, tier) in video_anchors.items()
        },
        "task_windows": [
            {"task": w["task"], "duration_s": w["duration_s"]}
            for w in windows
        ],
        "output_file_count": len(output_files),
    }
    meta_dir = session_dir / "video"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{base}_video_sync_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("  Metadata → %s", meta_path.name)

    result["success"] = True
    logger.info(
        "  Done: %d video clips written to %s",
        len(output_files), session_dir / "video",
    )
    return result


# ---------------------------------------------------------------------------
# Derive task windows inline (fallback when no sync_root)
# ---------------------------------------------------------------------------

def _derive_task_windows(
    data_root: Path,
    session_info: dict,
    group_id: str,
    session_id: str,
) -> tuple[list[dict[str, Any]], float | None]:
    """Derive T0–T4 task windows from XDF + stimuli events.

    This is a simplified inline version — imports ``compute_task_windows``
    from ``xdf_sync_pipeline`` to avoid duplication.
    """
    try:
        # Import task-window logic from sibling module
        xdf_mod_path = Path(__file__).resolve().parent / "xdf_sync_pipeline.py"
        if not xdf_mod_path.exists():
            logger.error("  xdf_sync_pipeline.py not found")
            return [], None

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "xdf_sync_pipeline", xdf_mod_path,
        )
        xdf_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xdf_mod)

        # Find stimuli events
        stimuli_dir = xdf_mod._find_stimuli_dir(data_root, session_info)
        if not stimuli_dir:
            logger.warning("  No stimuli directory found")
            return [], None

        experiment_events_paths = xdf_mod._find_all_stimuli_experiment_events(
            stimuli_dir, session_info,
        )
        if not experiment_events_paths:
            logger.warning("  No experiment events TSVs found")
            return [], None

        # Compute windows (pick best per task across TSVs)
        best_windows: dict[str, dict[str, Any]] = {}
        best_offset: float | None = None
        longest_set: list[dict[str, Any]] = []
        for ep in experiment_events_paths:
            ep_rows = xdf_mod._read_tsv(ep)
            ep_windows, ep_offset = xdf_mod.compute_task_windows(ep_rows)
            for w in ep_windows:
                prev = best_windows.get(w["task"])
                if prev is None or w["duration_s"] > prev["duration_s"]:
                    best_windows[w["task"]] = w
            if len(ep_windows) > len(longest_set):
                best_offset = ep_offset
                longest_set = ep_windows

        windows = [
            best_windows[t]
            for t in xdf_mod.TASK_ORDER
            if t in best_windows
        ]
        return windows, best_offset

    except Exception as exc:
        logger.error("  Task window derivation failed: %s", exc)
        return [], None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Split & synchronize multicam MKV videos into per-task MP4 clips "
            "aligned to the XDF/LSL clock."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory containing high_level_data_inventory.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed video clips",
    )
    p.add_argument(
        "--sync-root",
        type=Path,
        default=None,
        help=(
            "Path to existing xdf_sync_pipeline output (contains sub-01/ses-*/annot/). "
            "When provided, task windows and wall_minus_lsl offset are loaded "
            "from the pre-computed sync data instead of re-deriving from XDF. "
            "Recommended for consistency with other modalities."
        ),
    )
    p.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Process only these session IDs",
    )
    p.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Process only these group IDs (e.g. grp-07 grp-12)",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=18,
        help=(
            "H.264 CRF quality (0=lossless, 18=visually lossless, 23=default). "
            "Default: 18"
        ),
    )
    p.add_argument(
        "--preset",
        type=str,
        default="slow",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="H.264 encoding preset (default: slow)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without writing files",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    sync_root = args.sync_root.resolve() if args.sync_root else None

    logger.info("Data root:  %s", data_root)
    logger.info("Output dir: %s", output_dir)
    if sync_root:
        logger.info("Sync root:  %s", sync_root)
    logger.info("CRF: %d, Preset: %s", args.crf, args.preset)

    # Load inventory
    inventory = _load_inventory(data_root)
    sessions = inventory.get("sessions", [])
    logger.info("Inventory: %d sessions", len(sessions))

    # Filter
    if args.sessions:
        sessions = [s for s in sessions if s["session"] in args.sessions]
    if args.groups:
        sessions = [s for s in sessions if s.get("group_id") in args.groups]

    # Only sessions with AV video
    processable = []
    for s in sessions:
        av_dir = _find_av_session_dir(data_root, s)
        if av_dir:
            capture = _find_av_capture_dir(av_dir)
            if capture and (capture / "video").exists():
                mkv_count = len(list((capture / "video").glob("*.mkv")))
                if mkv_count > 0:
                    processable.append(s)
                    continue
        logger.debug("Skipping %s — no AV video", s["session"])

    logger.info("Sessions with AV video: %d", len(processable))

    if not processable:
        logger.warning("No sessions with AV video found. Nothing to process.")
        return 0

    if args.dry_run:
        logger.info("--- DRY RUN ---")
        for s in processable:
            av_dir = _find_av_session_dir(data_root, s)
            capture = _find_av_capture_dir(av_dir) if av_dir else None
            mkv_count = len(list((capture / "video").glob("*.mkv"))) if capture else 0
            logger.info(
                "  %s (group=%s): %d MKV files",
                s["session"], s.get("group_id", "?"), mkv_count,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for s in processable:
        try:
            r = process_session(
                s, data_root, output_dir,
                dry_run=args.dry_run,
                sync_root=sync_root,
                crf=args.crf,
                preset=args.preset,
            )
            results.append(r)
        except Exception as exc:
            logger.error("FAILED: %s — %s", s["session"], exc)
            results.append({
                "session": s["session"],
                "success": False,
                "error": str(exc),
            })

    # Pipeline summary
    summary = {
        "pipeline": "video_sync_pipeline",
        "version": "1.0",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "sync_root": str(sync_root) if sync_root else None,
        "crf": args.crf,
        "preset": args.preset,
        "total_sessions": len(processable),
        "succeeded": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }
    summary_path = output_dir / "video_sync_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Pipeline complete: %d/%d succeeded. Summary: %s",
        summary["succeeded"], summary["total_sessions"], summary_path,
    )

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
