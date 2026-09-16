"""Extract comprehensive task and phase timing from BIDS events files.

Reads per-task events.tsv and task_run_windows.tsv to produce a complete
timing reference for all sessions, including:
- Task-level boundaries (wall clock, LSL, duration)
- Phase-level boundaries within each task
- Discussion-only durations (excluding instruction/reading/calibration)
- T0 free_talk baseline window timing
- Data completeness and quality flags

Discussion window definitions:
    T0: free_talk → finish          (baseline casual conversation)
    T1: discussion_selection → finish  (after evidence card silent reading)
    T2: role_card → finish           (negotiation; actual speech onset may lag)
    T3: show_ideas_discussion → finish (after silent idea generation)
    T4: discussion → finish          (after contribution form reveal)

Usage:
    python tools/features/extract_session_timing.py \
        --beh-root  C:/Users/.../affectai_beh/bids_release_no_video \
        --annot-root C:/Users/.../affectai_annot/bids_release_no_video \
        --out-dir   analysis/results/session_timing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("extract_session_timing")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]

# Phase that marks the start of the discussion-only window per task
DISCUSSION_START_PHASE: dict[str, str] = {
    "T0": "free_talk",
    "T1": "discussion_selection",
    "T2": "role_card",
    "T3": "show_ideas_discussion",
    "T4": "discussion",
}

# All known moderator phases in expected order per task
EXPECTED_PHASES: dict[str, list[str]] = {
    "T0": ["welcome", "study_introduction", "vad_introduction",
            "postblock_introduction", "free_talk", "finish"],
    "T1": ["tobii_calibration", "intro", "evidence_card",
            "discussion_selection", "candidate_selection", "finish"],
    "T2": ["tobii_calibration", "shared_brief", "role_card",
            "settlement_form", "finish"],
    "T3": ["tobii_calibration", "instructions", "idea_generation",
            "show_ideas_discussion", "group_selection", "finish"],
    "T4": ["tobii_calibration", "shared_brief", "contribution_form",
            "discussion", "finish"],
}


def _parse_phase(value_json: str) -> str | None:
    """Extract 'phase' from the JSON value column."""
    try:
        d = json.loads(value_json)
        return d.get("phase")
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_session_date(session_id: str) -> str | None:
    """Extract YYYY-MM-DD date from session_id like ses-20260312_grp-07_run01."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})", session_id)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _parse_group(session_id: str) -> str | None:
    m = re.search(r"grp-(\d+)", session_id)
    return f"grp-{m.group(1)}" if m else None


def _wall_to_datetime(wall_clock: float) -> str:
    """Convert Unix wall clock to ISO datetime string."""
    try:
        dt = datetime.fromtimestamp(wall_clock, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OSError):
        return ""


def read_events(path: Path) -> list[dict]:
    """Read a BIDS events.tsv, return list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def read_task_run_windows(path: Path) -> dict[str, dict]:
    """Read task_run_windows.tsv, return dict keyed by task."""
    windows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            windows[row["task"]] = row
    return windows


def extract_moderator_phases(events: list[dict]) -> list[dict]:
    """Extract moderator push_content events with phase info, sorted by onset."""
    phases = []
    for ev in events:
        desc = ev.get("description", "")
        if "moderator" not in desc:
            continue
        trial_type = ev.get("trial_type", "")
        if trial_type != "push_content":
            continue
        phase = _parse_phase(ev.get("value", ""))
        if phase:
            phases.append({
                "phase": phase,
                "onset_s": float(ev["onset"]),
            })
    return sorted(phases, key=lambda x: x["onset_s"])


def compute_phase_durations(phases: list[dict], task_end_s: float | None = None) -> list[dict]:
    """Add duration to each phase (time until next phase or task end)."""
    result = []
    for i, p in enumerate(phases):
        if i + 1 < len(phases):
            dur = phases[i + 1]["onset_s"] - p["onset_s"]
        elif task_end_s is not None:
            dur = task_end_s - p["onset_s"]
        else:
            dur = None
        result.append({**p, "duration_s": dur})
    return result


def extract_recording_lsl_phases(path: Path, phase_name: str) -> float | None:
    """Extract a phase onset from recording-lsl_events.tsv (fallback source)."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if phase_name in line and ("Moderator" in line or "Experiment" in line):
                parts = line.split("\t")
                try:
                    return float(parts[0])
                except (ValueError, IndexError):
                    pass
    return None


def process_session(
    session_dir: Path,
    beh_dir: Path,
    annot_dir: Path | None,
) -> dict:
    """Process one session, return comprehensive timing dict."""
    session_id = session_dir.name
    group_id = _parse_group(session_id) or ""
    session_date = _parse_session_date(session_id) or ""

    result: dict = {
        "session_id": session_id,
        "group_id": group_id,
        "session_date": session_date,
        "session_datetime_utc": "",
        "tasks": {},
        "quality_flags": [],
    }

    # Load task_run_windows if available
    task_windows: dict[str, dict] = {}
    if annot_dir:
        tw_files = list(annot_dir.glob("*task_run_windows.tsv"))
        if tw_files:
            task_windows = read_task_run_windows(tw_files[0])

    # Get session start datetime from T0 wall clock
    if "T0" in task_windows:
        wall = float(task_windows["T0"]["start_wall_clock"])
        result["session_datetime_utc"] = _wall_to_datetime(wall)

    for task in TASK_ORDER:
        task_info: dict = {
            "task": task,
            "has_events": False,
            "has_task_window": task in task_windows,
            "task_duration_s": None,
            "task_start_wall": None,
            "task_end_wall": None,
            "task_start_lsl": None,
            "task_end_lsl": None,
            "phases": [],
            "discussion_start_phase": DISCUSSION_START_PHASE.get(task, ""),
            "discussion_onset_s": None,
            "discussion_duration_s": None,
            "missing_phases": [],
            "quality_notes": [],
        }

        # Task-level timing from task_run_windows
        if task in task_windows:
            tw = task_windows[task]
            task_info["task_duration_s"] = float(tw["duration_s"])
            task_info["task_start_wall"] = float(tw["start_wall_clock"])
            task_info["task_end_wall"] = float(tw["end_wall_clock"])
            task_info["task_start_lsl"] = float(tw["start_lsl"])
            task_info["task_end_lsl"] = float(tw["end_lsl"])

        # Per-task events
        events_files = sorted(beh_dir.glob(f"*task-{task}_run-01_events.tsv"))
        if events_files:
            events = read_events(events_files[0])
            if events:
                task_info["has_events"] = True
                phases = extract_moderator_phases(events)

                # Task duration from events (last event onset)
                all_onsets = [float(e["onset"]) for e in events
                              if e.get("onset", "").replace(".", "").replace("-", "").isdigit()]
                events_end_s = max(all_onsets) if all_onsets else None

                task_info["phases"] = compute_phase_durations(phases, events_end_s)

                # Check for expected phases
                found_phases = {p["phase"] for p in phases}
                expected = EXPECTED_PHASES.get(task, [])
                task_info["missing_phases"] = [p for p in expected if p not in found_phases]

                # Discussion timing
                disc_phase = DISCUSSION_START_PHASE.get(task)
                if disc_phase:
                    disc_entries = [p for p in phases if p["phase"] == disc_phase]
                    fin_entries = [p for p in phases if p["phase"] == "finish"]
                    if disc_entries and fin_entries:
                        disc_onset = disc_entries[0]["onset_s"]
                        fin_onset = fin_entries[-1]["onset_s"]
                        task_info["discussion_onset_s"] = disc_onset
                        task_info["discussion_duration_s"] = fin_onset - disc_onset
                    elif disc_entries and not fin_entries:
                        task_info["discussion_onset_s"] = disc_entries[0]["onset_s"]
                        if events_end_s:
                            task_info["discussion_duration_s"] = events_end_s - disc_entries[0]["onset_s"]
                        task_info["quality_notes"].append("finish marker missing; used last event")
            else:
                task_info["quality_notes"].append("events file exists but is empty")

        # Fallback: try recording-lsl for discussion phase
        if not task_info["has_events"] and task_info["has_task_window"]:
            rls_files = sorted(beh_dir.glob(f"*task-{task}_run-01_recording-lsl_events.tsv"))
            if rls_files:
                disc_phase = DISCUSSION_START_PHASE.get(task)
                if disc_phase:
                    lsl_onset = extract_recording_lsl_phases(rls_files[0], disc_phase)
                    if lsl_onset is not None and task_info["task_start_lsl"]:
                        rel_onset = lsl_onset - task_info["task_start_lsl"]
                        task_info["discussion_onset_s"] = rel_onset
                        if task_info["task_duration_s"]:
                            task_info["discussion_duration_s"] = task_info["task_duration_s"] - rel_onset
                        task_info["quality_notes"].append("recovered from recording-lsl")

        # Quality flags
        if not task_info["has_events"] and not task_info["has_task_window"]:
            task_info["quality_notes"].append("no data")
            result["quality_flags"].append(f"{task}: no data")
        elif task_info["has_task_window"] and task_info["task_duration_s"]:
            dur = task_info["task_duration_s"]
            if task == "T0" and dur > 3600:
                task_info["quality_notes"].append(f"duration {dur:.0f}s suspiciously long (overnight issue?)")
                result["quality_flags"].append(f"{task}: overnight issue")

        result["tasks"][task] = task_info

    return result


def write_task_timing_tsv(sessions: list[dict], out_path: Path) -> None:
    """Write one row per session × task with all timing info."""
    cols = [
        "session_id", "group_id", "session_date", "session_datetime_utc",
        "task", "task_duration_s",
        "task_start_wall", "task_end_wall", "task_start_lsl", "task_end_lsl",
        "has_events", "has_task_window",
        "discussion_start_phase", "discussion_onset_s", "discussion_duration_s",
        "missing_phases", "quality_notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for ses in sessions:
            for task in TASK_ORDER:
                if task not in ses["tasks"]:
                    continue
                t = ses["tasks"][task]
                row = {
                    "session_id": ses["session_id"],
                    "group_id": ses["group_id"],
                    "session_date": ses["session_date"],
                    "session_datetime_utc": ses["session_datetime_utc"],
                    "task": task,
                    "task_duration_s": f"{t['task_duration_s']:.1f}" if t["task_duration_s"] else "",
                    "task_start_wall": f"{t['task_start_wall']:.6f}" if t["task_start_wall"] else "",
                    "task_end_wall": f"{t['task_end_wall']:.6f}" if t["task_end_wall"] else "",
                    "task_start_lsl": f"{t['task_start_lsl']:.6f}" if t["task_start_lsl"] else "",
                    "task_end_lsl": f"{t['task_end_lsl']:.6f}" if t["task_end_lsl"] else "",
                    "has_events": str(t["has_events"]),
                    "has_task_window": str(t["has_task_window"]),
                    "discussion_start_phase": t["discussion_start_phase"],
                    "discussion_onset_s": f"{t['discussion_onset_s']:.1f}" if t["discussion_onset_s"] is not None else "",
                    "discussion_duration_s": f"{t['discussion_duration_s']:.1f}" if t["discussion_duration_s"] is not None else "",
                    "missing_phases": ";".join(t["missing_phases"]),
                    "quality_notes": ";".join(t["quality_notes"]),
                }
                w.writerow(row)
    LOG.info("Wrote task timing: %s", out_path)


def write_phase_timing_tsv(sessions: list[dict], out_path: Path) -> None:
    """Write one row per session × task × phase with onset and duration."""
    cols = [
        "session_id", "group_id", "task", "phase", "onset_s", "duration_s",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for ses in sessions:
            for task in TASK_ORDER:
                if task not in ses["tasks"]:
                    continue
                for phase in ses["tasks"][task].get("phases", []):
                    w.writerow({
                        "session_id": ses["session_id"],
                        "group_id": ses["group_id"],
                        "task": task,
                        "phase": phase["phase"],
                        "onset_s": f"{phase['onset_s']:.1f}",
                        "duration_s": f"{phase['duration_s']:.1f}" if phase["duration_s"] is not None else "",
                    })
    LOG.info("Wrote phase timing: %s", out_path)


def write_completeness_tsv(sessions: list[dict], out_path: Path) -> None:
    """Write a completeness summary: one row per session."""
    cols = [
        "session_id", "group_id", "session_date", "session_datetime_utc",
        "n_tasks_with_events", "n_tasks_with_windows",
        "n_discussion_durations", "has_free_talk_baseline",
        "quality_flags",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for ses in sessions:
            tasks = ses["tasks"]
            n_events = sum(1 for t in tasks.values() if t["has_events"])
            n_windows = sum(1 for t in tasks.values() if t["has_task_window"])
            n_disc = sum(1 for t in tasks.values() if t["discussion_duration_s"] is not None)
            has_ft = "T0" in tasks and tasks["T0"]["discussion_duration_s"] is not None
            w.writerow({
                "session_id": ses["session_id"],
                "group_id": ses["group_id"],
                "session_date": ses["session_date"],
                "session_datetime_utc": ses["session_datetime_utc"],
                "n_tasks_with_events": n_events,
                "n_tasks_with_windows": n_windows,
                "n_discussion_durations": n_disc,
                "has_free_talk_baseline": str(has_ft),
                "quality_flags": "; ".join(ses["quality_flags"]) if ses["quality_flags"] else "",
            })
    LOG.info("Wrote completeness summary: %s", out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beh-root", type=Path, required=True,
                   help="Path to bids_release_no_video containing sub-01/ses-*/beh/")
    p.add_argument("--annot-root", type=Path, default=None,
                   help="Path to bids_release_no_video containing sub-01/ses-*/annot/")
    p.add_argument("--out-dir", type=Path, default=Path("analysis/results/session_timing"),
                   help="Output directory for timing TSVs.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sub_dir = args.beh_root / "sub-01"
    if not sub_dir.exists():
        LOG.error("sub-01 directory not found under %s", args.beh_root)
        return 1

    session_dirs = sorted(sub_dir.iterdir())
    LOG.info("Found %d session directories.", len(session_dirs))

    results = []
    for ses_dir in session_dirs:
        if not ses_dir.is_dir():
            continue
        beh_dir = ses_dir / "beh"
        if not beh_dir.exists():
            LOG.warning("No beh/ in %s, skipping.", ses_dir.name)
            continue

        annot_dir = None
        if args.annot_root:
            candidate = args.annot_root / "sub-01" / ses_dir.name / "annot"
            if candidate.exists():
                annot_dir = candidate

        LOG.info("Processing %s ...", ses_dir.name)
        result = process_session(ses_dir, beh_dir, annot_dir)
        results.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_task_timing_tsv(results, args.out_dir / "task_timing.tsv")
    write_phase_timing_tsv(results, args.out_dir / "phase_timing.tsv")
    write_completeness_tsv(results, args.out_dir / "completeness_summary.tsv")

    LOG.info("Done. %d sessions processed.", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
