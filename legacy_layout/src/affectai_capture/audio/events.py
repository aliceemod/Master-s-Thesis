"""Parse events.tsv to extract moderator vs participant phase windows per task."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Phases where the moderator is speaking / giving instructions.
MODERATOR_PHASES = {
    "welcome", "study_introduction", "tobii_calibration", "intro",
    "instructions", "shared_brief", "evidence_card", "role_card",
    "vad_introduction", "postblock_introduction", "contribution_form",
    "finish",
}

# Phases where participants are actively discussing.
PARTICIPANT_PHASES = {
    "free_talk", "discussion", "discussion_selection", "idea_generation",
    "show_ideas_discussion", "candidate_selection", "group_selection",
    "settlement_form",
}


def _get_session_start_wall(segment_windows_path: Path) -> float | None:
    """Get the session start wall clock from the PRE segment window."""
    with segment_windows_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["task"] == "PRE":
                return float(row["start_wall_clock"])
    return None


def parse_phase_windows(
    events_path: Path,
    task_run_windows_path: Path,
    segment_windows_path: Path | None = None,
) -> dict[str, list[dict]]:
    """Parse events.tsv and return phase windows per task.

    Returns dict: task -> list of {start, end, phase, is_moderator}.
    Times are in audio-relative seconds (0 = start of that task's audio).

    events.tsv onset is seconds from session start (wall clock).
    Audio for each task starts at task_start_wall_clock - session_start_wall_clock.
    """
    # Get session start wall clock
    session_start_wall = None
    if segment_windows_path and segment_windows_path.exists():
        session_start_wall = _get_session_start_wall(segment_windows_path)

    # Read task windows (wall clock times)
    task_start_onset: dict[str, float] = {}  # task -> onset in events.tsv time
    task_end_onset: dict[str, float] = {}
    with task_run_windows_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            task = row["task"]
            start_wall = float(row["start_wall_clock"])
            end_wall = float(row["end_wall_clock"])

            if session_start_wall is not None:
                task_start_onset[task] = start_wall - session_start_wall
                task_end_onset[task] = end_wall - session_start_wall
            else:
                # Fallback: use the first moderator event as task start
                task_start_onset[task] = 0.0
                task_end_onset[task] = float(row["duration_s"])

    # Read moderator phase pushes from events.tsv
    phase_events: list[tuple[float, str, str]] = []  # (onset, task, phase)
    with events_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["trial_type"] != "push_content":
                continue
            desc = row.get("description", "")
            if "stream=moderator" not in desc:
                continue
            try:
                value = json.loads(row["value"])
            except (json.JSONDecodeError, KeyError):
                continue
            phase = value.get("phase")
            if not phase:
                continue
            task_match = re.search(r"task=(T\d+)", desc)
            task = task_match.group(1) if task_match else None
            onset = float(row["onset"])
            phase_events.append((onset, task, phase))

    # Build phase windows per task
    result: dict[str, list[dict]] = {}
    for task in sorted(task_start_onset.keys()):
        t_start = task_start_onset[task]
        t_end = task_end_onset[task]

        # Filter phase events for this task
        task_phases = sorted(
            [(t, phase) for (t, tk, phase) in phase_events
             if tk == task or (t_start <= t <= t_end)],
        )

        windows = []
        for i, (t, phase) in enumerate(task_phases):
            # Convert to audio-relative time
            w_start = t - t_start
            if i + 1 < len(task_phases):
                w_end = task_phases[i + 1][0] - t_start
            else:
                w_end = t_end - t_start

            windows.append({
                "start": max(0.0, w_start),
                "end": w_end,
                "phase": phase,
                "is_moderator": phase in MODERATOR_PHASES,
            })

        result[task] = windows
        log.info(
            "Task %s: %d phases (%d moderator, %d participant) over %.0fs",
            task, len(windows),
            sum(1 for w in windows if w["is_moderator"]),
            sum(1 for w in windows if not w["is_moderator"]),
            t_end - t_start,
        )

    return result


def is_moderator_phase(time_s: float, phase_windows: list[dict]) -> bool:
    """Check if a given audio time falls in a moderator phase."""
    for w in phase_windows:
        if w["start"] <= time_s <= w["end"]:
            return w["is_moderator"]
    return False
