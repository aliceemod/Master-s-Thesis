"""Extract turn-taking and interaction features from annotated transcript TSV files.

Input transcript files must match the pattern:
    transcript_grp-{grp}_T{N}_{date}*.tsv

These are the manually-annotated transcripts produced for groups 07, 10, 14, 15 and
stored in the ``tools/`` directory (or any directory passed via ``--transcript-dir``).

Expected columns (required):
    onset, duration, speaker, type

Optional enrichment columns (used when present):
    subtype, gap_prev_ms, trp_silence_ms, interruption_gap_ms, bclatency_ms, overlap_ms

Row type vocabulary:
    SPK  — speech segment (attributed to one participant P1–P4 or MODERATOR)
    BCK  — backchannel (attributed to one participant)
    OVL  — overlap annotation (speaker == GROUP; subtype encodes overlap kind)
    SIL  — silence (speaker == GROUP)
    LAU  — laughter (speaker == GROUP)
    FP   — filled pause (attributed to one participant)
    BRE  — breath (attributed to one participant)

Outputs written to ``--out-dir`` (default: ``features/``):
    transcript_participant_task.tsv       — one row per group × task × participant
    transcript_group_task.tsv             — one row per group × task
    transcript_feature_definitions.tsv    — human-readable feature dictionary

Usage:
    python tools/features/extract_transcript_features.py \\
        --transcript-dir tools \\
        --out-dir features

    # Restrict to specific groups or tasks:
    python tools/features/extract_transcript_features.py \\
        --transcript-dir tools \\
        --groups grp-07 grp-10 --tasks T1 T2

    # Verbose logging:
    python tools/features/extract_transcript_features.py \\
        --transcript-dir tools --verbose
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("extract_transcript_features")

# ── Constants ──────────────────────────────────────────────────────────────────

PARTICIPANTS = ("P1", "P2", "P3", "P4")

TYPE_SPK = "SPK"
TYPE_BCK = "BCK"
TYPE_OVL = "OVL"
TYPE_SIL = "SIL"
TYPE_LAU = "LAU"
TYPE_FP = "FP"
TYPE_BRE = "BRE"

OVL_COMPETITIVE: frozenset[str] = frozenset({"competitive", "floor_fight"})
OVL_SIMULTANEOUS: frozenset[str] = frozenset({"simultaneous"})
OVL_BACKCHANNEL_OVL: frozenset[str] = frozenset({"backchannel_ovl"})
OVL_COLLABORATIVE: frozenset[str] = frozenset({"collaborative"})
OVL_NEEDS_REVIEW: frozenset[str] = frozenset({"needs_review"})
OVL_COOPERATIVE: frozenset[str] = frozenset({"collaborative", "smooth", "simultaneous", "backchannel_ovl"})

# Filename pattern: transcript_grp[-]?{grp}_T{N}_{date}*.tsv (dash optional; some
# reviewed exports for grp-08/09/11/12/13 omit the dash, e.g. transcript_grp8_T1_...)
_FNAME_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)


def _normalize_group_id(raw: str) -> str:
    """Normalize e.g. 'grp8'/'grp-08' to 'grp-08'."""
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"

# Numeric columns to coerce on load
_NUMERIC_COLS = (
    "onset", "duration", "gap_prev_ms", "trp_silence_ms",
    "interruption_gap_ms", "bclatency_ms", "overlap_ms",
    "energy", "energy_ratio",
)


# ── File discovery ─────────────────────────────────────────────────────────────


def _discover_files(
    transcript_dir: Path,
    groups: list[str] | None,
    tasks: list[str] | None,
) -> list[tuple[Path, str, str]]:
    """Return (path, group_id, task_id) for each matching transcript TSV.

    Filenames must match ``transcript_grp-{grp}_T{N}_*.tsv``.
    """
    matches: list[tuple[Path, str, str]] = []
    for tsv in sorted(transcript_dir.glob("transcript_grp*.tsv")):
        if tsv.name.endswith(".bak.tsv"):
            continue
        m = _FNAME_RE.search(tsv.name)
        if m is None:
            LOG.debug("Skipping (no match): %s", tsv.name)
            continue
        grp, task = _normalize_group_id(m.group(1)), m.group(2)
        if groups and grp not in groups:
            continue
        if tasks and task not in tasks:
            continue
        matches.append((tsv, grp, task))
    return matches


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_tsv(path: Path) -> pd.DataFrame | None:
    """Load a transcript TSV and coerce numeric columns.  Returns None on error."""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8")
    except Exception as exc:
        LOG.warning("Could not read %s: %s", path, exc)
        return None
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Speaker/row helpers ────────────────────────────────────────────────────────


def _p_rows(df: pd.DataFrame, pid: str, row_type: str) -> pd.DataFrame:
    """Rows where speaker starts with *pid* and type == *row_type*."""
    mask = df["speaker"].str.startswith(pid, na=False) & (df["type"] == row_type)
    return df[mask]


def _group_rows(df: pd.DataFrame, row_type: str) -> pd.DataFrame:
    """Rows attributed to GROUP with the given type."""
    mask = (df["speaker"] == "GROUP") & (df["type"] == row_type)
    return df[mask]


def _task_duration(df: pd.DataFrame) -> float:
    """Task duration = span from earliest onset to latest offset across all rows."""
    if df.empty:
        return 0.0
    max_end = (df["onset"].fillna(0) + df["duration"].fillna(0)).max()
    min_onset = df["onset"].dropna().min()
    return float(max_end - min_onset)


# ── Statistical helpers ────────────────────────────────────────────────────────


def _gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative reals (0 = perfect equality)."""
    total = sum(values)
    if not values or total == 0:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    cum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(arr))
    return cum / (n * total)


def _transition_entropy(df: pd.DataFrame, participants: list[str]) -> float:
    """Shannon entropy (bits) of the inter-speaker turn-transition matrix.

    Only SPK rows for P1–P4 are used; consecutive same-speaker rows are treated
    as a single turn (self-transitions not counted).
    """
    spk = df[df["type"] == TYPE_SPK].copy()
    spk = spk[spk["speaker"].isin(participants)].sort_values("onset")
    speakers = spk["speaker"].tolist()
    if len(speakers) < 2:
        return 0.0
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for i in range(1, len(speakers)):
        if speakers[i] != speakers[i - 1]:
            counts[(speakers[i - 1], speakers[i])] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


# ── Participant-level features ─────────────────────────────────────────────────


def _extract_participant_features(
    df: pd.DataFrame,
    pid: str,
    task_duration_s: float,
) -> dict[str, object]:
    """Compute all participant-level features for one (group, task, participant).

    # Privacy: participant IDs are always P1–P4; no real names are written.
    """
    spk = _p_rows(df, pid, TYPE_SPK)
    bck = _p_rows(df, pid, TYPE_BCK)
    fp = _p_rows(df, pid, TYPE_FP)
    bre = _p_rows(df, pid, TYPE_BRE)

    duration_min = task_duration_s / 60.0 if task_duration_s > 0 else 1.0

    # ── Speaking time ──────────────────────────────────────────────────────────
    speaking_time_s = float(spk["duration"].sum()) if not spk.empty else 0.0
    speaking_share = (
        round(speaking_time_s / task_duration_s, 4) if task_duration_s > 0 else None
    )

    # ── Turn structure ─────────────────────────────────────────────────────────
    turn_count = len(spk)
    mean_turn_dur = float(spk["duration"].mean()) if not spk.empty else None
    median_turn_dur = float(spk["duration"].median()) if not spk.empty else None
    std_turn_dur = float(spk["duration"].std()) if len(spk) > 1 else None

    # ── Response gaps (gap_prev_ms > 0 on SPK = this speaker follows a pause) ──
    gaps: pd.Series = (
        spk["gap_prev_ms"].dropna() if "gap_prev_ms" in spk.columns else pd.Series(dtype=float)
    )
    pos_gaps = gaps[gaps > 0]
    mean_response_gap_ms = float(pos_gaps.mean()) if not pos_gaps.empty else None

    # ── TRP silence (trp_silence_ms > 0 = paused at transition-relevance place) ─
    trps: pd.Series = (
        spk["trp_silence_ms"].dropna()
        if "trp_silence_ms" in spk.columns
        else pd.Series(dtype=float)
    )
    pos_trps = trps[trps > 0]
    mean_trp_silence_ms = float(pos_trps.mean()) if not pos_trps.empty else None

    # ── Overlap burden (overlap_ms on SPK rows) ────────────────────────────────
    if "overlap_ms" in spk.columns:
        overlap_time_s = float(spk["overlap_ms"].fillna(0).sum()) / 1000.0
        overlap_fraction = (
            round(overlap_time_s / speaking_time_s, 4) if speaking_time_s > 0 else None
        )
    else:
        overlap_time_s = None
        overlap_fraction = None

    # ── Interruptions (interruption_gap_ms < 0 → this speaker was interrupted) ─
    if "interruption_gap_ms" in spk.columns:
        intr_mask = spk["interruption_gap_ms"].fillna(0) < 0
        interruption_count: int | None = int(intr_mask.sum())
        interruption_rate_per_min: float | None = round(
            interruption_count / duration_min, 3
        )
    else:
        interruption_count = None
        interruption_rate_per_min = None

    # ── Backchannels given ─────────────────────────────────────────────────────
    bck_count = len(bck)
    bck_rate_per_min = round(bck_count / duration_min, 3)
    bck_lat: pd.Series = (
        bck["bclatency_ms"].dropna()
        if "bclatency_ms" in bck.columns and not bck.empty
        else pd.Series(dtype=float)
    )
    bck_latency_mean_ms = float(bck_lat.mean()) if not bck_lat.empty else None

    # ── Disfluency ─────────────────────────────────────────────────────────────
    fp_count = len(fp)
    fp_rate_per_min = round(fp_count / duration_min, 3)
    bre_count = len(bre)

    return {
        "speaking_time_s": round(speaking_time_s, 3),
        "speaking_share": speaking_share,
        "turn_count": turn_count,
        "mean_turn_duration_s": round(mean_turn_dur, 3) if mean_turn_dur is not None else None,
        "median_turn_duration_s": (
            round(median_turn_dur, 3) if median_turn_dur is not None else None
        ),
        "std_turn_duration_s": round(std_turn_dur, 3) if std_turn_dur is not None else None,
        "mean_response_gap_ms": (
            round(mean_response_gap_ms, 1) if mean_response_gap_ms is not None else None
        ),
        "mean_trp_silence_ms": (
            round(mean_trp_silence_ms, 1) if mean_trp_silence_ms is not None else None
        ),
        "overlap_time_s": round(overlap_time_s, 3) if overlap_time_s is not None else None,
        "overlap_fraction": overlap_fraction,
        "interruption_count": interruption_count,
        "interruption_rate_per_min": interruption_rate_per_min,
        "backchannel_given_count": bck_count,
        "backchannel_given_rate_per_min": bck_rate_per_min,
        "backchannel_latency_mean_ms": (
            round(bck_latency_mean_ms, 1) if bck_latency_mean_ms is not None else None
        ),
        "filled_pause_count": fp_count,
        "filled_pause_rate_per_min": fp_rate_per_min,
        "breath_count": bre_count,
    }


# ── Group-level features ───────────────────────────────────────────────────────


def _extract_group_features(
    df: pd.DataFrame,
    participants: list[str],
    task_duration_s: float,
) -> dict[str, object]:
    """Compute group-level interaction features for one (group, task)."""
    duration_min = task_duration_s / 60.0 if task_duration_s > 0 else 1.0

    # ── Silence ────────────────────────────────────────────────────────────────
    sil = _group_rows(df, TYPE_SIL)
    sil_time_s = float(sil["duration"].sum()) if not sil.empty else 0.0
    sil_fraction = (
        round(sil_time_s / task_duration_s, 4) if task_duration_s > 0 else None
    )

    # ── Laughter ───────────────────────────────────────────────────────────────
    lau = _group_rows(df, TYPE_LAU)
    laughter_count = len(lau)
    laughter_time_s = float(lau["duration"].sum()) if not lau.empty else 0.0

    # ── Overlap events (OVL rows attributed to GROUP) ──────────────────────────
    ovl = _group_rows(df, TYPE_OVL)
    ovl_total = len(ovl)
    if "subtype" in ovl.columns and not ovl.empty:
        sub = ovl["subtype"].fillna("")
        ovl_competitive: int | None = int(sub.isin(OVL_COMPETITIVE).sum())
        ovl_simultaneous: int | None = int(sub.isin(OVL_SIMULTANEOUS).sum())
        ovl_backchannel: int | None = int(sub.isin(OVL_BACKCHANNEL_OVL).sum())
        ovl_collaborative: int | None = int(sub.isin(OVL_COLLABORATIVE).sum())
        ovl_needs_review: int | None = int(sub.isin(OVL_NEEDS_REVIEW).sum())
        ovl_cooperative: int | None = int(sub.isin(OVL_COOPERATIVE).sum())
        ovl_conflict_ratio: float | None = (
            round(float(ovl_competitive / (ovl_competitive + ovl_cooperative)), 4)
            if (ovl_competitive + ovl_cooperative) > 0
            else None
        )
    else:
        ovl_competitive = ovl_simultaneous = ovl_backchannel = None
        ovl_collaborative = ovl_needs_review = ovl_cooperative = ovl_conflict_ratio = None

    # ── Transition entropy ─────────────────────────────────────────────────────
    trans_entropy = _transition_entropy(df, participants)

    # ── Gini on speaking time ──────────────────────────────────────────────────
    speaking_times = [
        float(_p_rows(df, pid, TYPE_SPK)["duration"].sum())
        if not _p_rows(df, pid, TYPE_SPK).empty
        else 0.0
        for pid in participants
    ]
    gini_speaking = _gini(speaking_times)

    # ── Engagement density (all non-SIL events per minute) ────────────────────
    non_sil = df[df["type"] != TYPE_SIL]
    engagement_density = round(len(non_sil) / duration_min, 3)

    # ── Total turns across all participants ────────────────────────────────────
    total_turns = sum(len(_p_rows(df, pid, TYPE_SPK)) for pid in participants)
    n_active_speakers = sum(
        1 for pid in participants if len(_p_rows(df, pid, TYPE_SPK)) > 0
    )

    return {
        "task_duration_s": round(task_duration_s, 3),
        "n_active_speakers": n_active_speakers,
        "total_turns": total_turns,
        "sil_time_s": round(sil_time_s, 3),
        "sil_fraction": sil_fraction,
        "sil_event_count": len(sil),
        "laughter_count": laughter_count,
        "laughter_time_s": round(laughter_time_s, 3),
        "overlap_event_count": ovl_total,
        "competitive_overlap_count": ovl_competitive,
        "cooperative_overlap_count": ovl_cooperative,
        "collaborative_overlap_count": ovl_collaborative,
        "needs_review_overlap_count": ovl_needs_review,
        "conflict_overlap_ratio": ovl_conflict_ratio,
        "simultaneous_overlap_count": ovl_simultaneous,
        "backchannel_ovl_count": ovl_backchannel,
        "transition_entropy_bits": round(trans_entropy, 4),
        "gini_speaking_time": round(gini_speaking, 4),
        "engagement_density_per_min": engagement_density,
    }


# ── Per-file processing ────────────────────────────────────────────────────────


def process_file(
    path: Path,
    group_id: str,
    task_id: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Process one transcript TSV → participant rows and one group row.

    Returns (participant_rows, group_row).  Empty lists/dicts on failure.
    """
    df = _load_tsv(path)
    if df is None:
        return [], {}

    required = {"onset", "duration", "speaker", "type"}
    missing = required - set(df.columns)
    if missing:
        LOG.warning("Missing columns %s in %s — skipping", missing, path.name)
        return [], {}

    participants = [
        pid for pid in PARTICIPANTS
        if df["speaker"].str.startswith(pid, na=False).any()
    ]
    if not participants:
        LOG.warning("No P1–P4 speakers found in %s — skipping", path.name)
        return [], {}

    duration_s = _task_duration(df)
    LOG.debug("  %s: %d participants, %.1f s", path.name, len(participants), duration_s)

    group_row: dict[str, object] = {"group_id": group_id, "task_id": task_id}
    group_row.update(_extract_group_features(df, participants, duration_s))

    p_rows: list[dict[str, object]] = []
    for pid in participants:
        row: dict[str, object] = {
            "group_id": group_id,
            "task_id": task_id,
            "participant_id": pid,
        }
        row.update(_extract_participant_features(df, pid, duration_s))
        p_rows.append(row)

    return p_rows, group_row


# ── Feature definitions ────────────────────────────────────────────────────────


_PARTICIPANT_DEFS: list[tuple[str, str, str, str]] = [
    # (feature_name, level, units, description)
    ("speaking_time_s", "participant", "seconds",
     "Total duration of SPK segments attributed to this participant."),
    ("speaking_share", "participant", "proportion [0,1]",
     "Fraction of task duration this participant was speaking (SPK time / task duration)."),
    ("turn_count", "participant", "count",
     "Number of SPK annotation segments for this participant."),
    ("mean_turn_duration_s", "participant", "seconds",
     "Mean duration of this participant's SPK segments."),
    ("median_turn_duration_s", "participant", "seconds",
     "Median duration of this participant's SPK segments."),
    ("std_turn_duration_s", "participant", "seconds",
     "Standard deviation of this participant's SPK segment durations."),
    ("mean_response_gap_ms", "participant", "milliseconds",
     "Mean gap_prev_ms for SPK rows where gap > 0 ms; "
     "reflects how long this participant waits before taking a turn."),
    ("mean_trp_silence_ms", "participant", "milliseconds",
     "Mean trp_silence_ms for SPK rows where TRP silence > 0; "
     "silence held at a transition-relevance place before speaking."),
    ("overlap_time_s", "participant", "seconds",
     "Total time (from overlap_ms column) this participant's speech overlapped "
     "with someone else."),
    ("overlap_fraction", "participant", "proportion [0,1]",
     "Ratio of overlap_time_s to speaking_time_s."),
    ("interruption_count", "participant", "count",
     "Number of SPK rows with interruption_gap_ms < 0 "
     "(this participant was interrupted or started while someone else spoke)."),
    ("interruption_rate_per_min", "participant", "count/minute",
     "interruption_count divided by task duration in minutes."),
    ("backchannel_given_count", "participant", "count",
     "Number of BCK rows produced by this participant "
     "(e.g. mm-hmm, yeah, uh-huh while another speaks)."),
    ("backchannel_given_rate_per_min", "participant", "count/minute",
     "backchannel_given_count divided by task duration in minutes."),
    ("backchannel_latency_mean_ms", "participant", "milliseconds",
     "Mean bclatency_ms for this participant's BCK rows; "
     "how quickly they react with a backchannel after the main speaker's onset."),
    ("filled_pause_count", "participant", "count",
     "Number of FP (filled pause, e.g. um, uh) rows for this participant."),
    ("filled_pause_rate_per_min", "participant", "count/minute",
     "filled_pause_count divided by task duration in minutes."),
    ("breath_count", "participant", "count",
     "Number of BRE (audible breath) rows for this participant."),
]

_GROUP_DEFS: list[tuple[str, str, str, str]] = [
    ("task_duration_s", "group", "seconds",
     "Estimated task duration (max offset − min onset across all rows)."),
    ("n_active_speakers", "group", "count",
     "Number of P1–P4 participants who produced at least one SPK segment."),
    ("total_turns", "group", "count",
     "Sum of SPK segment counts across all active participants."),
    ("sil_time_s", "group", "seconds",
     "Total duration of GROUP SIL rows (annotated group silences)."),
    ("sil_fraction", "group", "proportion [0,1]",
     "sil_time_s divided by task_duration_s."),
    ("sil_event_count", "group", "count",
     "Number of GROUP SIL annotation rows."),
    ("laughter_count", "group", "count",
     "Number of GROUP LAU (laughter) events."),
    ("laughter_time_s", "group", "seconds",
     "Total duration of GROUP LAU events."),
    ("overlap_event_count", "group", "count",
     "Number of GROUP OVL (overlap) annotation rows."),
    ("competitive_overlap_count", "group", "count",
     "OVL rows with subtype 'competitive' or 'floor_fight' "
     "(speakers competing for the floor)."),
    ("cooperative_overlap_count", "group", "count",
     "OVL rows with cooperative subtypes: collaborative, smooth, simultaneous, "
     "or backchannel_ovl."),
    ("collaborative_overlap_count", "group", "count",
     "OVL rows with subtype 'collaborative'."),
    ("needs_review_overlap_count", "group", "count",
     "OVL rows with subtype 'needs_review' (ambiguous intent, requires manual review)."),
    ("conflict_overlap_ratio", "group", "proportion [0,1]",
     "competitive_overlap_count / (competitive_overlap_count + cooperative_overlap_count)."),
    ("simultaneous_overlap_count", "group", "count",
     "OVL rows with subtype 'simultaneous' (parallel speech without floor contest)."),
    ("backchannel_ovl_count", "group", "count",
     "OVL rows with subtype 'backchannel_ovl' "
     "(overlap caused by a backchannel during main speech)."),
    ("transition_entropy_bits", "group", "bits",
     "Shannon entropy of the inter-speaker turn-transition matrix. "
     "Higher values indicate more evenly distributed floor exchanges."),
    ("gini_speaking_time", "group", "Gini [0,1]",
     "Gini coefficient of speaking_time_s across active participants. "
     "0 = perfect equality; 1 = one participant speaks all the time."),
    ("engagement_density_per_min", "group", "count/minute",
     "Total non-SIL annotation rows divided by task duration in minutes; "
     "proxy for overall interaction density."),
]


def _write_definitions(out_dir: Path) -> None:
    """Write transcript_feature_definitions.tsv to *out_dir*."""
    rows = []
    for name, level, units, desc in _PARTICIPANT_DEFS:
        rows.append({
            "feature_name": name,
            "level": level,
            "units": units,
            "description": desc,
        })
    for name, level, units, desc in _GROUP_DEFS:
        rows.append({
            "feature_name": name,
            "level": level,
            "units": units,
            "description": desc,
        })
    df = pd.DataFrame(rows, columns=["feature_name", "level", "units", "description"])
    out = out_dir / "transcript_feature_definitions.tsv"
    df.to_csv(out, sep="\t", index=False)
    LOG.info("Written: %s", out)


# ── Main run ───────────────────────────────────────────────────────────────────


def run(
    transcript_dir: Path,
    out_dir: Path,
    groups: list[str] | None,
    tasks: list[str] | None,
) -> None:
    """Discover all matching transcript TSVs, extract features, write outputs."""
    if not transcript_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    files = _discover_files(transcript_dir, groups, tasks)
    if not files:
        LOG.error(
            "No matching transcript_grp-*.tsv files found in %s "
            "(groups=%s, tasks=%s)",
            transcript_dir, groups, tasks,
        )
        return

    LOG.info("Discovered %d transcript file(s)", len(files))

    all_p_rows: list[dict[str, object]] = []
    all_g_rows: list[dict[str, object]] = []

    for path, grp, task in files:
        LOG.info("Processing %s  [%s %s]", path.name, grp, task)
        p_rows, g_row = process_file(path, grp, task)
        all_p_rows.extend(p_rows)
        if g_row:
            all_g_rows.append(g_row)

    if not all_p_rows:
        LOG.error("No participant rows were produced — check input files.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    p_df = pd.DataFrame(all_p_rows)
    g_df = pd.DataFrame(all_g_rows)

    p_out = out_dir / "transcript_participant_task.tsv"
    g_out = out_dir / "transcript_group_task.tsv"

    p_df.to_csv(p_out, sep="\t", index=False)
    g_df.to_csv(g_out, sep="\t", index=False)

    LOG.info("Written: %s  (%d rows)", p_out, len(p_df))
    LOG.info("Written: %s  (%d rows)", g_out, len(g_df))

    _write_definitions(out_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("tools"),
        help="Directory containing transcript_grp-*.tsv files (default: tools/)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("features"),
        help="Output directory for feature TSVs (default: features/)",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        help="Restrict to specific groups, e.g. --groups grp-07 grp-10",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help="Restrict to specific tasks, e.g. --tasks T1 T2 T3",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    run(
        transcript_dir=args.transcript_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        groups=args.groups,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
