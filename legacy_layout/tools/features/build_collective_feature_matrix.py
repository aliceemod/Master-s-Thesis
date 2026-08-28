"""Build a multimodal collective feature matrix for latent group state discovery.

Covers all 10 groups with manual transcript annotations (grp-07..grp-16) and
tasks T1, T2, T3 (T0 = baseline, T4 = excluded).

Pipeline
--------
1. Load physio 30s window features; aggregate P1–P4 per window into group-level
   mean, std, and participant coverage count.
2. Bin transcript events (SPK, BCK, SIL, OVL, LAU, FP) into matching 30s
   windows; compute interaction-density features per bin. `window_index` is
   anchored to the task-split recording start (same anchor as physio/ET), not
   the first transcribed utterance — see `_bin_transcript_events` for the
   2026-08-12 fix and the `transcript_coverage`/jitter-tolerance handling of
   the untranscribed-overhead-vs-confirmed-silence distinction.
3. Load ET 30s window features; aggregate P1–P4 per window.
4. Outer-merge physio + transcript + ET on (group_id, task_id, window_index).
5. Load stimuli-answer self-report; compute mean VAD and post-task questionnaire
   items per group × task (for downstream validation of discovered states).

Outputs written to --out-dir (default: features/):
    collective_window_features.tsv   — one row per group × task × 30s window
    collective_task_selfreport.tsv   — one row per group × task

Usage:
    python tools/features/build_collective_feature_matrix.py

    # Restrict groups or tasks:
    python tools/features/build_collective_feature_matrix.py \\
        --groups grp-07 grp-10 --tasks T1 T2

    # Verbose logging:
    python tools/features/build_collective_feature_matrix.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("build_collective_feature_matrix")

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_GROUPS: list[str] = [
    "grp-07", "grp-08", "grp-09", "grp-10", "grp-11",
    "grp-12", "grp-13", "grp-14", "grp-15", "grp-16",
]
DEFAULT_TASKS: list[str] = ["T1", "T2", "T3"]   # T0 = baseline, T4 = excluded
PARTICIPANTS: tuple[str, ...] = ("P1", "P2", "P3", "P4")
WINDOW_S: float = 30.0

# Tolerance applied around the transcribed dialogue span when deciding transcript
# coverage per window (see _bin_transcript_events) — the transcript boundary is a
# manually-judged (and sometimes truncated-recording-driven) cut, not exact.
TRANSCRIPT_JITTER_S: float = 90.0

# Physio window columns to aggregate (group mean + std across participants)
PHYSIO_COLS: list[str] = [
    "hr_mean_bpm",
    "hrv_rmssd_ms",
    "eda_tonic_mean",
    "eda_phasic_rate_hz",
    "temp_mean",
]

# ET window columns to aggregate (group mean + std across participants)
ET_COLS: list[str] = [
    "pupil_mean",
    "pupil_std",
    "pupil_slope_per_s",
    "blink_rate_per_min",
    "gaze_dispersion",
    "gaze_valid_frac",
    "gaze_velocity_mean",
]

# Lexical window columns (group-aggregated in the source file)
LEXICAL_COLS: list[str] = [
    "lex_word_count",
    "lex_unique_words",
    "lex_ttr",
    "lex_bigram_entropy",
    "lex_trigram_entropy",
    "lex_agreement_count",
    "lex_hedging_count",
    "lex_certainty_count",
    "lex_positive_count",
    "lex_negative_count",
    "lex_question_count",
    "lex_suggestion_count",
    "lex_sentiment_ratio",
    "lex_social_composite",
]

# Post-task questionnaire items to extract per task (from stimuli answers files)
POST_TASK_ITEMS: dict[str, list[str]] = {
    "T1": [
        "engagement", "voice_inclusion", "team_coordination",
        "equality_of_contribution", "info_sharing",
        "mental_demand", "decision_confidence", "perceived_control",
    ],
    "T2": [
        "engagement", "voice_inclusion", "satisfaction", "cooperative",
        "mental_demand", "perceived_control",
        "trust_angle", "trust_front", "trust_next",
    ],
    "T3": [
        "engagement", "voice_inclusion", "team_coordination",
        "satisfaction", "confidence", "psych_safety", "fairness",
        "idea_quality", "idea_diversity", "mental_demand",
    ],
}

# Perceived-dominance item keys present in all three tasks
DOMINANCE_KEYS: list[str] = ["dominance_p1", "dominance_p2", "dominance_p3", "dominance_p4"]

# Regex helpers
_GRP_RE = re.compile(r"(grp-\d+)")
# Dash optional: reviewed exports for grp-08/09/11/12/13 omit the dash (transcript_grp8_T1_...).
_TRANSCRIPT_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)
_STIM_GRP_RE = re.compile(r"ses-\d+_(grp-\d+)_run")


# ── Generic helpers ────────────────────────────────────────────────────────────


def _extract_group(s: str) -> str | None:
    """Extract 'grp-NN' from a session_id or filename string."""
    m = _GRP_RE.search(str(s))
    return m.group(1) if m else None


def _normalize_group_id(raw: str) -> str:
    """Normalize e.g. 'grp8'/'grp-08' to 'grp-08'."""
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"


def _load_tsv(path: Path, label: str) -> pd.DataFrame | None:
    """Load a TSV file; return None and log a warning on failure."""
    if not path.is_file():
        LOG.warning("File not found: %s", path)
        return None
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
        LOG.debug("Loaded %s: %d rows × %d cols", label, len(df), len(df.columns))
        return df
    except Exception as exc:
        LOG.warning("Could not load %s: %s", label, exc)
        return None


# ── Physio aggregation ─────────────────────────────────────────────────────────


def _aggregate_physio(
    df: pd.DataFrame,
    groups: list[str],
    tasks: list[str],
) -> pd.DataFrame:
    """Aggregate per-participant physio windows to group level.

    Output columns per window:
      group_{col}_mean / _std  for each PHYSIO_COL
      n_physio_valid            number of participants with valid hr_mean_bpm
      window_start_s            approximate task-relative window start (seconds)
    """
    df = df.copy()
    df["group_id"] = df["session_id"].apply(_extract_group)
    df = df[df["group_id"].isin(groups) & df["task_id"].isin(tasks)].copy()
    if df.empty:
        LOG.warning("No physio rows matched the selected groups/tasks.")
        return pd.DataFrame()

    available = [c for c in PHYSIO_COLS if c in df.columns]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hr_mean_bpm" in df.columns:
        df["hr_mean_bpm"] = pd.to_numeric(df["hr_mean_bpm"], errors="coerce")

    key = ["group_id", "task_id", "window_index"]
    rows: list[dict] = []

    for keys, grp in df.groupby(key):
        row: dict = {
            "group_id": keys[0],
            "task_id": keys[1],
            "window_index": keys[2],
            "window_start_s": keys[2] * WINDOW_S,
            "n_participants": len(grp),
            "n_physio_valid": int(
                grp["hr_mean_bpm"].notna().sum()
            ) if "hr_mean_bpm" in grp.columns else 0,
        }
        for col in available:
            vals = grp[col].dropna()
            row[f"group_{col}_mean"] = round(float(vals.mean()), 4) if not vals.empty else None
            row[f"group_{col}_std"] = round(float(vals.std()), 4) if len(vals) > 1 else None
        rows.append(row)

    out = pd.DataFrame(rows)
    LOG.info("Physio aggregated: %d group-window rows", len(out))
    return out


# ── ET aggregation ─────────────────────────────────────────────────────────────


def _aggregate_et(
    df: pd.DataFrame,
    groups: list[str],
    tasks: list[str],
) -> pd.DataFrame:
    """Aggregate per-participant ET windows to group level (mean + std)."""
    df = df.copy()
    df["group_id"] = df["session_id"].apply(_extract_group)
    df = df[df["group_id"].isin(groups) & df["task_id"].isin(tasks)].copy()
    if df.empty:
        LOG.warning("No ET rows matched the selected groups/tasks.")
        return pd.DataFrame()

    available = [c for c in ET_COLS if c in df.columns]
    if not available:
        LOG.warning("No recognised ET columns found — skipping ET aggregation.")
        return pd.DataFrame()

    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    key = ["group_id", "task_id", "window_index"]
    rows: list[dict] = []

    for keys, grp in df.groupby(key):
        row: dict = {
            "group_id": keys[0],
            "task_id": keys[1],
            "window_index": keys[2],
        }
        for col in available:
            vals = grp[col].dropna()
            row[f"group_et_{col}_mean"] = round(float(vals.mean()), 4) if not vals.empty else None
            row[f"group_et_{col}_std"] = round(float(vals.std()), 4) if len(vals) > 1 else None
        rows.append(row)

    out = pd.DataFrame(rows)
    LOG.info("ET aggregated: %d group-window rows", len(out))
    return out


# ── Transcript binning ─────────────────────────────────────────────────────────


def _load_all_transcripts(
    transcript_dir: Path,
    groups: list[str],
    tasks: list[str],
) -> pd.DataFrame:
    """Load all matching transcript_grp[-]?*.tsv files into a single DataFrame."""
    frames: list[pd.DataFrame] = []
    for tsv in sorted(transcript_dir.glob("transcript_grp*.tsv")):
        if tsv.name.endswith(".bak.tsv"):
            continue
        m = _TRANSCRIPT_RE.search(tsv.name)
        if m is None:
            continue
        grp, task = _normalize_group_id(m.group(1)), m.group(2)
        if grp not in groups or task not in tasks:
            continue
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)
            for col in ("onset", "duration", "overlap_ms"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["group_id"] = grp
            df["task_id"] = task
            frames.append(df)
            LOG.debug("Loaded transcript: %s", tsv.name)
        except Exception as exc:
            LOG.warning("Could not load %s: %s", tsv.name, exc)

    if not frames:
        LOG.warning("No transcript files found for groups=%s tasks=%s", groups, tasks)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _speaking_entropy(speaker_durations: dict[str, float]) -> float:
    """Shannon entropy (bits) of speaking time distribution within a window."""
    total = sum(speaker_durations.values())
    if total == 0:
        return 0.0
    probs = [t / total for t in speaker_durations.values() if t > 0]
    return -sum(p * math.log2(p) for p in probs)


def _resolve_needs_review_by_duration(overlap_ms: float | None) -> str:
    """Fallback for any legacy needs_review rows not yet re-run through relabel_overlaps.py.

    subtype is always timing-based after relabel_overlaps.py is run, so this
    function should rarely be called in practice.
    """
    if overlap_ms is None or pd.isna(overlap_ms):
        return "smooth"
    overlap_ms = float(overlap_ms)
    if overlap_ms < 250:
        return "backchannel_ovl"
    if overlap_ms < 500:
        return "smooth"
    if overlap_ms < 1000:
        return "competitive"
    return "floor_fight"


def _dedup_ovl(ovl: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-speaker OVL rows sharing the same onset (rounded to 10ms) to one event.

    A single overlap between two speakers is logged as one OVL row per speaker, so raw
    OVL row counts double-count events. Ported from
    `icmi_paper/analysis/timing_based_overlap_features.ipynb` — keeps the longest-duration
    row per onset bucket as the representative event.
    """
    if ovl.empty:
        return ovl
    ovl = ovl.copy()
    ovl["_onset_key"] = ovl["onset"].round(2)
    idx = ovl.groupby("_onset_key")["duration"].idxmax()
    return ovl.loc[idx].drop(columns="_onset_key")


def _zero_window_counts() -> dict:
    """All transcript count/duration columns zeroed — used for confirmed-silent windows."""
    return {
        "tr_spk_count": 0,
        "tr_spk_duration_s": 0.0,
        "tr_silence_duration_s": 0.0,
        "tr_backchannel_count": 0,
        "tr_overlap_count": 0,
        "tr_competitive_overlap": 0,
        "tr_cooperative_overlap": 0,
        "tr_collaborative_overlap": 0,
        "tr_needs_review_overlap": 0,
        "tr_conflict_overlap_ratio": None,
        "tr_laughter_count": 0,
        "tr_filled_pause_count": 0,
        "tr_n_active_speakers": 0,
        "tr_speaking_entropy": 0.0,
        "tr_overlap_time_s": 0.0,
        "tr_ovl_count": 0,
        "tr_ovl_time_s": 0.0,
        "tr_ovl_simultaneous": 0,
        "tr_ovl_backchannel": 0,
        "tr_ovl_smooth": 0,
        "tr_ovl_competitive_timing": 0,
        "tr_ovl_floor_fight": 0,
        "tr_ovl_ctx_collaborative": 0,
        "tr_ovl_ctx_completion": 0,
        "tr_ovl_collaboration_index": None,
        "tr_ovl_completion_index": None,
    }


def _bin_transcript_events(
    events: pd.DataFrame,
    window_s: float = WINDOW_S,
    jitter_s: float = TRANSCRIPT_JITTER_S,
) -> pd.DataFrame:
    """Bin transcript events into 30s windows per (group_id, task_id).

    Window index 0 = first 30s of the task-split recording itself (onset == 0),
    the same anchor used by the physio/pupil/ET task-split files' `window_index`.

    Fixed 2026-08-12: previously rebased window_index to `onset.min()` (the first
    transcribed utterance), which silently offset every window_index relative to
    physio/pupil/ET by however long the pre-conversation overhead (calibration,
    briefing, settling-in) lasted before the first utterance — so the same
    `window_index` could mean a different real-world 30s slice per modality when
    outer-merged in `run()`.

    Because the transcribed span's start/end is itself a manually-judged boundary
    (and some underlying recordings were cut off/truncated), a `jitter_s`
    tolerance (default 90s, ~2-3 window steps) is applied around the first/last
    transcribed event when deciding `transcript_coverage`: windows within
    `[dialogue_onset_s - jitter_s, dialogue_offset_s + jitter_s]` are marked
    "covered" — including zero-event windows within that span, which are then
    real, confirmed-silent windows (all `tr_*` counts zeroed). Windows outside
    that range are omitted entirely, so they read as NaN (unknown/out-of-scope,
    never a silent 0) after the outer-merge in `run()`.

    **Auto-resolution of needs_review overlaps:**
    Any overlap with subtype='needs_review' is auto-classified by duration using
    _resolve_needs_review_by_duration() before counting into competitive/cooperative buckets.
    The raw (pre-resolution) count is preserved in `tr_needs_review_overlap` for transparency.

    **Fixed 2026-08-12 (same day): `tr_collaborative_overlap` subtype/context_label mix-up.**
    `"collaborative"` is a `context_label` value (a lexical-cue judgement, e.g. support words
    on a long overlap), never a `subtype` value (`subtype` is always one of the five pure-timing
    labels per `tools/relabel_overlaps.py`'s `VALID_SUBTYPES`) — checking `_subtype == "collaborative"`
    silently always returned zero rows. Now sourced from `context_label` instead.

    **Restored 2026-08-12: pure-timing-subtype and context-label OVL features** (`tr_ovl_*`),
    ported from `icmi_paper/analysis/timing_based_overlap_features.ipynb` (which computed them
    directly from raw transcripts for the published paper matrix) — with dedup of same-onset
    multi-speaker OVL rows via `_dedup_ovl`. Unlike that notebook, these are computed at the
    already-fixed 30s/task-recording-start window anchor, not the notebook's own `onset.min()`
    rebase (which has the same misalignment bug this script's window_index fix addressed).

    Output columns per window:
      tr_spk_count              number of SPK segments by participants
      tr_spk_duration_s         total speaking duration
      tr_silence_duration_s     total GROUP SIL duration
      tr_backchannel_count      BCK rows by participants
      tr_overlap_count          total OVL rows (not deduped)
      tr_competitive_overlap    OVL with subtype competitive or floor_fight
      tr_cooperative_overlap    OVL with subtype smooth/simultaneous/backchannel_ovl, or
                                 context_label collaborative
      tr_collaborative_overlap  OVL with context_label collaborative
      tr_needs_review_overlap   raw count of subtype=='needs_review' rows before resolution
      tr_laughter_count         GROUP LAU rows
      tr_filled_pause_count     FP rows by participants
      tr_n_active_speakers      participants with any SPK in this window
      tr_speaking_entropy       Shannon entropy of speaking-time distribution
      tr_overlap_time_s         total overlap_ms from SPK rows (seconds)
      tr_ovl_count              deduped OVL event count
      tr_ovl_time_s             deduped OVL event total duration (s)
      tr_ovl_simultaneous/_backchannel/_smooth/_competitive_timing/_floor_fight
                                deduped OVL event counts by pure timing subtype
      tr_ovl_ctx_collaborative / tr_ovl_ctx_completion
                                deduped OVL event counts by context_label
      tr_ovl_collaboration_index / tr_ovl_completion_index
                                context-label count / deduped OVL event count
      dialogue_onset_s          first transcribed event onset for this group/task
      dialogue_offset_s         last transcribed event end for this group/task
      transcript_coverage       always "covered" in this output — absence of a
                                 (group_id, task_id, window_index) row IS the
                                 out-of-scope signal after the outer-merge
    """
    if events.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (grp, task), grp_df in events.groupby(["group_id", "task_id"]):
        grp_df = grp_df.copy()
        dialogue_onset_s = float(grp_df["onset"].min())
        dialogue_offset_s = float((grp_df["onset"].fillna(0) + grp_df["duration"].fillna(0)).max())

        # Anchored to the task-split recording start (onset == 0) — matches the
        # physio/pupil/ET task-split file anchor, not the first utterance.
        grp_df["window_index"] = (grp_df["onset"] / window_s).apply(
            lambda x: int(x) if pd.notna(x) and x >= 0 else -1
        )
        grp_df = grp_df[grp_df["window_index"] >= 0]

        subtype_col = grp_df["subtype"].fillna("") if "subtype" in grp_df.columns else pd.Series("", index=grp_df.index)
        grp_df["_subtype"] = subtype_col.str.strip().str.lower()
        grp_df["_subtype_raw"] = grp_df["_subtype"]
        grp_df["_context"] = (
            grp_df["context_label"].fillna("").str.strip().str.lower()
            if "context_label" in grp_df.columns
            else pd.Series("", index=grp_df.index)
        )

        has_timing_class = "timing_class" in grp_df.columns
        # Resolve any legacy needs_review rows (should not occur after
        # running relabel_overlaps.py, but kept as a safety net).
        needs_review_mask = grp_df["_subtype"] == "needs_review"
        if needs_review_mask.sum() > 0:
            ovl_ms_col_name = "overlap_ms" if "overlap_ms" in grp_df.columns else "duration"
            grp_df.loc[needs_review_mask, "_subtype"] = grp_df.loc[
                needs_review_mask, ovl_ms_col_name
            ].apply(
                lambda v: _resolve_needs_review_by_duration(
                    float(v) * 1000 if ovl_ms_col_name == "duration" else v
                )
            )
            LOG.debug(
                "Resolved %d legacy needs_review overlaps in %s %s", needs_review_mask.sum(), grp, task
            )

        covered_indices: set[int] = set()
        for w_idx, w_df in grp_df.groupby("window_index"):
            covered_indices.add(int(w_idx))
            t = w_df["type"].fillna("")
            spk = w_df[t == "SPK"]
            p_spk = spk[spk["speaker"].isin(PARTICIPANTS)]
            sil = w_df[t == "SIL"]
            bck = w_df[(t == "BCK") & w_df["speaker"].isin(PARTICIPANTS)]
            ovl = w_df[t == "OVL"]
            comp_ovl = ovl[ovl["_subtype"].isin({"competitive", "floor_fight"})]
            collab_ovl = ovl[ovl["_context"] == "collaborative"]
            coop_ovl = ovl[ovl["_subtype"].isin({"smooth", "simultaneous", "backchannel_ovl"}) | (ovl["_context"] == "collaborative")]
            needs_review_ovl = ovl[ovl["_subtype_raw"] == "needs_review"]
            lau = w_df[t == "LAU"]
            fp = w_df[(t == "FP") & w_df["speaker"].isin(PARTICIPANTS)]

            speaker_durs = {
                pid: float(p_spk[p_spk["speaker"].str.startswith(pid, na=False)]["duration"].sum())
                for pid in PARTICIPANTS
            }

            ovl_ms_col = p_spk["overlap_ms"] if "overlap_ms" in p_spk.columns else pd.Series(dtype=float)

            ovl_dedup = _dedup_ovl(ovl)
            n_simult = int((ovl_dedup["_subtype"] == "simultaneous").sum())
            n_bcovl = int((ovl_dedup["_subtype"] == "backchannel_ovl").sum())
            n_smooth = int((ovl_dedup["_subtype"] == "smooth").sum())
            n_compet = int((ovl_dedup["_subtype"] == "competitive").sum())
            n_floor = int((ovl_dedup["_subtype"] == "floor_fight").sum())
            n_ctx_collab = int((ovl_dedup["_context"] == "collaborative").sum())
            n_ctx_completion = int((ovl_dedup["_context"] == "completion").sum())
            ovl_dedup_count = len(ovl_dedup)
            ovl_dedup_time_s = float(ovl_dedup["duration"].fillna(0).sum())
            rows.append({
                "group_id": grp,
                "task_id": task,
                "window_index": int(w_idx),
                "window_start_s": w_idx * window_s,
                "tr_spk_count": len(p_spk),
                "tr_spk_duration_s": round(float(p_spk["duration"].sum()), 3),
                "tr_silence_duration_s": round(float(sil["duration"].sum()), 3) if not sil.empty else 0.0,
                "tr_backchannel_count": len(bck),
                "tr_overlap_count": len(ovl),
                "tr_competitive_overlap": len(comp_ovl),
                "tr_cooperative_overlap": len(coop_ovl),
                "tr_collaborative_overlap": len(collab_ovl),
                "tr_needs_review_overlap": len(needs_review_ovl),
                "tr_conflict_overlap_ratio": round(
                    float(len(comp_ovl) / (len(comp_ovl) + len(coop_ovl))), 4
                ) if (len(comp_ovl) + len(coop_ovl)) > 0 else None,
                "tr_laughter_count": len(lau),
                "tr_filled_pause_count": len(fp),
                "tr_n_active_speakers": sum(1 for d in speaker_durs.values() if d > 0),
                "tr_speaking_entropy": round(_speaking_entropy(speaker_durs), 4),
                "tr_overlap_time_s": round(float(ovl_ms_col.fillna(0).sum()) / 1000.0, 3),
                "tr_ovl_count": ovl_dedup_count,
                "tr_ovl_time_s": round(ovl_dedup_time_s, 3),
                "tr_ovl_simultaneous": n_simult,
                "tr_ovl_backchannel": n_bcovl,
                "tr_ovl_smooth": n_smooth,
                "tr_ovl_competitive_timing": n_compet,
                "tr_ovl_floor_fight": n_floor,
                "tr_ovl_ctx_collaborative": n_ctx_collab,
                "tr_ovl_ctx_completion": n_ctx_completion,
                "tr_ovl_collaboration_index": round(n_ctx_collab / max(ovl_dedup_count, 1), 4),
                "tr_ovl_completion_index": round(n_ctx_completion / max(ovl_dedup_count, 1), 4),
                "dialogue_onset_s": round(dialogue_onset_s, 3),
                "dialogue_offset_s": round(dialogue_offset_s, 3),
                "transcript_coverage": "covered",
            })

        # Zero-filled but genuinely "covered" windows: inside the jittered
        # dialogue span with no coded events at all — real silence, not a gap.
        lo_idx = max(0, int((dialogue_onset_s - jitter_s) // window_s))
        hi_idx = int((dialogue_offset_s + jitter_s) // window_s)
        for w_idx in range(lo_idx, hi_idx + 1):
            if w_idx in covered_indices:
                continue
            rows.append({
                "group_id": grp,
                "task_id": task,
                "window_index": w_idx,
                "window_start_s": w_idx * window_s,
                **_zero_window_counts(),
                "dialogue_onset_s": round(dialogue_onset_s, 3),
                "dialogue_offset_s": round(dialogue_offset_s, 3),
                "transcript_coverage": "covered",
            })

    out = pd.DataFrame(rows)
    LOG.info("Transcript bins: %d group-window rows", len(out))
    return out


# ── Self-report (task-level) ───────────────────────────────────────────────────


def _load_selfreport(
    stimuli_dir: Path,
    groups: list[str],
    tasks: list[str],
) -> pd.DataFrame:
    """Extract VAD and post-task questionnaire items from stimuli_answers files.

    For in-task VAD (valence / arousal / dominance):
      - Compute per-participant mean across the 2-3 repeated measurements per task.
      - Then group mean and std across P1–P4.

    For post-task items:
      - Group mean of numeric responses.

    For perceived dominance (dominance_p1..p4):
      - Each participant rates how dominant every other participant was.
      - Output: perceived_dominance_{P1..P4} = mean of others' ratings for that participant.

    # Privacy: participant IDs are always P1–P4; no real names are written.
    """
    all_rows: list[dict] = []

    for tsv in sorted(stimuli_dir.glob("*stimuli_answers_extracted.tsv")):
        m = _STIM_GRP_RE.search(tsv.name)
        if m is None:
            continue
        grp = m.group(1)
        if grp not in groups:
            continue

        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)
        except Exception as exc:
            LOG.warning("Could not load %s: %s", tsv.name, exc)
            continue

        df["item_value"] = pd.to_numeric(df["item_value"], errors="coerce")
        task_col = "task" if "task" in df.columns else "task_id"

        for task in tasks:
            t_df = df[df[task_col] == task].copy()
            if t_df.empty:
                continue

            row: dict = {"group_id": grp, "task_id": task}

            # ── In-task VAD ────────────────────────────────────────────────────
            for dim in ("valence", "arousal", "dominance"):
                vad = t_df[t_df["item_key"] == dim]
                if vad.empty:
                    continue
                # Per-participant mean → group mean/std
                p_means = vad.groupby("participant")["item_value"].mean()
                row[f"vad_{dim}_mean"] = round(float(p_means.mean()), 3)
                row[f"vad_{dim}_std"] = round(float(p_means.std()), 3) if len(p_means) > 1 else None

            # ── Post-task questionnaire items ──────────────────────────────────
            for item in POST_TASK_ITEMS.get(task, []):
                item_df = t_df[t_df["item_key"] == item]
                if not item_df.empty:
                    val = item_df["item_value"].dropna()
                    if not val.empty:
                        row[f"{item}_mean"] = round(float(val.mean()), 3)

            # ── Perceived dominance (peer-rated, self excluded) ────────────────
            dom_others: dict[str, list[float]] = {f"P{i}": [] for i in range(1, 5)}
            for dom_key in DOMINANCE_KEYS:
                rated_pid = "P" + dom_key[-1]  # dominance_p1 → P1
                item_df = t_df[t_df["item_key"] == dom_key]
                for _, r in item_df.iterrows():
                    val = r["item_value"]
                    rater = str(r.get("participant", ""))
                    if pd.notna(val) and rater != rated_pid:
                        dom_others[rated_pid].append(float(val))
            for pid, ratings in dom_others.items():
                if ratings:
                    row[f"perceived_dominance_{pid}"] = round(
                        sum(ratings) / len(ratings), 3
                    )

            all_rows.append(row)
            LOG.debug("Self-report: %s %s → %d items", grp, task, len(row))

    out = pd.DataFrame(all_rows)
    LOG.info("Self-report: %d group-task rows", len(out))
    return out


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run(
    physio_path: Path,
    et_path: Path,
    transcript_dir: Path,
    stimuli_dir: Path,
    out_dir: Path,
    groups: list[str],
    tasks: list[str],
    transcript_jitter_s: float = TRANSCRIPT_JITTER_S,
) -> None:
    """Build and write both output tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = ["group_id", "task_id", "window_index"]

    # ── Physio ────────────────────────────────────────────────────────────────
    physio_raw = _load_tsv(physio_path, "physio_window_30s")
    physio_grp = (
        _aggregate_physio(physio_raw, groups, tasks) if physio_raw is not None else pd.DataFrame()
    )

    # ── ET ────────────────────────────────────────────────────────────────────
    et_raw = _load_tsv(et_path, "et_window_30s")
    et_grp = (
        _aggregate_et(et_raw, groups, tasks) if et_raw is not None else pd.DataFrame()
    )

    # ── Transcript bins ────────────────────────────────────────────────────────
    t_events = _load_all_transcripts(transcript_dir, groups, tasks)
    t_bins = _bin_transcript_events(t_events, jitter_s=transcript_jitter_s)

    # ── Lexical window features ────────────────────────────────────────────────
    lex_path = Path("analysis/results/lexical_window_30s.tsv")
    lex_grp = _load_tsv(lex_path, "lexical_window_30s")
    if lex_grp is not None:
        # Filter to requested groups/tasks and select columns
        lex_grp = lex_grp[
            lex_grp["group_id"].isin(groups) & lex_grp["task_id"].isin(tasks)
        ].copy()
        keep_cols = ["group_id", "task_id", "window_index"] + [
            c for c in LEXICAL_COLS if c in lex_grp.columns
        ]
        lex_grp = lex_grp[keep_cols]
        LOG.info("Loaded lexical window features: %d rows × %d cols", len(lex_grp), len(lex_grp.columns))
    else:
        lex_grp = pd.DataFrame()

    # ── Merge window-level features ────────────────────────────────────────────
    sources = [df for df in (physio_grp, t_bins) if not df.empty]
    if not sources:
        LOG.error("No window-level features produced — aborting.")
        return

    window_df = sources[0]
    for other in sources[1:]:
        window_df = pd.merge(window_df, other, on=key, how="outer", suffixes=("", "_dup"))
        # Resolve duplicate window_start_s columns if both sources have it
        if "window_start_s_dup" in window_df.columns:
            window_df["window_start_s"] = window_df["window_start_s"].fillna(
                window_df["window_start_s_dup"]
            )
            window_df.drop(columns=["window_start_s_dup"], inplace=True)

    if not et_grp.empty:
        window_df = pd.merge(window_df, et_grp, on=key, how="left")

    if not lex_grp.empty:
        window_df = pd.merge(window_df, lex_grp, on=key, how="left")

    window_df.sort_values(["group_id", "task_id", "window_index"], inplace=True)
    window_df.reset_index(drop=True, inplace=True)

    win_out = out_dir / "collective_window_features.tsv"
    window_df.to_csv(win_out, sep="\t", index=False)
    LOG.info(
        "Written: %s  (%d rows × %d columns)",
        win_out, len(window_df), len(window_df.columns),
    )

    # ── Self-report (task-level) ───────────────────────────────────────────────
    sr_df = _load_selfreport(stimuli_dir, groups, tasks)
    if not sr_df.empty:
        sr_out = out_dir / "collective_task_selfreport.tsv"
        sr_df.to_csv(sr_out, sep="\t", index=False)
        LOG.info("Written: %s  (%d rows × %d columns)", sr_out, len(sr_df), len(sr_df.columns))


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--physio-windows",
        type=Path,
        default=Path("features/physio_window_30s.tsv"),
        help="Physio 30s window file (default: features/physio_window_30s.tsv)",
    )
    p.add_argument(
        "--et-windows",
        type=Path,
        default=Path("features/et_window_30s.tsv"),
        help="ET 30s window file (default: features/et_window_30s.tsv)",
    )
    p.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("tools"),
        help="Directory containing transcript_grp-*.tsv files (default: tools/)",
    )
    p.add_argument(
        "--stimuli-dir",
        type=Path,
        default=Path("metadata/extracted stimuli answers"),
        help="Directory with stimuli_answers_extracted.tsv files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("features"),
        help="Output directory (default: features/)",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        metavar="GROUP",
        help="Groups to include (default: all 10, grp-07..grp-16)",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        metavar="TASK",
        help="Tasks to include (default: T1 T2 T3)",
    )
    p.add_argument(
        "--transcript-jitter-s",
        type=float,
        default=TRANSCRIPT_JITTER_S,
        help=(
            "Tolerance (s) around the transcribed dialogue span used to mark "
            "window transcript_coverage (default: 90s). Increase for sessions "
            "with imprecise transcript boundaries or truncated recordings."
        ),
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
        physio_path=args.physio_windows.resolve(),
        et_path=args.et_windows.resolve(),
        transcript_dir=args.transcript_dir.resolve(),
        stimuli_dir=args.stimuli_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        groups=args.groups,
        tasks=args.tasks,
        transcript_jitter_s=args.transcript_jitter_s,
    )


if __name__ == "__main__":
    main()
