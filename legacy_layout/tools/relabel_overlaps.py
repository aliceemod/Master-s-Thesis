"""Re-label OVL (overlap) entries in transcript TSVs.

Classification scheme
---------------------
**Primary label (`subtype`)** — always timing/duration based, no exceptions:

    simultaneous     start difference < 200 ms
    backchannel_ovl  overlap < 250 ms
    smooth           overlap 250–500 ms
    competitive      overlap 500–1000 ms
    floor_fight      overlap >= 1000 ms

**Context label (`context_label`)** — lexical cues from the speech surrounding the overlap.
Written whenever a clear functional signal is detected:

    completion       another speaker's utterance explicitly trails off (ends with '...',
                     '\u2026', or em-dash) AND this overlap continues with content words
                     (not mere agreement tokens). Captures collaborative sentence completions.
    collaborative    support words detected in overlapping speech (any timing subtype)
                     (e.g. "yes", "right", "mmh", "and" — supportive backchannels and
                      agreements; does not include pure completion cases)
    competitive      conflict words detected on a competitive/floor_fight overlap
                     (e.g. "no", "wait", "but" on a long contested overlap)

The `context_label` column is empty when no clear lexical signal is found.
The `confidence` column is never modified by this script.

Usage
-----
    # Dry-run: see what would change without writing files
    python tools/relabel_overlaps.py --dry-run

    # Apply relabeling in-place (backs up originals as .bak.tsv)
    python tools/relabel_overlaps.py

    # Target a specific directory
    python tools/relabel_overlaps.py --glob "transcripts/final/*.tsv"

Reversibility
-------------
Each modified file is backed up as ``<original_name>.bak.tsv`` before the first write.
To restore: copy transcript_grp-XX.bak.tsv transcript_grp-XX.tsv
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger("relabel_overlaps")

# Strong agreement signals — always count as support regardless of utterance length
SUPPORT_PATTERNS = [
    re.compile(r"\b(yeah|yep|mmh|mm-hmm|uh-huh|exactly|agree|agreed|good point|totally|absolutely|indeed)\b"),
]
# Context-sensitive words — only count as support in SHORT utterances (<= 4 words)
# Excludes 'yes' when embedded in conditional phrases like 'If yes, why?'
SHORT_SUPPORT_PATTERNS = [
    re.compile(r"^(yes|right|ok|okay|sure|correct|true|definitely)[!.?,]?\s*(please|thanks|thank you)?\s*$", re.IGNORECASE),
]
# Conflict signals — 'but', 'however', 'actually' are kept as weak conflict markers
# because they commonly co-occur with genuine disagreement. The heuristic is imperfect
# (many "but/actually" are discourse connectors, not conflict) but removing them causes
# more false positives in collaborative labels. Accept as a known limitation.
CONFLICT_PATTERNS = [
    re.compile(r"\b(no|not really|i disagree|disagree|wrong|hold on|wait|stop|but|however|actually)\b"),
    re.compile(r"\b(let me finish|don't interrupt|you are missing)\b"),
]

VALID_SUBTYPES = {
    "simultaneous",
    "backchannel_ovl",
    "smooth",
    "competitive",
    "floor_fight",
}

# Ellipsis / trailing-off markers in transcription indicating incomplete utterance
_ELLIPSIS_RE = re.compile(r"(\.{2,}|\u2026|[\u2014\u2013]\s*)$")  # ..., …, —, –

# Support words that are JUST agreement (not content completion)
_JUST_AGREE_RE = re.compile(
    r"^(yeah|yep|yes|mmh|mm-hmm|uh-huh|right|okay|ok|no|exactly)[.\s!]*$", re.IGNORECASE
)


def _has_cue(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(p.search(lower) for p in patterns)


def _has_support(text: str) -> bool:
    """Return True if text contains a clear support/agreement signal.

    Always-support words (yeah, mmh, exactly, agree…) match regardless of length.
    Context-sensitive words (right, ok, okay, yes, sure…) only count when the
    full utterance is very short (<=4 words), avoiding false positives like
    'relevant for all of us, right?' or 'Okay, so let me explain…'.
    """
    if not text:
        return False
    if _has_cue(text, SUPPORT_PATTERNS):
        return True
    # Short-utterance check for ambiguous words
    stripped = text.strip()
    if len(stripped.split()) <= 4 and any(p.match(stripped) for p in SHORT_SUPPORT_PATTERNS):
        return True
    return False


def classify_timing(overlap_ms: float, start_diff_ms: float | None) -> str:
    """Primary label: always duration-based, no lexical analysis.

    simultaneous     start_diff < 200 ms
    backchannel_ovl  overlap < 250 ms
    smooth           overlap 250–500 ms
    competitive      overlap 500–1000 ms
    floor_fight      overlap >= 1000 ms
    """
    if start_diff_ms is not None and start_diff_ms < 200:
        return "simultaneous"
    if overlap_ms < 250:
        return "backchannel_ovl"
    if overlap_ms < 500:
        return "smooth"
    if overlap_ms < 1000:
        return "competitive"
    return "floor_fight"


def _is_completion_context(df: pd.DataFrame, ovl_row: pd.Series) -> bool:
    """Return True if this OVL appears to be a collaborative sentence completion.

    Heuristic: another speaker's SPK utterance ends with an explicit ellipsis
    or em-dash (trailing-off marker) within 1.5 s before this OVL starts, and
    the OVL speaker's overlapping speech contains real content words (not merely
    agreement tokens like 'yeah').
    """
    onset = float(ovl_row["onset"])
    ovl_speaker = str(ovl_row.get("speaker", ""))

    spk = df[df["type"].astype(str).str.upper() == "SPK"]
    # Other speakers whose utterance ends just before / at the start of this OVL
    spk_end = spk["onset"] + spk["duration"].fillna(0)
    trailing = spk[
        (spk["speaker"] != ovl_speaker)
        & (spk_end >= onset - 1.5)
        & (spk_end <= onset + 0.3)
    ]
    if trailing.empty:
        return False
    # At least one of them must end with an ellipsis / dash
    if not trailing["text"].fillna("").apply(lambda t: bool(_ELLIPSIS_RE.search(str(t)))).any():
        return False
    return True


def classify_context(timing_label: str, text_a: str, text_b: str, is_completion: bool = False) -> str:
    """Secondary label: lexical cue from the speech surrounding the overlap.

    Returns 'completion' when another speaker's utterance trails off with an
    ellipsis/dash AND the OVL speaker continues with content words (not mere
    agreement). This captures collaborative sentence completions.

    Returns 'collaborative' when support words are found AND the overlap is
    long enough to be meaningful (smooth ≥250ms, competitive, or floor_fight).
    Brief backchannels (<250ms) and simultaneous starts carry their own timing
    signal and are excluded to avoid inflating the collaborative count with
    routine acoustic listener signals.

    Returns 'competitive' when conflict words are found on a long contested
    overlap (competitive or floor_fight timing).

    Returns '' (empty) when no clear lexical signal is found.
    """
    has_support = _has_support(text_a) or _has_support(text_b)
    has_conflict = _has_cue(text_a, CONFLICT_PATTERNS) or _has_cue(text_b, CONFLICT_PATTERNS)

    # Completion: takes priority — other speaker trailed off and this is content continuation
    ovl_text = text_b or text_a
    if is_completion and ovl_text and not _JUST_AGREE_RE.match(ovl_text.strip()):
        return "completion"

    # Collaborative: support words on a SUSTAINED overlap (≥250ms = smooth/competitive/floor_fight)
    # Brief backchannels and simultaneous starts are excluded — they carry their timing signal already
    if has_support and not has_conflict and timing_label in ("smooth", "competitive", "floor_fight"):
        return "collaborative"

    # Competitive: conflict words on a long contested overlap
    if timing_label in ("competitive", "floor_fight") and has_conflict and not has_support:
        return "competitive"
    return ""


def _find_context_spks(df: pd.DataFrame, ovl_row: pd.Series) -> tuple[pd.Series | None, pd.Series | None]:
    onset = float(ovl_row["onset"])
    end = onset + float(ovl_row.get("duration", 0) or 0)
    spk = df[df["type"].astype(str).str.upper() == "SPK"]
    overlapping = spk[(spk["onset"] < end) & (spk["onset"] + spk["duration"].fillna(0) > onset)]
    if len(overlapping) >= 2:
        overlapping = overlapping.sort_values("onset")
        return overlapping.iloc[0], overlapping.iloc[1]
    if len(overlapping) == 1:
        earlier = spk[spk["onset"] < onset].sort_values("onset").tail(1)
        if not earlier.empty:
            return earlier.iloc[0], overlapping.iloc[0]
        return overlapping.iloc[0], None
    earlier = spk[spk["onset"] < onset].sort_values("onset").tail(2)
    if len(earlier) == 2:
        return earlier.iloc[0], earlier.iloc[1]
    return None, None


def relabel_file(path: Path, dry_run: bool) -> dict:
    df = pd.read_csv(path, sep="\t", dtype={"text": str, "subtype": str, "type": str, "speaker": str})
    df["type"] = df["type"].fillna("").astype(str)
    df["subtype"] = df.get("subtype", pd.Series([""] * len(df))).fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce").fillna(0.0)

    # Ensure context_label column exists; confidence is never touched
    if "context_label" not in df.columns:
        df["context_label"] = ""
    df["context_label"] = df["context_label"].fillna("").astype(str)

    # Drop legacy timing_class column if present from a previous run
    if "timing_class" in df.columns:
        df.drop(columns=["timing_class"], inplace=True)

    counts_before = df[df["type"].str.upper() == "OVL"]["subtype"].value_counts().to_dict()
    changed = 0
    ctx_changed = 0

    for idx, row in df.iterrows():
        if str(row["type"]).upper() != "OVL":
            continue

        ovl_ms = float(row["duration"]) * 1000.0
        spk_a, spk_b = _find_context_spks(df, row)
        start_diff_ms: float | None = None
        if spk_a is not None and spk_b is not None:
            start_diff_ms = abs(float(spk_b["onset"]) - float(spk_a["onset"])) * 1000.0

        # Step 1: timing-based primary label
        new_sub = classify_timing(ovl_ms, start_diff_ms)

        # Step 2: lexical context — check for completion, then support/conflict
        text_a = str(spk_a["text"]) if spk_a is not None else ""
        text_b = str(spk_b["text"]) if spk_b is not None else ""
        is_completion = _is_completion_context(df, row)
        context = classify_context(new_sub, text_a, text_b, is_completion)

        current_sub = str(row["subtype"]).strip().lower()
        if new_sub != current_sub:
            changed += 1

        current_ctx = str(row.get("context_label", "")).strip()
        if context != current_ctx:
            ctx_changed += 1

        df.at[idx, "subtype"] = new_sub
        df.at[idx, "text"] = f"[OVL:{new_sub}]"
        df.at[idx, "context_label"] = context
        # confidence column is intentionally not modified

    counts_after = df[df["type"].str.upper() == "OVL"]["subtype"].value_counts().to_dict()
    ctx_counts_after = df[df["type"].str.upper() == "OVL"]["context_label"].value_counts().to_dict()

    if not dry_run and (changed or ctx_changed):
        backup = path.with_suffix(".bak.tsv")
        if not backup.exists():
            shutil.copy2(path, backup)
        df.to_csv(path, sep="\t", index=False)

    return {
        "file": str(path.name),
        "changed": changed,
        "ctx_changed": ctx_changed,
        "before": counts_before,
        "after": counts_after,
        "ctx_after": ctx_counts_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Transcript TSV files or globs")
    parser.add_argument("--glob", default="tools/transcript_grp-*.tsv", help="Fallback glob when no paths passed")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    files: list[Path] = []
    if args.paths:
        for p in args.paths:
            if p.is_dir():
                files.extend(sorted(p.glob("transcript_grp-*.tsv")))
            else:
                files.append(p)
    else:
        files = sorted(Path(".").glob(args.glob))

    if not files:
        logger.error("No transcript files found")
        return 1

    summary = []
    for path in files:
        if path.name.endswith(".bak.tsv"):
            continue
        result = relabel_file(path, args.dry_run)
        summary.append(result)
        logger.info(
            "%s: sub_changed=%d ctx_changed=%d after=%s ctx=%s",
            result["file"], result["changed"], result["ctx_changed"],
            result["after"], result["ctx_after"],
        )

    logger.info("Processed %d file(s); mode=%s", len(summary), "dry-run" if args.dry_run else "write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
