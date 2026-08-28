"""Per-participant transcript features (speaking time, backchannels, laughter, overlap
involvement) from `transcripts/final/*.tsv`.

Group-level features (silence, active-speaker count, collaboration index, etc.) are already
covered by `icmi_paper/results/hmm_input_features_final.tsv` — this script only extracts what
can be legitimately attributed to an individual speaker (P1-P4). Aggregated to task level
(not 30s windows) since participant-task is the granularity needed downstream.

Output: analysis/results/participant_transcript_features.tsv
    columns: group_id, task, participant, tr_speaking_time_s, tr_n_turns,
             tr_backchannel_count, tr_laughter_count, tr_overlap_count, tr_overlap_time_s,
             tr_overlap_competitive, tr_overlap_floor_fight, tr_overlap_smooth,
             tr_overlap_backchannel_ovl
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "transcripts" / "final"
OUT_PATH = REPO_ROOT / "analysis" / "results" / "participant_transcript_features.tsv"

_FNAME_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)
PARTICIPANTS = {"P1", "P2", "P3", "P4"}
OVL_SUBTYPES = ["competitive", "floor_fight", "smooth", "backchannel_ovl"]


def _normalize_group_id(raw: str) -> str:
    """Normalize e.g. 'grp8'/'grp-08' to 'grp-08'."""
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"


def _extract_one_file(fpath: Path) -> pd.DataFrame:
    """Return one row per participant with their transcript-derived task-level features."""
    m = _FNAME_RE.search(fpath.name)
    if m is None:
        logger.warning("Skipping unrecognized filename: %s", fpath.name)
        return pd.DataFrame()

    group_id = _normalize_group_id(m.group(1))
    task = m.group(2)

    df = pd.read_csv(fpath, sep="\t", dtype=str)
    df["duration"] = pd.to_numeric(df.get("duration"), errors="coerce").fillna(0.0)
    df["subtype"] = df.get("subtype", pd.Series("", index=df.index)).fillna("").str.strip().str.lower()

    rows = []
    for participant in sorted(PARTICIPANTS):
        p_df = df[df["speaker"] == participant]
        spk = p_df[p_df["type"] == "SPK"]
        ovl = p_df[p_df["type"] == "OVL"]

        row = {
            "group_id": group_id,
            "task": task,
            "participant": participant,
            "tr_speaking_time_s": round(spk["duration"].sum(), 3),
            "tr_n_turns": len(spk),
            "tr_backchannel_count": int((p_df["type"] == "BCK").sum()),
            "tr_laughter_count": int((p_df["type"] == "LAU").sum()),
            "tr_overlap_count": len(ovl),
            "tr_overlap_time_s": round(ovl["duration"].sum(), 3),
        }
        for subtype in OVL_SUBTYPES:
            row[f"tr_overlap_{subtype}"] = int((ovl["subtype"] == subtype).sum())
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    transcript_files = [
        f for f in sorted(TRANSCRIPT_DIR.glob("transcript_grp*.tsv"))
        if not f.name.endswith(".bak.tsv") and _FNAME_RE.search(f.name)
    ]
    logger.info("Found %d transcript files", len(transcript_files))

    all_rows = [_extract_one_file(f) for f in transcript_files]
    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["group_id", "task", "participant"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, sep="\t", index=False)
    logger.info("Wrote %d rows (%d groups x %d tasks x 4 participants) -> %s",
                len(result), result["group_id"].nunique(), result["task"].nunique(), OUT_PATH)


if __name__ == "__main__":
    main()
