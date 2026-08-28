"""Draw a stratified validation sample of OVL events for manual `context_label` ground-truth annotation.

Background
----------
`context_label` (collaborative / completion / competitive / empty) is currently assigned by a single
deterministic lexical-cue regex pass (`tools/relabel_overlaps.py::classify_context()`) with **no**
inter-rater validation against a human judgment. This script draws a stratified sample of OVL events —
balanced across the current `context_label` value and the timing `subtype`, spread across groups — and
exports the surrounding speech context for a human annotator to independently judge, **without** showing
the current automated label (to avoid anchoring bias). A separate answer-key file preserves the automated
label + sample provenance so agreement/kappa can be computed later without ever exposing it during annotation.

Scope: T1-T3 only (T4 is out of the analysis scope per 2026-08-12 decision).

Usage
-----
    python tools/sample_overlap_context_validation.py \\
        --transcript-dir transcripts/final \\
        --out-dir analysis/results/context_label_validation \\
        --n-collaborative 40 --n-completion 25 --n-competitive 30 --n-empty 30 \\
        --seed 42

Outputs
-------
    context_label_validation_BLIND.tsv   — for the human annotator. Columns:
        sample_id, group_id, task_id, onset, duration_ms, start_diff_ms, context_window_text
        human_context_label   (blank — fill in: collaborative / completion / competitive / empty)
        human_confidence       (blank — fill in: confident / uncertain)
        human_notes            (blank — optional)
    context_label_validation_ANSWER_KEY.tsv — NOT for the annotator. Same sample_id, plus the
        automated subtype/context_label that were sampled from, for later agreement computation.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("sample_overlap_context_validation")

_FNAME_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)
_TASKS_IN_SCOPE = {"T1", "T2", "T3"}  # T4 excluded — not used in analysis (2026-08-12)
_CONTEXT_WINDOW_S = 2.0  # seconds of surrounding speech shown to the annotator, each side


def _normalize_group_id(raw: str) -> str:
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"


def _load_transcripts(transcript_dir: Path) -> list[tuple[str, str, pd.DataFrame]]:
    """Return list of (group_id, task_id, dataframe) for all T1-T3 transcript files."""
    out = []
    for tsv in sorted(transcript_dir.glob("transcript_grp*.tsv")):
        m = _FNAME_RE.search(tsv.name)
        if m is None:
            continue
        task = m.group(2).upper()
        if task not in _TASKS_IN_SCOPE:
            continue
        grp = _normalize_group_id(m.group(1))
        df = pd.read_csv(tsv, sep="\t", dtype=str)
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce").fillna(0.0)
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
        df["type"] = df["type"].fillna("").astype(str).str.upper()
        df["text"] = df.get("text", pd.Series([""] * len(df))).fillna("").astype(str)
        df["subtype"] = df.get("subtype", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().str.lower()
        df["context_label"] = (
            df["context_label"].fillna("").astype(str).str.strip().str.lower()
            if "context_label" in df.columns
            else pd.Series([""] * len(df))
        )
        out.append((grp, task, df))
    return out


def _context_window_text(df: pd.DataFrame, onset: float, duration: float) -> str:
    """Render all SPK/BCK rows within +/- _CONTEXT_WINDOW_S of the overlap as a readable transcript snippet."""
    end = onset + duration
    lo = onset - _CONTEXT_WINDOW_S
    hi = end + _CONTEXT_WINDOW_S
    ctx = df[
        df["type"].isin({"SPK", "BCK"})
        & (df["onset"] + df["duration"] >= lo)
        & (df["onset"] <= hi)
    ].sort_values("onset")
    lines = []
    for _, row in ctx.iterrows():
        marker = ">>> " if row["onset"] < end and row["onset"] + row["duration"] > onset else "    "
        lines.append(f"{marker}[{row['speaker']}] {row['text']}")
    return "\n".join(lines) if lines else "(no surrounding SPK/BCK text found)"


def _start_diff_ms(df: pd.DataFrame, onset: float, duration: float) -> float | None:
    end = onset + duration
    spk = df[df["type"] == "SPK"]
    overlapping = spk[(spk["onset"] < end) & (spk["onset"] + spk["duration"] > onset)].sort_values("onset")
    if len(overlapping) < 2:
        return None
    return round(abs(float(overlapping.iloc[1]["onset"]) - float(overlapping.iloc[0]["onset"])) * 1000.0, 1)


def build_pool(transcript_dir: Path) -> pd.DataFrame:
    rows = []
    for grp, task, df in _load_transcripts(transcript_dir):
        ovl = df[df["type"] == "OVL"]
        for _, r in ovl.iterrows():
            onset = float(r["onset"])
            duration = float(r["duration"])
            rows.append({
                "group_id": grp,
                "task_id": task,
                "onset": onset,
                "duration_ms": round(duration * 1000.0, 1),
                "start_diff_ms": _start_diff_ms(df, onset, duration),
                "auto_subtype": r["subtype"],
                "auto_context_label": r["context_label"] or "(empty)",
                "context_window_text": _context_window_text(df, onset, duration),
            })
    return pd.DataFrame(rows)


def stratified_sample(
    pool: pd.DataFrame,
    quotas: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    picked_frames = []
    for label, n in quotas.items():
        stratum = pool[pool["auto_context_label"] == label]
        if stratum.empty:
            LOG.warning("No rows found for context_label=%r — skipping quota of %d", label, n)
            continue
        n_take = min(n, len(stratum))
        if n_take < n:
            LOG.warning(
                "Only %d rows available for context_label=%r (requested %d) — taking all of them",
                n_take, label, n,
            )
        idx = rng.sample(list(stratum.index), n_take)
        picked_frames.append(stratum.loc[idx])
    if not picked_frames:
        return pool.iloc[0:0]
    sampled = pd.concat(picked_frames, ignore_index=False)
    # Shuffle final order so the annotator can't infer the stratum from row order
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "sample_id", [f"S{i:04d}" for i in range(len(sampled))])
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript-dir", type=Path, default=Path("transcripts/final"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/results/context_label_validation"))
    ap.add_argument("--n-collaborative", type=int, default=40)
    ap.add_argument("--n-completion", type=int, default=25)
    ap.add_argument("--n-competitive", type=int, default=30)
    ap.add_argument("--n-empty", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)-5s %(message)s")

    pool = build_pool(args.transcript_dir)
    LOG.info("Pool: %d OVL events (T1-T3) across %d groups", len(pool), pool["group_id"].nunique())
    LOG.info("Pool breakdown by auto_context_label:\n%s", pool["auto_context_label"].value_counts().to_string())

    quotas = {
        "collaborative": args.n_collaborative,
        "completion": args.n_completion,
        "competitive": args.n_competitive,
        "(empty)": args.n_empty,
    }
    sampled = stratified_sample(pool, quotas, args.seed)
    LOG.info("Sampled %d events total", len(sampled))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    answer_key = sampled[["sample_id", "group_id", "task_id", "onset", "auto_subtype", "auto_context_label"]].copy()
    answer_key_path = args.out_dir / "context_label_validation_ANSWER_KEY.tsv"
    answer_key.to_csv(answer_key_path, sep="\t", index=False)

    blind = sampled[
        ["sample_id", "group_id", "task_id", "onset", "duration_ms", "start_diff_ms", "context_window_text"]
    ].copy()
    # PENDING (not blank) distinguishes "not yet annotated" from a human judgment of
    # "no clear signal", which must be entered as the literal word "empty".
    blind["human_context_label"] = "PENDING"
    blind["human_confidence"] = ""
    blind["human_notes"] = ""
    blind_path = args.out_dir / "context_label_validation_BLIND.tsv"
    blind.to_csv(blind_path, sep="\t", index=False)

    LOG.info(
        "Written: %s (%d rows) — annotate this one. Fill human_context_label with one of: "
        "collaborative / completion / competitive / empty (use the literal word 'empty', "
        "not a blank cell, when there is no clear signal).",
        blind_path, len(blind),
    )
    LOG.info("Written: %s (%d rows) — DO NOT open while annotating", answer_key_path, len(answer_key))


if __name__ == "__main__":
    main()
