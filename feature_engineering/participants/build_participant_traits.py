#!/usr/bin/env python3
"""Build an anonymised (group_id, participant P1-P4) -> personality/demographics table.

Source files (`metadata/high_level_session_inventory.csv`, `metadata/participants.tsv`)
carry real participant names. This script reads them ONLY to resolve the crosswalk and
NEVER writes any name column to its output — the output table is keyed purely on
`group_id` + `participant` (P1-P4).

ASSUMPTION (unverified, flag before trusting results): the order of names in
`high_level_session_inventory.csv`'s `participants_names` column reflects registration/
tablet order (position 1 -> P1, position 2 -> P2, ...), same convention as `tablet1..4`
seen in the stimuli-answers extracts. No ground-truth registration ledger was found in
this workspace to confirm this — treat any personality/demographic finding as provisional
until spot-checked.

Usage:
    python analysis/build_participant_traits.py
"""

from __future__ import annotations

import csv
import difflib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_INVENTORY = REPO_ROOT / "metadata" / "high_level_session_inventory.csv"
PARTICIPANTS_TSV = REPO_ROOT / "metadata" / "participants.tsv"
OUT_PATH = REPO_ROOT / "analysis" / "results" / "participant_traits.tsv"

GROUPS = [f"grp-{i:02d}" for i in range(7, 17)]  # grp-07 .. grp-16
TRAIT_COLS = [
    "age", "sex", "handedness", "english_proficiency", "education",
    "bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o",
]
MATCH_THRESHOLD = 0.75


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _group_name_order(session_inventory_path: Path) -> dict[str, list[str]]:
    """Return group_id -> ordered list of 4 names (position i = tablet i+1 = P{i+1})."""
    by_group: dict[str, list[str]] = {}
    with open(session_inventory_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row.get("group_id", "").strip()
            if gid not in GROUPS or gid in by_group:
                continue
            names_str = row.get("participants_names", "").strip()
            names = [n.strip() for n in names_str.split(";") if n.strip()]
            if len(names) == 4:
                by_group[gid] = names
    return by_group


def _load_participants(participants_path: Path) -> pd.DataFrame:
    return pd.read_csv(participants_path, sep="\t", dtype=str)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not SESSION_INVENTORY.exists():
        raise FileNotFoundError(f"Session inventory not found: {SESSION_INVENTORY}")
    if not PARTICIPANTS_TSV.exists():
        raise FileNotFoundError(f"Participants table not found: {PARTICIPANTS_TSV}")

    name_order = _group_name_order(SESSION_INVENTORY)
    missing_groups = sorted(set(GROUPS) - set(name_order))
    if missing_groups:
        logger.warning("No 4-name session row found for groups: %s", missing_groups)

    participants = _load_participants(PARTICIPANTS_TSV)
    session_names = participants["session_name"].tolist()

    rows: list[dict] = []
    n_exact, n_fuzzy, n_unmatched = 0, 0, 0

    for gid in GROUPS:
        names = name_order.get(gid)
        if not names:
            continue
        for pos, name in enumerate(names, start=1):
            participant = f"P{pos}"
            norm_targets = [_norm(s) for s in session_names]
            norm_name = _norm(name)

            if norm_name in norm_targets:
                idx = norm_targets.index(norm_name)
                n_exact += 1
            else:
                close = difflib.get_close_matches(norm_name, norm_targets, n=1, cutoff=MATCH_THRESHOLD)
                if close:
                    idx = norm_targets.index(close[0])
                    n_fuzzy += 1
                else:
                    idx = None
                    n_unmatched += 1

            rec = {"group_id": gid, "participant": participant}
            if idx is not None:
                src = participants.iloc[idx]
                for col in TRAIT_COLS:
                    rec[col] = src[col]
            else:
                for col in TRAIT_COLS:
                    rec[col] = "n/a"
            rows.append(rec)

    out_df = pd.DataFrame(rows, columns=["group_id", "participant"] + TRAIT_COLS)
    # Coerce numeric-looking columns
    for col in ["age", "bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, sep="\t", index=False)

    logger.info("Matched %d exact, %d fuzzy, %d unmatched (of %d)", n_exact, n_fuzzy, n_unmatched, len(rows))
    logger.info("Wrote %d rows -> %s", len(out_df), OUT_PATH)
    logger.info("No name columns included in output — only group_id/participant/trait values.")


if __name__ == "__main__":
    main()
