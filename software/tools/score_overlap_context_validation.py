"""Compute inter-rater agreement between the automated `context_label` heuristic and human ground truth.

Run this AFTER a human annotator has filled in `human_context_label` (and optionally
`human_confidence`/`human_notes`) in the BLIND validation TSV produced by
`tools/sample_overlap_context_validation.py`.

Usage
-----
    python tools/score_overlap_context_validation.py \\
        --blind analysis/results/context_label_validation/context_label_validation_BLIND.tsv \\
        --answer-key analysis/results/context_label_validation/context_label_validation_ANSWER_KEY.tsv

Reports
-------
    - Overall raw agreement % (automated vs. human)
    - Cohen's kappa (chance-corrected agreement)
    - Per-class precision/recall/F1 for the automated label against human ground truth
    - Confusion matrix (auto rows x human columns)

This does NOT modify any pipeline output. It only reports whether the current
`context_label` heuristic (tools/relabel_overlaps.py::classify_context) is trustworthy
enough to keep as-is, or should be replaced by a validated alternative (e.g. an
ensemble of models) — that decision is made by the user based on this report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _cohen_kappa(auto: pd.Series, human: pd.Series, labels: list[str]) -> float:
    n = len(auto)
    po = float((auto.values == human.values).sum()) / n
    pe = 0.0
    for label in labels:
        p_auto = float((auto == label).sum()) / n
        p_human = float((human == label).sum()) / n
        pe += p_auto * p_human
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _prf1(auto: pd.Series, human: pd.Series, label: str) -> tuple[float, float, float, int]:
    tp = int(((auto == label) & (human == label)).sum())
    fp = int(((auto == label) & (human != label)).sum())
    fn = int(((auto != label) & (human == label)).sum())
    support = int((human == label).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 and precision == precision and recall == recall
        else float("nan")
    )
    return precision, recall, f1, support


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind", type=Path, required=True)
    ap.add_argument("--answer-key", type=Path, required=True)
    args = ap.parse_args()

    blind = pd.read_csv(args.blind, sep="\t", dtype=str)
    key = pd.read_csv(args.answer_key, sep="\t", dtype=str)

    merged = blind.merge(key, on="sample_id", suffixes=("", "_key"))
    merged["human_context_label"] = merged["human_context_label"].fillna("").str.strip().str.lower()
    # "PENDING" (or a truly blank cell, for older exports) means not-yet-annotated;
    # a human judgment of "no clear signal" must be the literal word "empty".
    unfilled = merged["human_context_label"].isin({"", "pending"})
    if unfilled.any():
        print(f"WARNING: {unfilled.sum()} rows have no human_context_label filled in yet — excluding them.")
        merged = merged[~unfilled]

    if merged.empty:
        print("No annotated rows found. Fill in human_context_label in the BLIND tsv first.")
        return

    # Normalize both sides' "empty" representations to the same token for comparison
    auto = merged["auto_context_label"].str.strip().str.lower().replace("(empty)", "empty")
    human = merged["human_context_label"].replace("", "empty")
    labels = sorted(set(auto.unique()) | set(human.unique()))

    n = len(merged)
    agree = int((auto.values == human.values).sum())
    print(f"n annotated: {n}")
    print(f"Raw agreement: {agree}/{n} = {agree / n:.3%}")
    print(f"Cohen's kappa: {_cohen_kappa(auto, human, labels):.3f}")
    print()
    print(f"{'label':<15}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in labels:
        p, r, f1, support = _prf1(auto, human, label)
        label_disp = label if label else "(empty)"
        print(f"{label_disp:<15}{p:>10.3f}{r:>10.3f}{f1:>10.3f}{support:>10d}")
    print()
    print("Confusion matrix (rows=auto, cols=human):")
    conf = pd.crosstab(auto.replace("", "(empty)"), human.replace("", "(empty)"))
    print(conf.to_string())


if __name__ == "__main__":
    main()
