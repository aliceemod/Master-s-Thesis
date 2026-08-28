from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.features import analyze_task_response_qc as qc


def test_build_cross_stream_presence_marks_complete_blocks() -> None:
    df = pd.DataFrame(
        [
            {"group_id": "grp-01", "session_id": "ses-01", "task": "T1", "response_type": "form"},
            {"group_id": "grp-01", "session_id": "ses-01", "task": "T1", "response_type": "vad"},
            {"group_id": "grp-01", "session_id": "ses-01", "task": "T1", "response_type": "postblock"},
            {"group_id": "grp-01", "session_id": "ses-01", "task": "T2", "response_type": "vad"},
            {"group_id": "grp-01", "session_id": "ses-01", "task": "T2", "response_type": "postblock"},
        ]
    )

    out = qc.build_cross_stream_presence(df)
    t1 = out[out["task"] == "T1"].iloc[0]
    t2 = out[out["task"] == "T2"].iloc[0]

    assert int(t1["all_streams_present"]) == 1
    assert int(t2["all_streams_present"]) == 0
    assert int(t2["form"]) == 0


def test_build_postblock_item_completeness_uses_task_units_as_denominator() -> None:
    df_all = pd.DataFrame(
        [
            {
                "session_id": "ses-01",
                "group_id": "grp-01",
                "participant": "P1",
                "task": "T1",
                "response_type": "postblock",
            },
            {
                "session_id": "ses-01",
                "group_id": "grp-01",
                "participant": "P2",
                "task": "T1",
                "response_type": "postblock",
            },
            {
                "session_id": "ses-02",
                "group_id": "grp-02",
                "participant": "P1",
                "task": "T1",
                "response_type": "postblock",
            },
            {
                "session_id": "ses-02",
                "group_id": "grp-02",
                "participant": "P2",
                "task": "T1",
                "response_type": "postblock",
            },
        ]
    )
    pb = pd.DataFrame(
        [
            {
                "session_id": "ses-01",
                "participant": "P1",
                "task": "T1",
                "item_key": "engagement",
                "item_value_num": 5.0,
            },
            {
                "session_id": "ses-01",
                "participant": "P2",
                "task": "T1",
                "item_key": "engagement",
                "item_value_num": 4.0,
            },
            {
                "session_id": "ses-02",
                "participant": "P1",
                "task": "T1",
                "item_key": "engagement",
                "item_value_num": 6.0,
            },
        ]
    )

    out = qc.build_postblock_item_completeness(df_all, pb)
    row = out.iloc[0]

    assert row["expected_units"] == 4
    assert row["n_present"] == 3
    assert row["n_missing"] == 1
    assert np.isclose(row["pct_present"], 75.0)


def test_build_trust_and_familiarity_summaries_average_per_participant_task() -> None:
    pb = pd.DataFrame(
        [
            {
                "group_id": "grp-01",
                "session_id": "ses-01",
                "participant": "P1",
                "task": "T2",
                "item_key": "trust_next",
                "item_value_num": 4.0,
            },
            {
                "group_id": "grp-01",
                "session_id": "ses-01",
                "participant": "P1",
                "task": "T2",
                "item_key": "trust_front",
                "item_value_num": 6.0,
            },
            {
                "group_id": "grp-01",
                "session_id": "ses-01",
                "participant": "P1",
                "task": "T1",
                "item_key": "familiarity_p2",
                "item_value_num": 2.0,
            },
            {
                "group_id": "grp-01",
                "session_id": "ses-01",
                "participant": "P1",
                "task": "T1",
                "item_key": "familiarity_p3",
                "item_value_num": 4.0,
            },
        ]
    )

    trust = qc.build_trust_summary(pb)
    fam = qc.build_familiarity_summary(pb)

    assert np.isclose(trust.iloc[0]["trust_mean"], 5.0)
    assert np.isclose(fam.iloc[0]["familiarity_mean"], 3.0)
