"""Combine participant-level transcript, audio, physio, lexical, and questionnaire (target)
features into a single participant x task modeling table.

Data-prep only, mirrors `analysis/effectiveness_prediction_models.ipynb` but at participant
(P1-P4) granularity instead of group granularity. Sources:
    - analysis/results/participant_transcript_features.tsv          (this repo, real per-speaker)
    - analysis/results/participant_lexical_features.tsv             (this repo, real per-speaker)
    - analysis/results/participant_audio_features.tsv               (this repo, real per-speaker)
    - analysis/results/physio_features/physio_participant_task.tsv  (real, extracted from raw
      EmotiBit BIDS data via tools/features/extract_physio_features.py)
    - analysis/results/pupil_features/features_pupil_participant_task.tsv (real, extracted from
      raw Tobii BIDS data via tools/features/extract_pupil_features.py)
    - analysis/results/perceived_effectiveness_index_participant.tsv (target + questionnaire cols)

Output: analysis/results/participant_model_df.tsv
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
PHYSIO_PATH = RESULTS_DIR / "physio_features" / "physio_participant_task.tsv"
PUPIL_PATH = RESULTS_DIR / "pupil_features" / "features_pupil_participant_task.tsv"
OUT_PATH = RESULTS_DIR / "participant_model_df.tsv"

TRANSCRIPT_FEATS = [
    "tr_speaking_time_s", "tr_n_turns", "tr_backchannel_count", "tr_laughter_count",
    "tr_overlap_count", "tr_overlap_time_s", "tr_overlap_competitive",
    "tr_overlap_floor_fight", "tr_overlap_smooth", "tr_overlap_backchannel_ovl",
]
LEXICAL_FEATS = [
    # Basic metrics
    "lex_word_count", "lex_unique_words", "lex_ttr", "lex_mean_word_len",
    # N-gram counts and diversity
    "lex_n_bigrams", "lex_n_trigrams", "lex_unique_bigrams", "lex_unique_trigrams",
    "lex_bigram_diversity", "lex_trigram_diversity",
    # N-gram entropy (linguistic unpredictability)
    "lex_unigram_entropy", "lex_bigram_entropy", "lex_trigram_entropy",
    # Hapax legomena
    "lex_hapax_count", "lex_hapax_ratio",
    # Marker counts and rates
    "lex_agreement_count", "lex_hedging_count", "lex_certainty_count",
    "lex_question_count", "lex_suggestion_count", "lex_positive_count", "lex_negative_count",
    "lex_agreement_rate", "lex_question_rate_per_min", "lex_suggestion_rate_per_min",
    "lex_sentiment_ratio",
]
AUDIO_FEATS = ["audio_energy_mean", "audio_pitch_mean", "audio_pitch_sd", "audio_hnr_mean"]
PHYSIO_FEATS = ["hr_mean_bpm", "hrv_rmssd_ms", "eda_phasic_rate_hz", "temp_mean"]
PUPIL_FEATS = ["pupil_mean", "pupil_std", "pupil_slope_per_s"]
QUESTIONNAIRE_FEATS = [
    "cooperative", "decision_confidence", "idea_quality", "manipcheck_t2",
    "satisfaction", "team_coordination",
]
KEY = ["group_id", "task", "participant"]
_GRP_RE = re.compile(r"(grp-\d+)")


def _load_physio() -> pd.DataFrame:
    physio = pd.read_csv(PHYSIO_PATH, sep="\t")
    physio["group_id"] = physio["session_id"].str.extract(_GRP_RE)
    physio = physio.drop(columns=["task"]).rename(
        columns={"task_id": "task", "participant_id": "participant"}
    )
    return physio[KEY + PHYSIO_FEATS + ["coverage_pct", "qc_flag"]]


def _load_pupil() -> pd.DataFrame:
    pupil = pd.read_csv(PUPIL_PATH, sep="\t")
    pupil["group_id"] = pupil["session_id"].str.extract(_GRP_RE)
    pupil = pupil.rename(columns={"participant_id": "participant"})
    pupil = pupil.rename(columns={"gaze_valid_frac": "pupil_gaze_valid_frac"})
    return pupil[KEY + PUPIL_FEATS + ["pupil_gaze_valid_frac"]]


def _add_lexical_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Add theory-driven composite lexical scores (z-scored).

    Three interpretable dimensions:
    - lex_social_z: positive engagement (positive_count + agreement_count + sentiment_ratio)
    - lex_complexity_z: lexical complexity (unigram_entropy + ttr)
    - lex_verbosity_z: log-transformed word count
    """
    # Social engagement composite: positive language + agreement + sentiment
    social_cols = ["lex_positive_count", "lex_agreement_count", "lex_sentiment_ratio"]
    if all(c in df.columns for c in social_cols):
        social_z = df[social_cols].apply(lambda x: zscore(x, nan_policy="omit"), axis=0)
        df["lex_social_z"] = social_z.mean(axis=1)
    else:
        df["lex_social_z"] = np.nan

    # Lexical complexity composite: entropy + TTR
    complexity_cols = ["lex_unigram_entropy", "lex_ttr"]
    if all(c in df.columns for c in complexity_cols):
        complexity_z = df[complexity_cols].apply(lambda x: zscore(x, nan_policy="omit"), axis=0)
        df["lex_complexity_z"] = complexity_z.mean(axis=1)
    else:
        df["lex_complexity_z"] = np.nan

    # Verbosity composite: log word count
    if "lex_word_count" in df.columns:
        log_wc = np.log1p(df["lex_word_count"])
        df["lex_verbosity_z"] = zscore(log_wc, nan_policy="omit")
    else:
        df["lex_verbosity_z"] = np.nan

    return df


def main() -> None:
    transcript = pd.read_csv(RESULTS_DIR / "participant_transcript_features.tsv", sep="\t")
    lexical = pd.read_csv(RESULTS_DIR / "participant_lexical_features.tsv", sep="\t")
    audio = pd.read_csv(RESULTS_DIR / "participant_audio_features.tsv", sep="\t")
    physio = _load_physio()
    pupil = _load_pupil()
    target = pd.read_csv(RESULTS_DIR / "perceived_effectiveness_index_participant.tsv", sep="\t")

    logger.info("transcript: %s, lexical: %s, audio: %s, physio: %s, pupil: %s, target: %s",
                transcript.shape, lexical.shape, audio.shape, physio.shape, pupil.shape, target.shape)

    df = target.merge(transcript[KEY + TRANSCRIPT_FEATS], on=KEY, how="left")
    df = df.merge(lexical[KEY + LEXICAL_FEATS], on=KEY, how="left")
    df = df.merge(audio[KEY + AUDIO_FEATS], on=KEY, how="left")
    df = df.merge(physio, on=KEY, how="left")
    df = df.merge(pupil, on=KEY, how="left")

    # Add theory-driven lexical composite scores
    df = _add_lexical_composites(df)

    df = df[df["task"].isin(["T1", "T2", "T3"])].reset_index(drop=True)

    n_missing_transcript = df[TRANSCRIPT_FEATS[0]].isna().sum()
    n_missing_lexical = df[LEXICAL_FEATS[0]].isna().sum()
    n_missing_audio = df[AUDIO_FEATS[0]].isna().sum()
    n_missing_physio = df[PHYSIO_FEATS[0]].isna().sum()
    n_missing_pupil = df[PUPIL_FEATS[0]].isna().sum()
    n_low_coverage = (df["coverage_pct"] < 80).sum()
    n_low_gaze_valid = (df["pupil_gaze_valid_frac"] < 0.8).sum()
    logger.info(
        "Rows: %d (%d groups). Missing transcript: %d, missing lexical: %d, missing audio: %d, "
        "missing physio: %d, missing pupil: %d, physio coverage<80%%: %d, pupil gaze_valid<80%%: %d",
        len(df), df["group_id"].nunique(), n_missing_transcript, n_missing_lexical, n_missing_audio,
        n_missing_physio, n_missing_pupil, n_low_coverage, n_low_gaze_valid,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, sep="\t", index=False)
    logger.info("Wrote %d rows -> %s", len(df), OUT_PATH)


if __name__ == "__main__":
    main()
