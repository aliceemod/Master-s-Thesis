"""Primary raw-feature outcome-prediction analysis (2026-09-05).

Builds the group-task feature/target modeling matrix from the corrected
`collective_features_v20260905` matrix plus the per-family generated tables
(audio prosody, turn-taking, lexical, dialogue acts, semantic continuity, BERT
semantics), using the same group-task aggregation and eligibility screening
already established in `analysis/feature_eda_and_selection.ipynb`
(`to_group_task` / coverage-variance-sparsity screen), so the modeling feature
sets match the audited diagnostics in `analysis/results/feature_eda/`.

This supersedes `analysis/effectiveness_prediction_models.ipynb`, which still
reads `icmi_paper/results/hmm_input_features_final.tsv` — pre-baseline-fix,
pre-discussion-filter data. See docs/prediction_analysis_backlog.md Priority 1.

Design (implements docs/prediction_analysis_backlog.md Priority 1, in part):
    - Group-task is the prediction unit; validation is leave-one-group-out
      (GroupKFold with n_splits == n_groups).
    - Complete-case (listwise) per feature-set x target combination; no
      imputation. Effective n and n_groups are reported for every result row.
    - Preprocessing (StandardScaler) is fit only on the training fold, inside
      a scikit-learn Pipeline — never on the full dataset before splitting.
    - RidgeCV (tuned per fold) is the primary estimator.
    - Label-shuffle permutation test per feature-set x target pair.
    - Targets are grouped into primary (administered in all 3 tasks, n<=30),
      secondary (2 tasks, n<=20), and exploratory (1 task, n<=10) tiers by
      questionnaire coverage; exploratory-tier results are small-n and must
      not be read as robust evidence (see backlog Priority 4).

Not yet implemented here (tracked in docs/prediction_analysis_backlog.md
Priority 1): fold-local HMM fitting / state-vs-raw comparison, paired
fold-level bootstrap, and multiple-comparison correction across the full
feature-set x target grid.

Outputs (analysis/results/primary_prediction_analysis/):
    prediction_primary_results.tsv
    run_metadata.json
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
COLLECTIVE_DIR = RESULTS_DIR / "collective_features_v20260905"
OUTPUT_DIR = RESULTS_DIR / "primary_prediction_analysis"

SEED = 20260905
N_PERMUTATIONS = 200
MIN_GROUPS_TO_MODEL = 6  # need enough LOGO folds for a non-degenerate estimate
ALPHAS = np.logspace(-2, 3, 12)
TASKS = ["T1", "T2", "T3"]

# family -> source table (window/participant-task/group-task granularity, aggregated below)
FEATURE_SOURCES: dict[str, Path] = {
    "collective_core": COLLECTIVE_DIR / "collective_window_features.tsv",
    "audio_prosody": RESULTS_DIR / "participant_audio_features.tsv",
    "turn_taking": RESULTS_DIR / "transcript_features_full" / "transcript_group_task.tsv",
    "lexical": RESULTS_DIR / "lexical_window_30s.tsv",
    "bert_semantics": RESULTS_DIR / "bert_embeddings_window_30s.tsv",
    "dialogue_acts_keyword": RESULTS_DIR / "da_keyword_window_30s.tsv",
    "dialogue_acts_nli": RESULTS_DIR / "da_window_30s.tsv",
    "semantic_continuity": RESULTS_DIR / "semantic_continuity_window_30s.tsv",
}

IDENTIFIER_TOKENS = ("id", "group", "session", "task", "participant", "window", "source", "path")
EXCLUDED_TOKENS = (
    "qc", "flag", "available", "coverage", "duration", "sample_rate",
    "n_windows", "n_participants", "index", "label",
)

# target column -> tasks where the questionnaire item was administered (defines the tier)
TARGET_TASKS: dict[str, list[str]] = {
    "engagement_mean": TASKS,
    "voice_inclusion_mean": TASKS,
    "mental_demand_mean": TASKS,
    "vad_valence_mean": TASKS,
    "vad_arousal_mean": TASKS,
    "vad_dominance_mean": TASKS,
    "perceived_dominance_mean": TASKS,
    "team_coordination_mean": ["T1", "T3"],
    "perceived_control_mean": ["T1", "T2"],
    "satisfaction_mean": ["T2", "T3"],
    "equality_of_contribution_mean": ["T1"],
    "info_sharing_mean": ["T1"],
    "decision_confidence_mean": ["T1"],
    "cooperative_mean": ["T2"],
    "trust_angle_mean": ["T2"],
    "trust_front_mean": ["T2"],
    "trust_next_mean": ["T2"],
    "confidence_mean": ["T3"],
    "psych_safety_mean": ["T3"],
    "fairness_mean": ["T3"],
    "idea_quality_mean": ["T3"],
    "idea_diversity_mean": ["T3"],
}


def _tier(tasks: list[str]) -> str:
    if len(tasks) == 3:
        return "primary (all tasks, n<=30)"
    if len(tasks) == 2:
        return "secondary (2 tasks, n<=20)"
    return "exploratory (1 task, n<=10, small-n)"


def is_candidate_column(column: str, values: pd.Series) -> bool:
    """Return whether a numeric column is a feature rather than metadata or a QC field."""
    normalized = column.lower()
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return False
    if any(token in normalized for token in IDENTIFIER_TOKENS + EXCLUDED_TOKENS):
        return False
    return True


def to_group_task(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate a generated family table to one row per group-task (mean of numeric features).

    Mirrors `analysis/feature_eda_and_selection.ipynb::to_group_task` so the modeling
    feature sets match the audited eligibility diagnostics exactly.
    """
    task_column = next((column for column in ("task_id", "task") if column in frame), None)
    if "group_id" not in frame or task_column is None:
        return None
    features = [column for column in frame if is_candidate_column(column, frame[column])]
    if not features:
        return None
    aggregated = frame.groupby(["group_id", task_column], as_index=False)[features].mean(numeric_only=True)
    aggregated = aggregated.rename(columns={task_column: "task_id"})
    return aggregated[aggregated["task_id"].isin(TASKS)].reset_index(drop=True)


def eligible_features(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Keep only columns passing the same coverage/variance/sparsity screen as the EDA notebook."""
    keep = []
    for col in feature_cols:
        observed = frame[col].dropna()
        if observed.empty:
            continue
        coverage_pct = 100 * len(observed) / len(frame[col])
        near_constant = observed.nunique() <= 2 or observed.std() == 0
        sparse = (observed == 0).mean() >= 0.80
        if coverage_pct >= 80 and not near_constant and not sparse:
            keep.append(col)
    return keep


def build_feature_sets() -> dict[str, pd.DataFrame]:
    """Load, aggregate, and screen every feature family to group-task rows of eligible features."""
    feature_sets: dict[str, pd.DataFrame] = {}
    for family, path in FEATURE_SOURCES.items():
        if not path.exists():
            print(f"  [skip] {family}: source not found at {path}")
            continue
        raw = pd.read_csv(path, sep="\t")
        group_task = to_group_task(raw)
        if group_task is None or group_task.empty:
            print(f"  [skip] {family}: could not aggregate to group-task")
            continue
        feature_cols = [c for c in group_task.columns if c not in ("group_id", "task_id")]
        keep = eligible_features(group_task, feature_cols)
        if not keep:
            print(f"  [skip] {family}: no eligible features after screening")
            continue
        feature_sets[family] = group_task[["group_id", "task_id"] + keep]
        print(f"  [ok]   {family}: {len(keep)} eligible features, {len(group_task)} group-task rows")
    return feature_sets


def load_targets() -> pd.DataFrame:
    selfreport = pd.read_csv(COLLECTIVE_DIR / "collective_task_selfreport.tsv", sep="\t")
    dominance_cols = [c for c in selfreport.columns if c.startswith("perceived_dominance_P")]
    selfreport["perceived_dominance_mean"] = selfreport[dominance_cols].mean(axis=1)
    return selfreport


def logo_cv_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit Pipeline(StandardScaler, RidgeCV) fold-locally under leave-one-group-out CV."""
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=n_groups)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    oof_pred = cross_val_predict(pipe, x, y, cv=cv, groups=groups)
    return oof_pred, r2_score(y, oof_pred)


def permutation_p_value(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, observed_r2: float, rng: np.random.Generator
) -> float:
    """Label-shuffle permutation test: rerun the full LOGO-CV procedure on shuffled targets."""
    null_r2 = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_shuffled = rng.permutation(y)
        _, r2 = logo_cv_ridge(x, y_shuffled, groups)
        null_r2[i] = r2
    return float((1 + np.sum(null_r2 >= observed_r2)) / (N_PERMUTATIONS + 1))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Building feature sets...")
    feature_sets = build_feature_sets()

    print("\nLoading targets...")
    targets = load_targets()

    results: list[dict[str, object]] = []
    for family, feat_df in feature_sets.items():
        feature_cols = [c for c in feat_df.columns if c not in ("group_id", "task_id")]
        for target, tier_tasks in TARGET_TASKS.items():
            if target not in targets.columns:
                continue
            tier = _tier(tier_tasks)
            target_df = targets.loc[targets["task_id"].isin(tier_tasks), ["group_id", "task_id", target]]
            merged = target_df.merge(feat_df, on=["group_id", "task_id"], how="inner")
            complete = merged.dropna(subset=[target] + feature_cols)
            n_groups = complete["group_id"].nunique()

            if len(complete) < MIN_GROUPS_TO_MODEL or n_groups < MIN_GROUPS_TO_MODEL:
                results.append({
                    "feature_set": family, "target": target, "tier": tier,
                    "n_effective": len(complete), "n_groups": n_groups,
                    "n_features": len(feature_cols), "r2": np.nan, "mae": np.nan,
                    "spearman_rho": np.nan, "spearman_p": np.nan, "perm_p": np.nan,
                    "note": f"skipped: fewer than {MIN_GROUPS_TO_MODEL} complete-case groups",
                })
                continue

            x = complete[feature_cols].to_numpy(dtype=float)
            y = complete[target].to_numpy(dtype=float)
            groups = complete["group_id"].to_numpy()

            oof_pred, r2 = logo_cv_ridge(x, y, groups)
            mae = mean_absolute_error(y, oof_pred)
            rho, rho_p = spearmanr(y, oof_pred)
            perm_p = permutation_p_value(x, y, groups, r2, rng)

            results.append({
                "feature_set": family, "target": target, "tier": tier,
                "n_effective": len(complete), "n_groups": n_groups,
                "n_features": len(feature_cols), "r2": r2, "mae": mae,
                "spearman_rho": rho, "spearman_p": rho_p, "perm_p": perm_p,
                "note": "",
            })
            print(
                f"  {family:22s} -> {target:28s} [{tier.split(' ')[0]:11s}] "
                f"n={len(complete):2d} groups={n_groups:2d} R2={r2:+.3f} rho={rho:+.3f} perm_p={perm_p:.3f}"
            )

    results_df = pd.DataFrame(results).sort_values(["tier", "feature_set", "target"])
    results_df.to_csv(OUTPUT_DIR / "prediction_primary_results.tsv", sep="\t", index=False)

    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_permutations": N_PERMUTATIONS,
        "min_groups_to_model": MIN_GROUPS_TO_MODEL,
        "validation_scheme": "GroupKFold(n_splits=n_groups) == leave-one-group-out",
        "estimator": "Pipeline(StandardScaler(), RidgeCV(alphas=logspace(-2,3,12))), fit per fold",
        "missing_data_policy": "complete-case (listwise) per feature-set x target; no imputation",
        "input_matrix": str(COLLECTIVE_DIR.relative_to(REPO_ROOT)),
        "feature_families": {
            family: [c for c in df.columns if c not in ("group_id", "task_id")]
            for family, df in feature_sets.items()
        },
        "target_task_availability": TARGET_TASKS,
        "python_version": platform.python_version(),
        "known_gaps": [
            "Fold-local HMM fitting not yet implemented (see docs/prediction_analysis_backlog.md Priority 1).",
            "Lasso/ElasticNet alternative estimators not yet rerun on this matrix.",
            "No multiple-comparison correction across the feature-set x target grid; "
            "perm_p values here are per-comparison and exploratory.",
        ],
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nWrote {OUTPUT_DIR / 'prediction_primary_results.tsv'} ({len(results_df)} rows)")
    print(f"Wrote {OUTPUT_DIR / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
