"""Focused analysis of task-level tension, with emphasis on Task 2.

Uses transcript-derived conflict features and exports:
- task2_conflict_boxplots.png
- task2_conflict_bootstrap_means.png
- task2_conflict_stats.tsv
- task2_pairwise_tests.tsv

Default input: features/collective_window_features.tsv
Default output: figures/task2_focus/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


def _bootstrap_ci_mean(values: np.ndarray, n_boot: int = 4000, seed: int = 42) -> tuple[float, float]:
    """Return 95% bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (np.nan, np.nan)
    boots = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = float(np.mean(sample))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _fdr_bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction for a list of p-values."""
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        idx, p = indexed[rank - 1]
        val = min(prev, (p * n) / rank)
        adj[idx] = val
        prev = val
    return adj


def run_analysis(input_file: Path, out_dir: Path, n_boot: int) -> None:
    """Run Task 2 tension analysis and save figures/tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    df = pd.read_csv(input_file, sep="\t")
    df = df[df["task_id"].isin(["T1", "T2", "T3"])].copy()
    df = df.dropna(subset=["tr_spk_duration_s"]).copy()

    feature_cols = [
        "tr_competitive_overlap",
        "tr_cooperative_overlap",
        "tr_collaborative_overlap",
        "tr_needs_review_overlap",
        "tr_conflict_overlap_ratio",
        "tr_overlap_count",
        "tr_overlap_time_s",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Long format for plotting
    long_df = df[["task_id"] + feature_cols].melt(
        id_vars=["task_id"],
        value_vars=feature_cols,
        var_name="feature",
        value_name="value",
    )

    # Summary stats + bootstrap CI
    stat_rows: list[dict[str, object]] = []
    for feat in feature_cols:
        for task in ["T1", "T2", "T3"]:
            vals = pd.to_numeric(
                df.loc[df["task_id"] == task, feat], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            ci_lo, ci_hi = _bootstrap_ci_mean(vals, n_boot=n_boot)
            stat_rows.append(
                {
                    "feature": feat,
                    "task_id": task,
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    "median": float(np.median(vals)) if len(vals) else np.nan,
                    "ci95_mean_low": ci_lo,
                    "ci95_mean_high": ci_hi,
                }
            )

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(out_dir / "task2_conflict_stats.tsv", sep="\t", index=False)

    # Pairwise tests (Mann-Whitney U) for each feature
    pairs = [("T2", "T1"), ("T2", "T3"), ("T1", "T3")]
    test_rows: list[dict[str, object]] = []
    for feat in feature_cols:
        for a, b in pairs:
            a_vals = pd.to_numeric(
                df.loc[df["task_id"] == a, feat], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            b_vals = pd.to_numeric(
                df.loc[df["task_id"] == b, feat], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if len(a_vals) > 0 and len(b_vals) > 0:
                stat, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            else:
                stat, p = np.nan, np.nan
            test_rows.append(
                {
                    "feature": feat,
                    "task_a": a,
                    "task_b": b,
                    "n_a": len(a_vals),
                    "n_b": len(b_vals),
                    "mean_a": float(np.mean(a_vals)) if len(a_vals) else np.nan,
                    "mean_b": float(np.mean(b_vals)) if len(b_vals) else np.nan,
                    "delta_mean_a_minus_b": (
                        float(np.mean(a_vals) - np.mean(b_vals))
                        if len(a_vals) and len(b_vals)
                        else np.nan
                    ),
                    "u_stat": float(stat) if stat is not np.nan else np.nan,
                    "p_value": float(p) if p is not np.nan else np.nan,
                }
            )

    tests_df = pd.DataFrame(test_rows)
    valid_mask = tests_df["p_value"].notna()
    adj_vals = _fdr_bh(tests_df.loc[valid_mask, "p_value"].astype(float).tolist())
    tests_df["p_value_fdr_bh"] = np.nan
    tests_df.loc[valid_mask, "p_value_fdr_bh"] = adj_vals
    tests_df.to_csv(out_dir / "task2_pairwise_tests.tsv", sep="\t", index=False)

    # Figure 1: boxplots
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(5 * len(feature_cols), 4), sharex=False)
    if len(feature_cols) == 1:
        axes = [axes]

    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        sub = long_df[long_df["feature"] == feat]
        sns.boxplot(data=sub, x="task_id", y="value", ax=ax, order=["T1", "T2", "T3"])
        sns.stripplot(
            data=sub,
            x="task_id",
            y="value",
            ax=ax,
            order=["T1", "T2", "T3"],
            color="black",
            alpha=0.25,
            size=2,
            jitter=0.15,
        )
        ax.set_title(feat)
        ax.set_xlabel("task")

    fig.suptitle("Conflict Features by Task (Window-Level)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "task2_conflict_boxplots.png", dpi=220)
    plt.close(fig)

    # Figure 2: means + bootstrap CI
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(5 * len(feature_cols), 4), sharex=False)
    if len(feature_cols) == 1:
        axes = [axes]

    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        sub = stats_df[stats_df["feature"] == feat].set_index("task_id").loc[["T1", "T2", "T3"]]
        x = np.arange(len(sub))
        y = sub["mean"].to_numpy()
        lo = sub["ci95_mean_low"].to_numpy()
        hi = sub["ci95_mean_high"].to_numpy()
        yerr = np.vstack([y - lo, hi - y])
        ax.bar(x, y, color=["#4e9af1", "#f1a94e", "#7bc950"], alpha=0.85)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["T1", "T2", "T3"])
        ax.set_title(feat)
        ax.set_xlabel("task")
        ax.set_ylabel("mean ± 95% bootstrap CI")

    fig.suptitle("Task Means with Bootstrap 95% CI", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "task2_conflict_bootstrap_means.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("features") / "collective_window_features.tsv",
        help="Window-level feature file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "task2_focus",
        help="Output directory for figures/tables.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=4000,
        help="Number of bootstrap resamples for CI.",
    )
    args = parser.parse_args()

    run_analysis(args.input_file.resolve(), args.out_dir.resolve(), args.n_boot)


if __name__ == "__main__":
    main()
