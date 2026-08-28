"""Generate publication-ready figures for the GroupAffect-4 dataset paper.

This script reads from results/ TSV files and writes figures to figures/.
Run with: python tools/generate_paper_figures.py [--figures all|main|appendix]

Main-paper figures generated:
  F1  Benchmark results panel (B0â€“Bs AUC/accuracy vs baseline)
  F2  VAD manipulation-check panel (valence/arousal by task with Cohen's d)

Appendix figures generated:
  F3  Preprocessing effect panel (d before/after within-person z-score)
  F4  Group speech-balance dynamics (Gini by task)
  F5  Multimodal task profiles panel (radar/heatmap of normalised features)

Privacy: all identifiers shown in outputs are seat IDs (P1â€“P4) or group IDs
(grp-XX) only.  No real participant names are written to disk.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")  # non-interactive backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

# Palette consistent with paper (colourblind-friendly)
TASK_COLORS = {
    "T0": "#6baed6",  # light blue  â€“ baseline
    "T1": "#74c476",  # green       â€“ hidden-profile
    "T2": "#e6550d",  # orange-red  â€“ negotiation (emotional task)
    "T3": "#9ecae1",  # mid blue    â€“ idea generation
    "T4": "#a1d99b",  # light green â€“ public-goods
}
TASK_LABELS = {
    "T0": "T0\nBaseline",
    "T1": "T1\nHidden-Profile",
    "T2": "T2\nNegotiation",
    "T3": "T3\nIdea-Gen",
    "T4": "T4\nPublic-Goods",
}
BENCHMARK_PALETTE = {"baseline": "#bdbdbd", "model": "#2171b5"}

RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved â†’ %s", out_path.relative_to(REPO_ROOT))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d (two independent groups)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_sd = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# F1 â€” Benchmark results panel
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _f1_benchmark_panel(out_path: Path) -> None:
    """Bar chart of AUC / accuracy for each benchmark (baseline vs model)."""
    tsv = RESULTS / "benchmarks" / "logocv_baselines_preprocessed.tsv"
    if not tsv.exists():
        log.warning("F1: %s not found â€“ skipping", tsv)
        return

    df = pd.read_csv(tsv, sep="\t")

    # â”€â”€ select one primary metric per benchmark / target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rows = []

    def _pick(bid: str, tgt: str, pref_metric: str) -> None:
        sub = df[(df.benchmark_id == bid) & (df.target == tgt)]
        row = sub[sub.metric == pref_metric]
        if row.empty:
            return
        row = row.iloc[0]
        rows.append(
            {
                "label": tgt,
                "metric": pref_metric,
                "baseline": float(row.baseline) if str(row.baseline) not in ("", "nan") else float("nan"),
                "model": float(row.linear_model),
                "bid": bid,
            }
        )

    _pick("B0_task_classification", "Task label (T1--T4)", "accuracy")
    _pick("B1_affective_binary", "Valence (high/low)", "roc_auc")
    _pick("B1_affective_binary", "Arousal (high/low)", "roc_auc")
    _pick("B2_dominance_binary", "Dominance (high/low)", "roc_auc")
    _pick("B3_cognitive_state", "Engagement >= median", "roc_auc")
    _pick("B3_cognitive_state", "Mental demand >= median", "roc_auc")
    _pick("B4_personality_binary", "BFI Extraversion (high/low)", "roc_auc")
    _pick("B4_personality_binary", "BFI Openness (high/low)", "roc_auc")
    _pick("B5_performance_binary", "T4 Contribution (high/low)", "roc_auc")

    if not rows:
        log.warning("F1: no rows extracted â€“ skipping")
        return

    res = pd.DataFrame(rows)

    # Short display labels
    label_map = {
        "Task label (T1--T4)": "B0 Task class.\n(acc.)",
        "Valence (high/low)": "B1 Valence\n(AUC)",
        "Arousal (high/low)": "B1 Arousal\n(AUC)",
        "Dominance (high/low)": "B2 Dominance\n(AUC)",
        "Engagement >= median": "B3 Engagement\n(AUC)",
        "Mental demand >= median": "B3 Mental dem.\n(AUC)",
        "BFI Extraversion (high/low)": "B4 Extraversion\n(AUC)",
        "BFI Openness (high/low)": "B4 Openness\n(AUC)",
        "T4 Contribution (high/low)": "B5 Contribution\n(AUC)",
    }
    res["disp"] = res["label"].map(label_map).fillna(res["label"])
    res.loc[res["metric"] == "roc_auc", "baseline"] = res.loc[
        res["metric"] == "roc_auc", "baseline"
    ].fillna(0.5)

    with plt.rc_context(RCPARAMS):
        fig, ax = plt.subplots(figsize=(8.5, 3.2))

        x = np.arange(len(res))
        w = 0.35

        b1 = ax.bar(
            x - w / 2,
            res["baseline"].fillna(0),
            width=w,
            color=BENCHMARK_PALETTE["baseline"],
            label="Chance / mean baseline",
            zorder=2,
        )
        b2 = ax.bar(
            x + w / 2,
            res["model"],
            width=w,
            color=BENCHMARK_PALETTE["model"],
            label="Ridge (LOGO-CV)",
            zorder=2,
        )

        # Add value labels on bars
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if abs(h) > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#333333",
                )

        # Chance line for AUC (0.5) and accuracy guides
        ax.axhline(0.5, color="#e6550d", lw=0.8, ls="--", alpha=0.7, label="AUC chance (0.5)")
        ax.axhline(0.0, color="black", lw=0.5, ls="-", alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(res["disp"], fontsize=7.5)
        ax.set_ylabel("Performance (AUC / accuracy)")
        ax.set_title(
            "Benchmark feasibility baselines (LOGO-CV, preprocessed features)",
            fontsize=9,
        )
        ax.set_ylim(-0.25, 0.95)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(loc="upper right", fontsize=7.5)

        # Colour-code x-axis labels by benchmark category
        bid_to_color = {
            "B0_task_classification": "#1a9850",
            "B1_affective_binary": "#d73027",
            "B2_dominance_binary": "#4575b4",
            "B3_cognitive_state": "#9970ab",
            "B4_personality_binary": "#878787",
            "B5_performance_binary": "#878787",
        }
        for tick, (_, row) in zip(ax.get_xticklabels(), res.iterrows()):
            tick.set_color(bid_to_color.get(row["bid"], "black"))

        fig.tight_layout()
        _save(fig, out_path)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# F2 â€” VAD manipulation-check panel
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _f2_vad_manipulation_check(out_path: Path) -> None:
    """Valence & arousal by task (violin + meanÂ±CI) with d annotations."""
    pt_tsv = RESULTS / "task_responses" / "vad_participant_task.tsv"
    summary_tsv = RESULTS / "task_responses" / "vad_by_task.tsv"

    if not pt_tsv.exists():
        log.warning("F2: %s not found â€“ skipping", pt_tsv)
        return

    pt = pd.read_csv(pt_tsv, sep="\t")
    summary = pd.read_csv(summary_tsv, sep="\t")
    tasks = ["T0", "T1", "T2", "T3", "T4"]
    pt = pt[pt["task"].isin(tasks)]

    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=False)

        for ax, col, title, ref_task in [
            (axes[0], "valence", "Valence", "T0"),
            (axes[1], "arousal", "Arousal", "T0"),
        ]:
            vals_per_task = [
                pt.loc[pt["task"] == t, col].dropna().values for t in tasks
            ]

            # violin
            parts = ax.violinplot(
                vals_per_task,
                positions=range(len(tasks)),
                widths=0.6,
                showmedians=False,
                showextrema=False,
            )
            colors = [TASK_COLORS[t] for t in tasks]
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.55)
                pc.set_edgecolor("none")

            # mean Â± 95 % CI
            for i, (t, vals) in enumerate(zip(tasks, vals_per_task)):
                if len(vals) < 2:
                    continue
                m = np.mean(vals)
                se = stats.sem(vals)
                ci = se * stats.t.ppf(0.975, df=len(vals) - 1)
                ax.errorbar(
                    i, m, yerr=ci,
                    fmt="D",
                    color="black",
                    markersize=4,
                    capsize=3,
                    lw=1.2,
                    zorder=5,
                )

            # Cohen's d vs T0 annotations above each active task
            ref_vals = pt.loc[pt["task"] == ref_task, col].dropna().values
            for i, t in enumerate(tasks[1:], start=1):
                task_vals = pt.loc[pt["task"] == t, col].dropna().values
                d = _cohens_d(task_vals, ref_vals)
                if not np.isnan(d):
                    ax.text(
                        i,
                        9.5,
                        f"d={d:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color="#333333",
                    )

            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=7.5)
            ax.set_ylabel("SAM score (1â€“9)")
            ax.set_title(f"{title} by task", fontsize=9)
            ax.set_ylim(1, 10.5)
            ax.set_yticks([1, 3, 5, 7, 9])
            ax.grid(axis="y", alpha=0.3, zorder=0)

        # T2 annotation arrow
        ax0 = axes[0]
        T2_idx = tasks.index("T2")
        t2_row = summary[summary["task"] == "T2"]
        t2_mean = float(t2_row["valence_mean"].iloc[0]) if not t2_row.empty else 5.72
        ax0.annotate(
            f"T2: mean={t2_mean:.2f}\nlowest valence block",
            xy=(T2_idx, t2_mean),
            xytext=(T2_idx + 0.6, t2_mean - 1.3),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="#e6550d", lw=1.0),
            color="#e6550d",
        )

        fig.suptitle(
            "Affective self-report by task (SAM; n=28â€“38 per task)",
            fontsize=9,
            y=1.01,
        )
        fig.tight_layout()
        _save(fig, out_path)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# F3 â€” Preprocessing effect panel  (appendix)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _f3_preprocessing_effect(out_path: Path) -> None:
    """Cohen's d (T2 vs T1) before and after within-person z-score."""
    # Data from PREPROCESSING_REPORT.md (hard-coded audit values)
    # d(T2,T1) raw vs preprocessed for key features
    features = [
        "HR mean (bpm)",
        "HRV RMSSD (ms)",
        "EDA tonic (ÂµS)",
        "EDA SCR rate (Hz)",
        "Skin temp (Â°C)",
        "Pupil diam. (mm)",
        "Speaking fraction",
    ]
    d_raw = [0.159, 0.329, 0.244, 0.130, -0.005, 0.280, 0.440]
    d_prep = [0.318, 0.280, -0.018, 0.268, -0.066, 1.130, 0.880]

    # d(T3 vs T1) raw vs preprocessed
    features_t3 = features[:5]
    d_raw_t3 = [0.304, 0.206, 0.289, 0.091, -0.094]
    d_prep_t3 = [0.362, -0.378, -0.294, -0.017, -0.486]

    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))

        for ax, feats, dr, dp, contrast in [
            (axes[0], features, d_raw, d_prep, "T2 vs T1 (negotiation vs reading)"),
            (axes[1], features_t3, d_raw_t3, d_prep_t3, "T3 vs T1 (idea-gen vs reading)"),
        ]:
            y = np.arange(len(feats))
            ax.barh(y - 0.18, dr, height=0.35, color="#bdbdbd", label="Raw features", zorder=2)
            ax.barh(y + 0.18, dp, height=0.35, color="#2171b5", label="Within-person z-score", zorder=2)
            ax.axvline(0, color="black", lw=0.7)
            ax.axvspan(xmin=-0.2, xmax=0.2, alpha=0.06, color="grey", zorder=0)
            ax.set_yticks(y)
            ax.set_yticklabels(feats, fontsize=8)
            ax.set_xlabel("Cohen's d")
            ax.set_title(contrast, fontsize=8.5)
            ax.grid(axis="x", alpha=0.3, zorder=0)
            if ax is axes[0]:
                ax.legend(loc="lower right", fontsize=7.5)

        fig.suptitle(
            "Effect of within-person robust z-score preprocessing on task separation",
            fontsize=9,
        )
        fig.tight_layout()
        _save(fig, out_path)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# F4 â€” Group speech-balance dynamics  (appendix)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _f4_speech_balance(out_path: Path) -> None:
    """Gini coefficient and dominant-speaker share by task (all groups)."""
    tsv = RESULTS / "expanded_stats" / "group_speech_dynamics.tsv"
    if not tsv.exists():
        log.warning("F4: %s not found â€“ skipping", tsv)
        return

    df = pd.read_csv(tsv, sep="\t")
    tasks = ["T0", "T1", "T2", "T3", "T4"]
    df = df[df["task_id"].isin(tasks)]

    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))

        for ax, col, ylabel, title in [
            (axes[0], "gini_speaking", "Gini coefficient", "Speaking inequality (Gini)"),
            (
                axes[1],
                "dominant_speaker_share",
                "Fraction of total speech",
                "Dominant-speaker share",
            ),
        ]:
            vals_per_task = [df.loc[df["task_id"] == t, col].dropna().values for t in tasks]
            colors = [TASK_COLORS[t] for t in tasks]

            parts = ax.violinplot(
                vals_per_task,
                positions=range(len(tasks)),
                widths=0.55,
                showmedians=True,
                showextrema=False,
            )
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.6)
                pc.set_edgecolor("none")
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_lw(1.5)

            # Group-level means
            means = [np.mean(v) if len(v) > 0 else np.nan for v in vals_per_task]
            ax.plot(range(len(tasks)), means, "k--o", ms=4, lw=1.2, zorder=5)

            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=7.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3, zorder=0)

        # add T0 caveat note
        axes[0].text(
            0,
            0.97,
            "T0 baseline artefact\n(lapel VAD inflation)",
            ha="center",
            va="top",
            fontsize=6,
            color="#e6550d",
            transform=axes[0].get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e6550d", alpha=0.7),
        )

        fig.suptitle(
            "Group speech-balance dynamics across tasks (n=10 groups)",
            fontsize=9,
        )
        fig.tight_layout()
        _save(fig, out_path)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# F5 â€” Multimodal task profiles  (appendix)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _f5_multimodal_task_profiles(out_path: Path) -> None:
    """Heatmap of mean normalised features across tasks (T0â€“T4)."""
    tsv = RESULTS / "expanded_stats" / "multimodal_task_profiles.tsv"
    if not tsv.exists():
        log.warning("F5: %s not found â€“ skipping", tsv)
        return

    df = pd.read_csv(tsv, sep="\t")
    tasks = ["T0", "T1", "T2", "T3", "T4"]

    # Select interpretable delta features present in the file
    wanted_cols = [
        "vad_valence_self_num", "vad_arousal_self_num",
        "hr_mean_bpm_delta_t0", "eda_tonic_mean_delta_t0",
        "eda_phasic_rate_hz_delta_t0", "temp_mean_delta_t0",
        "pupil_mean_delta_t0",
        "audio_speaking_fraction_delta_t0", "audio_overlap_fraction_delta_t0",
        "audio_pitch_mean_delta_t0",
    ]
    feat_labels = [
        "VAD Valence", "VAD Arousal",
        "Î”HR (bpm)", "Î”EDA tonic",
        "Î”SCR rate", "Î”Skin temp",
        "Î”Pupil diam.",
        "Î”Speaking frac.", "Î”Overlap frac.",
        "Î”Pitch mean",
    ]

    present = [c for c in wanted_cols if c in df.columns]
    feat_labels_used = [feat_labels[wanted_cols.index(c)] for c in present]

    if not present:
        log.warning("F5: no matching feature columns â€“ skipping")
        return

    # Build task Ã— feature mean matrix
    matrix = np.full((len(tasks), len(present)), np.nan)
    for i, t in enumerate(tasks):
        sub = df[df["task"] == t] if "task" in df.columns else pd.DataFrame()
        if sub.empty and "task_id" in df.columns:
            sub = df[df["task_id"] == t]
        for j, col in enumerate(present):
            vals = sub[col].dropna()
            if len(vals) > 0:
                matrix[i, j] = vals.mean()

    # z-score each column across tasks for visual clarity
    matrix_z = np.full_like(matrix, np.nan)
    for j in range(matrix.shape[1]):
        col_vals = matrix[:, j]
        valid = ~np.isnan(col_vals)
        if valid.sum() >= 2:
            mu, sd = col_vals[valid].mean(), col_vals[valid].std()
            if sd > 1e-9:
                matrix_z[valid, j] = (col_vals[valid] - mu) / sd

    with plt.rc_context(RCPARAMS):
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        im = ax.imshow(matrix_z.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=8)
        ax.set_yticks(range(len(feat_labels_used)))
        ax.set_yticklabels(feat_labels_used, fontsize=7.5)
        ax.set_title(
            "Task-level multimodal feature profiles (z-scored across tasks)", fontsize=9
        )

        # Value annotations
        for i in range(len(tasks)):
            for j in range(len(present)):
                val = matrix_z[i, j]
                if not np.isnan(val):
                    ax.text(
                        i, j, f"{val:.1f}",
                        ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(val) > 1.2 else "black",
                    )

        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("z-score", fontsize=8)
        fig.tight_layout()
        _save(fig, out_path)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# F6 â€” Modality coverage heatmap (enhance existing, appendix-ready version)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _f6_modality_coverage(out_path: Path) -> None:
    """Per-session Ã— modality coverage fractions as annotated heatmap."""
    tsv = RESULTS / "dataset_stats" / "usability_by_task.tsv"
    if not tsv.exists():
        log.warning("F6: %s not found â€“ skipping", tsv)
        return

    df = pd.read_csv(tsv, sep="\t")

    # Expect columns: session or group, modality, task, fraction_usable
    # Adapt column names robustly
    group_col = next((c for c in df.columns if "group" in c.lower() or "session" in c.lower()), None)
    mod_col = next((c for c in df.columns if "modal" in c.lower() or "stream" in c.lower() or "modality" in c.lower()), None)
    task_col = next((c for c in df.columns if "task" in c.lower()), None)
    frac_col = next((c for c in df.columns if "frac" in c.lower() or "usab" in c.lower() or "coverage" in c.lower()), None)

    if any(x is None for x in [group_col, mod_col, task_col, frac_col]):
        log.warning("F6: cannot identify columns in %s â€“ columns: %s", tsv, list(df.columns))
        return

    pivot = df.pivot_table(
        values=frac_col, index=mod_col, columns=task_col, aggfunc="mean"
    ).fillna(0)

    with plt.rc_context(RCPARAMS):
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r", vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7.5)
        ax.set_title("Modality usability by task (fraction of participant-task rows)", fontsize=9)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                ax.text(
                    j, i, f"{v:.0%}",
                    ha="center", va="center",
                    fontsize=6.5,
                    color="white" if v < 0.4 else "black",
                )

        fig.colorbar(im, ax=ax, shrink=0.8, label="Usable fraction")
        fig.tight_layout()
        _save(fig, out_path)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

FIGURE_REGISTRY: dict[str, tuple] = {
    "F1": (
        _f1_benchmark_panel,
        FIGURES / "benchmarks" / "benchmark_panel.png",
        "main",
    ),
    "F2": (
        _f2_vad_manipulation_check,
        FIGURES / "task_responses" / "vad_manipulation_check.png",
        "main",
    ),
    "F3": (
        _f3_preprocessing_effect,
        FIGURES / "preprocessing" / "preprocessing_effect_panel.png",
        "appendix",
    ),
    "F4": (
        _f4_speech_balance,
        FIGURES / "audio" / "group_speech_dynamics.png",
        "appendix",
    ),
    "F5": (
        _f5_multimodal_task_profiles,
        FIGURES / "expanded_stats" / "multimodal_task_radar.png",
        "appendix",
    ),
    "F6": (
        _f6_modality_coverage,
        FIGURES / "dataset_stats" / "usability_by_task_enhanced.png",
        "appendix",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--figures",
        default="all",
        choices=["all", "main", "appendix"] + list(FIGURE_REGISTRY.keys()),
        help="Which figures to generate (default: all).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Debug-level logging."
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sel = args.figures
    for fid, (fn, path, category) in FIGURE_REGISTRY.items():
        if sel in ("all", category, fid):
            log.info("Generating %s (%s) â€¦", fid, category)
            try:
                fn(path)
            except Exception as exc:
                log.error("Error generating %s: %s", fid, exc, exc_info=args.verbose)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

