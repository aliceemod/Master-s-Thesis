"""Dialogue act feature extraction — runs standalone, logs to file, safe to leave overnight."""
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no popup windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # must be set before importing transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"    # suppress transformers warnings/progress
os.environ["TQDM_DISABLE"] = "1"                  # disable all tqdm bars (they hang in hidden processes)
from transformers import pipeline

REPO_ROOT = Path(__file__).parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "transcripts" / "final"
OUT_TSV   = REPO_ROOT / "analysis" / "results" / "da_window_30s.tsv"
OUT_PLOT1 = REPO_ROOT / "analysis" / "results" / "da_task_profile.png"
OUT_PLOT2 = REPO_ROOT / "analysis" / "results" / "da_group_heatmap.png"
OUT_PLOT3 = REPO_ROOT / "analysis" / "results" / "da_eta2.png"
LOG_FILE  = REPO_ROOT / "analysis" / "results" / "da_run_log.txt"

OUT_TSV.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

WINDOW_S     = 30.0
PARTICIPANTS = ("P1", "P2", "P3", "P4")
MODEL        = "MoritzLaurer/deberta-v3-base-zeroshot-v1"  # ~500MB, fast, purpose-built for zero-shot
BATCH_SIZE   = 32

DA_LABELS = {
    "statement":    "expressing an opinion, asserting a fact, or sharing a personal view",
    "proposal":     "suggesting a new option, plan, or course of action for the group to consider",
    "agreement":    "endorsing, accepting, or affirming a prior idea or proposal from someone else",
    "disagreement": "rejecting, opposing, or pushing back against an idea or proposal",
    "compromise":   "proposing to blend or combine competing options to reach common ground",
    "question":     "asking for information, clarification, or a decision from the group",
    "answer":       "responding directly to a question that was just asked",
    "backchannel":  "giving a brief acknowledgment like mm-hmm or yeah to signal attention without adding new content",
    "hedge":        "expressing uncertainty or tentatively softening a claim with words like maybe or probably",
    "decision":     "formally or informally moving the group toward a final decision or agreement",
    "offtask":      "making a joke, engaging in social talk, or saying something unrelated to the current task",
}

DA_SHORT      = list(DA_LABELS.keys())
DA_HYPOTHESES = list(DA_LABELS.values())
DA_PROB_COLS  = [f"da_{k}" for k in DA_SHORT]
HYP_TO_SHORT  = {v: k for k, v in DA_LABELS.items()}

# ── Load transcripts ────────────────────────────────────────────────────────

_FNAME_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)

def _normalize_group_id(raw: str) -> str:
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"

log.info("Loading transcripts from %s", TRANSCRIPT_DIR)
all_events = []
for fpath in TRANSCRIPT_DIR.glob("*.tsv"):
    match = _FNAME_RE.search(fpath.name)
    if not match:
        continue
    df = pd.read_csv(fpath, sep="\t")
    df["group_id"] = _normalize_group_id(match.group(1))
    df["task_id"]  = match.group(2).upper()
    all_events.append(df)

events_df = pd.concat(all_events, ignore_index=True)
log.info("Loaded %d transcript rows", len(events_df))

spk_events = events_df[
    (events_df["type"] == "SPK") &
    (events_df["speaker"].isin(PARTICIPANTS)) &
    (events_df["text"].notna()) &
    (events_df["text"].str.strip() != "")
].copy()
spk_events["window_index"] = (spk_events["onset"] / WINDOW_S).apply(
    lambda x: int(x) if pd.notna(x) and x >= 0 else -1
)
spk_events = spk_events[spk_events["window_index"] >= 0].reset_index(drop=True)
log.info("Utterances to classify: %d", len(spk_events))

# ── Load model ──────────────────────────────────────────────────────────────

log.info("Loading model %s (downloading if first run) ...", MODEL)
zsc = pipeline("zero-shot-classification", model=MODEL, device=-1)
log.info("Model ready.")

# ── Classify ────────────────────────────────────────────────────────────────

utterances = spk_events["text"].tolist()
results = []
n_batches = (len(utterances) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_i in range(n_batches):
    start = batch_i * BATCH_SIZE
    batch = utterances[start : start + BATCH_SIZE]
    batch_results = zsc(batch, candidate_labels=DA_HYPOTHESES, multi_label=False)
    if isinstance(batch_results, dict):
        batch_results = [batch_results]
    results.extend(batch_results)
    if (batch_i + 1) % 5 == 0 or batch_i == n_batches - 1:
        log.info("  batch %d / %d  (%d utterances done)", batch_i + 1, n_batches, len(results))

log.info("Classification done.")

# ── Build utterance DA frame ─────────────────────────────────────────────────

utt_rows = []
for i, res in enumerate(results):
    row = {"utt_idx": i}
    for label, score in zip(res["labels"], res["scores"]):
        row[f"da_{HYP_TO_SHORT[label]}"] = score
    row["da_dominant"] = HYP_TO_SHORT[res["labels"][0]]
    utt_rows.append(row)

utt_da_df = pd.DataFrame(utt_rows)
log.info("Dominant act distribution:\n%s", utt_da_df["da_dominant"].value_counts().to_string())

# ── Aggregate per window ─────────────────────────────────────────────────────

spk_with_da = pd.concat(
    [spk_events[["group_id", "task_id", "window_index", "speaker"]].reset_index(drop=True),
     utt_da_df.drop(columns="utt_idx")],
    axis=1
)

win_rows = []
for (grp, task, win), win_df in spk_with_da.groupby(["group_id", "task_id", "window_index"]):
    row = {
        "group_id": grp, "task_id": task, "window_index": win,
        "da_n_utterances": len(win_df),
        "da_n_speakers": win_df["speaker"].nunique(),
    }
    for col in DA_PROB_COLS:
        row[col] = win_df[col].mean()
    row["da_dominant"] = win_df["da_dominant"].mode().iloc[0]
    win_rows.append(row)

da_df = pd.DataFrame(win_rows)
da_df.to_csv(OUT_TSV, sep="\t", index=False)
log.info("Saved %d windows → %s", len(da_df), OUT_TSV)

# ── Plots (saved to PNG, no popup) ───────────────────────────────────────────

task_profile = da_df.groupby("task_id")[DA_PROB_COLS].mean()
fig, ax = plt.subplots(figsize=(13, 5))
task_profile.T.plot(kind="bar", ax=ax, colormap="tab10")
ax.set_xticklabels([c.replace("da_", "") for c in DA_PROB_COLS], rotation=35, ha="right")
ax.set_title("Dialogue Act Profile by Task")
ax.set_ylabel("Mean probability")
ax.legend(title="Task")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(OUT_PLOT1, dpi=150)
log.info("Saved %s", OUT_PLOT1)

grp_profile = da_df.groupby("group_id")[DA_PROB_COLS].mean()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(grp_profile, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=[c.replace("da_", "") for c in DA_PROB_COLS], ax=ax)
ax.set_title("Mean Dialogue Act Profile per Group")
plt.tight_layout()
fig.savefig(OUT_PLOT2, dpi=150)
log.info("Saved %s", OUT_PLOT2)

eta_rows = []
for col in DA_PROB_COLS:
    vals = da_df[col].values
    grand_mean = vals.mean()
    ss_total = ((vals - grand_mean) ** 2).sum()
    if ss_total == 0:
        continue
    task_masks = [(da_df["task_id"] == t).values for t in da_df["task_id"].unique()]
    ss_task = sum((vals[m].mean() - grand_mean) ** 2 * m.sum() for m in task_masks)
    grp_masks = [(da_df["group_id"] == g).values for g in da_df["group_id"].unique()]
    ss_grp = sum((vals[m].mean() - grand_mean) ** 2 * m.sum() for m in grp_masks)
    eta_rows.append({"Feature": col.replace("da_", ""),
                     "η²_task": ss_task / ss_total, "η²_group": ss_grp / ss_total})

eta_df = pd.DataFrame(eta_rows)
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(eta_df))
w = 0.35
ax.bar(x - w/2, eta_df["η²_task"],  w, label="η² task",  color="steelblue")
ax.bar(x + w/2, eta_df["η²_group"], w, label="η² group", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(eta_df["Feature"], rotation=30, ha="right")
ax.axhline(0.05, color="grey", linestyle="--", alpha=0.6)
ax.set_title("DA Features: variance explained by task vs. group")
ax.set_ylabel("η²")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(OUT_PLOT3, dpi=150)
log.info("Saved %s", OUT_PLOT3)

log.info("ALL DONE.")
