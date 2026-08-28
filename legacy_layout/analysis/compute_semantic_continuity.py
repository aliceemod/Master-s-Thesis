"""Compute per-window semantic continuity: mean cosine similarity between consecutive speakers.
Also computes a zero-shot completion score for each overlap candidate using the cached DeBERTa model.
Outputs:
  analysis/results/semantic_continuity_window_30s.tsv  — window-level tr_semantic_continuity
  analysis/results/completion_scores_nli.tsv           — per-candidate NLI completion probability
"""
import logging
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "transcripts" / "final"
WINDOW_S = 30.0
PARTICIPANTS = ("P1", "P2", "P3", "P4")

_FNAME_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)

def norm_grp(raw):
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"

# ── Load transcripts ──────────────────────────────────────────────────────────
log.info("Loading transcripts...")
all_events = []
for fpath in TRANSCRIPT_DIR.glob("*.tsv"):
    m = _FNAME_RE.search(fpath.name)
    if not m:
        continue
    df = pd.read_csv(fpath, sep="\t")
    df["group_id"] = norm_grp(m.group(1))
    df["task_id"]  = m.group(2).upper()
    all_events.append(df)

events_df = pd.concat(all_events, ignore_index=True)
spk = events_df[
    (events_df["type"] == "SPK") &
    (events_df["speaker"].isin(PARTICIPANTS)) &
    (events_df["text"].notna()) &
    (events_df["text"].str.strip() != "") &
    (events_df["task_id"] != "T4")
].copy()
spk["text"] = spk["text"].str.strip()
spk["window_index"] = (spk["onset"] / WINDOW_S).apply(
    lambda x: int(x) if pd.notna(x) and x >= 0 else -1
)
spk = spk[spk["window_index"] >= 0].sort_values(["group_id", "task_id", "onset"]).reset_index(drop=True)
log.info("Utterances: %d", len(spk))

# ── Load sentence-BERT (already cached) ──────────────────────────────────────
log.info("Loading sentence-BERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")
log.info("Model ready. Embedding %d utterances...", len(spk))
embeddings = model.encode(spk["text"].tolist(), show_progress_bar=True, batch_size=64)
spk["_emb_idx"] = range(len(spk))
log.info("Embeddings done: %s", embeddings.shape)

# ── Semantic continuity per window ───────────────────────────────────────────
log.info("Computing per-window semantic continuity...")
win_rows = []
for (grp, task, win), win_df in spk.groupby(["group_id", "task_id", "window_index"]):
    win_df = win_df.sort_values("onset").reset_index(drop=True)
    if len(win_df) < 2:
        win_rows.append({"group_id": grp, "task_id": task, "window_index": win,
                         "tr_semantic_continuity": np.nan,
                         "tr_cross_speaker_similarity": np.nan})
        continue
    
    # Consecutive pairs (respects turn order)
    embs = embeddings[win_df["_emb_idx"].values]
    consecutive_sims = []
    for i in range(len(win_df) - 1):
        if win_df.iloc[i]["speaker"] != win_df.iloc[i+1]["speaker"]:
            sim = cosine_similarity([embs[i]], [embs[i+1]])[0, 0]
            consecutive_sims.append(float(sim))
    
    # All cross-speaker pairs (any combination)
    cross_sims = []
    speakers = win_df["speaker"].values
    for i in range(len(win_df)):
        for j in range(i + 1, len(win_df)):
            if speakers[i] != speakers[j]:
                sim = cosine_similarity([embs[i]], [embs[j]])[0, 0]
                cross_sims.append(float(sim))
    
    win_rows.append({
        "group_id": grp,
        "task_id": task,
        "window_index": win,
        "tr_semantic_continuity": np.mean(consecutive_sims) if consecutive_sims else np.nan,
        "tr_cross_speaker_similarity": np.mean(cross_sims) if cross_sims else np.nan,
    })

cont_df = pd.DataFrame(win_rows)
out_cont = REPO_ROOT / "analysis" / "results" / "semantic_continuity_window_30s.tsv"
cont_df.to_csv(out_cont, sep="\t", index=False)
log.info("Saved semantic continuity: %s rows → %s", len(cont_df), out_cont)

# ── Zero-shot NLI completion score ───────────────────────────────────────────
from transformers import pipeline as hf_pipeline

cand_path = REPO_ROOT / "analysis" / "results" / "completion_candidates_broad.tsv"
if not cand_path.exists():
    log.warning("Broad candidates file not found, skipping NLI step")
    sys.exit(0)

cands = pd.read_csv(cand_path, sep="\t", index_col=0)
cands.columns = cands.columns.str.strip()
for col in ["text_a", "text_b"]:
    cands[col] = cands[col].astype(str).str.strip()

HYPOTHESES = [
    "the second speaker is completing the first speaker's unfinished sentence",
    "the second speaker is repeating or echoing what the first speaker said",
]

log.info("Loading DeBERTa zero-shot classifier (cached)...")
zsc = hf_pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1",
    device=-1,
)
log.info("Classifier ready. Scoring %d candidates...", len(cands))

texts = [f'{r["speaker_a"]} said: "{r["text_a"]}" and then {r["speaker_b"]} said: "{r["text_b"]}"'
         for _, r in cands.iterrows()]

BATCH = 16
results = []
for i in range(0, len(texts), BATCH):
    batch = texts[i:i + BATCH]
    br = zsc(batch, candidate_labels=HYPOTHESES, multi_label=True)
    if isinstance(br, dict):
        br = [br]
    results.extend(br)
    if (i // BATCH + 1) % 5 == 0:
        log.info("  batch %d / %d", i // BATCH + 1, (len(texts) + BATCH - 1) // BATCH)

for i, res in enumerate(results):
    label_scores = dict(zip(res["labels"], res["scores"]))
    cands.at[cands.index[i], "nli_completion_score"] = label_scores.get(HYPOTHESES[0], 0.0)
    cands.at[cands.index[i], "nli_repetition_score"] = label_scores.get(HYPOTHESES[1], 0.0)

out_nli = REPO_ROOT / "analysis" / "results" / "completion_scores_nli.tsv"
cands.to_csv(out_nli, sep="\t", index_label="#")
log.info("Saved NLI scores: %s → %s", len(cands), out_nli)

# Validate: confirmed completions should score high
confirmed = {
    ("grp-10", "T3", 595.174),
    ("grp-15", "T3", 697.302),
    ("grp-11", "T1", 268.696),
}
log.info("Validation — confirmed completions NLI scores:")
for _, row in cands.iterrows():
    key = (row["group_id"].strip(), row["task_id"].strip(), float(row["onset_b"]))
    if key in confirmed:
        log.info("  %s %s t=%.1f: completion=%.3f  repetition=%.3f",
                 row["group_id"], row["task_id"], row["onset_b"],
                 row.get("nli_completion_score", 0), row.get("nli_repetition_score", 0))

log.info("ALL DONE.")
