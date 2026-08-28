import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

warnings.filterwarnings('ignore')

REPO_ROOT = Path.cwd().parent if Path.cwd().name == 'analysis' else Path.cwd()
TRANSCRIPT_DIR = REPO_ROOT / 'transcripts' / 'final'
OUT_PATH = REPO_ROOT / 'analysis' / 'results' / 'da_window_30s.tsv'

WINDOW_S = 30.0
PARTICIPANTS = ('P1', 'P2', 'P3', 'P4')

print(f'Repo root: {REPO_ROOT}')
print(f'Transcripts: {TRANSCRIPT_DIR}')

# Hypotheses are phrased as what the utterance IS doing — the NLI model checks entailment.
# Cue words from the codebook are embedded in the hypothesis text to guide the model.
DA_LABELS = {
    'statement':    'expressing an opinion, asserting a fact, or sharing a personal view',
    'proposal':     'suggesting a new option, plan, or course of action for the group to consider',
    'agreement':    'endorsing, accepting, or affirming a prior idea or proposal from someone else',
    'disagreement': 'rejecting, opposing, or pushing back against an idea or proposal',
    'compromise':   'proposing to blend or combine competing options to reach common ground',
    'question':     'asking for information, clarification, or a decision from the group',
    'answer':       'responding directly to a question that was just asked',
    'backchannel':  'giving a brief acknowledgment like mm-hmm or yeah to signal attention without adding new content',
    'hedge':        'expressing uncertainty or tentatively softening a claim with words like maybe or probably',
    'decision':     'formally or informally moving the group toward a final decision or agreement',
    'offtask':      'making a joke, engaging in social talk, or saying something unrelated to the current task',
}

DA_SHORT = list(DA_LABELS.keys())
DA_HYPOTHESES = list(DA_LABELS.values())

print(f'{len(DA_LABELS)} dialogue act categories:')
for k, v in DA_LABELS.items():
    print(f'  {k:15s} → "{v}"')

_FNAME_RE = re.compile(r'transcript_(grp[_-]?\d+)_(T\d+)_', re.IGNORECASE)

def _normalize_group_id(raw: str) -> str:
    digits = re.sub(r'^grp[_-]?', '', raw.lower())
    return f'grp-{int(digits):02d}'

all_events = []
for fpath in TRANSCRIPT_DIR.glob('*.tsv'):
    match = _FNAME_RE.search(fpath.name)
    if not match:
        continue
    df = pd.read_csv(fpath, sep='\t')
    df['group_id'] = _normalize_group_id(match.group(1))
    df['task_id']  = match.group(2).upper()
    all_events.append(df)

events_df = pd.concat(all_events, ignore_index=True)
print(f'Loaded {len(events_df)} transcript rows')

# Filter to non-empty SPK utterances with window assignment
spk_events = events_df[
    (events_df['type'] == 'SPK') &
    (events_df['speaker'].isin(PARTICIPANTS)) &
    (events_df['text'].notna()) &
    (events_df['text'].str.strip() != '')
].copy()

spk_events['window_index'] = (spk_events['onset'] / WINDOW_S).apply(
    lambda x: int(x) if pd.notna(x) and x >= 0 else -1
)
spk_events = spk_events[spk_events['window_index'] >= 0].reset_index(drop=True)

print(f'Utterances to classify: {len(spk_events)}')
print(f'Groups: {spk_events["group_id"].nunique()}, Tasks: {spk_events["task_id"].unique().tolist()}')

MODEL = 'facebook/bart-large-mnli'  # or 'typeform/distilbart-mnli-12-1' for speed

print(f'Loading {MODEL} ...')
zsc = pipeline('zero-shot-classification', model=MODEL, device=-1)  # device=-1 = CPU
print('Model ready.')
print(f'Classifying {len(spk_events)} utterances × {len(DA_LABELS)} labels ...')
print('(Expect 20–40 min on CPU for bart-large; 7–12 min for distilbart)')

from tqdm.auto import tqdm

BATCH_SIZE = 16
utterances = spk_events['text'].tolist()

# Each result: {'labels': [...], 'scores': [...]} in descending score order
results = []
for i in tqdm(range(0, len(utterances), BATCH_SIZE), desc='Classifying'):
    batch = utterances[i : i + BATCH_SIZE]
    batch_results = zsc(batch, candidate_labels=DA_HYPOTHESES, multi_label=False)
    # zsc returns a list when given a list
    if isinstance(batch_results, dict):
        batch_results = [batch_results]
    results.extend(batch_results)

print(f'Done. Classified {len(results)} utterances.')

# Build a DataFrame of per-utterance DA probability distributions
# Map hypothesis strings back to short DA names
hyp_to_short = {v: k for k, v in DA_LABELS.items()}

utt_da_rows = []
for i, res in enumerate(results):
    row = {'utt_idx': i}
    for label, score in zip(res['labels'], res['scores']):
        short = hyp_to_short[label]
        row[f'da_{short}'] = score
    row['da_dominant'] = hyp_to_short[res['labels'][0]]  # highest-scoring label
    utt_da_rows.append(row)

utt_da_df = pd.DataFrame(utt_da_rows)
da_prob_cols = [f'da_{k}' for k in DA_SHORT]

print(utt_da_df.head())
print(f'\nDominant act distribution across all utterances:')
print(utt_da_df['da_dominant'].value_counts())

# Attach metadata to utterance DA scores
spk_with_da = spk_events[['group_id', 'task_id', 'window_index', 'speaker']].copy()
spk_with_da = pd.concat([spk_with_da.reset_index(drop=True), utt_da_df.drop(columns='utt_idx')], axis=1)

# Mean-pool probability scores per window (preserves uncertainty, better than argmax counting)
win_rows = []
for (grp, task, win), win_df in spk_with_da.groupby(['group_id', 'task_id', 'window_index']):
    row = {
        'group_id': grp,
        'task_id': task,
        'window_index': win,
        'da_n_utterances': len(win_df),
        'da_n_speakers': win_df['speaker'].nunique(),
    }
    for col in da_prob_cols:
        row[col] = win_df[col].mean()  # mean probability across utterances
    row['da_dominant'] = win_df['da_dominant'].mode().iloc[0]  # most common dominant act
    win_rows.append(row)

da_df = pd.DataFrame(win_rows)
print(f'Windows: {len(da_df)}')
print(da_df.head())

da_df.to_csv(OUT_PATH, sep='\t', index=False)
print(f'Saved {len(da_df)} rows → {OUT_PATH}')
print(f'Columns: {da_df.columns.tolist()}')

# Mean DA profile per task — shows which acts dominate each task type
task_profile = da_df.groupby('task_id')[da_prob_cols].mean()

fig, ax = plt.subplots(figsize=(13, 5))
task_profile.T.plot(kind='bar', ax=ax, colormap='tab10')
ax.set_xlabel('Dialogue Act')
ax.set_ylabel('Mean probability')
ax.set_title('Dialogue Act Profile by Task')
ax.set_xticklabels([c.replace('da_', '') for c in da_prob_cols], rotation=35, ha='right')
ax.legend(title='Task')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Heatmap: mean DA profile per group (averaged across all tasks and windows)
grp_profile = da_df.groupby('group_id')[da_prob_cols].mean()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    grp_profile,
    annot=True, fmt='.2f', cmap='YlOrRd',
    xticklabels=[c.replace('da_', '') for c in da_prob_cols],
    ax=ax
)
ax.set_title('Mean Dialogue Act Profile per Group')
plt.tight_layout()
plt.show()

# η² decomposition — same check as for BERT embeddings
# Tells us whether DA features discriminate tasks vs. groups
results_eta = []
for col in da_prob_cols:
    vals = da_df[col].values
    grand_mean = vals.mean()
    ss_total = ((vals - grand_mean) ** 2).sum()
    if ss_total == 0:
        continue

    task_masks = [(da_df['task_id'] == t).values for t in da_df['task_id'].unique()]
    ss_task = sum((vals[m].mean() - grand_mean) ** 2 * m.sum() for m in task_masks)

    grp_masks = [(da_df['group_id'] == g).values for g in da_df['group_id'].unique()]
    ss_grp = sum((vals[m].mean() - grand_mean) ** 2 * m.sum() for m in grp_masks)

    results_eta.append({
        'Feature': col.replace('da_', ''),
        'η²_task': ss_task / ss_total,
        'η²_group': ss_grp / ss_total,
    })

eta_df = pd.DataFrame(results_eta)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(eta_df))
w = 0.35
ax.bar(x - w/2, eta_df['η²_task'],  w, label='η² task',  color='steelblue')
ax.bar(x + w/2, eta_df['η²_group'], w, label='η² group', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(eta_df['Feature'], rotation=30, ha='right')
ax.set_ylabel('η²')
ax.set_title('DA Features: variance explained by task vs. group')
ax.axhline(0.05, color='grey', linestyle='--', alpha=0.6)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"Mean η²_task  : {eta_df['η²_task'].mean():.3f}")
print(f"Mean η²_group : {eta_df['η²_group'].mean():.3f}")
print()
print('Compare with BERT-UMAP:')
print('  BERT η²_task=0.36, η²_group=0.06')
print('  If DA η²_group >> BERT η²_group: DA features carry more group-level signal')
