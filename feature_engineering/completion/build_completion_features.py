"""Build window- and task-level completion/repetition rate features from labelled candidates."""
import pandas as pd
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
WINDOW_S  = 30.0

# Load the labelled candidates (round 1 + round 2)
r1 = pd.read_csv(REPO_ROOT / 'analysis' / 'results' / 'completion_candidates_labelled.tsv', sep='\t', index_col=0)
r1.columns = r1.columns.str.strip()
for col in r1.select_dtypes('object').columns:
    r1[col] = r1[col].astype(str).str.strip()

r2 = pd.read_csv(REPO_ROOT / 'analysis' / 'results' / 'completion_candidates_round2.tsv', sep='\t', index_col=0)
r2.columns = r2.columns.str.strip()
for col in r2.select_dtypes('object').columns:
    r2[col] = r2[col].astype(str).str.strip()

# Align columns and concat
shared = ['group_id', 'task_id', 'onset_b', 'is_completion', 'is_repetition']
for df in [r1, r2]:
    df['group_id'] = df['group_id'].astype(str).str.strip()
    df['task_id']  = df['task_id'].astype(str).str.strip()

# Round 1 used different label convention (yes/no/repetition in one column)
if 'is_completion_labelled' in r1.columns:
    r1 = r1.rename(columns={
        'is_completion_labelled': 'is_completion',
        'is_repetition_labelled': 'is_repetition',
    })

combined = pd.concat([
    r1[shared].assign(source='round1'),
    r2[shared].assign(source='round2'),
], ignore_index=True)

combined['is_completion'] = pd.to_numeric(combined['is_completion'], errors='coerce').fillna(0).astype(int)
combined['is_repetition']  = pd.to_numeric(combined['is_repetition'],  errors='coerce').fillna(0).astype(int)
combined['onset_b'] = pd.to_numeric(combined['onset_b'], errors='coerce')

print(f'Total candidates: {len(combined)}')
print(f'  Completions : {combined["is_completion"].sum()}')
print(f'  Repetitions : {combined["is_repetition"].sum()}')

# Assign to window
combined['window_index'] = (combined['onset_b'] / WINDOW_S).apply(
    lambda x: int(x) if pd.notna(x) and x >= 0 else -1
)
valid = combined[combined['window_index'] >= 0]

# Window-level counts
win_feats = (
    valid.groupby(['group_id', 'task_id', 'window_index'])[['is_completion', 'is_repetition']]
    .sum()
    .reset_index()
    .rename(columns={'is_completion': 'tr_completion_count', 'is_repetition': 'tr_repetition_count'})
)

out_win = REPO_ROOT / 'analysis' / 'results' / 'completion_repetition_window_30s.tsv'
win_feats.to_csv(out_win, sep='\t', index=False)
print(f'\nWindow-level features saved: {len(win_feats)} rows → {out_win.name}')
print(win_feats[win_feats[['tr_completion_count','tr_repetition_count']].sum(axis=1) > 0].to_string())

# Task-level aggregation (for prediction model)
task_feats = (
    valid.groupby(['group_id', 'task_id'])[['is_completion', 'is_repetition']]
    .sum()
    .reset_index()
    .rename(columns={'is_completion': 'n_completions', 'is_repetition': 'n_repetitions'})
)

out_task = REPO_ROOT / 'analysis' / 'results' / 'completion_repetition_task.tsv'
task_feats.to_csv(out_task, sep='\t', index=False)
print(f'\nTask-level features saved: {len(task_feats)} rows → {out_task.name}')
print(task_feats[task_feats[['n_completions','n_repetitions']].sum(axis=1) > 0].to_string())
