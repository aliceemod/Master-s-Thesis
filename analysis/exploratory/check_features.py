import pandas as pd

b = pd.read_csv('features_before/collective_window_features.tsv', sep='\t')
a = pd.read_csv('features_after/collective_window_features.tsv', sep='\t')

print(f'BEFORE: {len(b)} windows, {len(b["group_id"].unique())} groups')
print(f'Groups: {sorted(b["group_id"].unique())}')
print(f'\nAFTER: {len(a)} windows, {len(a["group_id"].unique())} groups')
print(f'Groups: {sorted(a["group_id"].unique())}')
