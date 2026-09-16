import pandas as pd
import os

# Check all available feature matrices
paths = [
    'features/collective_window_features.tsv',
    'features_before/collective_window_features.tsv',
    'features_after/collective_window_features.tsv',
    '_tmp_collective_fullsample_export/features/collective_window_features.tsv',
    '_tmp_collective_fullsample_completecase_export/features/collective_window_features.tsv'
]

for path in paths:
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t')
        groups = sorted(df['group_id'].unique())
        print(f'{path}: {len(df):3d} windows, {len(groups):2d} groups')
        print(f'  Groups: {groups}')
