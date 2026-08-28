import pandas as pd
from pathlib import Path

files = sorted(Path('tools').glob('transcript_grp*.tsv'))
files = [f for f in files if not f.name.endswith('.bak.tsv')]

rows = []
for f in files:
    try:
        df = pd.read_csv(f, sep='\t', dtype=str)
        if 'type' not in df.columns:
            continue
        ovl = df[df['type']=='OVL'].copy()
        if ovl.empty:
            continue
        g = f.name.split('_')[1].replace('grp-','grp').replace('grp','grp-').lstrip('grp-')
        # extract group/task from filename
        import re
        m = re.search(r'grp-?(\d+)_T(\d)', f.name, re.I)
        grp = f'grp-{m.group(1):0>2}' if m else f.name
        task = f'T{m.group(2)}' if m else '?'
        subtype_counts = ovl['subtype'].fillna('(blank)').value_counts().to_dict()
        for sub, cnt in sorted(subtype_counts.items()):
            rows.append({'group': grp, 'task': task, 'subtype': sub, 'count': cnt})
    except Exception as e:
        print(f'ERROR {f.name}: {e}')

df = pd.DataFrame(rows)
print('=== OVL subtype counts by subtype across all groups ===')
print(df.groupby('subtype')['count'].sum().sort_values(ascending=False).to_string())
print()
print('=== competitive + floor_fight per group ===')
comp = df[df['subtype'].isin(['competitive','floor_fight','needs_review'])]
print(comp.groupby(['group','subtype'])['count'].sum().unstack(fill_value=0).to_string())
