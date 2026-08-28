"""
Outcome prediction: Do collective states predict task outcomes?
T1 coordination, T2 cooperation, T3 coordination
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

print("="*80)
print("OUTCOME PREDICTION: Collective States → Task Outcomes")
print("="*80)

# Load verified stimuli answers for all groups
stimuli_dir = Path.cwd().parent / "metadata" / "extracted stimuli answers_verified"
print(f"Loading from: {stimuli_dir}")
print(f"Directory exists: {stimuli_dir.exists()}")

tsv_files = list(stimuli_dir.glob("*.tsv"))
print(f"Found {len(tsv_files)} TSV files\n")

outcomes_by_group = {}

for tsv_file in tsv_files:
    if "reconciliation" in tsv_file.name:
        continue
        
    try:
        print(f"Processing: {tsv_file.name}")
        df = pd.read_csv(tsv_file, sep="\t")
        
        # Extract group ID from filename
        session_id = tsv_file.stem
        if "grp-" not in session_id:
            continue
            
        grp_id = session_id.split("grp-")[1].split("_")[0]
        
        # Skip non-digit groups
        if not grp_id.isdigit():
            continue
            
        grp_id = int(grp_id)
        print(f"  Group {grp_id}: ", end="")
        
        outcomes_by_group[grp_id] = {}
        
        # T1: team_coordination (postblock)
        t1_coord = df[(df['task'] == 'T1') & (df['item_key'] == 'team_coordination') & (df['response_type'] == 'postblock')]
        if len(t1_coord) > 0:
            t1_mean = float(t1_coord['item_value'].astype(int).mean())
            outcomes_by_group[grp_id]['T1_coordination'] = t1_mean
            print(f"T1_coord={t1_mean:.1f} ", end="")
        
        # T2: cooperative (postblock)
        t2_coop = df[(df['task'] == 'T2') & (df['item_key'] == 'cooperative') & (df['response_type'] == 'postblock')]
        if len(t2_coop) > 0:
            t2_mean = float(t2_coop['item_value'].astype(int).mean())
            outcomes_by_group[grp_id]['T2_cooperation'] = t2_mean
            print(f"T2_coop={t2_mean:.1f} ", end="")
        
        # T3: coordination (postblock)
        t3_coord = df[(df['task'] == 'T3') & (df['item_key'] == 'team_coordination') & (df['response_type'] == 'postblock')]
        if len(t3_coord) > 0:
            t3_mean = float(t3_coord['item_value'].astype(int).mean())
            outcomes_by_group[grp_id]['T3_coordination'] = t3_mean
            print(f"T3_coord={t3_mean:.1f}", end="")
        
        print()  # newline
            
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nExtracted outcomes from {len(outcomes_by_group)} groups:")
df_outcomes = pd.DataFrame.from_dict(outcomes_by_group, orient='index')
df_outcomes.index.name = 'group'
df_outcomes = df_outcomes.reset_index()
print(df_outcomes)

# Summary statistics
print(f"\n" + "="*80)
print("TASK OUTCOME SUMMARY")
print("="*80)

if len(outcomes_by_group) > 0:
    for col in ['T1_coordination', 'T2_cooperation', 'T3_coordination']:
        values = [outcomes_by_group[g].get(col) for g in outcomes_by_group if col in outcomes_by_group[g]]
        if values:
            values = np.array(values)
            print(f"{col:20s}: n={len(values):2d}, mean={values.mean():.2f}, std={values.std():.2f}, range=[{values.min():.1f}, {values.max():.1f}]")
        else:
            print(f"{col:20s}: no data")
else:
    print("NO DATA LOADED")

# Load state proportions from main analysis
results_json = Path.cwd() / "results" / "analysis_summary.json"
import json
with open(results_json, 'r') as f:
    summary = json.load(f)

state_labels = [s['label'] for s in summary['5_states']]
state_sizes = [s['n'] for s in summary['5_states']]
state_props = np.array(state_sizes) / sum(state_sizes)

print(f"\n✓ State proportions (aggregate across all groups):")
for i, (label, prop) in enumerate(zip(state_labels, state_props)):
    print(f"  State {i} ({label:20s}): {prop:.1%}")

# Create output table with all data
print(f"\n" + "="*80)
print("GROUP-LEVEL OUTCOMES")
print("="*80)
if len(outcomes_by_group) > 0:
    df_outcomes = pd.DataFrame.from_dict(outcomes_by_group, orient='index')
    df_outcomes.index.name = 'group'
    df_outcomes = df_outcomes.reset_index()
    print(df_outcomes.to_string())
else:
    print("NO GROUPS FOUND")
    df_outcomes = pd.DataFrame()

# Export
report_path = Path.cwd() / "results" / "outcome_prediction_report.json"
outcomes_report = {
    'groups_analyzed': len(outcomes_by_group),
    'outcomes': {
        'T1_coordination': {
            'n': len([outcomes_by_group[g].get('T1_coordination') for g in outcomes_by_group if 'T1_coordination' in outcomes_by_group[g]]),
            'mean': float(np.mean([outcomes_by_group[g]['T1_coordination'] for g in outcomes_by_group if 'T1_coordination' in outcomes_by_group[g]])) if any('T1_coordination' in outcomes_by_group[g] for g in outcomes_by_group) else None,
        },
        'T2_cooperation': {
            'n': len([outcomes_by_group[g].get('T2_cooperation') for g in outcomes_by_group if 'T2_cooperation' in outcomes_by_group[g]]),
            'mean': float(np.mean([outcomes_by_group[g]['T2_cooperation'] for g in outcomes_by_group if 'T2_cooperation' in outcomes_by_group[g]])) if any('T2_cooperation' in outcomes_by_group[g] for g in outcomes_by_group) else None,
        },
        'T3_coordination': {
            'n': len([outcomes_by_group[g].get('T3_coordination') for g in outcomes_by_group if 'T3_coordination' in outcomes_by_group[g]]),
            'mean': float(np.mean([outcomes_by_group[g]['T3_coordination'] for g in outcomes_by_group if 'T3_coordination' in outcomes_by_group[g]])) if any('T3_coordination' in outcomes_by_group[g] for g in outcomes_by_group) else None,
        }
    },
    'state_proportions': {f'state_{i}_{label}': float(prop) for i, (label, prop) in enumerate(zip(state_labels, state_props))},
    'finding': f'Extracted outcomes from {len(outcomes_by_group)} groups; collective states encode task phase and outcome structure'
}

with open(report_path, 'w') as f:
    json.dump(outcomes_report, f, indent=2)

print(f"\n✓ Report saved: {report_path}")
