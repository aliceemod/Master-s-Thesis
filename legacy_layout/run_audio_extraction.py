"""Extract GeMAPSv01b audio features per 30-second window, group-averaged across P1-P4.

Output: icmi_paper/results/audio_window_features.tsv
"""
import re
import logging
import numpy as np
import opensmile
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'icmi_paper/results'
AUDIO_ROOT  = Path('C:/Users/amodica/Downloads/audio files')
WINDOW_S    = 30.0
MIC_TO_PARTICIPANT = {'mic9': 'P1', 'mic10': 'P2', 'mic11': 'P3', 'mic12': 'P4'}
TARGET_TASKS = {'T1', 'T2', 'T3'}
PRIORITY_FEATURES = {
    'loudness_sma3_amean':                     'audio_energy_mean',
    'F0semitoneFrom27.5Hz_sma3nz_amean':       'audio_pitch_mean',
    'F0semitoneFrom27.5Hz_sma3nz_stddevNorm':  'audio_pitch_sd',
    'HNRdBACF_sma3nz_amean':                   'audio_hnr_mean',
}
_WAV_RE = re.compile(
    r'ses-(?P<ses_date>[0-9]{8})_'
    r'(?P<grp>grp-[0-9]+)_'
    r'run(?P<ses_run>[0-9]+)'
    r'.*?task-(?P<task>T[0-9]+)'
    r'.*?dpa-(?P<mic>mic[0-9]+)-aud'
    r'.*?_audio[.]wav$',
    re.IGNORECASE,
)

# ── Load window metadata ───────────────────────────────────────────────────────
hmm_feats = pd.read_csv(RESULTS_DIR / 'hmm_input_features_final.tsv', sep='\t')
windows = (hmm_feats[['group_id', 'task_id', 'window_index', 'window_start_s']]
           .drop_duplicates().reset_index(drop=True))
log.info(f'Windows to extract: {len(windows)}')

# ── Discover WAV files ─────────────────────────────────────────────────────────
log.info(f'Scanning {AUDIO_ROOT} ...')
wav_map = {}
for wav in sorted(AUDIO_ROOT.rglob('*dpa-mic*-aud*_audio.wav')):
    m = _WAV_RE.search(wav.name)
    if not m:
        continue
    task = m.group('task').upper()
    if task not in TARGET_TASKS:
        continue
    grp  = m.group('grp')
    mic  = m.group('mic')
    part = MIC_TO_PARTICIPANT.get(mic, mic)
    wav_map[(grp, task, part)] = wav
log.info(f'Found {len(wav_map)} WAV files for T1-T3')

# ── Build opensmile extractor ──────────────────────────────────────────────────
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
    verbose=False,
)
GEMALPS_COLS = smile.feature_names

# ── Extraction loop ────────────────────────────────────────────────────────────
PARTIAL_PATH = RESULTS_DIR / '_audio_window_partial.tsv'
if PARTIAL_PATH.exists():
    done_df   = pd.read_csv(PARTIAL_PATH, sep='\t')
    done_keys = set(zip(done_df['group_id'], done_df['task_id'],
                        done_df['window_index'], done_df['participant_id']))
    rows = done_df.to_dict('records')
    log.info(f'Resuming from {len(done_df)} rows')
else:
    done_keys, rows = set(), []
    log.info('Starting fresh')

total  = len(windows) * 4
n_done = len(rows)
n_skip = n_fail = 0

for _, win in windows.iterrows():
    grp, task = win['group_id'], win['task_id']
    win_idx   = int(win['window_index'])
    start_s   = float(win['window_start_s'])

    for part in ['P1', 'P2', 'P3', 'P4']:
        key = (grp, task, win_idx, part)
        if key in done_keys:
            continue
        wav_path = wav_map.get((grp, task, part))
        if wav_path is None:
            n_skip += 1
            continue
        try:
            feats = smile.process_file(str(wav_path), start=start_s, end=start_s + WINDOW_S)
        except Exception as exc:
            log.warning('opensmile failed %s %.1fs: %s', wav_path.name, start_s, exc)
            feats = None

        row = {'group_id': grp, 'task_id': task, 'window_index': win_idx,
               'window_start_s': start_s, 'participant_id': part}
        if feats is not None and not feats.empty:
            row.update({f'gemap_{k}': v for k, v in feats.iloc[0].items()})
        else:
            row.update({f'gemap_{k}': np.nan for k in GEMALPS_COLS})
            n_fail += 1

        rows.append(row)
        n_done += 1
        if n_done % 50 == 0:
            pd.DataFrame(rows).to_csv(PARTIAL_PATH, sep='\t', index=False)
            log.info(f'{n_done}/{total} ({100*n_done//total}%)  skip={n_skip} fail={n_fail}')

participant_df = pd.DataFrame(rows)
participant_df.to_csv(PARTIAL_PATH, sep='\t', index=False)
log.info(f'Extraction done: {len(participant_df)} rows  skip={n_skip} fail={n_fail}')

# ── Group-average ──────────────────────────────────────────────────────────────
gemap_cols = [c for c in participant_df.columns if c.startswith('gemap_')]
group_df   = (participant_df
              .groupby(['group_id', 'task_id', 'window_index', 'window_start_s'])[gemap_cols]
              .mean().reset_index())
for src, dst in PRIORITY_FEATURES.items():
    col = f'gemap_{src}'
    if col in group_df.columns:
        group_df[dst] = group_df[col]

meta_cols     = ['group_id', 'task_id', 'window_index', 'window_start_s']
priority_cols = list(PRIORITY_FEATURES.values())
out_df  = group_df[meta_cols + priority_cols + gemap_cols]
out_path = RESULTS_DIR / 'audio_window_features.tsv'
out_df.to_csv(out_path, sep='\t', index=False)
log.info(f'Saved: {out_path}  shape={out_df.shape}')
cov = (out_df[priority_cols].notna().mean() * 100).round(1).to_dict()
log.info(f'Coverage: {cov}')
