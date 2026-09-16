"""Search for genuine sentence completion cases across all transcripts.

Patterns to look for:
1. SPK row ends with '…', '...', or a dangling connector ('and', 'or', 'but', 'so', 'that', 'which', 'who')
   AND another speaker starts talking within ~1s (OVL or SPK)
2. OVL context_label = 'collaborative' where B's content continues A's thought
"""
import pandas as pd
from pathlib import Path
import re

final = Path('transcripts/final')

# Trailing-off patterns: sentence ends with connector/incomplete marker
TRAIL_PATTERNS = [
    re.compile(r'\.\.\.$'),                               # ends with ...
    re.compile(r'[—–…]$'),                                # em-dash or ellipsis char
    re.compile(r'\b(and|or|but|so|that|which|who|when|where|because|if|as)\s*[,]?\s*$', re.IGNORECASE),  # trailing connector
    re.compile(r'\b(I think|I mean|like|you know|kind of|sort of)\s*$', re.IGNORECASE),  # trailing hedge
]

def trails_off(text: str) -> tuple[bool, str]:
    text = text.strip()
    for p in TRAIL_PATTERNS:
        if p.search(text):
            return True, p.pattern
    return False, ''

results = []

for fpath in sorted(final.glob('transcript_grp*.tsv')):
    if fpath.name.endswith('.bak.tsv'):
        continue
    df = pd.read_csv(fpath, sep='\t')
    df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
    df['text'] = df['text'].fillna('')
    df = df.sort_values('onset').reset_index(drop=True)

    spk_rows = df[df['type'] == 'SPK']
    ovl_rows = df[df['type'] == 'OVL']
    participants = {'P1', 'P2', 'P3', 'P4'}

    for idx, row in spk_rows.iterrows():
        if row['speaker'] not in participants:
            continue
        text = str(row['text']).strip()
        # Skip very short texts (single words), OVL placeholders
        if len(text) < 8 or text.startswith('['):
            continue
        trails, pattern = trails_off(text)
        if not trails:
            continue

        spk_end = row['onset'] + row['duration']

        # Look for another participant starting within 1.5s of A ending
        for _, row2 in df.iterrows():
            if row2['type'] not in ('SPK', 'OVL'):
                continue
            if row2['speaker'] == row['speaker']:
                continue
            if row2['speaker'] not in participants and 'MODERATOR' not in str(row2['speaker']):
                continue
            if not (row['onset'] <= row2['onset'] <= spk_end + 1.5):
                continue

            text2 = str(row2['text']).strip()
            # B's text should be real content (not just yeah/mmh/placeholder)
            if text2.startswith('['):
                continue
            JUST_AGREE = re.compile(r'^(yeah|yep|yes|mmh|mm-hmm|uh-huh|right|okay|ok|no|exactly)[.\s!]*$', re.IGNORECASE)
            is_content = not JUST_AGREE.match(text2) and len(text2) > 5

            # Is there an OVL event for B during A's speech?
            b_ovl = ovl_rows[
                (ovl_rows['speaker'] == row2['speaker']) &
                (ovl_rows['onset'] >= row['onset'] - 0.2) &
                (ovl_rows['onset'] <= spk_end + 0.5)
            ]
            is_overlapping = not b_ovl.empty

            if is_content or is_overlapping:
                results.append({
                    'file': fpath.name[:40],
                    'spk_a': row['speaker'],
                    'text_a': text[:70],
                    'trail_pattern': pattern[:30],
                    'spk_b': row2['speaker'],
                    'text_b': text2[:70],
                    'b_type': row2['type'],
                    'overlapping': is_overlapping,
                    'onset_a': row['onset'],
                    'onset_b': row2['onset'],
                })

# Filter to overlapping completions only (most interesting)
overlapping = [r for r in results if r['overlapping']]
print(f'Total trailing-off SPK rows with overlapping continuation: {len(overlapping)}')
print()

# Show the best examples — content continuations during overlap
content_cont = [r for r in overlapping if not re.match(r'^(yeah|yep|yes|mmh|right|okay)', r['text_b'], re.IGNORECASE)]
print(f'With CONTENT words from B (potential true completions): {len(content_cont)}')
print()
print('='*72)
print('BEST EXAMPLES OF POTENTIAL SENTENCE COMPLETION:')
print('='*72)

shown = set()
for r in content_cont[:25]:
    key = (r['file'], r['onset_a'])
    if key in shown:
        continue
    shown.add(key)
    print(f"\n[{r['file']}]")
    print(f"  A ({r['spk_a']}) onset={r['onset_a']:.2f}s: \"{r['text_a']}\"")
    print(f"  B ({r['spk_b']}) onset={r['onset_b']:.2f}s [{r['b_type']}{'+ OVL' if r['overlapping'] else ''}]: \"{r['text_b']}\"")
    print(f"  Pattern: {r['trail_pattern']}")
