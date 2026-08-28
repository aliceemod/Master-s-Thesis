# OVL Annotation Methodology — Overlap Subtype & Context Classification

## Overview

All manual transcripts contain OVL rows marking segments where two or more speakers speak simultaneously. The July 2026 reclassification pass replaced a mixed timing+semantic labeling scheme with a **two-layer architecture** separating objective acoustic features from functional/lexical interpretation:

| Column | Layer | Source | Description |
|---|---|---|---|
| `subtype` | Timing (objective) | Acoustic duration | 5 fixed categories based purely on overlap duration and start-time difference |
| `context_label` | Lexical (interpretive) | Adjacent speech text | Collaborative, completion, competitive, or empty — assigned by LLM-assisted rule-based detection |

---

## Layer 1: Timing Subtypes (`subtype`)

The `subtype` field is **always determined by acoustic timing alone** — no text analysis, no annotator judgment about intent.

| Subtype | Rule | Interpretation |
|---|---|---|
| `simultaneous` | Start-time difference < 200 ms | Both speakers began at virtually the same moment — joint floor entry |
| `backchannel_ovl` | Overlap duration < 250 ms | Very brief acoustic overlap — listener signal or quick interjection |
| `smooth` | Overlap 250–500 ms | Managed turn transition with short overlap — normal conversational handoff |
| `competitive` | Overlap 500–1000 ms | Contested floor — one speaker continues while another attempts to take over |
| `floor_fight` | Overlap ≥ 1000 ms | Sustained simultaneous speech — extended contest for the conversational floor |

**Key principle:** `competitive` and `floor_fight` label the *acoustic duration* of the overlap, not the communicative intent. A long overlap can still be collaborative in spirit (see `context_label` below).

### Implementation

Assigned by `classify_timing()` in `tools/relabel_overlaps.py`:

```python
def classify_timing(overlap_ms: float, start_diff_ms: float | None) -> str:
    if start_diff_ms is not None and start_diff_ms < 200:
        return "simultaneous"
    if overlap_ms < 250:
        return "backchannel_ovl"
    if overlap_ms < 500:
        return "smooth"
    if overlap_ms < 1000:
        return "competitive"
    return "floor_fight"
```

---

## Layer 2: Context Labels (`context_label`)

The `context_label` field captures the **functional/communicative intent** of the overlap based on the speech content of the overlapping speakers. It is populated only when a clear lexical signal is detected; otherwise it is empty.

| Value | Meaning | How detected |
|---|---|---|
| `collaborative` | Supportive overlapping speech — agreement, affirmation, or backchannel signals during another's turn | Support words present in the overlapping speaker's concurrent utterance |
| `completion` | Collaborative sentence completion — one speaker finishes another's trailing utterance | Prior speaker's utterance ends with an explicit ellipsis/em-dash marker AND the overlapping speaker continues with content words (not mere agreement) |
| `competitive` | Conflictual intent — challenge or contradiction during an already-contested overlap | Conflict words present on a `competitive` or `floor_fight` timing overlap |
| *(empty)* | No clear lexical signal — neutral or ambiguous | No support or conflict cue detected |

---

## Detection Algorithm

### Step 1 — Find concurrent speech

For each OVL row, `_find_context_spks()` locates the 1–2 SPK rows that are acoustically coincident with the overlap window, providing the speech text for cue detection.

### Step 2 — Completion detection (priority check)

`_is_completion_context()` checks whether another speaker's utterance **trails off** with an explicit marker (`...`, `…`, `—`, `–`) within 1.5 s before the overlap onset, and the overlapping speaker's text contains content words (not just bare agreement tokens like "yeah").

```
If prior speaker ends with ellipsis/dash AND
   overlap speaker text is NOT a bare agreement token (yeah/mmh/right...)
→ context_label = "completion"
```

### Step 3 — Support/conflict cue detection

Lexical cue lists were developed with LLM assistance (iterative testing and refinement against the AffectAI corpus) and applied as regular-expression pattern matching:

**Strong support signals** (always counted regardless of utterance length):
> *yeah, yep, mmh, mm-hmm, uh-huh, exactly, agree, agreed, good point, totally, absolutely, indeed*

**Context-sensitive support signals** (only counted in short utterances ≤ 4 words, to avoid false positives from discourse markers mid-sentence):
> *yes, right, ok, okay, sure, correct, true, definitely*

**Note on disambiguation:** A "yeah" or "right" that appears as a standalone SPK turn (no concurrent OVL) is a genuine turn-response and is **not** considered a collaborative overlap signal. The algorithm only inspects text from SPK rows that are time-coincident with the OVL event.

**Conflict signals** (only applied to `competitive`/`floor_fight` timing overlaps):
> *no, not really, i disagree, disagree, wrong, hold on, wait, stop, but, however, actually, let me finish, don't interrupt*

### Step 4 — Label assignment

```
If completion context detected AND overlap text has content words:
    context_label = "completion"
Else if support cues present AND no conflict cues:
    context_label = "collaborative"
Else if conflict cues present AND timing is competitive/floor_fight:
    context_label = "competitive"
Else:
    context_label = "" (empty)
```

---

## How Annotation Was Done

### Phase 1 — Original manual transcription
Transcripts for groups 07, 10, 14, 15, 16 were produced by a human annotator using the `tools/transcript_corrector.html` browser tool. The annotator reviewed each OVL row against the audio and labeled overlaps using a combined timing+semantic vocabulary (including a `collaborative` subtype for sentence completions and agreement signals). Groups 08, 09, 11, 12, 13 were programmatically reclassified from raw diarization output.

### Phase 2 — July 2026 reclassification
The `tools/relabel_overlaps.py` script was run on all transcripts in `transcripts/final/`, replacing the mixed vocabulary with the two-layer system:
1. All OVL `subtype` fields rewritten using pure timing rules (Phase 1 semantic labels discarded)
2. `context_label` column added and populated via LLM-assisted rule-based detection

The LLM (GitHub Copilot / Claude Sonnet) contributed to:
- Designing the support/conflict cue word lists through iterative testing
- Identifying disambiguation cases (e.g., "right?" as a question tag vs. agreement)
- Writing and refining the `_is_completion_context()` heuristic
- Reviewing edge cases and adjusting the short-utterance threshold for context-sensitive words

The pattern-matching rules are fully deterministic and reproducible; the LLM role was in **authoring the rules**, not in making per-overlap decisions.

### Canonical source
All final transcripts are stored in `transcripts/final/`. The `transcripts/final_backup_20260714/` folder preserves the pre-reclassification state.

---

## Final Label Distribution (Post-Reclassification, All Tasks)

### Timing subtypes (`subtype`), all groups

| Subtype | Count | % |
|---|---|---|
| `floor_fight` | 313 | 33.3% |
| `backchannel_ovl` | 192 | 20.4% |
| `competitive` | 189 | 20.1% |
| `smooth` | 140 | 14.9% |
| `simultaneous` | 106 | 11.3% |
| **Total OVL rows** | **940** | |

### Context labels (`context_label`), T1–T3 only (HMM analysis subset)

| Context | Count | % of OVL |
|---|---|---|
| `collaborative` | 313 | 40.7% |
| `completion` | 26 | 3.4% |
| `competitive` | 76 | 9.9% |
| *(empty)* | 354 | 46.0% |
| **Total OVL rows (T1–T3)** | **769** | |

---

## Feature Pipeline Implications

### Features derived from `subtype` (timing-based, used in HMM)

| Feature | Definition |
|---|---|
| `tr_ovl_simultaneous` | Count of simultaneous overlaps per window |
| `tr_ovl_backchannel` | Count of backchannel_ovl per window |
| `tr_ovl_smooth` | Count of smooth overlaps per window |
| `tr_ovl_competitive_timing` | Count of timing-competitive overlaps per window |
| `tr_ovl_floor_fight` | Count of floor_fight overlaps per window |
| `tr_ovl_long_ratio` | (competitive + floor_fight) / total — long-overlap prevalence |
| `tr_ovl_short_ratio` | (simultaneous + backchannel) / total — brief overlap prevalence |

### Features derived from `context_label` (lexical, used in HMM)

| Feature | Definition |
|---|---|
| `tr_ovl_context_collaborative` | Count of overlaps with collaborative context_label per window |
| `tr_ovl_context_completion` | Count of sentence-completion overlaps per window |
| `tr_ovl_collaboration_index` | collaborative / total — fraction of overlaps with collaborative intent |

### Legacy features (available but not in primary HMM feature set)

| Feature | Note |
|---|---|
| `tr_overlap_count` | Total OVL count regardless of subtype |
| `tr_overlap_time_s` | Total overlap duration (seconds) |
| `tr_competitive_overlap` | Old mixed count (competitive + floor_fight, semantic labeling) — superseded |
| `tr_collaborative_overlap` | Old semantic collaborative count — superseded by `tr_ovl_collaboration_index` |

---

## Paper Methods Note (suggested text)

> Overlap segments were classified using a two-layer scheme. First, each overlap was assigned a **timing subtype** (`simultaneous`, `backchannel_ovl`, `smooth`, `competitive`, `floor_fight`) based purely on acoustic overlap duration and speaker start-time difference, with no text analysis. Second, a **context label** (`collaborative`, `completion`, or `competitive`) was applied using LLM-assisted rule-based detection on the speech text of the overlapping speakers: support cues (e.g., *yeah, exactly, agree, mmh*) triggered a `collaborative` label on any timing subtype; overlaps occurring when a prior speaker's utterance trailed off with an explicit ellipsis marker and the incoming speech contained content words received a `completion` label; and conflict cues (e.g., *no, wait, hold on*) triggered a `competitive` label on long-duration overlaps. Context-sensitive words (e.g., *right*, *okay*) were only counted as support when the speaker's full utterance was four words or fewer, to avoid false positives from discourse markers mid-sentence. All 37 transcripts in `transcripts/final/` were processed with `tools/relabel_overlaps.py` (July 2026).


---

## Classification Algorithm

The algorithm mirrors the `classifyOVLSubtype()` function in `tools/transcript_corrector.html`:

```
input: duration_ms, start_diff_ms, text_a, text_b

if start_diff_ms < 200       → simultaneous
if dur < 250 and support_cue  → backchannel_ovl
if dur < 300                  → smooth
if dur > 1000:
    if conflict_cue           → floor_fight
    if support_cue            → collaborative
    else                      → needs_review
if dur > 500:
    if conflict_cue           → competitive
    if support_cue            → collaborative
    else                      → needs_review
if conflict_cue               → competitive
if support_cue                → collaborative
else                          → backchannel_ovl
```

**Support cue words:** yes, yeah, yep, right, exactly, true, agree, correct, ok, okay, mmh, mm-hmm, uh-huh, i see, good point, and, plus, also

**Conflict cue words:** no, not really, i disagree, disagree, wrong, hold on, wait, stop, but, however, actually, let me finish, don't interrupt, you are missing the point

---

## Reclassification Pass (July 2026)

### Background

The original auto-classification pass in the corrector tool assigned `floor_fight` to all overlaps >1 s with no text context check, and `competitive` to overlaps 500 ms–1 s similarly. This resulted in systematic over-labelling of long overlaps as adversarial.

### Groups reviewed manually via corrector UI
Groups 07, 10, 14, 15, 16 — annotator reviewed each OVL row in the corrector against the audio and surrounding speech context.

### Groups reclassified programmatically (July 2026)
Groups 08, 09, 11, 12, 13 — same text-context algorithm applied via `_tmp_reclassify_ovl.py`. Originals preserved; reviewed copies stored as `*_reviewed.tsv` in `tools/`.

**Rows targeted per group (only `floor_fight`/`competitive` rows re-evaluated):**

| Group | Targeted | Reclassified |
|---|---|---|
| grp-08 | 68 | 42 |
| grp-09 | 23 | 17 |
| grp-11 | 47 | 36 |
| grp-12 | 46 | 23 |
| grp-13 | 7 | 5 |

---

## Final OVL Subtype Distribution (Full Sample, Post-Review)

### Totals across all groups

| Subtype | Count |
|---|---|
| collaborative | 363 |
| smooth | 247 |
| simultaneous | 208 |
| backchannel_ovl | 177 |
| floor_fight | 142 |
| competitive | 110 |
| needs_review | 100 |
| **Total OVL rows** | **1347** |

### By group

| Group | backchannel_ovl | collaborative | competitive | floor_fight | needs_review | simultaneous | smooth |
|---|---|---|---|---|---|---|---|
| grp-07 | 22 | 54 | 15 | 31 | 7 | 15 | 7 |
| grp-08 | 42 | 66 | 24 | 28 | 18 | 74 | 100 |
| grp-09 | 6 | 26 | 2 | 10 | 6 | 10 | 18 |
| grp-10 | 18 | 28 | 4 | 6 | 7 | 13 | 6 |
| grp-11 | 28 | 46 | 12 | 10 | 18 | 40 | 52 |
| grp-12 | 14 | 32 | 22 | 24 | 12 | 14 | 28 |
| grp-13 | 0 | 10 | 2 | 2 | 0 | 8 | 10 |
| grp-14 | 16 | 16 | 9 | 6 | 15 | 9 | 4 |
| grp-15 | 10 | 27 | 5 | 5 | 4 | 9 | 6 |
| grp-16 | 21 | 58 | 15 | 20 | 13 | 16 | 16 |

---

## Feature Pipeline Implications

- `tr_competitive_overlap` counts `competitive` + `floor_fight` (pooled, see `extract_transcript_features.py: OVL_COMPETITIVE`)
- `tr_cooperative_overlap` counts `collaborative` + `smooth` + `simultaneous` + `backchannel_ovl`
- `tr_conflict_overlap_ratio` = competitive / (competitive + cooperative); `needs_review` rows excluded from denominator
- `needs_review` rows are never double-counted — they contribute to `tr_overlap_count` and `tr_overlap_time_s` only

---

## Paper Methods Note (suggested text)

> Overlap segments were classified into seven subtypes using a rule-based algorithm applied to overlap duration and lexical cues in adjacent speech segments (support cues: *yeah, right, exactly, and*; conflict cues: *no, wait, hold on, but*). Overlaps shorter than 300 ms were classified as smooth or backchannel; overlaps exceeding 1 s with conflict cues as floor-fight; ambiguous cases were marked `needs_review` and excluded from the competitive/cooperative ratio. Groups 07, 10, 14, 15, and 16 were reviewed manually against audio; remaining groups received the same rule-based classification applied programmatically.
