# Labels Codebook — AffectAI Transcripts

> This codebook defines every annotation label used in transcript TSV files produced by the
> **AffectAI Transcript Corrector** (`tools/transcript_corrector.html`) and stored as
> `tools/transcript_grp-{grp}_{task}_{date}.tsv`.
>
> Last updated: 2026-07-01  
> Applies to all groups 07, 10, 14, 15 (T1–T4).

---

## 1. Row types (`type` column)

| Code | Full name | Speaker attribution | Notes |
|------|-----------|---------------------|-------|
| `SPK` | Speech segment | Individual (`P1`–`P4`) or `MODERATOR` | Core transcription unit. `text` = transcript of speech. |
| `SIL` | Silence / pause | `GROUP` | Auto-inserted by **Auto-fill SIL**. `text` = pause notation (see §4). |
| `BRE` | Audible breath | Individual | `text = [BRE]` |
| `FP` | Filler / hesitation | Individual | uh, um, er, hm |
| `LAU` | Laughter | Individual or `GROUP` | `text` = `[LAUGHTER]` / `[SOFT LAUGHTER]` / `[GROUP LAUGHTER]` |
| `BCK` | Backchannel | Individual | Brief supportive interjection while another speaker holds the floor. |
| `OVL` | Overlap | The interrupting speaker | Two or more speakers simultaneously active. Subtype encodes function (see §2). |
| `NOI` | Noise / artefact | `GROUP` | Chair movements, mic bumps, external sounds. Subtype encodes source (see §3). |
| `MOD_EVT` | Moderator event | `MODERATOR` | Non-verbal moderator actions (instructions, warnings, timer callouts). |

---

## 2. OVL subtypes (`subtype` column when `type == OVL`)

### Design rationale (2026-07-01 revision)

The original auto-classifier used **duration alone** as a proxy for conflict intent:
long overlap → `competitive`; short overlap → `smooth`. This was observed to be incorrect
because a long overlap can be affiliative (co-construction, enthusiastic agreement) or
ambiguous (neutral task progression with two speakers talking at once).

The revised classifier uses **both timing and lexical cues** from the overlapping SPK entries
to assign function, and falls back to `needs_review` when intent is ambiguous rather than
forcing a conflict label.

### Subtype definitions

| Subtype | Timing condition | Lexical condition | Confidence | Interpretation |
|---------|-----------------|-------------------|------------|----------------|
| `simultaneous` | start difference < 200 ms | any | `confident` | Both speakers begin at almost the same time; neither interrupts the other. Often happens at turn boundaries or during group-simultaneous responses. |
| `smooth` | overlap < 300 ms | no clear support or conflict | `confident` | Very short temporal overlap at a natural handoff point. Typical smooth turn transition. |
| `backchannel_ovl` | overlap < 250 ms | support cues present | `confident` | Short supportive interjection (yeah, right, exactly, mm-hmm) while original speaker continues. Does not signal attempt to take floor. |
| `collaborative` | any | support cues present, no conflict | `uncertain` | Completion or co-construction of the previous speaker's point; affiliative continuation; enthusiastic agreement causing overlap. Long overlaps with support cues land here instead of `floor_fight`. |
| `competitive` | > 500 ms or conflict cue present | conflict cue present | `confident` if cue present, `uncertain` otherwise | Interruption with a clear challenge or contradiction signal. Intent to take the floor or redirect discussion. |
| `floor_fight` | > 1000 ms | conflict cues present | `confident` | Sustained mutual overlap where both parties hold ground; associated with explicit disagreement, correction, or challenge. Requires conflict cue — long duration alone is not sufficient. |
| `needs_review` | > 500 ms | no support OR conflict cue | `uncertain` | Duration suggests a substantive overlap but the text contains neither clear support nor conflict markers. Annotator should review and reclassify manually if context clarifies intent. |

### Cue word lists

**Support cues** (indicate affiliative / cooperative intent):
- Agreement/acknowledgement: *yes, yeah, yep, right, exactly, true, agree, correct, ok, okay, mmh, mm-hmm, uh-huh, i see, good point*
- Additive connectives: *and, plus, also*

**Conflict cues** (indicate competitive / challenge intent):
- Disagreement/challenge: *no, not really, i disagree, disagree, wrong, hold on, wait, stop, but, however, actually*
- Floor-defence phrases: *let me finish, don't interrupt, you are missing*

> **Note:** These are heuristic cues applied to the SPK entries that bracket the OVL entry.
> A word like "actually" can be non-confrontational in some contexts; entries where the
> classification feels uncertain should use `confidence = uncertain` and, for important analyses,
> be manually reviewed.

### Classification decision tree

```
OVL detected (start_diff, duration, text_a, text_b)
│
├─ start_diff < 200 ms ─────────────────────────────── simultaneous  [confident]
│
├─ duration < 250 ms AND support cue ───────────────── backchannel_ovl [confident]
│
├─ duration < 300 ms ───────────────────────────────── smooth  [confident]
│
├─ duration > 1000 ms
│   ├─ conflict cue ────────────────────────────────── floor_fight  [confident]
│   ├─ support cue ─────────────────────────────────── collaborative  [uncertain]
│   └─ no cue ──────────────────────────────────────── needs_review  [uncertain]
│
├─ duration > 500 ms
│   ├─ conflict cue ────────────────────────────────── competitive  [confident]
│   ├─ support cue ─────────────────────────────────── collaborative  [uncertain]
│   └─ no cue ──────────────────────────────────────── needs_review  [uncertain]
│
└─ 300–500 ms
    ├─ conflict cue ────────────────────────────────── competitive  [uncertain]
    ├─ support cue ─────────────────────────────────── collaborative  [confident]
    └─ no cue ──────────────────────────────────────── backchannel_ovl [uncertain]
```

### Historical note: pre-2026-07-01 labels

Before the July 2026 revision, the auto-classifier used duration thresholds only:
- > 1000 ms → `floor_fight`
- > 500 ms → `competitive`
- < 300 ms → `smooth`
- otherwise → `backchannel_ovl`

This resulted in many affiliative long overlaps being incorrectly coded as `competitive`
or `floor_fight`. The `.bak.tsv` backups (`tools/transcript_grp-*.bak.tsv`) preserve the
pre-revision labels for traceability.

---

## 3. NOI subtypes

| Subtype | Description |
|---------|-------------|
| `noise` | General unidentified background noise |
| `artefact` | Recording artefact (clipping, dropout) |
| `cough` | Cough or sneeze |
| `chair` | Chair or table movement |
| `mic_bump` | Microphone bump or handling |
| `door` | Door or ambient room sound |

---

## 4. Pause notation (SIL text field)

| Notation | Duration |
|----------|----------|
| `(.)` | < 200 ms |
| `(0.3)` | 200–500 ms (actual seconds to 1 decimal) |
| `(0.7)` | 500 ms – 1 s |
| `(1.2)` | 1–2 s |
| `(2.5)` | > 2 s |

---

## 5. Confidence field values

| Value | Meaning |
|-------|---------|
| `confident` | Label is derived from clear lexical or timing evidence |
| `uncertain` | Label is a best estimate; context or intent is unclear |
| `manual` | Label was assigned manually by the annotator, overriding auto-classification |

---

## 6. Downstream analysis mapping

For analysis scripts, overlap subtypes are grouped as follows:

| Analysis category | Subtypes included |
|-------------------|-------------------|
| **Conflict / tension** | `competitive`, `floor_fight` |
| **Cooperative overlap** | `collaborative`, `backchannel_ovl`, `smooth`, `simultaneous` |
| **Ambiguous / exclude** | `needs_review` |

`needs_review` entries should be treated as missing data in significance tests unless
a manual review pass has resolved them.

---

## 7. Tools and scripts

| Tool | Purpose |
|------|---------|
| `tools/transcript_corrector.html` | Web-based annotation UI; **Classify OVL** button applies the decision tree above automatically |
| `tools/relabel_overlaps.py` | Batch-relabels existing TSV files using the same decision logic; preserves `.bak.tsv` originals |
| `tools/features/extract_transcript_features.py` | Aggregates TSV rows into per-group-task and per-participant feature tables |
| `tools/features/analyze_task2_tension.py` | Statistical analysis of conflict features across tasks |

---

## 8. Re-running relabeling

```bash
# Dry-run — shows what would change without writing
python tools/relabel_overlaps.py --dry-run

# Apply relabeling to all transcript_grp-*.tsv (backs up originals as .bak.tsv)
python tools/relabel_overlaps.py

# Preserve any manually-set collaborative labels during relabeling
python tools/relabel_overlaps.py --preserve-collaborative

# Restore original labels for one file
copy tools\transcript_grp-07_T2_2026-06-17.bak.tsv tools\transcript_grp-07_T2_2026-06-17.tsv
```

After relabeling, rebuild the derived feature tables:

```bash
python tools/features/extract_transcript_features.py --transcript-dir tools/_transcript_current
python tools/features/build_collective_feature_matrix.py
```
