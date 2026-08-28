# Audio Annotation Pipeline

Automatically transcribe multi-speaker group recordings into a time-stamped, speaker-labeled transcript.

## What it does

In the AffectAI study, each session records 4 participants sitting around a table. Each person wears a **DPA 4060 close-talk microphone** clipped near their mouth. The recordings are split by task (T0-T4), producing 20 WAV files per session (4 mics x 5 tasks).

This pipeline takes those raw WAV files and produces a **readable conversation transcript**:

```
[02:35] Olena (P3): That's an interesting one.

[03:05] Daniela (P4): Yeah, I actually gave two to the community.

[03:14] Olena (P3): Ah, I got it. Oh, wow. So person one gave all.

[05:06] Jana (P2): So in reality, you were selfish.

[05:08] Harshavardhan (P1): No.
```

## How it works

The pipeline decides who said what using three layers — each one catches what the previous one missed.

### Step 1: Transcribe each mic independently

Each of the 4 microphone WAV files is transcribed using **WhisperX**, which combines:

- **faster-whisper** (large-v3-turbo) for speech-to-text
- **wav2vec2 forced alignment** for accurate word-level timestamps

### Step 2: Bleed rejection (energy comparison across close-talk mics)

When one person speaks, their voice bleeds into the other 3 mics at lower volume. For each transcribed segment, we compare loudness across all 4 mics. The loudest mic's owner is the real speaker — copies on other mics are discarded.

### Step 3: Confident vs uncertain (energy ratio)

For each surviving segment, we check how much louder the winning mic was vs the next loudest:

- **Ratio >= 2.0**: confident — clearly close-talk speech, labeled as that mic's participant
- **Ratio < 2.0**: uncertain — could be a participant speaking quietly or moderator from room speaker

### Step 4: Room mic resolution (PanaCast 50)

If a PanaCast 50 room mic audio file is available (`*panacast*50*audio*.wav` in the session directory), uncertain segments are checked against it:

- **Close-talk mic louder than room mic** (mic/p50 > 0.3): confirmed as participant
- **Room mic louder** (mic/p50 < 0.3): still uncertain, passed to next step

### Step 5: Speaker embedding verification (pyannote ECAPA-TDNN)

Voice profiles are built for each participant from their clearest segments (ratio >= 3.0, duration >= 1.5s) pooled across all tasks. Remaining uncertain segments are compared:

- **Voice matches a participant AND that participant's mic is loudest**: confirmed as that participant
- **No match**: labeled as MODERATOR (last resort — no close-talk signal, no voice match)

## Mic-to-participant mapping

Fixed by the lab setup:

| Mic | File suffix | Seat | Position |
|-----|------------|------|----------|
| Mic 9 | `_mic9_aud.wav` | **P1** | back-right |
| Mic 10 | `_mic10_aud.wav` | **P2** | front-right |
| Mic 11 | `_mic11_aud.wav` | **P3** | front-left |
| Mic 12 | `_mic12_aud.wav` | **P4** | back-left |

Participant names are read from `participant_map.tsv` in each session directory.

## Setup

### Prerequisites

- Python 3.10+

### Install

```bash
cd affectai-data-processing
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install whisperx torch torchaudio soundfile
```

## Usage

### Single session

```bash
python tools/audio_annotate.py /path/to/ses-20260312_grp-07_run01
```

### Multiple sessions

```bash
python tools/audio_annotate.py /data/ses-20260312_grp-07_run01 /data/ses-20260315_grp-08_run01

# Or use a glob
python tools/audio_annotate.py /data/ses-*
```

### BIDS-format sessions (e.g. from `bids_release_no_video.zip`)

The same `audio_annotate.py` works directly on BIDS-style session directories
(audio filenames `..._acq-av-dpa-mic9-aud-dpa-mic9-aud_audio.wav` and the
`annot/*_group_participants.tsv` participant map). No special flag needed —
just point at the session directory:

```bash
# After extracting the zip into a flat layout, e.g.
#   ~/Projects/affectai_sessions/ses-20260312_grp-07_run01/
#   ~/Projects/affectai_sessions/ses-20260313_grp-08_run01/
#   ...
caffeinate -i -s python tools/audio_annotate.py ~/Projects/affectai_sessions/ses-* \
  --device cpu --compute-type int8
```

If the session has no `participant_map.tsv` and no `annot/*_group_participants.tsv`,
participants are labeled with their seat ID (`P1`, `P2`, `P3`, `P4`).

### Options

```bash
# Default (large-v3-turbo, int8, CPU)
python tools/audio_annotate.py /path/to/session

# With NVIDIA GPU
python tools/audio_annotate.py /path/to/session --device cuda --compute-type float16

# Smaller model for speed
python tools/audio_annotate.py /path/to/session --model medium

# Prevent Mac sleep
caffeinate -i -s python tools/audio_annotate.py /path/to/session

# Force re-transcription (ignore cache)
python tools/audio_annotate.py /path/to/session --no-skip-existing
```

### All CLI options

```
python tools/audio_annotate.py --help

positional arguments:
  session_dirs          One or more session directories

options:
  --output-dir, -o      Output directory (default: session_dir/audio_annot/)
  --tasks, -t           Specific tasks (e.g. T0 T1). Default: all
  --device, -d          cpu or cuda (default: cpu)
  --model, -m           Whisper model (default: large-v3-turbo)
  --compute-type        float32, float16, int8 (default: int8)
  --language, -l        Language code (default: en)
  --batch-size, -b      WhisperX batch size (default: 8)
  --energy-ratio        Bleed rejection threshold (default: 1.0)
  --no-skip-existing    Re-transcribe even if cached results exist
  --verbose, -v         Enable debug logging
```

## Output files

For each session, outputs are saved under `audio_annot/`:

```
ses-20260312_grp-07_run01/
  audio_annot/
    T0/
      transcript.txt            <-- clean readable transcript
      master_transcript.tsv     <-- detailed TSV for analysis
      master_words.tsv          <-- word-level timestamps
      mic9_transcript.json      <-- cached per-mic (P1)
      mic10_transcript.json     <-- cached per-mic (P2)
      mic11_transcript.json     <-- cached per-mic (P3)
      mic12_transcript.json     <-- cached per-mic (P4)
    T1/ ... T4/                 <-- same structure
```

### transcript.txt

Clean conversation format:

```
[01:15] Olena (P3): That's an interesting one.

[02:08] Harshavardhan (P1): Okay.

[02:19] Moderator: Now we move to the discussion part.
```

### master_transcript.tsv

| Column | Description |
|--------|-------------|
| `onset` | Start time in seconds |
| `duration` | Duration in seconds |
| `speaker` | `P1_firstname_lastname`, `P2_...`, or `MODERATOR` |
| `text` | Transcribed text |
| `energy` | RMS energy |
| `energy_ratio` | How much louder this mic was vs the next loudest |
| `mic` | Source microphone or `shared` for moderator |
| `confidence` | `confident` or `uncertain` |

### Room mic (optional but recommended)

Place the PanaCast 50 room mic audio file in the session directory:

```
ses-20260312_grp-07_run01/
  jabra_panacast_50_vid_audio.wav     <-- room mic
  audio/                               <-- close-talk mics
```

The pipeline automatically detects it and uses it to resolve uncertain segments. Without it, uncertain segments are resolved by speaker embeddings only.

## Backchannel recovery (second pass)

The bleed-rejection step is conservative — it drops a segment unless the
owner mic is the loudest. That's correct for full speech turns, but it
silently removes backchannels (short "yeah", "mhm", "okay", "right" while
another participant is the dominant speaker). After the main run, a
second pass over the cached per-mic WhisperX output rescues these:

```bash
python tools/audio_refine_backchannels.py /path/to/ses-* 
```

For each task, this produces (alongside the originals — originals untouched):

- `master_transcript_with_backchannels.tsv` — original master + recovered
  segments, marked `confidence=backchannel`
- `transcript_with_backchannels.txt` — readable conversation with `[bc]`
  markers on recovered lines

The refinement is energy-only (no GPU, no embeddings). It looks for short
backchannel-type words (yeah, yes, mhm, okay, right, sure, exactly, no, …)
that exist in the per-mic WhisperX cache but aren't already covered by a
master segment from the same speaker, and accepts them when the owner mic
is meaningfully louder than its own quiet baseline AND at least roughly
comparable to the loudest competing mic in the tight word window. Tunables
are at the top of `tools/audio_refine_backchannels.py`.

## Caching

Per-mic WhisperX transcription is cached as JSON. On re-runs:
- Transcription is skipped if cache exists (fast)
- Bleed filter, merge, P50, and embeddings re-run every time (instant)
- Use `--no-skip-existing` to force full re-transcription

## Source code

```
src/affectai_capture/audio/
  __init__.py
  transcribe.py     # WhisperX transcription + energy-based bleed rejection
  merge.py          # Energy ratio classification (confident vs uncertain)
  room_mic.py       # PanaCast 50 room mic comparison
  embeddings.py     # Speaker profile building + voice verification
  events.py         # Phase window parsing from events.tsv
  io.py             # Save/load TSV, JSON, readable transcript

tools/
  audio_annotate.py # CLI entry point
```

## Processing times

| Model | Compute | Device | Time per mic (8 min audio) |
|-------|---------|--------|---------------------------|
| large-v3-turbo | int8 | CPU | ~5-8 min |
| large-v3-turbo | float16 | CUDA | ~30 sec |
| medium | int8 | CPU | ~2-3 min |

First run transcribes all mics (~2 hours on CPU for a full session). Re-runs with cached transcripts complete in seconds.
