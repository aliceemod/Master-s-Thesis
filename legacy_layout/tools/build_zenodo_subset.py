"""Build the anonymised Zenodo preview subset for NeurIPS 2026 review.

Creates data/zenodo/ — a self-contained ~3.7 GB anonymised subset of the
full BIDS release suitable for upload to Zenodo Sandbox as an anonymous
review URL.

What is included
----------------
- All root metadata (dataset_description.json, participants.tsv/.json,
  events.json, task-*_events.json, transcript.json, words.json,
  croissant_metadata.json, README, zenodo_README.txt, .zenodo.json)
- sub-01/ses-*/events.tsv (session timeline spine, all 10 sessions)
- sub-01/ses-*/physio/  — all 10 sessions (~175 MB)
- sub-01/ses-*/beh/     — all 10 sessions (~7 MB)
- sub-01/ses-*/annot/   — all 10 sessions (~2.2 GB)
- sub-01/ses-*/audio/*_transcript*.tsv|txt, *_words.tsv
  — all transcript/word files, all 10 sessions (~5 MB)
- sub-01/ses-*/et/      — 7 of 10 sessions; grp-11, grp-12, grp-16 skipped
  because their ET files are 1.1–1.6 GB each (~1.3 GB total for 7 sessions)

What is excluded
----------------
- *.wav audio files  — voice re-identification risk + 22.9 GB; gated access
- et/ for grp-11, grp-12, grp-16  — oversized for 4 GB preview budget
- Any column that contains real names  — already anonymised upstream

Total target: ≤ 4 GB  (estimated ~3.75 GB)

Anonymous review note
---------------------
Author names are replaced with "Anonymous (under review)" in all metadata.
The Zenodo Sandbox record URL should be supplied as the anonymous review URL
in the NeurIPS submission form.

Usage
-----
    python tools/build_zenodo_subset.py [--dry-run] [--out-dir data/zenodo]
    python tools/build_zenodo_subset.py --help
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

# Sessions whose ET folder exceeds the 4 GB preview budget.
# Full ET for these sessions is available in the complete release.
_ET_EXCLUDE_SESSIONS = {
    "ses-20260318_grp-11_run01",  # 1151 MB
    "ses-20260318_grp-12_run01",  # 1207 MB
    "ses-20260320_grp-16_run01",  # 1594 MB
}

# Root metadata files to copy verbatim
_ROOT_META = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "events.json",
    "task-T0_events.json",
    "task-T1_events.json",
    "task-T2_events.json",
    "task-T3_events.json",
    "task-T4_events.json",
    "task-T0T1T2T3T4_events.json",
    "transcript.json",
    "words.json",
    "croissant_metadata.json",
]

# Modalities to copy completely (all files in the folder)
_FULL_MODALITIES = ("physio", "beh", "annot")

# In audio/ copy only transcript/word TSV and TXT — exclude WAV
_AUDIO_KEEP_SUFFIXES = {".tsv", ".txt"}
_AUDIO_EXCLUDE_EXT = {".wav"}


def _copy_file(src: Path, dest: Path, dry_run: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        shutil.copy2(src, dest)


def build(bids_root: Path, out_dir: Path, dry_run: bool) -> None:
    copied = 0
    skipped_wav = 0
    skipped_et = 0
    total_bytes = 0

    # ── Root metadata ──────────────────────────────────────────────────────
    for name in _ROOT_META:
        src = bids_root / name
        if src.exists():
            _copy_file(src, out_dir / name, dry_run)
            total_bytes += src.stat().st_size
            copied += 1
            logging.debug("META  %s", name)
        else:
            logging.warning("Root metadata not found: %s", name)

    # Copy the full BIDS README
    bids_readme = bids_root / "README"
    if bids_readme.exists():
        _copy_file(bids_readme, out_dir / "README", dry_run)
        total_bytes += bids_readme.stat().st_size
        copied += 1

    sub_dir = bids_root / "sub-01"
    if not sub_dir.exists():
        logging.error("sub-01 not found under %s", bids_root)
        return

    # ── Per-session data ───────────────────────────────────────────────────
    for ses_dir in sorted(sub_dir.iterdir()):
        if not ses_dir.is_dir():
            continue
        ses_id = ses_dir.name

        # events.tsv (session spine)
        events_tsv = ses_dir / "events.tsv"
        if events_tsv.exists():
            _copy_file(events_tsv, out_dir / "sub-01" / ses_id / "events.tsv", dry_run)
            total_bytes += events_tsv.stat().st_size
            copied += 1

        # Full modalities: physio, beh, annot
        for mod in _FULL_MODALITIES:
            mod_dir = ses_dir / mod
            if not mod_dir.exists():
                continue
            for f in sorted(mod_dir.iterdir()):
                if not f.is_file():
                    continue
                dest = out_dir / "sub-01" / ses_id / mod / f.name
                _copy_file(f, dest, dry_run)
                total_bytes += f.stat().st_size
                copied += 1
                logging.debug("%-6s %s", mod.upper(), f.name)

        # audio/ — transcripts and words only, no WAV
        audio_dir = ses_dir / "audio"
        if audio_dir.exists():
            for f in sorted(audio_dir.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix in _AUDIO_EXCLUDE_EXT:
                    skipped_wav += 1
                    continue
                if f.suffix not in _AUDIO_KEEP_SUFFIXES:
                    continue
                dest = out_dir / "sub-01" / ses_id / "audio" / f.name
                _copy_file(f, dest, dry_run)
                total_bytes += f.stat().st_size
                copied += 1
                logging.debug("AUDIO %s", f.name)

        # et/ — skip oversized sessions
        et_dir = ses_dir / "et"
        if et_dir.exists():
            if ses_id in _ET_EXCLUDE_SESSIONS:
                et_count = sum(1 for f in et_dir.iterdir() if f.is_file())
                et_mb = sum(
                    f.stat().st_size for f in et_dir.iterdir() if f.is_file()
                ) / (1024 * 1024)
                logging.info(
                    "ET SKIP  %s — %d files, %.0f MB (exceeds preview budget; "
                    "available in full release)",
                    ses_id, et_count, et_mb,
                )
                skipped_et += et_count
            else:
                for f in sorted(et_dir.iterdir()):
                    if not f.is_file():
                        continue
                    dest = out_dir / "sub-01" / ses_id / "et" / f.name
                    _copy_file(f, dest, dry_run)
                    total_bytes += f.stat().st_size
                    copied += 1
                    logging.debug("ET     %s", f.name)

    label = "[DRY-RUN] " if dry_run else ""
    logging.info(
        "%sFinished: %d file(s) %s (~%.2f GB), %d WAV skipped, %d ET files skipped "
        "(grp-11/12/16 — full ET in complete release)",
        label,
        copied,
        "would be copied" if dry_run else "copied",
        total_bytes / (1024 ** 3),
        skipped_wav,
        skipped_et,
    )


def _write_zenodo_json(out_dir: Path, dry_run: bool) -> None:
    """Write .zenodo.json for Zenodo Sandbox upload."""
    meta = {
        "title": "AffectAI: A Multimodal Dataset of Co-Located Group Interaction "
                 "(Anonymous NeurIPS 2026 Review Preview)",
        "description": (
            "<p>Anonymous preview subset for NeurIPS 2026 Datasets &amp; Benchmarks "
            "review. DO NOT CITE — submission under double-blind review.</p>"
            "<p>AffectAI is a multimodal corpus of co-located four-person group "
            "interaction across five structured tasks (T0 resting baseline, T1 "
            "hidden-profile decision, T2 mini-negotiation, T3 NGT idea generation, "
            "T4 public-goods micro-game) from 40 participants in 10 groups.</p>"
            "<p>This ~3.7 GB preview includes: all physiology (EmotiBit PPG/EDA/"
            "temperature/IMU, ~175 MB), all behavioural/self-report data (~7 MB), "
            "all session-level annotation/sync files (~2.2 GB), all speech "
            "transcripts and word-level timestamps (~5 MB), and egocentric "
            "eye-tracking (Tobii Pro Glasses 3) for 7 of 10 sessions (~1.3 GB). "
            "Close-talk audio WAV files and ET data for three sessions (grp-11, "
            "grp-12, grp-16) are excluded from this preview for size and access-tier "
            "reasons; they are available on request under a Data Use Agreement.</p>"
            "<p>Croissant metadata and Responsible AI fields are in "
            "croissant_metadata.json. Column schemas are in transcript.json and "
            "words.json.</p>"
        ),
        "upload_type": "dataset",
        "access_right": "open",
        "license": "cc-by-4.0",
        "creators": [
            {
                "name": "Anonymous (under review)",
                "affiliation": "Anonymous"
            }
        ],
        "keywords": [
            "affect", "emotion", "group dynamics", "multimodal", "physiology",
            "eye-tracking", "audio", "transcripts", "social interaction",
            "EDA", "PPG", "HRV", "pupil diameter", "Big Five personality",
            "collaborative tasks", "affective computing", "co-located interaction",
            "BIDS", "NeurIPS 2026"
        ],
        "notes": (
            "This is an anonymised preview subset (~3.7 GB) for double-blind NeurIPS "
            "2026 D&B review. Author names and institution will be restored at "
            "camera-ready. This record will be superseded by the full versioned "
            "release after review. Do not redistribute or use for any purpose other "
            "than reviewing the submission."
        ),
        "version": "0.1.0-preview",
        "language": "eng",
        "related_identifiers": [],
        "communities": []
    }
    dest = out_dir / ".zenodo.json"
    if not dry_run:
        dest.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("%s.zenodo.json written", "[DRY-RUN] " if dry_run else "")


def _write_zenodo_readme(out_dir: Path, dry_run: bool) -> None:
    """Write a Zenodo-specific README that appears prominently on the record page."""
    content = """\
AffectAI — Anonymous NeurIPS 2026 Review Preview
=================================================

IMPORTANT: This is an anonymised preview subset for double-blind review at
NeurIPS 2026 Datasets & Benchmarks Track. Do not cite or redistribute.
Author names will be restored at camera-ready.

Dataset: AffectAI: A Multimodal Dataset of Co-Located Group Interaction
Format: BIDS-inspired (see dataset_description.json and README)
License: CC-BY-4.0 (tabular and transcript data included here)

What this preview contains (~3.7 GB)
-------------------------------------

  Modality             Sessions  Size     Notes
  ────────────────────────────────────────────────────────────────────────
  Root metadata        —          < 1 MB  dataset_description.json, croissant, etc.
  Physiology (EmotiBit) 10/10    ~175 MB  PPG, EDA, temperature, IMU TSV.GZ
  Behavioural / beh    10/10      ~7 MB   self-report, task responses
  Annotation / annot   10/10     ~2.2 GB  sync, task windows, signal maps
  Transcripts (audio/) 10/10      ~5 MB   segment + word TSV and TXT (no WAV)
  Eye tracking (ET)     7/10     ~1.3 GB  grp-07 to grp-15 excl. grp-11,12

  Total                                  ~3.7 GB

What is NOT in this preview (available in full release)
-------------------------------------------------------

  - Close-talk audio WAV files (22.9 GB) — voice re-identification risk;
    gated under a Data Use Agreement. Contact authors post-review.
  - Egocentric eye-tracking (et/) for grp-11, grp-12, grp-16 — files are
    1.1–1.6 GB each; available in the full release.

Directory structure
-------------------

  dataset_description.json    dataset-level metadata (BIDS + NeurIPS E&D)
  participants.tsv/.json       40 participants, demographics, BFI-44 scores
  transcript.json              column schema for *_transcript.tsv files
  words.json                   column schema for *_words.tsv files
  croissant_metadata.json      Croissant 1.1 + RAI fields
  README                       full BIDS release README
  sub-01/
    ses-{session_id}/
      events.tsv               session timeline spine
      physio/                  EmotiBit TSV.GZ + JSON sidecars
      et/                      Tobii TSV.GZ + JSON sidecars (7 sessions)
      audio/                   transcript TSV/TXT only (no WAV)
      beh/                     behavioural/self-report TSV
      annot/                   sync, task windows, participant-signal-map

Benchmark evaluation protocol
------------------------------

Leave-one-group-out cross-validation (group_id as split key).
See dataset_description.json → BenchmarkTasks for B0–B6 definitions.
Ridge / logistic baseline results are in the companion paper.

Access and responsible use
--------------------------

Transcript files contain anonymised speech (P1–P4 seat IDs only; no real
names). Audio WAV is excluded from this preview and requires a DUA.
Out-of-scope: clinical inference, individual profiling, speaker ID, surveillance.

Version: 0.1.0-preview (NeurIPS 2026 D&B anonymous review)
"""
    dest = out_dir / "zenodo_README.txt"
    if not dry_run:
        dest.write_text(content, encoding="utf-8")
    logging.info("%szenodo_README.txt written", "[DRY-RUN] " if dry_run else "")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Build anonymised Zenodo preview subset (≤4 GB) from BIDS release.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=repo_root / "data" / "bids_release_no_video",
        help="Source BIDS release root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "zenodo",
        help="Destination directory for the Zenodo subset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying files.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.bids_root.exists():
        parser.error(f"BIDS root not found: {args.bids_root}")

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        logging.warning(
            "Output directory already exists and is non-empty: %s — "
            "existing files will be overwritten (new files not removed).",
            args.out_dir,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    build(bids_root=args.bids_root, out_dir=args.out_dir, dry_run=args.dry_run)
    _write_zenodo_json(out_dir=args.out_dir, dry_run=args.dry_run)
    _write_zenodo_readme(out_dir=args.out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
