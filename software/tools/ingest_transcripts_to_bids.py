"""Ingest audio transcripts into the BIDS release tree.

Copies and renames transcript files from:
    results/audio/transcripts/ses-{id}/T{0-4}/
to:
    data/bids_release_no_video/sub-01/ses-{id}/audio/

BIDS naming:
    master_transcript.tsv
        → sub-01_ses-{id}_task-T{n}_run-01_transcript.tsv
    master_transcript_with_backchannels.tsv
        → sub-01_ses-{id}_task-T{n}_run-01_desc-withBackchannels_transcript.tsv
    master_words.tsv
        → sub-01_ses-{id}_task-T{n}_run-01_words.tsv
    transcript.txt
        → sub-01_ses-{id}_task-T{n}_run-01_transcript.txt
    transcript_with_backchannels.txt
        → sub-01_ses-{id}_task-T{n}_run-01_desc-withBackchannels_transcript.txt

Usage:
    python tools/ingest_transcripts_to_bids.py [--dry-run]
    python tools/ingest_transcripts_to_bids.py --bids-root <path> --transcripts-root <path>
"""

import argparse
import logging
import shutil
from pathlib import Path

# Source file → (desc entity, suffix, extension)
# desc=None means no desc- entity in the BIDS name
_FILE_MAP: dict[str, tuple[str | None, str, str]] = {
    "master_transcript.tsv":                  (None,                "transcript", ".tsv"),
    "master_transcript_with_backchannels.tsv": ("withBackchannels", "transcript", ".tsv"),
    "master_words.tsv":                        (None,                "words",      ".tsv"),
    "transcript.txt":                          (None,                "transcript", ".txt"),
    "transcript_with_backchannels.txt":        ("withBackchannels", "transcript", ".txt"),
}

# T0..T4 folder names in the transcripts tree
_TASK_DIRS = {"T0", "T1", "T2", "T3", "T4"}


def _bids_name(sub: str, ses: str, task: str, run: str,
               desc: str | None, suffix: str, ext: str) -> str:
    entities = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"
    if desc:
        entities += f"_desc-{desc}"
    return f"{entities}_{suffix}{ext}"


def ingest(transcripts_root: Path, bids_root: Path, dry_run: bool) -> None:
    copied = 0
    skipped = 0
    missing_dest = 0

    for ses_dir in sorted(transcripts_root.iterdir()):
        if not ses_dir.is_dir():
            continue
        ses_id = ses_dir.name  # e.g. ses-20260312_grp-07_run01

        bids_audio_dir = bids_root / "sub-01" / ses_id / "audio"
        if not bids_audio_dir.exists():
            logging.warning("BIDS audio dir not found, skipping: %s", bids_audio_dir)
            missing_dest += 1
            continue

        # ses_id folder name is "ses-20260312_grp-07_run01"; strip the "ses-" prefix
        # so the BIDS name builder can re-attach it: sub-01_ses-20260312_grp-07_run01_...
        ses_value = ses_id.removeprefix("ses-")

        for task_dir in sorted(ses_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name not in _TASK_DIRS:
                continue
            task = task_dir.name  # T0..T4

            for src_name, (desc, suffix, ext) in _FILE_MAP.items():
                src = task_dir / src_name
                if not src.exists():
                    logging.debug("Source file not found (expected for grp-13 T3/T4): %s", src)
                    continue

                dest_name = _bids_name(
                    sub="01",
                    ses=ses_value,
                    task=task,
                    run="01",
                    desc=desc,
                    suffix=suffix,
                    ext=ext,
                )
                dest = bids_audio_dir / dest_name

                if dest.exists():
                    logging.debug("Already exists, skipping: %s", dest.name)
                    skipped += 1
                    continue

                logging.info("%s → %s", src.relative_to(transcripts_root.parent.parent.parent),
                             dest.relative_to(bids_root.parent))
                if not dry_run:
                    shutil.copy2(src, dest)
                copied += 1

    label = "[DRY-RUN] " if dry_run else ""
    logging.info(
        "%sFinished: %d file(s) %s, %d skipped (already present), %d session(s) with missing BIDS audio dir",
        label,
        copied,
        "would be copied" if dry_run else "copied",
        skipped,
        missing_dest,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Ingest audio transcripts into the BIDS release tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transcripts-root",
        type=Path,
        default=repo_root / "results" / "audio" / "transcripts",
        help="Root of the per-session transcript tree.",
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=repo_root / "data" / "bids_release_no_video",
        help="BIDS dataset root to write into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying any files.",
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

    if not args.transcripts_root.exists():
        parser.error(f"Transcripts root not found: {args.transcripts_root}")
    if not args.bids_root.exists():
        parser.error(f"BIDS root not found: {args.bids_root}")

    ingest(
        transcripts_root=args.transcripts_root,
        bids_root=args.bids_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
