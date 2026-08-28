#!/usr/bin/env python3
"""CLI tool to run the audio annotation pipeline on one or more sessions.

Usage:
    # Single session
    python tools/audio_annotate.py /path/to/ses-20260312_grp-07_run01

    # Multiple sessions
    python tools/audio_annotate.py /path/to/ses-20260312_grp-07_run01 /path/to/ses-20260315_grp-08_run01

    # All sessions in a directory
    python tools/audio_annotate.py /path/to/data/ses-*

    # Specific tasks and model options
    python tools/audio_annotate.py /path/to/session --tasks T0 T1 --model large-v3 --compute-type int8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src/ to path so we can import the audio module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from affectai_capture.audio.pipeline import run_pipeline

log = logging.getLogger(__name__)


def process_session(
    session_dir: Path, args: argparse.Namespace
) -> dict[str, list] | None:
    """Run the pipeline on a single session. Returns results or None on error."""
    if not session_dir.is_dir():
        log.error("Not a directory: %s", session_dir)
        return None

    if not (session_dir / "audio").is_dir():
        log.error("No audio/ folder in %s", session_dir)
        return None

    output_dir = args.output_dir if args.output_dir else None

    return run_pipeline(
        session_dir=session_dir,
        output_dir=output_dir,
        tasks=args.tasks,
        device=args.device,
        model_name=args.model,
        compute_type=args.compute_type,
        language=args.language,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        energy_ratio_threshold=args.energy_ratio,
        skip_existing=not args.no_skip_existing,
    )


def print_summary(
    all_results: dict[str, dict[str, list]],
    elapsed: float,
) -> None:
    """Print a summary of all processed sessions."""
    print("\n" + "=" * 70)
    print("AUDIO ANNOTATION SUMMARY")
    print("=" * 70)

    grand_total = 0
    for session_name, results in sorted(all_results.items()):
        session_total = 0
        print(f"\n  Session: {session_name}")
        print(f"  {'─' * 50}")
        for task, segments in sorted(results.items()):
            speakers = {}
            for seg in segments:
                speakers[seg.speaker] = speakers.get(seg.speaker, 0) + 1
            session_total += len(segments)
            print(f"    {task}: {len(segments)} segments")
            for spk, count in sorted(speakers.items()):
                print(f"      {spk}: {count}")
        print(f"    {'─' * 40}")
        print(f"    Session total: {session_total} segments")
        grand_total += session_total

    print(f"\n  {'=' * 50}")
    print(f"  Grand total: {grand_total} segments across {len(all_results)} session(s)")
    print(f"  Elapsed: {elapsed / 60:.1f} minutes")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio annotation pipeline: transcribe close-talk mics and merge into master transcript.",
        epilog="Examples:\n"
               "  %(prog)s /data/ses-20260312_grp-07_run01\n"
               "  %(prog)s /data/ses-* --model large-v3 --compute-type int8\n"
               "  %(prog)s session1/ session2/ --tasks T0 T2\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "session_dirs",
        type=Path,
        nargs="+",
        help="One or more session directories (each must contain audio/ and participant_map.tsv)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: each session_dir/audio_annot/)",
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        default=None,
        help="Specific tasks to process (e.g. T0 T1). Default: all",
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--model", "-m",
        default="large-v3",
        help="Whisper model name (default: large-v3)",
    )
    parser.add_argument(
        "--compute-type",
        default="float32",
        choices=["float32", "float16", "int8"],
        help="Compute type (default: float32). Use int8 for faster CPU inference.",
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="WhisperX batch size (default: 8)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote models",
    )
    parser.add_argument(
        "--energy-ratio",
        type=float,
        default=1.0,
        help="Energy ratio threshold for bleed rejection (default: 1.5). "
             "A mic must be this factor louder than all others to keep a segment.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-transcribe even if per-mic JSON already exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start = time.time()
    all_results: dict[str, dict[str, list]] = {}
    failed: list[str] = []

    for session_dir in args.session_dirs:
        session_name = session_dir.name
        log.info("=" * 60)
        log.info("Processing session: %s", session_name)
        log.info("=" * 60)

        results = process_session(session_dir, args)
        if results is None:
            failed.append(session_name)
            continue

        all_results[session_name] = results

    elapsed = time.time() - start
    print_summary(all_results, elapsed)

    if failed:
        print(f"\n  FAILED sessions: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
