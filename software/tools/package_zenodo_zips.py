"""Package the GroupAffect-4 BIDS release into Zenodo-compatible zip files.

Zenodo limits each record to 100 files.  This script creates a small set of
logically grouped zip archives for two separate Zenodo records:

Subset packaging  (~3.7 GB, 6 files)  --mode subset
  → data/zenodo_zips_subset/
---------------------------------------------
  affectai_metadata.zip          Root JSON/TSV + README_subset (as README)
                                 + README_full for reference          < 1 MB
  affectai_physio.zip            All physio TSV.GZ + sidecars       ~175 MB
  affectai_beh.zip               All beh TSV                          ~7 MB
  affectai_transcripts.zip       All transcript/word TSV+TXT           ~5 MB
  affectai_annot.zip             All annot files                     ~2.2 GB
  affectai_et_grp07to15.zip      ET 7 sessions (grp-11/12/16 excl.) ~1.3 GB

Full release packaging  (~30 GB, 16 files)  --mode full
  → data/zenodo_zips_full/
--------------------------------------------------
  affectai_metadata.zip          Root JSON/TSV + README (as README)
                                 + README_subset for reference        < 1 MB
  affectai_physio.zip            All physio TSV.GZ + sidecars       ~175 MB
  affectai_beh.zip               All beh TSV                          ~7 MB
  affectai_transcripts.zip       All transcript/word TSV+TXT           ~5 MB
  affectai_annot.zip             All annot files                     ~2.2 GB
  affectai_et_all.zip            ET all 10 sessions                  ~5.3 GB
  affectai_audio_grp-07.zip  }
  ...                        }   one WAV zip per session            ~22.9 GB
  affectai_audio_grp-16.zip  }

Usage
-----
    python tools/package_zenodo_zips.py --mode subset --dry-run
    python tools/package_zenodo_zips.py --mode subset
    python tools/package_zenodo_zips.py --mode full --dry-run
    python tools/package_zenodo_zips.py --mode full
    python tools/package_zenodo_zips.py --help
"""

import argparse
import logging
import zipfile
from pathlib import Path

# Sessions whose ET is too large for the 4 GB subset budget
_ET_EXCLUDE_SUBSET = {
    "ses-20260318_grp-11_run01",
    "ses-20260318_grp-12_run01",
    "ses-20260320_grp-16_run01",
}

# Root metadata files included in every release mode (READMEs handled separately)
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
    ".zenodo.json",
]


def _add_file(zf: zipfile.ZipFile, src: Path, arcname: str, dry_run: bool,
              stats: dict) -> None:
    size = src.stat().st_size
    stats["files"] += 1
    stats["bytes"] += size
    if not dry_run:
        zf.write(src, arcname)


def _open_zip(path: Path, dry_run: bool) -> zipfile.ZipFile:
    if dry_run:
        return zipfile.ZipFile(path, "w", zipfile.ZIP_STORED)
    path.parent.mkdir(parents=True, exist_ok=True)
    return zipfile.ZipFile(path, "w", zipfile.ZIP_STORED)


def _log_zip(name: str, stats: dict, dry_run: bool) -> None:
    label = "[DRY-RUN] " if dry_run else ""
    logging.info(
        "%s%-45s  %4d files  %6.0f MB",
        label, name, stats["files"], stats["bytes"] / (1024 * 1024),
    )


def _iter_mod(ses_dir: Path, modality: str):
    """Yield all files under ses_dir/modality/."""
    mod_dir = ses_dir / modality
    if mod_dir.exists():
        yield from sorted(f for f in mod_dir.rglob("*") if f.is_file())


def _arcname(bids_root: Path, f: Path) -> str:
    """Relative arc-path inside the zip, rooted at bids_root parent."""
    return str(f.relative_to(bids_root.parent)).replace("\\", "/")


def _bids_arc(bids_root: Path, filename: str) -> str:
    """Arc-path for a root-level BIDS file by name."""
    rel = str(bids_root.relative_to(bids_root.parent)).replace("\\", "/")
    return f"{rel}/{filename}"


def build_metadata_zip(bids_root: Path, out_dir: Path, mode: str,
                       dry_run: bool, repo_root: Path | None = None) -> None:
    """Build affectai_metadata.zip.

    The primary README inside the archive matches the release:
      subset mode → README_subset stored as README; full README stored as README_full
      full mode   → README stored as README; README_subset also included as README_subset
    """
    name = "affectai_metadata.zip"
    if repo_root is None:
        repo_root = bids_root.parent
    stats = {"files": 0, "bytes": 0}
    dest = out_dir / name
    with _open_zip(dest, dry_run) as zf:
        # Core metadata files
        for fname in _ROOT_META:
            src = bids_root / fname
            if src.exists():
                _add_file(zf, src, _arcname(bids_root, src), dry_run, stats)

        # READMEs — mode-specific primary README + cross-reference copy
        readme_full = bids_root / "README"
        readme_subset = bids_root / "README_subset"
        if mode == "subset":
            # Primary README for this Zenodo record is the subset README
            if readme_subset.exists():
                _add_file(zf, readme_subset, _bids_arc(bids_root, "README"),
                          dry_run, stats)
            if readme_full.exists():
                _add_file(zf, readme_full, _bids_arc(bids_root, "README_full"),
                          dry_run, stats)
        else:
            # Primary README for this Zenodo record is the full README
            if readme_full.exists():
                _add_file(zf, readme_full, _bids_arc(bids_root, "README"),
                          dry_run, stats)
            if readme_subset.exists():
                _add_file(zf, readme_subset, _bids_arc(bids_root, "README_subset"),
                          dry_run, stats)

        # events.tsv per session (timeline spine)
        for ses_dir in sorted((bids_root / "sub-01").iterdir()):
            if not ses_dir.is_dir():
                continue
            ev = ses_dir / "events.tsv"
            if ev.exists():
                _add_file(zf, ev, _arcname(bids_root, ev), dry_run, stats)

        # analysis/ scripts — if present at repo root
        analysis_dir = repo_root / "analysis"
        if analysis_dir.exists():
            for f in sorted(analysis_dir.rglob("*")):
                if f.is_file():
                    arc = "analysis/" + str(f.relative_to(analysis_dir)).replace("\\", "/")
                    _add_file(zf, f, arc, dry_run, stats)

    _log_zip(name, stats, dry_run)


def build_modality_zip(bids_root: Path, out_dir: Path, modality: str,
                       zip_name: str, sessions: list[str] | None,
                       dry_run: bool) -> None:
    stats = {"files": 0, "bytes": 0}
    dest = out_dir / zip_name
    with _open_zip(dest, dry_run) as zf:
        for ses_dir in sorted((bids_root / "sub-01").iterdir()):
            if not ses_dir.is_dir():
                continue
            if sessions and ses_dir.name not in sessions:
                continue
            for f in _iter_mod(ses_dir, modality):
                _add_file(zf, f, _arcname(bids_root, f), dry_run, stats)
    _log_zip(zip_name, stats, dry_run)


def build_audio_transcript_zip(bids_root: Path, out_dir: Path,
                                dry_run: bool) -> None:
    """Transcript/word TSV+TXT from audio/ — no WAV."""
    name = "affectai_transcripts.zip"
    stats = {"files": 0, "bytes": 0}
    dest = out_dir / name
    with _open_zip(dest, dry_run) as zf:
        for ses_dir in sorted((bids_root / "sub-01").iterdir()):
            if not ses_dir.is_dir():
                continue
            audio_dir = ses_dir / "audio"
            if not audio_dir.exists():
                continue
            for f in sorted(audio_dir.iterdir()):
                if f.is_file() and f.suffix in {".tsv", ".txt"}:
                    _add_file(zf, f, _arcname(bids_root, f), dry_run, stats)
    _log_zip(name, stats, dry_run)


def build_audio_wav_zip(bids_root: Path, out_dir: Path, ses_id: str,
                        dry_run: bool) -> None:
    """One zip per session containing only WAV files."""
    grp = next(
        (part for part in ses_id.split("_") if part.startswith("grp-")), ses_id
    )
    name = f"affectai_audio_{grp}.zip"
    stats = {"files": 0, "bytes": 0}
    dest = out_dir / name
    ses_dir = bids_root / "sub-01" / ses_id
    audio_dir = ses_dir / "audio"
    if not audio_dir.exists():
        logging.warning("audio/ not found for %s — skipping", ses_id)
        return
    with _open_zip(dest, dry_run) as zf:
        for f in sorted(audio_dir.iterdir()):
            if f.is_file() and f.suffix == ".wav":
                _add_file(zf, f, _arcname(bids_root, f), dry_run, stats)
    _log_zip(name, stats, dry_run)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Package GroupAffect-4 release into Zenodo-compatible zip files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["subset", "full"],
        default="subset",
        help=(
            "subset: <4 GB Zenodo record (no WAV, ET 7 sessions, 6 files) "
            "→ data/zenodo_zips_subset/. "
            "full: complete Zenodo record with all WAV and ET (16 files) "
            "→ data/zenodo_zips_full/."
        ),
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
        default=None,
        help="Destination directory (default: data/zenodo_zips_subset or data/zenodo_zips_full).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned zip contents and sizes without writing files.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = repo_root / "data" / f"zenodo_zips_{args.mode}"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.bids_root.exists():
        parser.error(f"BIDS root not found: {args.bids_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    label = "[DRY-RUN] " if args.dry_run else ""
    logging.info("%sBuilding %s zips → %s", label, args.mode, args.out_dir)
    logging.info("-" * 70)

    all_sessions = sorted(
        d.name for d in (args.bids_root / "sub-01").iterdir() if d.is_dir()
    )

    # 1. Metadata zip (mode-specific primary README)
    build_metadata_zip(args.bids_root, args.out_dir, args.mode,
                       args.dry_run, repo_root=repo_root)

    # 2. Physio
    build_modality_zip(args.bids_root, args.out_dir, "physio",
                       "affectai_physio.zip", None, args.dry_run)

    # 3. Behavioural
    build_modality_zip(args.bids_root, args.out_dir, "beh",
                       "affectai_beh.zip", None, args.dry_run)

    # 4. Transcripts (TSV/TXT only, no WAV)
    build_audio_transcript_zip(args.bids_root, args.out_dir, args.dry_run)

    # 5. Annotations
    build_modality_zip(args.bids_root, args.out_dir, "annot",
                       "affectai_annot.zip", None, args.dry_run)

    # 6. Eye tracking
    if args.mode == "subset":
        et_sessions = [s for s in all_sessions if s not in _ET_EXCLUDE_SUBSET]
        build_modality_zip(args.bids_root, args.out_dir, "et",
                           "affectai_et_grp07to15.zip", et_sessions, args.dry_run)
        logging.info(
            "  ET for grp-11, grp-12, grp-16 excluded — in full release only"
        )
    else:
        build_modality_zip(args.bids_root, args.out_dir, "et",
                           "affectai_et_all.zip", None, args.dry_run)

    # 7. Audio WAV — full mode only, one zip per session
    if args.mode == "full":
        logging.info("Building per-session audio WAV zips...")
        for ses_id in all_sessions:
            build_audio_wav_zip(args.bids_root, args.out_dir, ses_id, args.dry_run)

    # Summary
    if not args.dry_run:
        zips = list(args.out_dir.glob("*.zip"))
        total = sum(z.stat().st_size for z in zips)
        logging.info("-" * 70)
        logging.info(
            "Done: %d zip file(s), %.2f GB total → %s",
            len(zips), total / (1024 ** 3), args.out_dir,
        )
        if len(zips) > 100:
            logging.warning(
                "WARNING: %d zip files exceed Zenodo 100-file limit!", len(zips)
            )
        else:
            logging.info("File count %d <= 100 (Zenodo limit OK)", len(zips))
    else:
        logging.info("-" * 70)
        logging.info("[DRY-RUN] Complete. No files written.")


if __name__ == "__main__":
    main()
