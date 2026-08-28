#!/usr/bin/env python3
"""Recover backchannels lost by the main pipeline's bleed filter.

The main pipeline (`tools/audio_annotate.py`) drops short utterances like
"yeah", "mhm", "okay" when another participant is the dominant speaker,
because the owner mic's energy fails the "must be loudest" check. This
script does a second pass over the cached per-mic WhisperX output and
rescues short backchannel-type words using a looser, word-level energy
test: the word must be louder than the mic's quiet baseline (proves the
participant said something) AND the owner mic must be at least roughly
comparable to the loudest other mic in the tight window around the word
(allows for cross-talk where another speaker dominates the wider segment
but not the brief backchannel).

Inputs (per task, under <session>/audio_annot/<task>/):
  - mic9_transcript.json, mic10_..., mic11_..., mic12_...  (cached WhisperX)
  - master_transcript.tsv                                  (kept segments)
Audio:
  - <session>/audio/*_acq-(av-)?dpa(_|-)mic{9-12}*.wav

Outputs (per task, written alongside the originals — originals untouched):
  - master_transcript_with_backchannels.tsv
  - transcript_with_backchannels.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("refine_backchannels")

# Inlined here so this script has no dependency on whisperx/torch
# (which the main pipeline imports). Keep behaviour in sync with
# src/affectai_capture/audio/pipeline.py and io.py.
MIC_TO_SEAT = {"mic9": "P1", "mic10": "P2", "mic11": "P3", "mic12": "P4"}


def _format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _short_name(speaker: str) -> str:
    if speaker == "MODERATOR":
        return "Moderator"
    parts = speaker.split("_", 1)
    if len(parts) < 2:
        return speaker
    seat = parts[0]
    name_parts = parts[1].split("_")
    first = name_parts[0].capitalize()
    return f"{first} ({seat})"


def parse_participant_map(session_dir: Path) -> dict[str, str]:
    pmap_path = session_dir / "participant_map.tsv"
    out: dict[str, str] = {}
    if pmap_path.exists():
        for line in pmap_path.read_text().strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                out[parts[0]] = parts[2]
        return out
    annot_dir = session_dir / "annot"
    if annot_dir.is_dir():
        cands = sorted(annot_dir.glob("*_group_participants.tsv"))
        if cands:
            lines = cands[0].read_text().strip().split("\n")
            header = lines[0].split("\t")
            si = header.index("seat") if "seat" in header else 2
            pi = header.index("participant_id") if "participant_id" in header else 3
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) > max(si, pi):
                    out[parts[si]] = parts[pi]
    return out

# Word-level rescue tunables.
BACKCHANNEL_TOKENS = {
    "yeah", "yes", "yep", "yup", "yah",
    "mhm", "mhmm", "mm", "hm", "hmm",
    "ok", "okay", "alright", "right",
    "sure", "exactly", "true", "definitely", "absolutely",
    "no", "nope",
    "uh-huh", "uhhuh", "huh",
    "wow", "oh",
}
MAX_BACKCHANNEL_DURATION = 1.5  # seconds — anything longer isn't a backchannel
WORD_PADDING = 0.05  # seconds — pad the word window slightly for energy
MIN_BASELINE_RATIO = 2.0  # word energy must be 2x the mic's quiet baseline
MIN_CROSS_MIC_RATIO = 0.5  # owner mic must be >= 0.5x of loudest competing mic
OVERLAP_THRESHOLD = 0.5  # if a master segment covers >50% of the word, skip
BASELINE_WINDOW = 30.0  # seconds — use this much surrounding audio to estimate mic noise floor


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    speaker: str
    mic: str
    energy: float
    energy_ratio: float


def _normalize_token(word: str) -> str:
    return re.sub(r"[^a-z\-]", "", word.lower())


def _rms(audio: np.ndarray, sr: int, start: float, end: float) -> float:
    s = max(0, min(int(start * sr), len(audio)))
    e = max(s, min(int(end * sr), len(audio)))
    if e <= s:
        return 0.0
    chunk = audio[s:e].astype(np.float64)
    return float(np.sqrt(np.mean(chunk ** 2)))


def _quiet_baseline(audio: np.ndarray, sr: int, center: float) -> float:
    """Estimate mic noise floor near `center` time using lowest-energy 0.3s
    windows over a +/-15s span. This excludes loud speech and gives a stable
    reference for "is something happening on this mic?"."""
    half = BASELINE_WINDOW / 2
    s = max(0.0, center - half)
    e = min(len(audio) / sr, center + half)
    win = 0.3
    energies = []
    t = s
    while t + win <= e:
        energies.append(_rms(audio, sr, t, t + win))
        t += win
    if not energies:
        return 0.0
    energies.sort()
    # Use the 10th percentile as the "quiet" baseline.
    idx = max(0, int(len(energies) * 0.1))
    return energies[idx]


def _load_master(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["start"] = float(row["onset"])
            row["end"] = row["start"] + float(row["duration"])
            rows.append(row)
    return rows


def _word_already_covered(word_start: float, word_end: float, master: list[dict], speaker: str) -> bool:
    """Is the word already represented by an existing master segment from the same speaker?"""
    word_dur = max(word_end - word_start, 1e-6)
    for row in master:
        if row["speaker"] != speaker:
            continue
        overlap = max(0.0, min(row["end"], word_end) - max(row["start"], word_start))
        if overlap / word_dur > OVERLAP_THRESHOLD:
            return True
    return False


def _build_mic_to_speaker(session_dir: Path) -> dict[str, str]:
    pmap = parse_participant_map(session_dir)
    mic_to_speaker = {}
    for mic, seat in MIC_TO_SEAT.items():
        name = pmap.get(seat, seat)
        mic_to_speaker[mic] = (
            f"{seat}_{name.replace(' ', '_')}" if name != seat else seat
        )
    return mic_to_speaker


def _load_mic_audio(session_dir: Path, task: str, mic: str) -> tuple[np.ndarray, int] | None:
    audio_dir = session_dir / "audio"
    candidates = list(audio_dir.glob(f"*task-{task}_*{mic}*.wav"))
    candidates = [c for c in candidates if "panacast" not in c.name.lower()]
    if not candidates:
        return None
    data, sr = sf.read(candidates[0], dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def refine_task(session_dir: Path, task_dir: Path, mic_to_speaker: dict[str, str]) -> int:
    """Returns count of backchannels recovered."""
    master_path = task_dir / "master_transcript.tsv"
    if not master_path.exists():
        log.warning("[%s] no master_transcript.tsv — skipping", task_dir.name)
        return 0

    master = _load_master(master_path)
    task = task_dir.name

    # Load all 4 mic audios for this task.
    mic_audio: dict[str, tuple[np.ndarray, int]] = {}
    for mic in MIC_TO_SEAT:
        loaded = _load_mic_audio(session_dir, task, mic)
        if loaded is not None:
            mic_audio[mic] = loaded
    if not mic_audio:
        log.warning("[%s] no audio found — skipping", task)
        return 0

    candidates: list[Candidate] = []

    for mic, (audio, sr) in mic_audio.items():
        cache = task_dir / f"{mic}_transcript.json"
        if not cache.exists():
            continue
        speaker = mic_to_speaker.get(mic, mic)
        per_mic = json.loads(cache.read_text())
        for seg in per_mic.get("segments", []):
            for word in seg.get("words", []):
                w_text = _normalize_token(word.get("word", ""))
                if w_text not in BACKCHANNEL_TOKENS:
                    continue
                w_start = float(word.get("start", seg["start"]))
                w_end = float(word.get("end", seg["end"]))
                w_dur = w_end - w_start
                if w_dur <= 0 or w_dur > MAX_BACKCHANNEL_DURATION:
                    continue
                if _word_already_covered(w_start, w_end, master, speaker):
                    continue

                # Energy check on a slightly padded window.
                start = max(0.0, w_start - WORD_PADDING)
                end = w_end + WORD_PADDING

                own_e = _rms(audio, sr, start, end)
                if own_e <= 0:
                    continue

                baseline = _quiet_baseline(audio, sr, (start + end) / 2)
                if baseline > 0 and own_e / baseline < MIN_BASELINE_RATIO:
                    continue  # mic didn't capture anything above its noise floor

                # Cross-mic check: owner mic should be roughly competitive
                # in this short window (it doesn't have to be the loudest).
                max_other = max(
                    (_rms(other, other_sr, start, end)
                     for m, (other, other_sr) in mic_audio.items() if m != mic),
                    default=0.0,
                )
                ratio = own_e / max_other if max_other > 0 else float("inf")
                if ratio < MIN_CROSS_MIC_RATIO:
                    continue

                candidates.append(Candidate(
                    start=w_start, end=w_end, text=word["word"].strip(),
                    speaker=speaker, mic=mic,
                    energy=own_e, energy_ratio=ratio,
                ))

    if not candidates:
        log.info("[%s] no backchannels recovered", task)
    else:
        log.info("[%s] recovered %d backchannel candidate(s)", task, len(candidates))

    # Merge with master and sort by time.
    merged = [
        {
            "onset": float(r["onset"]),
            "duration": float(r["duration"]),
            "speaker": r["speaker"],
            "text": r["text"],
            "energy": float(r["energy"]),
            "energy_ratio": float(r.get("energy_ratio", 0.0)),
            "mic": r["mic"],
            "confidence": r["confidence"],
        }
        for r in master
    ]
    for c in candidates:
        merged.append({
            "onset": c.start,
            "duration": c.end - c.start,
            "speaker": c.speaker,
            "text": c.text,
            "energy": c.energy,
            "energy_ratio": c.energy_ratio,
            "mic": c.mic,
            "confidence": "backchannel",
        })
    merged.sort(key=lambda r: (r["onset"], r["speaker"]))

    # TSV
    out_tsv = task_dir / "master_transcript_with_backchannels.tsv"
    with out_tsv.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["onset", "duration", "speaker", "text", "energy",
                         "energy_ratio", "mic", "confidence"])
        for r in merged:
            writer.writerow([
                f"{r['onset']:.3f}", f"{r['duration']:.3f}",
                r["speaker"], r["text"],
                f"{r['energy']:.6f}", f"{r['energy_ratio']:.2f}",
                r["mic"], r["confidence"],
            ])

    # Readable
    out_txt = task_dir / "transcript_with_backchannels.txt"
    lines = []
    prev_speaker = None
    for r in merged:
        ts = _format_timestamp(r["onset"])
        name = _short_name(r["speaker"])
        if prev_speaker is not None and r["speaker"] != prev_speaker:
            lines.append("")
        marker = ""
        if r["confidence"] == "backchannel":
            marker = " [bc]"
        elif r["confidence"] == "uncertain":
            marker = " [?]"
        lines.append(f"[{ts}] {name}{marker}: {r['text']}")
        prev_speaker = r["speaker"]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return len(candidates)


def refine_session(session_dir: Path) -> int:
    annot_root = session_dir / "audio_annot"
    if not annot_root.is_dir():
        log.error("[%s] no audio_annot/ — run audio_annotate.py first", session_dir.name)
        return 0

    mic_to_speaker = _build_mic_to_speaker(session_dir)
    log.info("[%s] mic→speaker: %s", session_dir.name, mic_to_speaker)

    total = 0
    task_dirs = sorted(d for d in annot_root.iterdir() if d.is_dir() and re.match(r"T\d+$", d.name))
    for task_dir in task_dirs:
        total += refine_task(session_dir, task_dir, mic_to_speaker)
    log.info("[%s] total backchannels recovered: %d", session_dir.name, total)
    return total


def main() -> None:
    p = argparse.ArgumentParser(
        description="Backchannel-recovery second pass over annotated sessions.",
    )
    p.add_argument("session_dirs", type=Path, nargs="+",
                   help="Session directories that already have audio_annot/ from audio_annotate.py")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    grand_total = 0
    for session_dir in args.session_dirs:
        if not session_dir.is_dir():
            log.error("Not a directory: %s", session_dir)
            continue
        grand_total += refine_session(session_dir)

    print(f"\nGrand total backchannels recovered: {grand_total}")


if __name__ == "__main__":
    main()
