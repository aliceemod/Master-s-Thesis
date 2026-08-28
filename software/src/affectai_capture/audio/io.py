"""I/O helpers: save/load transcripts as BIDS-compatible TSV and JSON."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from .merge import SpeakerSegment
from .transcribe import Segment, WordSegment

log = logging.getLogger(__name__)


def save_per_mic_json(segments: list[Segment], path: Path, mic: str, speaker: str) -> None:
    """Save per-mic transcript as JSON (intermediate format)."""
    records = []
    for seg in segments:
        records.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text,
            "energy": round(seg.energy, 6),
            "energy_ratio": round(seg.energy_ratio, 3),
            "words": [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word,
                    "score": round(w.score, 3),
                }
                for w in seg.words
            ],
        })
    out = {"mic": mic, "speaker": speaker, "segments": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("Saved %d segments to %s", len(records), path)


def load_per_mic_json(path: Path) -> tuple[str, str, list[Segment]]:
    """Load per-mic JSON. Returns (mic, speaker, segments)."""
    data = json.loads(path.read_text())
    segments = []
    for s in data["segments"]:
        words = [
            WordSegment(start=w["start"], end=w["end"], word=w["word"], score=w.get("score", 0.0))
            for w in s.get("words", [])
        ]
        segments.append(Segment(
            start=s["start"], end=s["end"], text=s["text"],
            energy=s.get("energy", 0.0), energy_ratio=s.get("energy_ratio", 0.0),
            words=words,
        ))
    return data["mic"], data["speaker"], segments


def save_master_tsv(segments: list[SpeakerSegment], path: Path) -> None:
    """Save the merged master transcript as a BIDS-compatible TSV.

    Columns: onset, duration, speaker, text, energy, mic
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["onset", "duration", "speaker", "text", "energy", "energy_ratio", "mic", "confidence"])
        for seg in segments:
            writer.writerow([
                f"{seg.start:.3f}",
                f"{seg.duration:.3f}",
                seg.speaker,
                seg.text,
                f"{seg.energy:.6f}",
                f"{getattr(seg, 'energy_ratio', 0):.2f}",
                seg.mic,
                getattr(seg, "confidence", "confident"),
            ])
    log.info("Saved master transcript (%d rows) to %s", len(segments), path)


def save_master_words_tsv(segments: list[SpeakerSegment], path: Path) -> None:
    """Save word-level master transcript as TSV.

    Columns: onset, duration, speaker, word, score, mic
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["onset", "duration", "speaker", "word", "score", "mic"])
        count = 0
        for seg in segments:
            for w in seg.words:
                writer.writerow([
                    f"{w.start:.3f}",
                    f"{w.end - w.start:.3f}",
                    seg.speaker,
                    w.word,
                    f"{w.score:.3f}",
                    seg.mic,
                ])
                count += 1
    log.info("Saved word-level transcript (%d words) to %s", count, path)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _short_name(speaker: str) -> str:
    """Extract a clean first name from speaker label like 'P1_harshavardhan_reddy'."""
    if speaker == "MODERATOR":
        return "Moderator"
    parts = speaker.split("_", 1)
    if len(parts) < 2:
        return speaker
    seat = parts[0]
    name_parts = parts[1].split("_")
    first_name = name_parts[0].capitalize()
    return f"{first_name} ({seat})"


def save_readable_transcript(segments: list[SpeakerSegment], path: Path) -> None:
    """Save a clean, human-readable transcript.

    Format:
        [01:15] Olena (P3): That's an interesting one.
        [02:08] Harshavardhan (P1): Okay.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    prev_speaker = None

    for seg in segments:
        ts = _format_timestamp(seg.start)
        name = _short_name(seg.speaker)

        # Add a blank line when the speaker changes for readability
        if prev_speaker is not None and seg.speaker != prev_speaker:
            lines.append("")

        uncertain = " [?]" if getattr(seg, "confidence", "confident") == "uncertain" else ""
        lines.append(f"[{ts}] {name}{uncertain}: {seg.text}")
        prev_speaker = seg.speaker

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved readable transcript (%d lines) to %s", len(segments), path)
