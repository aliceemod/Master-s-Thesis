"""Merge per-mic transcripts into a single chronological master transcript.

Simple energy-ratio classification:
- ratio >= 2.0: confident participant (P50 and embeddings cannot override)
- ratio < 2.0: uncertain — P50 and embeddings will resolve
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .transcribe import Segment, WordSegment

log = logging.getLogger(__name__)

MODERATOR_LABEL = "MODERATOR"
CONFIDENT_RATIO = 2.0


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    text: str
    energy: float = 0.0
    energy_ratio: float = 0.0
    words: list[WordSegment] = field(default_factory=list)
    mic: str = ""
    confidence: str = "confident"

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_transcripts(
    mic_transcripts: dict[str, list[Segment]],
    mic_to_speaker: dict[str, str],
    phase_windows: list[dict] | None = None,
) -> list[SpeakerSegment]:
    all_segments: list[SpeakerSegment] = []
    confident_count = 0
    uncertain_count = 0

    for mic, segments in mic_transcripts.items():
        speaker = mic_to_speaker[mic]
        for seg in segments:
            if seg.energy_ratio >= CONFIDENT_RATIO:
                label = speaker
                conf = "confident"
                confident_count += 1
            else:
                label = speaker
                conf = "uncertain"
                uncertain_count += 1

            all_segments.append(SpeakerSegment(
                start=seg.start, end=seg.end,
                speaker=label, text=seg.text,
                energy=seg.energy, energy_ratio=seg.energy_ratio,
                words=seg.words, mic=mic, confidence=conf,
            ))

    all_segments.sort(key=lambda s: (s.start, s.speaker))
    log.info(
        "Merged: %d segments — %d confident, %d uncertain",
        len(all_segments), confident_count, uncertain_count,
    )
    return all_segments
