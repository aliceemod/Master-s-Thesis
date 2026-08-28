"""Room mic (PanaCast 50) support for resolving uncertain segments.

Only processes segments marked 'uncertain'. Never touches 'confident' ones.
Segments with mic/p50 > 0.3 are confirmed as participant.
Segments with mic/p50 < 0.3 remain uncertain (for embeddings to resolve).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .transcribe import compute_rms_energy

log = logging.getLogger(__name__)

# Participant close-talk: mic/p50 typically 0.4-1.5
# Moderator from room: mic/p50 typically 0.05-0.15
# Two thresholds:
#   > 0.3: confirmed participant
#   < 0.15: confirmed moderator
#   0.15-0.3: uncertain, let embeddings decide
ROOM_MIC_PARTICIPANT_THRESHOLD = 0.3
ROOM_MIC_MODERATOR_THRESHOLD = 0.15


def find_room_mic(session_dir: Path) -> Path | None:
    """Find a single continuous room mic file in the session root."""
    candidates = list(session_dir.glob("*panacast*50*audio*")) + \
                 list(session_dir.glob("*panacast*50*aud*")) + \
                 list(session_dir.glob("*p50*audio*"))
    for c in candidates:
        if c.suffix == ".wav":
            return c
    return None


def find_per_task_room_mics(audio_dir: Path) -> dict[str, Path]:
    """Find per-task room mic files in the audio directory.

    Looks for files with 'panacast-50' in the name, split by task.
    Returns dict: task -> path (e.g. {"T0": Path(...), "T1": Path(...)}).
    """
    import re
    task_pattern = re.compile(r"_task-(T\d+)_")
    result = {}
    for wav in sorted(audio_dir.glob("*panacast*50*")):
        if wav.suffix != ".wav":
            continue
        match = task_pattern.search(wav.name)
        if match:
            result[match.group(1)] = wav
    return result


def load_room_mic(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def compute_p50_offsets(
    session_dir: Path,
    room_audio: np.ndarray,
    room_sr: int,
) -> dict[str, float]:
    audio_dir = session_dir / "audio"
    offsets = {}
    cumulative = 0.0
    for task in ["T0", "T1", "T2", "T3", "T4"]:
        wavs = sorted(audio_dir.glob(f"*{task}*mic9*"))
        if not wavs:
            continue
        import wave
        f = wave.open(str(wavs[0]), "rb")
        dur = f.getnframes() / f.getframerate()
        f.close()
        offsets[task] = cumulative
        cumulative += dur
    log.info("Room mic task offsets: %s", {t: f"{o:.1f}s" for t, o in offsets.items()})
    return offsets


def resolve_with_room_mic(
    segments: list,
    room_audio: np.ndarray,
    room_sr: int,
    p50_offset: float,
    mic_audio: dict[str, tuple[np.ndarray, int]],
) -> tuple[int, int]:
    """Resolve uncertain segments using room mic comparison.

    - mic/p50 > 0.3: confirmed participant (confidence → confident)
    - mic/p50 < 0.3: labeled MODERATOR (no close-talk signal detected)

    Never touches confident segments.
    """
    confirmed = 0
    relabeled = 0

    for seg in segments:
        if seg.confidence != "uncertain":
            continue

        mic = seg.mic
        if mic not in mic_audio:
            continue
        audio, sr = mic_audio[mic]
        mic_energy = compute_rms_energy(audio, sr, seg.start, seg.end)

        room_start = p50_offset + seg.start
        room_end = p50_offset + seg.end
        room_energy = compute_rms_energy(room_audio, room_sr, room_start, room_end)

        if room_energy <= 0:
            continue

        ratio = mic_energy / room_energy

        if ratio > ROOM_MIC_PARTICIPANT_THRESHOLD:
            # Clearly close-talk → confirmed participant
            seg.confidence = "confident"
            confirmed += 1
        elif ratio < ROOM_MIC_MODERATOR_THRESHOLD:
            # Clearly room audio → moderator
            seg.speaker = "MODERATOR"
            seg.mic = "shared"
            seg.confidence = "confident"
            relabeled += 1
        else:
            # Borderline (0.15-0.3) → stay uncertain for embeddings
            pass

    log.info(
        "Room mic: %d confirmed participant, %d still uncertain",
        confirmed, relabeled,
    )
    return confirmed, relabeled
