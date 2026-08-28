"""Per-mic transcription with cross-mic energy-based bleed rejection.

Architecture (per AMI/ICSI corpus methodology):
1. Transcribe each mic fully with WhisperX (let it segment naturally)
2. For each resulting segment, compute energy on this mic vs all others
3. Keep only segments where this mic is the loudest → the real speaker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import whisperx

log = logging.getLogger(__name__)

WHISPER_SR = 16_000


@dataclass
class WordSegment:
    start: float
    end: float
    word: str
    score: float = 0.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[WordSegment] = field(default_factory=list)
    energy: float = 0.0
    energy_ratio: float = 0.0  # how much louder this mic was vs the next loudest


def compute_rms_energy(audio: np.ndarray, sr: int, start: float, end: float) -> float:
    """Compute RMS energy of a time window in the audio signal."""
    s = max(0, min(int(start * sr), len(audio)))
    e = max(s, min(int(end * sr), len(audio)))
    if e <= s:
        return 0.0
    chunk = audio[s:e].astype(np.float64)
    return float(np.sqrt(np.mean(chunk ** 2)))


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file. Returns (mono float32 samples, sample_rate)."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def transcribe_full(
    wav_path: Path,
    device: str = "cpu",
    model_name: str = "large-v3",
    compute_type: str = "int8",
    language: str = "en",
    batch_size: int = 8,
) -> list[Segment]:
    """Transcribe a full mic WAV with WhisperX. Returns all segments with energy."""
    raw_audio, native_sr = load_audio(wav_path)
    audio_16k = whisperx.load_audio(str(wav_path))

    log.info("Loading WhisperX model: %s (device=%s, compute=%s)", model_name, device, compute_type)
    model = whisperx.load_model(
        model_name, device=device, compute_type=compute_type, language=language,
    )

    log.info("Transcribing %s...", wav_path.name)
    result = model.transcribe(audio_16k, batch_size=batch_size, language=language)

    log.info("Aligning word timestamps...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=device,
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio_16k, device=device,
    )

    segments = []
    for seg in result["segments"]:
        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append(WordSegment(
                    start=w["start"], end=w["end"],
                    word=w["word"], score=w.get("score", 0.0),
                ))
        energy = compute_rms_energy(raw_audio, native_sr, seg["start"], seg["end"])
        segments.append(Segment(
            start=seg["start"], end=seg["end"],
            text=seg.get("text", "").strip(),
            words=words, energy=energy,
        ))

    log.info("Transcribed %d segments from %s", len(segments), wav_path.name)

    del model, align_model
    import gc
    gc.collect()

    return segments


def filter_bleed(
    segments: list[Segment],
    mic_audio: np.ndarray,
    mic_sr: int,
    other_audios: list[np.ndarray],
    energy_ratio_threshold: float = 1.5,
) -> list[Segment]:
    """Keep only segments where this mic has the highest energy.

    For each transcribed segment, compare energy on this mic vs the same
    time window on all other mics. If this mic is not the loudest by
    at least the threshold ratio, the segment is bleed — reject it.

    Parameters
    ----------
    segments : Transcribed segments from this mic.
    mic_audio : Raw audio for this mic (native sample rate).
    mic_sr : Sample rate.
    other_audios : Raw audio for all OTHER mics (same sr).
    energy_ratio_threshold : This mic must be this factor louder than the
        loudest other mic. 1.5 = 50% louder (~3.5 dB), conservative for
        close-talk. Use lower values (1.2-1.3) if speakers are quiet.
    """
    kept = []
    rejected = 0

    for seg in segments:
        my_energy = compute_rms_energy(mic_audio, mic_sr, seg.start, seg.end)
        max_other = max(
            compute_rms_energy(other, mic_sr, seg.start, seg.end)
            for other in other_audios
        )

        if my_energy <= 0:
            rejected += 1
            continue

        ratio = my_energy / max_other if max_other > 0 else float("inf")

        if ratio >= energy_ratio_threshold:
            seg.energy_ratio = ratio
            kept.append(seg)
        else:
            rejected += 1
            log.debug(
                "REJECT bleed %.1f-%.1fs: this=%.4f, loudest_other=%.4f, ratio=%.2f, text=%r",
                seg.start, seg.end, my_energy, max_other, ratio, seg.text[:50],
            )

    log.info("Bleed filter: %d kept, %d rejected (of %d)", len(kept), rejected, len(segments))
    return kept
