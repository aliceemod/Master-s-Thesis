"""Speaker embeddings — final step to resolve remaining uncertain segments.

After energy ratio and P50 room mic, some segments are still uncertain.
Embeddings compare the voice against participant profiles to decide:
- Voice matches a participant whose mic is loudest → PARTICIPANT
- No match → MODERATOR (last resort)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

MIN_REFERENCE_DURATION = 1.5
SIMILARITY_THRESHOLD = 0.5


def _load_embedding_model():
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=False)
    inference = Inference(model, window="whole")
    return inference


def extract_embedding(inference, audio, sr, start, end):
    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    if (e - s) / sr < 0.5:
        return None
    chunk = audio[s:e].astype(np.float32)
    waveform = torch.from_numpy(chunk).unsqueeze(0)
    audio_dict = {"waveform": waveform, "sample_rate": sr}
    embedding = inference(audio_dict)
    return embedding.flatten()


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_speaker_profiles(
    mic_segments: dict[str, list],
    mic_audio: dict[str, tuple[np.ndarray, int]],
    mic_to_speaker: dict[str, str],
    top_n: int = 5,
) -> dict[str, np.ndarray]:
    """Build profiles from pre-filtered segments (discussion phase, high ratio)."""
    log.info("Building speaker profiles...")
    inference = _load_embedding_model()
    profiles = {}

    for mic, segments in mic_segments.items():
        speaker = mic_to_speaker.get(mic)
        if not speaker:
            continue
        audio, sr = mic_audio[mic]
        candidates = sorted(segments, key=lambda s: s.energy_ratio, reverse=True)[:top_n]
        if not candidates:
            log.warning("[%s] No reference segments for %s", mic, speaker)
            continue

        embeddings = []
        total_dur = 0.0
        for seg in candidates:
            emb = extract_embedding(inference, audio, sr, seg.start, seg.end)
            if emb is not None:
                embeddings.append(emb)
                total_dur += seg.end - seg.start

        if embeddings:
            profiles[speaker] = np.mean(embeddings, axis=0)
            log.info("[%s] %s: %d segs, %.1fs, best ratio %.1f",
                     mic, speaker, len(embeddings), total_dur, candidates[0].energy_ratio)

    del inference
    import gc; gc.collect()
    log.info("Built %d speaker profiles", len(profiles))
    return profiles


def build_moderator_profile(
    room_mic_audio: dict[str, tuple[np.ndarray, int]],
    close_talk_audio: dict[str, dict[str, tuple[np.ndarray, int]]],
    top_n: int = 10,
) -> np.ndarray | None:
    """Build moderator voice profile from P50 room mic during instruction phases.

    The first ~60s of each task is typically moderator instructions.
    We find segments where P50 is loud and all close-talk mics are quiet
    (confirming it's the moderator, not a participant).

    Parameters
    ----------
    room_mic_audio : task -> (p50_audio, sr)
    close_talk_audio : task -> {mic -> (audio, sr)}
    """
    log.info("Building moderator profile from room mic...")
    inference = _load_embedding_model()

    # Scan first 120s of each task for moderator speech on P50
    from .transcribe import compute_rms_energy
    mod_embeddings = []

    for task, (p50, p50_sr) in room_mic_audio.items():
        mics = close_talk_audio.get(task, {})
        # Scan in 3-second windows
        for start in range(0, min(120, int(len(p50) / p50_sr) - 3), 3):
            end = start + 3
            p50_e = compute_rms_energy(p50, p50_sr, start, end)
            if p50_e < 0.01:
                continue  # silence

            # Check all close-talk mics are much quieter
            max_mic_e = 0.0
            for mic, (audio, sr) in mics.items():
                mic_e = compute_rms_energy(audio, sr, start, end)
                max_mic_e = max(max_mic_e, mic_e)

            if max_mic_e > 0 and max_mic_e / p50_e < 0.15:
                # Close-talk mics are very quiet relative to P50 → moderator
                emb = extract_embedding(inference, p50, p50_sr, start, end)
                if emb is not None:
                    mod_embeddings.append(emb)

    del inference
    import gc; gc.collect()

    if mod_embeddings:
        # Take the best ones (most consistent)
        profile = np.mean(mod_embeddings[:top_n], axis=0)
        log.info("Moderator profile: %d segments used", min(len(mod_embeddings), top_n))
        return profile
    else:
        log.warning("Could not build moderator profile")
        return None


def resolve_with_embeddings(
    segments: list,
    profiles: dict[str, np.ndarray],
    all_mic_audio: dict[str, tuple[np.ndarray, int]],
    mic_to_speaker: dict[str, str],
    room_mic_audio: tuple[np.ndarray, int] | None = None,
) -> tuple[int, int]:
    """Final resolution of uncertain segments using speaker embeddings.

    Compares against ALL profiles (participants + moderator).
    For each still-uncertain segment:
    - Extract embedding from the loudest close-talk mic AND from P50
    - Compare against all profiles (participants + moderator)
    - Best match wins

    Returns (confirmed_participant, labeled_moderator).
    """
    if not profiles:
        mod_count = 0
        for seg in segments:
            if seg.confidence == "uncertain":
                seg.speaker = "MODERATOR"
                seg.mic = "shared"
                seg.confidence = "confident"
                mod_count += 1
        log.info("No profiles — %d uncertain → MODERATOR", mod_count)
        return 0, mod_count

    speaker_to_mic = {speaker: mic for mic, speaker in mic_to_speaker.items()}

    # Add moderator to profiles if present
    mod_profile = profiles.get("MODERATOR")
    all_profiles = dict(profiles)  # includes MODERATOR if built

    inference = _load_embedding_model()
    confirmed = 0
    moderator = 0

    for seg in segments:
        if seg.confidence != "uncertain":
            continue

        if (seg.end - seg.start) < 0.8:
            # Too short for embedding — default to moderator
            seg.speaker = "MODERATOR"
            seg.mic = "shared"
            seg.confidence = "confident"
            moderator += 1
            continue

        # Extract embedding from loudest close-talk mic
        from .transcribe import compute_rms_energy
        mic_energies = {}
        for mic, (audio, sr) in all_mic_audio.items():
            mic_energies[mic] = compute_rms_energy(audio, sr, seg.start, seg.end)
        best_mic = max(mic_energies, key=mic_energies.get)
        audio, sr = all_mic_audio[best_mic]
        emb_mic = extract_embedding(inference, audio, sr, seg.start, seg.end)

        # Also extract from P50 if available (moderator voice is clearer there)
        emb_p50 = None
        if room_mic_audio is not None:
            p50_audio, p50_sr = room_mic_audio
            emb_p50 = extract_embedding(inference, p50_audio, p50_sr, seg.start, seg.end)

        # Compare against ALL profiles (participants + moderator)
        best_match = None
        best_score = -1.0
        for speaker, profile in all_profiles.items():
            # For participant profiles, use close-talk mic embedding
            # For moderator profile, use P50 embedding (clearer for moderator)
            if speaker == "MODERATOR" and emb_p50 is not None:
                emb = emb_p50
            elif emb_mic is not None:
                emb = emb_mic
            else:
                continue
            s = cosine_similarity(emb, profile)
            if s > best_score:
                best_score = s
                best_match = speaker

        if best_match == "MODERATOR":
            seg.speaker = "MODERATOR"
            seg.mic = "shared"
            seg.confidence = "confident"
            moderator += 1
            log.debug("MODERATOR %.1f-%.1fs (score=%.2f vs mod profile)", seg.start, seg.end, best_score)
        elif best_match and best_score >= SIMILARITY_THRESHOLD:
            expected_mic = speaker_to_mic.get(best_match)
            if expected_mic == best_mic:
                seg.speaker = best_match
                seg.mic = best_mic
                seg.confidence = "confident"
                confirmed += 1
                log.debug("CONFIRMED %.1f-%.1fs → %s (score=%.2f)", seg.start, seg.end, best_match, best_score)
            else:
                # Voice match but wrong mic — likely moderator
                seg.speaker = "MODERATOR"
                seg.mic = "shared"
                seg.confidence = "confident"
                moderator += 1
                log.debug("MODERATOR %.1f-%.1fs (matched %s but mic=%s not %s)",
                          seg.start, seg.end, best_match, best_mic, expected_mic)
        else:
            seg.speaker = "MODERATOR"
            seg.mic = "shared"
            seg.confidence = "confident"
            moderator += 1
            log.debug("MODERATOR %.1f-%.1fs (no strong match, best=%.2f)", seg.start, seg.end, best_score)

    del inference
    import gc; gc.collect()
    log.info("Embeddings: %d confirmed participant, %d → MODERATOR", confirmed, moderator)
    return confirmed, moderator
