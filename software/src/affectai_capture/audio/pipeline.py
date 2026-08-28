"""Orchestrator: runs the full audio annotation pipeline for a session.

Flow:
1. Transcribe each mic fully with WhisperX
2. Bleed filter: keep only segments where that mic is loudest
3. Merge: ratio >= 2.0 = confident participant, else uncertain
4. P50 room mic: uncertain + mic/p50 > 0.3 = confirmed participant
5. Speaker embeddings: remaining uncertain → match voice or label MODERATOR
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .embeddings import build_speaker_profiles, build_moderator_profile, resolve_with_embeddings
from .io import load_per_mic_json, save_master_tsv, save_master_words_tsv, save_per_mic_json, save_readable_transcript
from .merge import SpeakerSegment, merge_transcripts
from .room_mic import find_room_mic, find_per_task_room_mics, load_room_mic, compute_p50_offsets, resolve_with_room_mic
from .transcribe import Segment, filter_bleed, load_audio, transcribe_full

log = logging.getLogger(__name__)

MIC_TO_SEAT = {
    "mic9": "P1", "mic10": "P2", "mic11": "P3", "mic12": "P4",
}
# Audio filename conventions seen in the wild:
#   local           : sub-..._task-T0_..._acq-dpa_mic9_aud.wav
#   BIDS export     : sub-..._task-T0_..._acq-av-dpa-mic9-aud-dpa-mic9-aud_audio.wav
#   BIDS variant 1  : sub-..._task-T0_..._acq-dpa_mic9_audio.wav        (e.g. grp-09 T0)
#   BIDS variant 2  : sub-..._task-T1_..._acq-av-dpa-mic9-aud-dpa-mic9-aud-cap-<group>-<date>_audio.wav (e.g. grp-09 T1+)
MIC_PATTERN = re.compile(
    r"_acq-dpa_(mic\d+)_aud(?:io)?\.wav$"
    r"|_acq-av-dpa-(mic\d+)-aud-dpa-mic\d+-aud(?:-cap-[a-z0-9-]+)?_audio\.wav$"
)
TASK_PATTERN = re.compile(r"_task-(T\d+)_")


def parse_participant_map(session_dir):
    """Read seat→display-name mapping.

    Tries `participant_map.tsv` (local layout) first, then falls back to
    `annot/*_group_participants.tsv` (BIDS export). The BIDS file has no
    `name` column, so we fall back to `participant_id` (e.g. sub-017).
    """
    pmap_path = session_dir / "participant_map.tsv"
    result = {}
    if pmap_path.exists():
        for line in pmap_path.read_text().strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                result[parts[0]] = parts[2]
        return result

    annot_dir = session_dir / "annot"
    if annot_dir.is_dir():
        candidates = sorted(annot_dir.glob("*_group_participants.tsv"))
        if candidates:
            lines = candidates[0].read_text().strip().split("\n")
            header = lines[0].split("\t")
            seat_idx = header.index("seat") if "seat" in header else 2
            pid_idx = header.index("participant_id") if "participant_id" in header else 3
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) > max(seat_idx, pid_idx):
                    result[parts[seat_idx]] = parts[pid_idx]
    return result


def discover_audio_files(audio_dir):
    tasks = {}
    for wav in sorted(audio_dir.glob("*.wav")):
        mic_match = MIC_PATTERN.search(wav.name)
        task_match = TASK_PATTERN.search(wav.name)
        if mic_match and task_match:
            mic = mic_match.group(1) or mic_match.group(2)
            tasks.setdefault(task_match.group(1), {})[mic] = wav
    return tasks


def run_pipeline(
    session_dir: Path,
    output_dir: Path | None = None,
    tasks: list[str] | None = None,
    device: str = "cpu",
    model_name: str = "large-v3",
    compute_type: str = "int8",
    language: str = "en",
    batch_size: int = 8,
    hf_token: str | None = None,
    energy_ratio_threshold: float = 1.0,
    skip_existing: bool = True,
) -> dict[str, list[SpeakerSegment]]:

    audio_dir = session_dir / "audio"
    if output_dir is None:
        output_dir = session_dir / "audio_annot"
    output_dir.mkdir(parents=True, exist_ok=True)

    pmap = parse_participant_map(session_dir)
    mic_to_speaker = {}
    for mic, seat in MIC_TO_SEAT.items():
        name = pmap.get(seat, seat)
        mic_to_speaker[mic] = f"{seat}_{name.replace(' ', '_')}" if name != seat else seat
    log.info("Mic-to-speaker mapping: %s", mic_to_speaker)

    all_tasks = discover_audio_files(audio_dir)
    if not all_tasks:
        log.error("No audio files found in %s", audio_dir)
        return {}
    log.info("Found tasks: %s", sorted(all_tasks.keys()))
    if tasks:
        all_tasks = {t: m for t, m in all_tasks.items() if t in tasks}

    # Load room mic (P50) — either per-task files or one continuous file
    per_task_p50 = find_per_task_room_mics(audio_dir)
    room_mic_path = find_room_mic(session_dir) if not per_task_p50 else None
    room_audio, room_sr, p50_offsets = None, 0, {}
    per_task_room_audio: dict[str, tuple] = {}  # task -> (audio, sr)
    if per_task_p50:
        log.info("Found per-task room mic files: %s", sorted(per_task_p50.keys()))
        for task, path in per_task_p50.items():
            audio, sr = load_room_mic(path)
            per_task_room_audio[task] = (audio, sr)
    elif room_mic_path:
        log.info("Found continuous room mic: %s", room_mic_path.name)
        room_audio, room_sr = load_room_mic(room_mic_path)
        p50_offsets = compute_p50_offsets(session_dir, room_audio, room_sr)
    else:
        log.info("No room mic found")

    # === PASS 1: Transcribe + bleed filter ===
    all_mic_transcripts = {}
    all_mic_raw = {}
    for task_name in sorted(all_tasks.keys()):
        mics = all_tasks[task_name]
        log.info("=" * 60)
        log.info("[Pass 1] %s (%d mics)", task_name, len(mics))
        log.info("=" * 60)
        task_output = output_dir / task_name
        task_output.mkdir(parents=True, exist_ok=True)

        mic_raw = {}
        for mic, wav_path in sorted(mics.items()):
            raw, sr = load_audio(wav_path)
            mic_raw[mic] = (raw, sr)

        mic_transcripts_raw = {}
        for mic, wav_path in sorted(mics.items()):
            mic_json = task_output / f"{mic}_transcript.json"
            speaker = mic_to_speaker.get(mic, mic)
            if skip_existing and mic_json.exists():
                log.info("[%s] Loading cached", mic)
                _, _, segments = load_per_mic_json(mic_json)
                mic_transcripts_raw[mic] = segments
                continue
            log.info("[%s] Transcribing (%s)...", mic, speaker)
            segments = transcribe_full(
                wav_path, device=device, model_name=model_name,
                compute_type=compute_type, language=language, batch_size=batch_size,
            )
            mic_transcripts_raw[mic] = segments
            save_per_mic_json(segments, mic_json, mic, speaker)

        mic_transcripts = {}
        for mic in sorted(mics.keys()):
            raw, sr = mic_raw[mic]
            other_raws = [mic_raw[m][0] for m in sorted(mics.keys()) if m != mic]
            filtered = filter_bleed(mic_transcripts_raw[mic], raw, sr, other_raws,
                                    energy_ratio_threshold=energy_ratio_threshold)
            mic_transcripts[mic] = filtered

        all_mic_transcripts[task_name] = mic_transcripts
        all_mic_raw[task_name] = mic_raw

    # === BUILD SPEAKER PROFILES from best segments across all tasks ===
    # Only use segments confirmed as close-talk by BOTH energy ratio AND P50
    log.info("=" * 60)
    log.info("Building speaker profiles from ALL tasks (P50-verified)")
    log.info("=" * 60)
    combined_segments = {}
    combined_audio = {}
    from .transcribe import compute_rms_energy
    for task_name in sorted(all_tasks.keys()):
        # Get P50 audio for this task
        task_p50, task_p50_sr, task_p50_offset = None, 0, 0.0
        if task_name in per_task_room_audio:
            task_p50, task_p50_sr = per_task_room_audio[task_name]
            task_p50_offset = 0.0
        elif room_audio is not None and task_name in p50_offsets:
            task_p50, task_p50_sr = room_audio, room_sr
            task_p50_offset = p50_offsets[task_name]

        for mic, segments in all_mic_transcripts[task_name].items():
            if mic not in combined_segments:
                combined_segments[mic] = []
                combined_audio[mic] = all_mic_raw[task_name][mic]
            mic_audio_data, mic_sr = all_mic_raw[task_name][mic]
            for seg in segments:
                if seg.energy_ratio < 3.0 or (seg.end - seg.start) < MIN_REF_DUR:
                    continue
                # P50 check: must be confirmed close-talk
                if task_p50 is not None:
                    mic_e = compute_rms_energy(mic_audio_data, mic_sr, seg.start, seg.end)
                    p50_e = compute_rms_energy(task_p50, task_p50_sr,
                                               task_p50_offset + seg.start,
                                               task_p50_offset + seg.end)
                    if p50_e > 0 and mic_e / p50_e < 0.3:
                        continue  # room audio, not close-talk — skip
                combined_segments[mic].append(seg)
    profiles = build_speaker_profiles(combined_segments, combined_audio, mic_to_speaker)

    # Build moderator profile from P50 room mic (first ~120s of each task)
    if per_task_room_audio:
        mod_profile = build_moderator_profile(per_task_room_audio, all_mic_raw)
        if mod_profile is not None:
            profiles["MODERATOR"] = mod_profile
    elif room_audio is not None:
        # Continuous P50 — build from first 120s of each task
        room_per_task = {}
        for task_name in sorted(all_tasks.keys()):
            if task_name in p50_offsets:
                offset = p50_offsets[task_name]
                # Extract the first 120s of this task from the continuous recording
                start_sample = int(offset * room_sr)
                end_sample = int((offset + 120) * room_sr)
                task_chunk = room_audio[start_sample:end_sample]
                room_per_task[task_name] = (task_chunk, room_sr)
        if room_per_task:
            mod_profile = build_moderator_profile(room_per_task, all_mic_raw)
            if mod_profile is not None:
                profiles["MODERATOR"] = mod_profile

    # === PASS 2: Merge + resolve ===
    results = {}
    for task_name in sorted(all_tasks.keys()):
        log.info("=" * 60)
        log.info("[Pass 2] %s", task_name)
        log.info("=" * 60)
        task_output = output_dir / task_name
        mic_raw = all_mic_raw[task_name]

        # Step 3: Merge (ratio >= 2.0 = confident, else uncertain)
        master = merge_transcripts(all_mic_transcripts[task_name], mic_to_speaker)

        # Step 4: P50 confirms uncertain participants (mic/p50 > 0.3)
        if task_name in per_task_room_audio:
            # Per-task P50 — already aligned, offset = 0
            p50_audio, p50_sr = per_task_room_audio[task_name]
            resolve_with_room_mic(master, p50_audio, p50_sr, 0.0, mic_raw)
        elif room_audio is not None and task_name in p50_offsets:
            # Continuous P50 — use computed offset
            resolve_with_room_mic(master, room_audio, room_sr, p50_offsets[task_name], mic_raw)

        # Step 5: Embeddings resolve remaining uncertain → participant or moderator
        # Pass P50 audio so moderator profile comparison uses the room mic
        task_room = None
        if task_name in per_task_room_audio:
            task_room = per_task_room_audio[task_name]
        elif room_audio is not None and task_name in p50_offsets:
            offset = p50_offsets[task_name]
            mic_dur = max((seg.end for segs in all_mic_transcripts[task_name].values() for seg in segs), default=0)
            s = int(offset * room_sr)
            e = int((offset + mic_dur) * room_sr)
            task_room = (room_audio[s:e], room_sr)
        resolve_with_embeddings(master, profiles, mic_raw, mic_to_speaker, room_mic_audio=task_room)

        save_master_tsv(master, task_output / "master_transcript.tsv")
        save_master_words_tsv(master, task_output / "master_words.tsv")
        save_readable_transcript(master, task_output / "transcript.txt")

        results[task_name] = master
        log.info("Task %s complete: %d segments", task_name, len(master))

    return results


MIN_REF_DUR = 1.5
