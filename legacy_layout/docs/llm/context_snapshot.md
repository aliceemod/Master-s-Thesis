# Context snapshot

> **Purpose:** Copilot reads this first every session. Current state only — history is in `CHANGES.md`.

## Doc routing (what to read when)

| Working on… | Read |
|---|---|
| Any code/tool change | This file + `.github/copilot-instructions.md` + `docs/decisions.md` |
| System design | `docs/architecture.md`, `docs/data_flow.md` |
| Task/stimulus logic | `docs/stimulus_design.md`, `docs/stimulus_design_small.md`, `stimuli/STRUCTURE.md` |
| Event markers | `docs/tasks_description.md` |
| Data collection | `docs/data_collection_guide.md` (end-to-end session protocol), `docs/raw_data_upload_and_bids_conversion.md` (post-session upload + Tobii ingest + raw→BIDS) |
| Post-processing merge | `tools/multisource_to_bids_runs.py` (merge AV/Recording/Stimuli/Tobii sourcedata into BIDS, chunk into task runs T0–T4, optionally split media per task with `--split-media`, emit normalized stimuli answers table, and write participant signal map TSV) |
| Printable role guides | `docs/sources/moderator_guide.tex` (scripts, checklists) · `docs/sources/data_collector_guide.tex` (CLI, devices, archival) |
| Offline compute stack | `docs/offline_compute_stack.md` (two-workstation post-collection processing layout: RTX 5080 primary GPU worker, Phoenix RTX A2000 staging/secondary worker, 14 TB USB archive, direct-cable networking, and command workflow), `tools/offline_compute_stack.py` (stack init, queue tracking, per-session plan generation, and stage execution helper) |
| Device setup | `docs/emotibit_quickstart.md`, `docs/jabra_panacast_quickstart.md`, `docs/vicon_tobii_lsl_capture.md` |
| Jabra cameras | `docs/jabra_recording_checklist.md`, `docs/jabra_panacast_guide.md` |
| Calibration | `docs/calibration_usage.md` (**primary reference — 4-phase lifecycle**), `tools/calibrate_charuco.py`, `tools/validate_calibration_robust.py` (desk marker validation via `--marker-map`; camera orientation from `configs/ffmpeg_multicap.json` `rotate_180`; extra dicts via `--aruco-dicts`), `tools/visualize_calibration.py`, `configs/camera_specs.json` (P20 center-crop: `expected_fx_1080p=950`; per-camera overrides for cam5/cam6 in `camera_overrides` block; see ADR-0004/0005), `tools/merge_calibration_tomls.py` (transplant cameras from prior TOML), `tools/recenter_calibration.py` |
| Glasses localization | `docs/calibration_usage.md` Phase 4, `tools/tobii_multicam_glasses_tracker.py` (Approach A: ArUco markers on glasses, triangulated by fixed cams → 6-DoF pose per frame), `tools/tobii_multi_glasses_world_align.py` (Approach B: PnP from scene-video markers → pose per frame), Approach C (fusion) not yet implemented |
| Online calibration | `docs/online_calibration.md`, `tools/online_calibration.py` (live ChArUco detection, marker-aware audio feedback, per-frame sync logs in `frame_logs/` with wall-clock `pts_time`, strict configured-camera presence + recording-name checks, DirectShow identity-based index resolution via `video_alt_name` to keep camera labels aligned with ffmpeg_multicap, UTF-8-safe export subprocess capture, robust validation fallback, fast-recording mode that skips live validation and relies on post-hoc export, seeded export intrinsics enabled by default) |
| 3D pose/face/hand | `tools/multicam_pose3d.py`, `tools/face_hand_pipeline.py`, `tools/video_only_3d_pipeline.py`, `docs/video_only_3d_pipeline.md` |
| Video feature extraction | `docs/video_feature_extraction.md`, `tools/extract_video_features.py` (compact per-camera frame sync, ArUco detections, optional MediaPipe body/face/hand arrays, optional RTMPose/RTMW-via-MMPose body extraction, `--dry-run` preflight summary generation, and automatic split-clip timing recovery from `annot/*_task_run_windows.tsv` when frame logs are absent) |
| Physio feature extraction | `tools/features/extract_physio_features.py`, `tools/features/extract_pupil_features.py`, `tools/features/visualize_physio_features.py`, `tools/features/analyze_physio_paper.py`, `tools/features/analyze_autonomic_paper.py`, `paper/PHYSIO_ANALYSIS_PLAN.md` |
| OpenPose scripts | `scripts/README.md`, `docs/openpose_integration.md` |
| Sync/timing | `docs/SYNC_BEST_PRACTICES.md`, `docs/recording_sync_calibration_pipeline.md` |
| Known bugs | `docs/known_issues.md` |
| Labels/annotation | `docs/labels_codebook.md` — full type/subtype definitions incl. revised OVL taxonomy (2026-07-01) |
| Overlap relabeling | `tools/relabel_overlaps.py` — batch-applies timing+lexical OVL classifier to transcript TSVs; see `docs/labels_codebook.md` §2 for design rationale |
| Prompt templates | `docs/llm/prompt_playbook.md`, `docs/llm/DEVOPS_PROMPT_LIBRARY.md` |

## Current state (what exists, what works)

### Experiment design
- **4 participants** per session (P1–P4), **4 tasks** (T1–T4)
- T1 Hidden-Profile Decision, T2 Mini-Negotiation, T3 Idea Generation (NGT), T4 Public-Goods Micro-Game
- Session ~75–90 min (main) or ~60–75 min (small/pilot)
- **Phase durations (active timers):** T1 discussion+selection 420 s; T2 negotiation 480 s + settlement 60 s; T3 idea-gen 150 s (silent) + ranking/discussion 420 s + group-selection 60 s; T4 contribution 60 s (private) + outcome reveal 60 s + discussion 180 s
- **VAD schedule:** T0 [60,150,240] · T1 [60,210,360] · T2 [60,240,420] · T3 [210,360,510] (skips silent idea-gen) · T4 [90*,270] (* simultaneous at outcome reveal, then towards end of discussion); jitter ±5 s

### Hardware & device inventory

| Device | Count | Signals | PC | LSL pathway |
|---|---|---|---|---|
| Tobii Pro Glasses 3 | 4 (one per P) | gaze, pupil, egocentric video, IMU | Recording PC | Tobii SDK → `tobii_glasses_lsl_bridge.py` → LSL; also via Vicon SDK |
| EmotiBit | 4 (one per P) | PPG, GSR/EDA, temp, IMU | Recording PC | **Preferred:** Oscilloscope LSL push → `tools/emotibit_lsl_merger.py` → merged `Emotibit_P#_stream` (hardware ID `MD-V7-XXXX` in `source_id` maps to P1–P4 via `configs/emotibit_participants.json`). **Fallback:** UDP → `emotibit.py` → LSL. |
| Cameras (Jabra P20/P50) | 7 main + 2 optional extra | video + embedded audio | AV PC | clocks -> LSL via `ffmpeg_multicap.py` frame-logs |
| DPA 4060 mics | 5 (4 close-talk + 1 room/spare) | audio | AV PC | clocks → LSL via `dpa_recorder.py` |

> **Overwrite guard:** `ffmpeg_multicap.py` appends `_HHMMSS` to every session directory at launch, so re-runs never clobber earlier data.
>
> **Session GUI:** `tools/session_orchestrator_gui.py` provides step-by-step launch with live status dots, per-step Redo buttons, and a dedicated Start/Stop Recording toggle that writes `recording_start`/`recording_stop` events to `events.tsv`.
| Vicon optical | 6 cameras | 3D markers | Recording PC | Vicon DataStream SDK → `vicon_nexus_lsl_bridge.py` → LSL |
| Tablets | 4 (one per P) | self-report responses | Wi-Fi → Recording PC | SSE from `display_server.py`; responses → LSL |
| Big Screen | 1 | shared stimulus display | Recording PC (HDMI) | SSE from `display_server.py` |

### Two-PC architecture
- **Recording PC** — runs LSL central recorder, stimuli server, Tobii bridge, Vicon bridge, EmotiBit listener. All LSL streams converge here → XDF.
- **AV PC** -- runs `ffmpeg_multicap.py` (7 cameras) and `dpa_recorder.py` (5 mics). Publishes clock/progress streams to LSL on Recording PC.

### Participant registration
- `src/affectai_capture/registration.py` — maps real names → P1–P4, stores demographics/personality linkage
- `participants.tsv` at study root (BIDS-compliant, anonymised IDs only)
- Per-session `participants.json` sidecar with group assignment

### Session orchestrator
- `tools/session_orchestrator.py` — automates full data-collection pipeline across two PCs (AV PC + Recording PC)
  - Reads `configs/session_schedule.example.tsv` (group_id + 4 participant names + optional demographics)
  - Deterministic random P1–P4 seat assignment (SHA-256 seed from group_id — both PCs agree)
  - Creates BIDS session skeleton, registers participants, writes events.tsv
  - **AV PC** (`--role av-pc`): spawns `ffmpeg_multicap.py` (video/audio + LSL) + starts Tobii on-device recording via G3SDK `Recorder.Start()`
  - **Recording PC** (`--role recording-pc`): spawns `display_server.py` (stimuli + LSL markers) + `tobii_glasses_lsl_bridge.py` (gaze → LSL) + LabRecorder (XDF)
  - `ProcessSupervisor` monitors child health, restarts nothing (fail-loud), graceful Ctrl-C → `CTRL_BREAK_EVENT` on Windows
  - Writes `ses-*_orchestrator_summary.json` on shutdown
- `tools/session_orchestrator_gui.py` — tkinter GUI wrapper (dark mode, no new deps)
  - New Data Ops panel (both roles): Upload Raw to Azure, Ingest Tobii manual download, Raw→BIDS conversion
  - **Session naming lock across two PCs**: writes a shared lock file under `<out_root>/_session_locks/` and blocks start on mismatched session/group/seat mapping
  - **AV lock fallback for split filesystems**: if Recording-PC lock is not visible from AV (e.g., non-shared `out_root`), AV launcher can continue only after explicit operator confirmation (manual-sync mode)
  - **Lock status visibility**: control center shows compact lock state text, full lock details on hover, live role refresh from lock file, and `updated X ago` freshness
  - **Control-center stop closes all managed processes**: supervisor children, feed, calibration, Tobii on-device recording, and active data-op subprocesses
  - **Step-level process controls**: each process-backed session step has both `Redo` and `Stop` actions
  - **Critical-step protection**: stopping FFmpeg/Display Server via step controls requires confirmation
  - **Stop Session behavior**: explicitly stops active recording, then closes stimuli and all remaining managed processes
  - Load/browse schedule TSV, view groups + randomised seat table
  - **Role-specific config**: switching AV PC / Recording PC shows only relevant fields
    - AV PC: FFmpeg config
    - Recording PC: LabRecorder path, Display host/port
    - Common (always): Output root, Tobii config, EmotiBit participant-map path, Tobii SDK DLL
  - **Auto-run numbering**: session ID = `YYYYMMDD_{group}_run{NN}`, auto-increments to avoid overwrite
  - **Overwrite guard**: warns if session directory already exists before starting
  - **Pre-flight device check**: probes Tobii IPs (TCP :80), enumerates dshow cameras, verifies LabRecorder/SDK/config/port
  - Pre-flight also verifies EmotiBit participant-map file path exists
  - **Live LSL stream monitor (both roles)**: auto-scans every 5 s, scrollable treeview + device-type summary (Tobii:N | EmotiBit:N | …); AV shows this as diagnostics panel
  - Start/Stop session with live process health dots (green/red per child PID)
  - Prominent session-ID banner on start — both PCs must use same ID
  - **Recording status details**: shows live LSL recording timer and XDF file size
  - **Python XDF recorder (`tools/lsl_xdf_recorder.py`)**: pure-Python LabRecorder replacement using **single-writer-thread architecture**: N puller threads → per-stream `collections.deque` (GIL-atomic append) → 1 writer thread drains all queues every 100 ms → file. Eliminates GIL/lock convoy starvation that occurred with 79+ concurrent write-lock threads. XDFRecorder resolves all LSL streams (or prefix-filtered subset) at start() and records them to a single .xdf file. Default resolve-timeout in GUI: 30 s. Late-stream discovery now rescans every 5 s by default (configurable via CLI) and dynamically adds new inlets/StreamHeader chunks mid-recording. Stream continuity uses a session-stable identity key `(name, type, source_id, hostname, channel_count, channel_format)` so outlet restarts (new UID) can still reconnect as the same logical stream; matching is no longer name-only. Pull loops use tighter polling and larger chunks for better high-rate capture robustness, and use `StreamInlet.pull_chunk` for both numeric and string streams (pylsl 1.18 compatible). Per-stream micro-batching (target 32 samples, max 100 ms latency) increases samples-per-chunk and reduces file overhead while flushing safely on reconnect/stop. Recorder logs a deterministic per-stream manifest at start and stop (stop includes per-stream sample totals) for quick parity checks against LabRecorder. Boundary + flush done inside writer thread every 30 s (`--boundary-interval`, default 30 s). Late-stream discovery scan interval is configurable via `--late-discovery-interval` (default 5 s).
  - **Azure blob configuration**: accessible in End of Session section during data ops (not in Configuration section)
  - **Blob credentials file**: "Load" supports JSON and `.env` formats; default path auto-prefers `tools/azure_strorage/azure_strorage/.env` when present (fallback to `configs/azure_blob_credentials*.json`)
  - **Default Azure destination**: GUI pre-fills `https://affectai.blob.core.windows.net/raw` for AV/Recording raw uploads
  - **Configuration panel visibility**: Configuration frame auto-hides during active session, re-appears after session ends
  - **Calibration import**: can import saved calibration material files into session/output calibration folder
  - **Calibration recording foldering**: calibration capture now auto-resolves/creates the session folder from Session ID and writes into `<session_dir>/sourcedata/av/calibration/` (`calibration-video*` and `calibration-validation-live*`), keeping calibration clips inside session-scoped AV sourcedata; calibration panel shows a live `Output dir` preview, supports browsing `Capture dir`, and step-2 auto-discovery accepts both legacy and session-stamped run folder names; "Find Latest" and "2) Calibrate" now also fall back to a broad `rglob("video")` scan across all sessions under `{out_root}/{recording_type}` so captures are found even when no session is loaded in the GUI; if the operator pastes the `video/` subfolder path the GUI automatically goes up one level to the run dir
  - Real-time scrolling log with colour-coded levels + elapsed timer
  - Default paths: Tobii SDK `tools/vendor/tobii_sdk/net472/G3SDK.dll`, LabRecorder `tools/LabRecorder/LabRecorderCLI.exe`
  - Background-thread execution — UI never freezes
  - Launch: `python tools/session_orchestrator_gui.py`

### Protocols
- Main: `docs/sources/main/AffectAI_protocol.tex` (4-task variant)
- Small: `docs/sources/aux/Affect_AI_data_collection_small.tex` (MSc pilot, T1–T4)

### Stimuli system
- `stimuli/display_server.py` — HTTP/SSE server → Tablets 1–4, Big Screen, Moderator (`/moderator`).
  **mDNS / stable tablet URLs**: when `zeroconf` is installed (`pip install zeroconf`), the server registers itself as `affectai-display._http._tcp.local.` on startup so tablets can permanently bookmark `http://affectai-display.local:8080/tablet/N` — survives DHCP IP changes.  Moderator panel shows `.local` URLs in green when active; IP-based fallback URLs shown below.  `/network_info` response includes `mdns_hostname` field.
  **Android companion app**: `stimuli/android_tablet_app/` — Kotlin WebView app; uses `NsdManager` to auto-discover the server, falls back to last-known IP, then manual URL entry.  One-time tablet-number setup (1-4) stored in SharedPreferences.
  Moderator console has numbered phase buttons, untimed/timed timer logic, per-task step-by-step instructions, `/clear_cache` on task switch.
  **Tobii calibration step**: each task T1–T4 now has a purple "0. Tobii Calibration" button that precedes the brief. Clicking it pushes a full-screen SVG crosshair/aiming-point to bigscreen + all 4 tablets, and emits a `tobii_calibration` LSL event to every stream (moderator, participant_1–4, bigscreen) for post-processing gaze alignment.  
  **Participant names**: when launched with `--out-root`, automatically loads participant names from `.private/registration_ledger.jsonl` for the current session. Names are normalized to first-name only for display (tablet headers and bigscreen participant line) and are NEVER sent to LSL markers or event logs (only anonymous P1-P4 IDs are logged). Moderator console also includes manual name input fields. Endpoints: `GET /get_participant_names`, `POST /set_participant_names`.
  Finish workflow: each task (T0–T4) has a red "✔ Finish" button → pushes per-task `TASK_FINISH_CONTENT[task]` to all devices + triggers post-block survey (T1–T4).
  Final decision: `logFinalDecision` logs to server AND broadcasts summary card to bigscreen (T1: candidate, T2: topic+format, T3: idea+author).
  T1: evidence_card phase merges evidence push (tablets) + silent-reading display + 75 s timer (bigscreen) in one button; discussion phase = 420 s (7 min); new `candidate_selection` phase (60 s) pushes `T1_CANDIDATE_SELECTION_FORM` to all tablets + position-banner bigscreen.
  T2: role_card push auto-sends 8-min negotiation timer to bigscreen.
  T3: merged "Ideas Board + Discussion (7 min)" button (`show_ideas_discussion`) pushes ideas grid + 420 s timer to bigscreen; idea generation 180 s; group_selection 60 s.
  T4: contribution 60 s, outcome reveal 120 s, discussion 120 s; outcome preserved on bigscreen during discussion via `T4_LAST_OUTCOME`; "Reveal Outcome" button calls `showT4Outcome()` → `POST /t4_outcome` — moderator can force-reveal even with <4 contributions (confirm dialog), no timer wait for manual reveal. Outcome display includes per-participant table.
  WRAPUP pseudo-task: 3 phases (intro, send_final_vad, thanks) — final VAD collection + session-complete screen.
  **VAD schedule:** phase-aligned probe schedule via `_VAD_TASK_SCHEDULE` (≥2 probes/task): T0 [60,150,240], T1 [60,210,360], T2 [60,240,420], T3 [210,360,510], T4 [90☆,270] (☆=simultaneous for all tablets at outcome reveal). Per-tablet stagger 0–20 s; T4 first probe simultaneous. Jitter ±5 s. Finish/postblock paths force-stop the timer server-side; manual all-tablet VAD sends use staggered dispatch.
  **T4 display:** reduced font sizes in shared instructions table (0.92em, padding:7px 10px) and outcome reveal screens (bigscreen title:34px, main:22px; tablet title:20px) to prevent text cutoff.
  **Post-task questionnaire:** T1 includes per-participant familiarity items (`familiarity_p1`..`familiarity_p4`) asked only after Task 1. Tablet rendering personalizes participant-referenced labels as `P# (FirstName)` when names are available; logging remains anonymized by item key and P1-P4 IDs.
- `stimuli/task_content.py` — all task materials (T0–T4), timer durations, per-task `TASK_FINISH_CONTENT` dict, `WRAPUP_CONTENT`, `WRAPUP_THANKS`. Contains only objects/fields that are actually rendered on tablets or bigscreen. Live objects: T0 (T0_WELCOME…T0_FREE_TALK), T1 (T1_INTRO, T1_SILENT_READING, T1_EVIDENCE_CARDS, T1_CANDIDATE_SELECTION_FORM, T1_POSITION_BANNER_HTML), T2 (T2_SHARED_BRIEF[topics/formats only], T2_ROLE_CARDS[html only], T2_SETTLEMENT_FORM), T3 (T3_SHARED_INSTRUCTIONS, T3_IDEA_GENERATION_SHEET, T3_GROUP_SELECTION_FORM), T4 (T4_SHARED_INSTRUCTIONS, T4_CONTRIBUTION_FORM).
- `stimuli/probe_definitions.py` — VAD probes with emoji anchors, rotating cognitive/social probes
- `stimuli/tasks/` — PsychoPy T1–T4 implementations (entry: `stimuli/task_runner.py`)
- `stimuli/event_logger.py` — `ExperimentEventLogger`: dual-write (local TSV + LSL) event logging. 7 streams: `participant_1`–`participant_4`, `moderator`, `bigscreen`, `experiment` (cumulative). Each event carries `wall_clock`, `lsl_clock`, `session_id`, `group_id`, `stream`, `event_type`, `task`, `phase`, `participant`, `device_id`, JSON `detail`. LSL `<desc>` embeds group/session. CLI: `--group-id grp-A`.
- Per-device LSL streams: 7 outlets (`AffectAI_Participant_1–4`, `_Moderator`, `_BigScreen`, `_Experiment`)

### Capture
- `tools/ffmpeg_multicap.py` — multi-device capture (P20/P50 + DPA), LSL clock, frame logs, progress streams; Windows preflight now aborts if any configured `video_alt_name` is enumerated by DirectShow as `(none)`/missing, and `input_video_codec` defaults to auto-negotiate unless explicitly set. LSL sidecar recorder teardown now stops before capture shutdown and uses non-recovering inlets to avoid benign reconnect spam at end-of-run.
  - **Recording duration:** Default 2 hours (7200s) via `--max-duration` (use `--max-duration 0` for unlimited). Removed hardcoded 1-hour limit from all FFmpeg commands.
  - **Pixel format control:** Added `pixel_format` config option to force camera output format (e.g., `yuyv422` instead of `nv12` for better quality). PanaCast 50 defaults to `nv12` (4:2:0 chroma, 1.5 bytes/pixel) while PanaCast 20 uses `yuyv422` (4:2:2 chroma, 2 bytes/pixel) — this results in ~55% lower bitrate for P50 if not overridden. Config now forces P50 to `yuyv422`. See [docs/camera_quality_diagnostics.md](docs/camera_quality_diagnostics.md) for troubleshooting.
- `tools/check_usb_distribution.ps1` — Windows preflight helper: resolves configured camera `video_alt_name` identities to live PnP location paths, groups by USB host controller key, flags missing devices and controllers above threshold (`-MaxCamerasPerController`, default 2)
- `tools/dpa_recorder.py` — 5× DPA mics via RME Fireface 802
- `devices/jabra_panacast.py` — USB/network Jabra capture
- `devices/emotibit.py` — 4× EmotiBit: PPG, EDA, temp, IMU → LSL + JSONL
- `tools/tobii_glasses_lsl_bridge.py` — 4× Tobii Glasses → LSL (gaze, pupil, 3D gaze, IMU, events); uses `System.Reactive.Observer.Create[T]` for pythonnet-compatible SDK subscription (extension methods don't work), explicit .NET assembly resolution from the SDK folder, and skips unreachable configured IP devices so one offline unit does not terminate the whole bridge
- Tobii participant stream naming (default): one unified participant stream (`Tobii_P1_stream` … `Tobii_P4_stream`) with regular sampled channels only (gaze+pupil, optional `--with-3d`, optional `--with-imu`) and nominal LSL rate `50 Hz` by default (`--nominal-srate` to override). Irregular Tobii packets (`event`, `sync_port`) are published to global stream `evetns_tobii` at irregular rate (`0 Hz`). Use `--split-streams` for legacy separate `Event`/`IMU`/`SyncPort` outlets.
- EmotiBit participant stream naming (default): one unified stream per participant (`Emotibit_P1_stream` … `Emotibit_P4_stream`) with all physio channels as vector channels; default hardware-ID map: P1 `MD-V7-0001141`, P2 `MD-V7-0001160`, P3 `MD-V7-0001409`, P4 `MD-V7-0000837`
- EmotiBit participant mapping is configurable via `configs/emotibit_participants.json` (CLI: `python -m affectai_capture.devices.emotibit --participant-map <path>`), supporting legacy hardware-ID map and optional `by_source` (source IP/name → participant) for deterministic participant assignment when app-side names collide; example: `configs/emotibit_participants_by_source.example.json`
- EmotiBit startup diagnostics now warn clearly when UDP listener is active but no packets have arrived yet (first warning ~10 s, then ~15 s cadence), and include expected `by_source` sender hints when configured
- EmotiBit can optionally publish source-keyed fallback LSL streams when participant mapping fails via `--allow-unmapped-lsl` (also available in `tools/emotibit_one_command.py`); useful for diagnostics or bridge mode where packets should still be visible in LSL
- EmotiBit one-command fixed-setup launch: `python tools/emotibit_one_command.py --session <session_dir> --participant-map <map.json>` validates the map and starts UDP→LSL capture (practical mode when devices are preconfigured once in Oscilloscope)
- `tools/tobii_multi_glasses_world_align.py` — offline marker-based world alignment for recorded 4× Tobii (scene video + gaze NDJSON → world gaze; `gaze2d` or `gaze3d` rays)
- `tools/tobii_multicam_glasses_tracker.py` — track 4× glasses 6-DoF pose via small ArUco markers on glasses frames, detected by fixed multicam rig; triangulates markers then transforms Tobii gaze to world; config loader accepts both `table_markers` and `world.marker_map` schemas; video-to-calibration camera mapping is separator-insensitive (`-`/`_` and long prefixes) for BIDS-style filenames; `configs/tobii_multicam_glasses_tracker.example.yaml`, `configs/desk_markers_large.yaml`
- `tools/test_mediapipe_pose.py` — helper script to generate OpenPose-compatible BODY_25-style per-frame JSON from videos; now creates parent JSON directories before writing (Windows path/output safety)
- `tools/generate_aruco_marker_sheet.py` — generate printable ArUco marker sheets (table corners + glasses frames); supports `--profile lab_dual_board` for 6 desk-edge 50mm markers (4 corners + left/right side centers) + 4 glasses pairs at 25mm, `--paper-size a3` for A3-ready paged outputs (`table_markers_a3_page1.png`, `table_markers_a3_page2.png`, `glasses_markers_a3.png`), and exports machine-readable layout JSON (`lab_dual_board_layout.json`) plus tracker-ready config JSON (`tobii_multicam_glasses_tracker_lab.json`) with desk/fixed-board/camera/participant geometry metadata
- `tools/qc/qc_tobii_world_gaze.py` — QC summary + scatter/timeseries plots for world-aligned multi-glasses gaze outputs (optional marker/bounds overlay via align config)
- `tools/vicon_nexus_lsl_bridge.py` — Vicon DataStream → LSL (segments, marker trajectories, devices, eye-trackers); captures raw subject markers + labeled/unlabeled marker trajectories into NDJSON, supports marker toggles (`--no-markers`), marker-frame JSON LSL (`--markers-lsl`), and per-marker numeric LSL streams with `--per-stream-lsl`; JSON stream load controls: `--frame-json-lsl-max-hz` (default 50 Hz), `--markers-json-lsl-max-hz` (default 50 Hz), `--output-flush-interval-s` (default 1.0 s), `--loop-sleep-ms` (default 1.0 ms)
- `tools/upload_raw_data.py` — uploads complete session raw tree to Azure Blob with sync-artifact manifest; supports `azcopy` destination URL mode and Azure SDK account/key/container mode, including fully-offline SDK `--dry-run` preflight
- `tools/ingest_tobii_downloads.py` — ingests manually downloaded Tobii files into `sourcedata/tobii_device/`
- `tools/raw_to_bids.py` — converts AV/Recording/Tobii raw sources to BIDS-oriented modality outputs (optional XDF extraction via `pyxdf`)
- `tools/multisource_to_bids_runs.py` — merges split source folders (AV/Recording/Stimuli/Tobii) into one output `ses-*`, runs raw→BIDS conversion, derives deterministic task windows from stimuli experiment events with phase-aware boundaries (`T0`: intro→finish, `T1`–`T4`: tobii_calibration→finish when markers exist), writes per-task run outputs (`task-T0`…`task-T4`, `run-01`) including run-sliced LSL-derived tables, exports normalized readable stimuli answers (`beh/*_stimuli_answers.tsv`), writes participant signal mapping (`annot/*_participant_signal_map.tsv`), and avoids media clip overwrites for repeated captures by adding deterministic capture tokens in `acq-*` names
- Sync: 4-tier frame synchronisation (frame logs → LSL → progress TSV → events JSONL)

### BIDS output
- `sub-{id}/ses-{id}/` with modality dirs: `eeg/`, `et/`, `physio/`, `audio/`, `video/`, `mocap/`, `beh/`, `annot/`
- One authoritative `events.tsv` per session (timeline spine)
- `participants.tsv` at study root for cross-session participant roster

### Physio features and paper analysis
- `tools/features/extract_physio_features.py` extracts task-level and 30 s rolling-window EmotiBit features from split `physio/*_task-T*_run-01_acq-P*_emotibit.tsv.gz` files, including PPG/HR/HRV, EDA tonic/phasic/SCR, temperature, and IMU summaries with QC flags.
- Canonical outputs are `features/physio_participant_task.tsv`, `features/physio_window_30s.tsv`, `features/physio_qc_summary.tsv`, and `features/physio_feature_definitions.tsv`; use `--include-missing-qc` for expected participant-task coverage tables.
- `tools/features/extract_pupil_features.py` extracts Tobii pupil size features from task-split `et/*_task-T*_run-01_acq-P*_tobii.tsv.gz` files into `features_pupil_participant_task.tsv` and `features_pupil_window_30s.tsv`.
- `tools/features/visualize_physio_features.py` creates quick QC/review PNGs, `tools/features/analyze_physio_paper.py` writes paper-facing EmotiBit tables under `results/physio/`, and `tools/features/analyze_autonomic_paper.py` combines EmotiBit + pupil features into task fingerprints, modality coverage, pupil-physio links, and temporal-profile figures under `results/autonomic/` and `figures/autonomic/`.
- `tools/features/run_neurips_benchmark_baselines.py` writes NeurIPS E&D feasibility baselines under `results/benchmarks/logocv_baselines.tsv` and `paper/tables/benchmark_baselines.tex` using leave-one-group-out splits on the no-video release.

### 3D reconstruction
- `tools/multicam_pose3d.py` — multicam 3D pose (zone-aware, face-cameras, distortion-robust)
- `tools/face_hand_pipeline.py` — 478 face landmarks, 21×2 hand landmarks, 52 blendshapes
- `tools/refine_skeleton_3d.py` — quality gate → velocity filter → interpolation → smoothing
- `tools/video_only_3d_pipeline.py` — end-to-end offline video-only workflow that runs Tobii glasses world alignment (`tobii_multicam_glasses_tracker.py`), multicam 3D reconstruction (`multicam_pose3d.py`), optional skeleton refinement (`refine_skeleton_3d.py`), and per-participant gesture event extraction (`gestures_events.ndjson` + `gestures_summary.json`) in the shared calibrated coordinate system; supports `--dry-run` prerequisite validation and writes `pipeline_dry_run.json`; if `--calibration` is missing, can auto-calibrate from six P20 videos in `--videos-dir` (cam1..cam6) via `tools/calibrate_charuco.py`
- `tools/extract_video_features.py` is now the recommended first heavy video pass before Pipeline 2: run `--dry-run`, then extract reusable frame sync, ArUco, body, face, and hand features under `features_video/`. `video_only_3d_pipeline.py` still consumes OpenPose-compatible `--pose-root` JSON; feature-native downstream adapters should use `features_video/feature_manifest.json` plus per-camera `.npz`/JSONL artifacts.
- `tools/online_multicam_feed.py` — live multi-camera preview with ArUco overlays, zone-based MediaPipe pose per person, and optional 3D triangulation preview using calibration TOML; supports `--mode live` (test feed from DirectShow cameras) and `--mode surveillance` (read from growing MKV files written by `ffmpeg_multicap.py` for non-blocking monitoring during recording); `--fast` mode enables frame skipping, preview downscaling, and lighter model for low-overhead surveillance; zone rectangles are now scaled with downscaled previews so skeleton overlays stay aligned in fast mode, per-person triangulation stores keypoints keyed by camera label (stable cam-map lookups), camera-specific zone overrides resolve through camera aliases from capture settings (`cam1`, `camera1`, `cam_0`) and calibration keys, glasses-marker detection runs on full-resolution frames with multi-dictionary ArUco support (`--aruco-dicts`, default `DICT_4X4_50,DICT_4X4_250`), and 2D overlays draw landmark connections + points for clearer skeleton display; example zone config: `configs/desk_zones.json`; integrated into session orchestrator GUI with feed mode selection and full CLI options
- `tools/session_orchestrator_gui.py` — Tkinter GUI for session orchestration with recording_type selector at top (test/pilot/final → top-level folder), schedule loading (reads `date`, `start_time`, `end_time` from TSV), BIDS init with hierarchical folder structure (`{out_root}/{recording_type}/sub-XX/ses-{session_id}/`), auto-generated session IDs from schedule (`{group_id}_{date}_{start_time}` e.g., `grp-01_20260305_1300`), enhanced seat assignment display showing "Group | Date | Time" header, session history tracking (logs each recording start to `sessions/session_history.tsv` with timestamp, group, date, P1-P4 names, role, session_dir), per-session capture manifest tracking in `ses-<session_id>_capture_tracking.json` (XDF/AV/Tobii artifacts + stage checkpoints), process health monitoring, calibration controls (AV 3-step workflow: Record Calibration Video -> Calibrate -> Live Validation with auto short capture), feed controls (surveillance/test), FFmpeg options panel, and Recording PC action buttons (start stimuli, Tobii LSL, LSL recording). Calibration consumers now auto-resolve the newest calibration TOML (when field value is empty/stale) across session and out-root scopes, and step-3 live validation forwards multi-dictionary marker detection (`--aruco-dicts DICT_4X4_50,DICT_4X4_250`) to match feed robustness. `Start LSL Recording` verifies LabRecorder is alive before flagging recording start; AV launch validates lock readiness first and can run in explicit operator-confirmed manual-sync mode when shared lock files are not visible across PCs. Data Ops Azure upload now supports both destination-URL (`azcopy`) mode and account/key/container JSON credentials (SDK fallback) without exposing account keys in command-line logs.
- `tools/session_orchestrator_gui.py` schedule panel now supports long-list navigation with a dedicated vertical scrollbar, mouse-wheel scrolling, and quick `Top` / `Bottom` row-jump buttons that auto-select the target group.
- `tools/session_orchestrator_gui_av.py` — AV PC GUI entrypoint with role locked to AV (calibration + feed controls enabled) and AV-specific startup defaults that enforce sync-safe capture settings (`frame-log`, `record-lsl`, markers, `ffmpeg_clock@100Hz`, `ffmpeg_progress_`, `stabilization-delay 2.0`, `sequential-start-delay 0.3`), auto-applies P50 color preset via `tools/lock_exposure.py` (brightness 160, contrast 150, saturation 138, sharpness 170) immediately before FFmpeg capture start, suppresses camera dialog popups in AV runs, enables surveillance feed based on FFmpeg process health (not recording-marker state), and exposes calibration as explicit 3-step buttons where step 3 captures a fresh short validation clip (`calibration-validation-live*`) before desk-marker validation.
- `tools/session_orchestrator_gui_recording.py` — Recording PC GUI entrypoint with role locked to Recording (capture-only controls)
- Recording-PC GUI session flow now includes a managed **Vicon LSL Bridge** step (like Tobii bridge), configurable via `Vicon server` (`host:port`, default `localhost:801`), launching `tools/vicon_nexus_lsl_bridge.py` with marker-focused forwarding (`--per-stream-lsl --markers-lsl --structured-lsl --no-devices --no-eye-trackers`) and session-scoped raw output under `sourcedata/vicon_lsl/vicon_datastream_raw.ndjson`; capture manifests include `vicon_lsl_bridge` artifacts
- Recording-PC GUI now exposes **Vicon Bridge Settings** (enable step + stream/output toggles + performance controls for `Frame JSON Hz`, `Marker JSON Hz`, `Flush (s)`, `Loop sleep (ms)`), performs preflight endpoint reachability checks for the configured Vicon server, and surfaces Vicon streams in the LSL monitor summary/list (`Vicon:N`); default Vicon server is `10.145.48.221:801`; Vicon bridge is OFF by default at GUI startup (operator opt-in)
- Recording-PC GUI Tobii readiness wait now uses robust Tobii stream matching (name/type/source-id variants) and excludes irregular Tobii outlets (`_Event`, `_Imu`, `_SyncPort`, `evetns_tobii`) to reduce false "0/N streams" detections when participant Tobii streams are already present
- `tools/calibrate_charuco.py` -- spatial calibration (Charuco + anipose + ground-plane + --init-focal); `record` uses sync-safe ffmpeg_multicap settings (`group-id`, `frame-log`, `record-lsl`, `stabilization-delay 2.0`, `sequential-start-delay 0.3`, `ffmpeg_clock@100Hz`, `ffmpeg_progress_`, markers), defaults to 75s capture with 15s cue beeps, supports `--board-type auto` (5x3/7x5 auto-select), reads board square sizes from config (`--board-config`), and uses Windows-safe process-tree shutdown to prevent orphan ffmpeg captures. During `calibrate`, MKV->MP4 preprocessing attempts event-based temporal pre-alignment from `ffmpeg_multicap_events.jsonl`; recordings should include sync artifacts (`frame-log`, `record-lsl`) for robust multi-camera overlap. `calibrate` auto-excludes cameras with fewer than `--min-charuco-frames` detections (default: 15) before graph construction, printing `EXCLUDED <cam>: N frames` warnings; prevents "Could not build calibration graph" when a camera has 0 or near-0 board detections.
- `tools/calibrate_cameras.py` -- drift/delay calibration recording aligned with the same ffmpeg_multicap sync command + 75s/15s cue-beep protocol; timestamp parsing accepts `unix_time_s`/`unix_time`/`host_time`, and recording stop uses Windows-safe process-tree shutdown
- `tools/online_calibration.py` -- real-time multi-camera ChArUco detection with audio feedback (success beeps on 2+ camera detection, completion chord, per-camera statistics and validation), supports auto board mode (`5x3` and `7x5`) and live feed overlays (`--show-feed`); reprojection validation is optional (`--enable-reprojection-validation`) for faster end-to-end runs; can export session artifacts via `--export-calibration` to produce `video_camera_calibration.toml` plus validation/log reports under `<session>/video/`, with sync-aware preprocessing enabled by default for non-hardware-synced cameras (`frame -> lsl -> events` fallback), encoder fallback for sync-trim export (`libx264 -> h264_mf -> mjpeg`), short default sync-trim clips (defaults to `--duration`) to avoid long full-session re-encodes, UTF-8 subprocess env for export on Windows to prevent cp1252 Unicode crashes, and optional `table_marker_map.yaml` export (`--export-table-marker-map`) to define table-based coordinates.
- `tools/lock_exposure.py` -- exposure/WB/gain lock for Jabra cameras (**standalone preflight only** — `camera_setup_script` is disabled in `configs/ffmpeg_multicap.json`; inline use causes DirectShow handle conflict blocking ffmpeg; AV GUI now invokes it as a separate pre-start step for P50 color consistency)
- `tools/recenter_calibration.py` -- re-express TOML with chosen camera as origin
- `configs/camera_specs.json` -- known FOV/focal length per camera model (P20: 120-deg/554px, P50: 133-deg/418px)

### Camera layout
- 6× PanaCast 20: cam1–cam4 (per-participant face, ceiling-mounted **upside-down** — flip in post), cam5 (front-center overview, upright), cam6 (back-middle rear overview, upright)
- 1× PanaCast 50: wide-angle room view (upright)
- Flip flags: `--flip-cameras cam_0 cam_1 cam_2 cam_3` in `multicam_pose3d.py`, `recenter_calibration.py`, `layout_video_3d.py`
- `tools/annotation_gui/` now auto-rotates cam1-cam4 playback feeds 180° for upright review, including face overlays/crops.
- `tools/annotation_gui/` now defaults visible playback to cam5 when available (fallback: first camera), supports live face-detection pause/resume with `F`, and preserves face controls across session/group reloads.
- `tools/annotation_gui/` face detection now uses temporal box stabilization + brief miss-hold to reduce flicker, sensitivity-aware YuNet confidence filtering, and resilient Haar fallback when DNN misses/unavailable.
- `tools/annotation_gui/` supports optional stronger OpenCV DNN Res10 SSD backbone in `Auto`/`DNN` mode when model files are present under `configs/models/face/` or `models/face/` (`deploy.prototxt` + `res10_300x300_ssd_iter_140000_fp16.caffemodel`).
- `tools/annotation_gui/` now shows a live face-detector backend status label in the camera controls, reflecting active visible-panel backends (`Res10`, `YuNet`, `Haar`, `Off`, `Unavailable`, `Waiting`).
- For flipped cam1-cam4 playback feeds, `tools/annotation_gui/` now runs face detection only in the upper half of the frame (upper-left/upper-right quadrants) after flip.
- `tools/annotation_gui/` now supports manual per-feed face detection ROI selection in GUI (`ROI Edit` + drag on each feed, `Clear ROI` to reset). Manual ROI takes precedence over automatic ROI rules.
- `tools/annotation_gui/` now auto-seeds missing per-feed sync offsets from task sync logs (`annot/*_acq-lsl_sync.tsv`) using LSL-clock-based projection of `ffmpeg_progress_*` media time to a shared LSL reference (task `start_lsl` from `annot/*_task_run_windows.tsv` when available), so DPA/Jabra feeds load closer to aligned before manual fine-tuning.
- `tools/annotation_gui/` now also infers Tobii scene-video offsets from per-participant Tobii ET tables (`et/*_acq-P*_tobii.tsv[.gz]`) via participant first-`lsl_time` anchoring, improving within-Tobii video alignment.
- `tools/annotation_gui/` now falls back to source-session capture logs (`sourcedata/*/lsl/ffmpeg_progress_*.jsonl`) when available, resolved via `AFFECTAI_SOURCE_SESSIONS_ROOT` or default local capture root, to improve initial AV/audio alignment in copied/derived datasets.
- `tools/annotation_gui/` sync controls now include grouped lock toggles: `Lock V` shifts all video feed offsets together, `Lock A` shifts all audio feed offsets together; either can be unchecked for independent adjustment.
- `tools/annotation_gui/` now includes Tobii scene MP4 files from `et/` (`*_task-*_run-*_acq-*_tobii.mp4`) as video feeds in the review grid by default.
- `tools/annotation_gui/` loads Tobii scene feeds with face detection disabled; AV camera feeds keep existing face-detection behavior.
- `tools/annotation_gui/` transcript auto-load now prefers structured `audio_annot/<task>/` inputs: `master_transcript.tsv` (rendered transcript), `master_words.tsv` (word summary), and `mic*_transcript.json` mapped to loaded DPA mic channels.
- `tools/annotation_gui/` transcript panel now includes editable segment/word tables, word-level playhead-follow highlighting during playback, and a horizontal `Now speaking` strip directly below the video grid.
- **Quality expectations:** 60-85 Mbps for MJPEG at 1920x1080@30fps (~200-300 MB/min). P50 now configured with forced `yuyv422` pixel format (was defaulting to `nv12` → 32 Mbps). AV GUI applies P50 preflight controls automatically (brightness 160, contrast 150, saturation 138, sharpness 170) and runs without camera property dialog popups by default.

### Quality
- `make check` passes (ruff + pytest) ✅
- `tests/test_calibrate_charuco.py` — 39 tests: board defs, camera-spec matching, intrinsic matrices, video discovery, encoder detection, CLI dispatch, repo specs
- `tests/test_online_calibration.py` — regression tests for export encoder fallback detection and sync-trim command construction
- `tests/test_tobii_multicam_glasses_tracker.py` — 48 tests: CameraCalibration, config parsing (including `world.marker_map` compatibility), DLT triangulation, marker corner triangulation, glasses pose estimation, gaze-to-world transform, gaze loading, video discovery, video/calibration name mapping robustness, TOML loading, repo config validation
- `tests/test_multisource_to_bids_runs.py` — regression tests for phase-aware task-window derivation, LSL run-splitting, stimuli answer normalization, and participant signal-map generation used by multisource merge/chunk workflow

## How to verify
```bash
make check                # required before any commit
make sync-sources         # after adding PDFs/source cards
```

## Non-negotiables
- Protocol is source of truth (`docs/sources/main/AffectAI_protocol.tex`)
- Code changes → update `CHANGES.md` + this file
- One `events.tsv` per session (timeline spine)
- Self-annotations are individual-level (`docs/labels_codebook.md`)
- **OVL taxonomy revised 2026-07-01**: overlap subtype classification now uses timing + lexical cues; old duration-only labels (competitive/floor_fight over-assigned) replaced with revised labels including `collaborative` and `needs_review`; `.bak.tsv` originals preserved in `tools/`
- Never overwrite raw vendor files; derived outputs only
- Marker payloads must be machine-parseable (no ad-hoc `str(dict)`)
- Core package runs without vendor SDKs; adapters are optional extras
