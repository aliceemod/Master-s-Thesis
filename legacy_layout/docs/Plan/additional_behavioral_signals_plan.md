# AffectAI — Additional Behavioral Signals: What to Extract and How to Use It

> **Context**: This document identifies behavioral signals that can be derived from
> sources already being collected — tablet interaction logs, VAD self-reports,
> post-block questionnaires, task decision logs, LSL event streams, and the stimuli
> system — but are not yet being used in the 3D modeling or analysis pipelines.
> It is a companion to the three existing improvement plans (gaze, ROI/skeleton/gesture,
> and multi-modal fusion).
>
> **Philosophy**: Every signal here comes for free from infrastructure already in place.
> No new hardware is needed. The cost is parsing and alignment.

---

## Table of Contents

1. [Tablet Interaction Micro-Behaviors](#1-tablet-interaction-micro-behaviors)
2. [VAD Trajectory Features](#2-vad-trajectory-features)
3. [Post-Block Questionnaire Signals](#3-post-block-questionnaire-signals)
4. [Task Decision Logs](#4-task-decision-logs)
5. [Big Screen Attention Events](#5-big-screen-attention-events)
6. [Turn-Taking and Speech Structure](#6-turn-taking-and-speech-structure)
7. [Probe Response Latency](#7-probe-response-latency)
8. [Device Connection and Engagement Proxy](#8-device-connection-and-engagement-proxy)
9. [Cross-Participant Behavioral Synchrony](#9-cross-participant-behavioral-synchrony)
10. [Task-Phase Behavioral Profiles](#10-task-phase-behavioral-profiles)
11. [Integration with 3D Pipeline](#11-integration-with-3d-pipeline)
12. [Priority Order](#12-priority-order)

---

## 1. Tablet Interaction Micro-Behaviors

### 1.1 What the Logs Actually Contain

Every response JSONL record from `display_server.py` contains rich timing metadata
that is currently only used to extract the submitted value. The full record structure is:

```json
{
  "device_id": "tablet2",
  "participant": 2,
  "task": "T2",
  "type": "vad",
  "valence": 7,
  "arousal": 5,
  "dominance": 4,
  "client_timestamp": "2026-03-18T13:45:22.341Z",
  "client_perf_ms": 45823.4,
  "clock_offset_ms": -12.3,
  "received_at": 1742300722.895,
  "server_received_lsl": 4521.234,
  "session_id": "ses-20260318_grp-13_run01",
  "probe_schema": {...}
}
```

### 1.2 Response Latency as Engagement Proxy

The time between `probe_displayed` (SSE push timestamp `server_sent_lsl`) and
`server_received_lsl` (HTTP receive) gives **response latency**. Fast responses
suggest high engagement or decisiveness; very slow or absent responses suggest
distraction, avoidance, or low engagement:

```python
def compute_response_latencies(
    responses_jsonl: list[dict],
    sse_events_tsv: pd.DataFrame,
    participant: str,
) -> list[dict]:
    """
    For each VAD/probe response, compute:
      - raw_rt_s: server_received_lsl - server_sent_lsl (conservative lower bound)
      - est_rt_s: finer estimate using browser perf clock
      - skipped: True if participant skipped the prompt
    """
    latencies = []
    for rec in responses_jsonl:
        if rec.get("participant") != participant:
            continue

        # Find matching push event
        push = sse_events_tsv[
            (sse_events_tsv["event_type"] == "push_content") &
            (sse_events_tsv["participant"] == participant) &
            (sse_events_tsv["task"] == rec["task"])
        ]
        if push.empty:
            continue

        sent_lsl = float(push.iloc[-1]["lsl_clock"])
        recv_lsl = float(rec["server_received_lsl"])
        raw_rt = recv_lsl - sent_lsl

        # Finer estimate using browser clock calibration
        est_submit_wall_ms = rec["client_perf_ms"] + rec["clock_offset_ms"]
        est_rt = (est_submit_wall_ms / 1000.0) - sent_lsl

        latencies.append({
            "participant": participant,
            "task": rec["task"],
            "type": rec["type"],
            "raw_rt_s": raw_rt,
            "est_rt_s": est_rt,
            "valence": rec.get("valence"),
            "arousal": rec.get("arousal"),
            "lsl_ts": recv_lsl,
        })
    return latencies
```

### 1.3 Skip Rate as Disengagement Signal

Every skipped VAD prompt is logged as `vad-skip` click in the tablet JS. Track
skip rate per participant per task:

```python
def skip_rate(events_tsv: pd.DataFrame, participant: str, task: str) -> float:
    """
    Proportion of VAD prompts that were skipped rather than submitted.
    High skip rate (> 30%) = likely disengagement or distraction.
    """
    sent = events_tsv[
        (events_tsv["event_type"] == "push_vad") &
        (events_tsv["task"] == task)
    ]
    # Skips are logged as response type="skip" in responses JSONL
    # or as 'vad_skip' event in event log
    skipped_events = events_tsv[
        (events_tsv["event_type"].str.contains("skip", na=False)) &
        (events_tsv["participant"] == participant) &
        (events_tsv["task"] == task)
    ]
    n_sent = max(len(sent), 1)
    return len(skipped_events) / n_sent
```

### 1.4 Tablet Dwell Time per Form Field

T1 candidate selection, T2 settlement, T3 idea generation, and T4 contribution
forms all have specific fields. The form is shown at a known time (push event LSL
timestamp). The response is submitted at a known time (response received LSL
timestamp). The gap is **total form dwell time**, which varies meaningfully:

| Task | Form | Long dwell → | Short dwell → |
|------|------|-------------|---------------|
| T1 | Candidate selection | Uncertainty, deliberation | Quick consensus |
| T2 | Settlement form | Hard negotiation, re-evaluation | Quick agreement |
| T3 | Idea generation | Creative engagement | Low engagement |
| T4 | Contribution | Social deliberation, guilt | Predetermined choice |

```python
def form_dwell_times(
    responses_jsonl: list[dict],
    push_events_tsv: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each form submission, compute dwell time in seconds.
    """
    records = []
    form_types = {"candidate_selection", "settlement", "idea_generation", "contribution"}

    for rec in responses_jsonl:
        rtype = rec.get("type", "")
        if rtype not in form_types and "contribution" not in rtype:
            continue
        task = rec.get("task", "")
        pid = rec.get("participant")
        recv_lsl = float(rec.get("server_received_lsl", 0))

        # Find when this form was pushed
        push = push_events_tsv[
            (push_events_tsv["task"] == task) &
            (push_events_tsv["event_type"] == "push_content")
        ]
        if push.empty:
            continue
        sent_lsl = float(push.iloc[-1]["lsl_clock"])
        dwell_s = recv_lsl - sent_lsl

        records.append({
            "participant": pid,
            "task": task,
            "form_type": rtype,
            "dwell_s": dwell_s,
            "lsl_ts": recv_lsl,
        })
    return pd.DataFrame(records)
```

### 1.5 T4 Contribution as Behavioral Measure

The T4 contribution value (0–10 tokens) is a direct behavioral measure of
**cooperativeness vs. free-riding**. It is the most objectively anchored behavior
in the entire protocol:

```python
THEORETICAL_PREDICTIONS = {
    "purely_selfish":      0,   # contribute nothing
    "purely_cooperative": 10,   # contribute everything
    "nash_equilibrium":    0,   # game-theoretic prediction
    "empirical_average":   5,   # typical lab finding ~50%
}

def contribution_profile(contributions: dict[str, int]) -> dict:
    """
    Given {participant: contribution} for one T4 round, compute:
    - deviation from group mean (free-rider vs over-contributor)
    - cooperation index (0=selfish, 1=fully cooperative)
    - whether participant matched/exceeded/fell below expectations
    """
    values = list(contributions.values())
    group_mean = sum(values) / len(values)
    return {
        p: {
            "contribution": c,
            "deviation_from_mean": c - group_mean,
            "cooperation_index": c / 10.0,
            "role": "over_contributor" if c > group_mean else
                    ("free_rider" if c < group_mean else "matched_group"),
        }
        for p, c in contributions.items()
    }
```

### 1.6 T3 Idea Text Features

Participants type their three ideas into a text form. The idea text is logged in
the response JSONL. This gives free-text behavioral data for each participant:

```python
def idea_text_features(idea_text: str) -> dict:
    """
    Basic NLP features from T3 idea submission text.
    No external libraries needed for these basic features.
    """
    words = idea_text.strip().split()
    sentences = [s.strip() for s in idea_text.split('.') if s.strip()]
    return {
        "word_count": len(words),
        "char_count": len(idea_text),
        "n_sentences": len(sentences),
        "specificity_proxy": len(words),  # more words = more specific
        "has_location": any(w.lower() in {"outdoor", "indoor", "venue", "park", "office"}
                           for w in words),
        "has_activity": any(w.lower() in {"game", "sport", "cook", "paint", "hike", "race"}
                           for w in words),
    }
```

---

## 2. VAD Trajectory Features

### 2.1 Beyond the Raw Rating

Currently VAD values are stored as raw Likert ratings. The trajectory of VAD
across a task carries substantially more behavioral information than any single
rating:

```python
import numpy as np
from scipy import stats

def vad_trajectory_features(
    vad_records: list[dict],   # sorted by lsl_ts, one task, one participant
    task_duration_s: float,
) -> dict:
    """
    Extract trajectory-level features from a sequence of VAD ratings.
    Requires >= 2 measurements.
    """
    if len(vad_records) < 2:
        return {}

    ts   = np.array([r["lsl_ts"] for r in vad_records])
    val  = np.array([r["valence"] for r in vad_records], dtype=float)
    aro  = np.array([r["arousal"] for r in vad_records], dtype=float)
    dom  = np.array([r.get("dominance", np.nan) for r in vad_records], dtype=float)

    def safe_slope(x, y):
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
        return float(slope)

    return {
        # Trend across task (positive = improving, negative = declining)
        "valence_slope":     safe_slope(ts, val),
        "arousal_slope":     safe_slope(ts, aro),
        "dominance_slope":   safe_slope(ts, dom),

        # Variability — high = emotionally unstable during task
        "valence_std":       float(np.nanstd(val)),
        "arousal_std":       float(np.nanstd(aro)),

        # Peak and trough
        "valence_min":       float(np.nanmin(val)),
        "valence_max":       float(np.nanmax(val)),
        "arousal_peak":      float(np.nanmax(aro)),

        # Recovery: was end-valence higher than mid-valence?
        "valence_recovery":  float(val[-1] - val[len(val)//2]) if len(val) > 2 else np.nan,

        # Absolute change from start to end
        "valence_delta":     float(val[-1] - val[0]),
        "arousal_delta":     float(aro[-1] - aro[0]),

        # Time-weighted average
        "valence_twmean":    float(np.trapz(val, ts) / (ts[-1] - ts[0])) if ts[-1] > ts[0] else float(np.nanmean(val)),
    }
```

### 2.2 Emotional Convergence Across Participants

If four participants in the same group show VAD ratings converging over time,
this indicates **emotional alignment** — a group-level phenomenon relevant to
cohesion, trust, and decision quality:

```python
def group_emotional_convergence(
    vad_by_participant: dict[str, list[dict]],  # {pid: vad_records}
    dimension: str = "valence",
) -> list[dict]:
    """
    At each shared time point, compute the standard deviation across
    participants on the given VAD dimension. Decreasing std = convergence.
    Returns list of {lsl_ts, std, mean, n_valid} per measurement round.
    """
    # Align by measurement round (1st probe, 2nd probe, 3rd probe per task)
    max_rounds = max(len(v) for v in vad_by_participant.values())
    convergence = []
    for round_idx in range(max_rounds):
        values = []
        ts_vals = []
        for pid, records in vad_by_participant.items():
            if round_idx < len(records):
                v = records[round_idx].get(dimension)
                if v is not None:
                    values.append(float(v))
                    ts_vals.append(records[round_idx]["lsl_ts"])
        if len(values) >= 2:
            convergence.append({
                "round": round_idx,
                "lsl_ts": float(np.mean(ts_vals)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_valid": len(values),
                "dimension": dimension,
            })
    return convergence
```

### 2.3 VAD-Behavior Coupling

Correlate VAD ratings with concurrent behavioral signals from the 3D pipeline:

```python
# Cross-modal coupling table (features to correlate with VAD at each measurement)
VAD_BEHAVIOR_COUPLINGS = {
    # VAD dimension    : behavioral signal (from multimodal_feature_frame)
    "valence":  [
        "skeleton_conf",           # high valence → more upright posture?
        "gaze_roi_fused_desk",     # low valence → more desk gaze (task disengagement)?
        "speaking",                # high valence → more verbal participation?
        "head_nod_rate",           # high valence → more head nods?
    ],
    "arousal": [
        "rw_mean_velocity",        # high arousal → more hand movement?
        "imu_motion_score",        # high arousal → more head movement?
        "mutual_gaze_rate",        # high arousal → more mutual gaze?
        "audio_energy_db",         # high arousal → louder speech?
    ],
    "dominance": [
        "speaking_proportion",     # high dominance → more speaking time?
        "turn_initiations",        # high dominance → more turn initiation?
        "pointing_gesture_rate",   # high dominance → more pointing?
    ],
}
```

---

## 3. Post-Block Questionnaire Signals

### 3.1 Dominance Perception — Sociometric Data

The post-block questionnaire asks each participant to rate every other participant's
dominance on a 1–7 scale (`dominance_p1` through `dominance_p4`). This creates a
**4×4 peer-perception sociometric matrix** per task — one of the most valuable
social behavior measures in the dataset:

```python
def build_dominance_matrix(
    postblock_responses: list[dict],  # all postblock responses for one task
    participant_ids: list[str] = ["P1", "P2", "P3", "P4"],
) -> np.ndarray:
    """
    Build 4×4 matrix where M[i,j] = mean rating of Pj's dominance by Pi.
    Diagonal = self-rating.
    Returns (4, 4) float array. NaN = missing.
    """
    n = len(participant_ids)
    matrix = np.full((n, n), np.nan)

    for rec in postblock_responses:
        rater_pid = str(rec.get("participant", ""))
        rater_idx = next((i for i, p in enumerate(participant_ids)
                         if str(p[-1]) == str(rater_pid)), None)
        if rater_idx is None:
            continue
        for target_idx, target_pid in enumerate(participant_ids):
            key = f"dominance_{target_pid.lower()}"  # e.g. "dominance_p1"
            val = rec.get("responses", {}).get(key) or rec.get(key)
            if val is not None:
                matrix[rater_idx, target_idx] = float(val)

    return matrix


def dominance_indices(dominance_matrix: np.ndarray) -> dict:
    """
    Compute aggregate dominance indices from the sociometric matrix.
    """
    # Perceived dominance: mean rating received from others (column mean, excl. diagonal)
    n = dominance_matrix.shape[0]
    perceived = []
    for j in range(n):
        others = [dominance_matrix[i, j] for i in range(n) if i != j
                  and not np.isnan(dominance_matrix[i, j])]
        perceived.append(np.mean(others) if others else np.nan)

    # Self-perceived dominance: diagonal
    self_rated = [dominance_matrix[i, i] for i in range(n)]

    # Dominance overestimation: self - mean(others)
    overestimation = [s - p if not np.isnan(s) and not np.isnan(p) else np.nan
                      for s, p in zip(self_rated, perceived)]

    return {
        "perceived_dominance":   perceived,
        "self_dominance":        self_rated,
        "dominance_overestimation": overestimation,
        "group_dominance_std":   float(np.nanstd(perceived)),  # inequality in group
        "most_dominant_idx":     int(np.nanargmax(perceived)) if any(~np.isnan(perceived)) else None,
    }
```

### 3.2 Familiarity Network (T1 only)

T1 asks each participant how well they knew each other before the session
(`familiarity_p1` through `familiarity_p4`). This creates a **pre-existing
social network** that can explain variance in gaze, turn-taking, and dominance:

```python
def build_familiarity_graph(
    t1_postblock: list[dict],
    participant_ids: list[str] = ["P1", "P2", "P3", "P4"],
) -> dict:
    """
    Returns adjacency dict: {(Pi, Pj): mean_familiarity_rating}
    Asymmetric: P1 rating P2 ≠ P2 rating P1 in general.
    """
    graph = {}
    for rec in t1_postblock:
        rater = f"P{rec.get('participant')}"
        for target in participant_ids:
            if target == rater:
                continue
            key_name = f"familiarity_{target.lower()}"
            val = rec.get("responses", {}).get(key_name) or rec.get(key_name)
            if val is not None:
                graph[(rater, target)] = float(val)
    return graph
```

### 3.3 Trust Network (T2 and T4)

T2 and T4 ask directional trust ratings by seating position ("participant next to
you", "in front of you", "at an angle"). These can be converted to a directed
trust graph using the known seating layout:

```python
# Known seating: P1=back-right, P2=front-right, P3=front-left, P4=back-left
# For each seat, "next to" and "in front of" and "at angle" correspond to:
SEATING_RELATIONS = {
    "P1": {"next": "P4", "front": "P2", "angle": "P3"},
    "P2": {"next": "P3", "front": "P1", "angle": "P4"},
    "P3": {"next": "P2", "front": "P4", "angle": "P1"},
    "P4": {"next": "P1", "front": "P3", "angle": "P2"},
}

def build_trust_network(
    postblock_responses: list[dict],
    task: str,  # "T2" or "T4"
) -> dict[tuple[str, str], float]:
    """
    Returns directed trust graph {(rater, target): trust_rating}.
    """
    trust_graph = {}
    for rec in postblock_responses:
        if rec.get("task") != task:
            continue
        rater = f"P{rec.get('participant')}"
        relations = SEATING_RELATIONS.get(rater, {})
        responses = rec.get("responses", {}) or rec

        for relation, target in relations.items():
            key = f"trust_{relation}"   # trust_next, trust_front, trust_angle
            val = responses.get(key)
            if val is not None:
                trust_graph[(rater, target)] = float(val)

    return trust_graph
```

### 3.4 Information-Sharing Quality (T1 manipcheck)

`manipcheck_t1` asks: "Information that was unique to individual group members
influenced the final decision." This is the **task manipulation check** — the
theoretically correct hidden-profile outcome (Candidate C) requires full
information pooling. High agreement on this item validates the hidden-profile
manipulation; low agreement suggests information pooling failed.

Combine with the actual decision (from `t1_candidate` decision log) to classify
sessions by hidden-profile success:

```python
def hidden_profile_outcome(
    decision: str,          # "Candidate A", "B", or "C"
    manipcheck_mean: float, # mean across 4 participants (1-7 scale)
) -> str:
    """
    Classify T1 session outcome.
    """
    correct = decision == "Candidate C"
    pooled  = manipcheck_mean >= 5.0   # "agree" that unique info was shared
    if correct and pooled:
        return "full_pooling_success"
    if correct and not pooled:
        return "correct_by_luck"
    if not correct and pooled:
        return "pooled_but_wrong"
    return "failed_to_pool"
```

---

## 4. Task Decision Logs

### 4.1 Decision Content Analysis

The moderator logs the final decision for each task in LSL and in the response
JSONL. These decisions are currently used only for record-keeping. They contain
behavioral signal:

```python
# T1 candidate decision → hidden profile success (§3.4)
# T2 topic + format → integrative vs distributive negotiation outcome
# T3 winning idea + author → whose ideas are selected (influence measure)
# T4 contribution amounts → cooperation level

T2_INTEGRATIVE_COMBINATIONS = {
    # These combinations maximise total utility across all participants
    # (topic-priority P2+P4 get their topic; format-priority P1+P3 get their format)
    ("How to use AI for productivity in daily tasks",
     "Interactive Informal Workshop"): "integrative",
    ("How to use AI for productivity in daily tasks",
     "Cross-Department Working Group"): "integrative",
    ("How to work better across teams",
     "Interactive Informal Workshop"): "integrative",
    ("How to work better across teams",
     "Cross-Department Working Group"): "integrative",
    # Distributive = one side wins on both issues
}

def t2_outcome_quality(topic: str, format_: str) -> str:
    return T2_INTEGRATIVE_COMBINATIONS.get((topic, format_), "distributive_or_compromise")
```

### 4.2 T3 Idea Authorship — Influence Tracking

The `idea_author` field in T3 group selection tells you whose idea won. Combine
with speaking time (from VAD), gaze received (from 3D pipeline), and dominance
ratings to build an **influence model**:

```python
def influence_index(
    participant: str,
    t3_winner: str,           # e.g. "P2 — Idea 1"
    speaking_proportion: float,
    perceived_dominance: float,
    gaze_received_proportion: float,
) -> dict:
    """
    Composite influence score combining objective (idea won) and
    perceptual (dominance, gaze received) measures.
    """
    idea_winner = t3_winner.startswith(participant)
    return {
        "participant":              participant,
        "idea_selected":            idea_winner,
        "speaking_proportion":      speaking_proportion,
        "perceived_dominance":      perceived_dominance,
        "gaze_received_proportion": gaze_received_proportion,
        "composite_influence":      (
            0.4 * float(idea_winner) +
            0.2 * speaking_proportion +
            0.2 * (perceived_dominance / 7.0) +
            0.2 * gaze_received_proportion
        ),
    }
```

### 4.3 Decision Timing

The moderator logs decisions to LSL at specific timestamps. The time from
task start to final decision (`decision_lag_s`) is a group-level measure of
**decision speed** that varies with task difficulty, group consensus, and conflict:

```python
def decision_timing(
    task_start_lsl: float,
    decision_lsl: float,
    task_nominal_duration_s: dict,  # {"T1": 600, "T2": 480, "T3": 600, "T4": 300}
    task: str,
) -> dict:
    lag_s = decision_lsl - task_start_lsl
    nominal = task_nominal_duration_s.get(task, 600)
    return {
        "task": task,
        "decision_lag_s": lag_s,
        "decision_lag_norm": lag_s / nominal,  # 0=immediate, 1=at deadline
        "decided_early": lag_s < 0.6 * nominal,
        "decided_late":  lag_s > 0.9 * nominal,
    }
```

---

## 5. Big Screen Attention Events

### 5.1 Tobii Calibration Aiming Point as Gaze Baseline

The moderator pushes a `tobii_calibration` event before each task. All participants
see the same fixation cross at the center of all screens. The LSL timestamp is
emitted to **every stream simultaneously** (`tobii_calibration` in moderator,
participant_1–4, and bigscreen streams).

This gives a **synchronized multi-person fixation event** that can validate gaze
calibration per session and per task. If any participant's gaze during this event
is more than 2° from the screen center, re-calibration is recommended:

```python
def validate_tobii_calibration_event(
    gaze_records_by_participant: dict[str, list[dict]],
    calibration_lsl_ts: float,
    window_s: float = 3.0,
    screen_center_world: np.ndarray = np.array([0.0, 0.8, 1.0]),  # big screen center
    max_angle_deg: float = 2.0,
) -> dict[str, dict]:
    """
    For the 3-second window around each tobii_calibration event,
    compute mean gaze deviation from screen center per participant.
    """
    results = {}
    for pid, gaze in gaze_records_by_participant.items():
        window = [g for g in gaze
                  if abs(g["timestamp_s"] - calibration_lsl_ts) < window_s
                  and g.get("gaze_validity") == "valid"]
        if not window:
            results[pid] = {"n_samples": 0, "mean_angle_deg": np.nan, "pass": False}
            continue

        angles = []
        for g in window:
            pt = np.array(g.get("gaze_point_world", [np.nan]*3))
            if np.any(np.isnan(pt)):
                continue
            vec_to_pt = pt - np.array(g.get("gaze_origin_world", pt))
            vec_to_screen = screen_center_world - np.array(g.get("gaze_origin_world", pt))
            cos_a = np.dot(vec_to_pt, vec_to_screen) / (
                np.linalg.norm(vec_to_pt) * np.linalg.norm(vec_to_screen) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

        mean_angle = float(np.mean(angles)) if angles else np.nan
        results[pid] = {
            "n_samples": len(angles),
            "mean_angle_deg": mean_angle,
            "pass": mean_angle < max_angle_deg if not np.isnan(mean_angle) else False,
        }
    return results
```

### 5.2 Phase Transition Orienting Responses

Every time the moderator pushes new content to the big screen, all participants
are expected to orient toward it. The gaze data in the 2-second window after each
`push_content` event reveals:

- Who oriented toward the screen first (quickest to notice = most attentive)
- Who delayed or didn't orient (distracted or disengaged)
- Group synchrony of orienting (all look at once vs. staggered)

```python
def orienting_response_features(
    push_events: pd.DataFrame,
    gaze_frames: list[dict],    # unified per-frame gaze ROI records
    fps: float = 30.0,
    window_s: float = 2.0,
) -> list[dict]:
    """
    For each bigscreen content push, compute orienting latency per participant.
    """
    results = []
    screen_pushes = push_events[push_events["event_type"] == "push_content"]

    for _, push in screen_pushes.iterrows():
        push_ts = float(push["lsl_clock"])
        end_ts  = push_ts + window_s

        window_frames = [f for f in gaze_frames
                         if push_ts <= f["timestamp_s"] <= end_ts]

        for pid in ["P1", "P2", "P3", "P4"]:
            oriented_ts = None
            for frame in window_frames:
                if frame["participants"][pid]["gaze_roi_fused"] == "big_screen":
                    oriented_ts = frame["timestamp_s"]
                    break

            results.append({
                "push_ts":         push_ts,
                "task":            push.get("task"),
                "phase":           push.get("phase"),
                "participant":     pid,
                "oriented":        oriented_ts is not None,
                "orienting_lag_s": (oriented_ts - push_ts) if oriented_ts else np.nan,
            })

    return results
```

---

## 6. Turn-Taking and Speech Structure

### 6.1 Interruption Detection

From per-participant VAD (when a participant speaks, VAD on adjacent microphones
rises slightly due to crosstalk, but mainly the speaker's own mic dominates).
When two mics show high energy simultaneously (above crosstalk threshold), an
**interruption or overlap** is occurring:

```python
def detect_interruptions(
    vad_by_participant: dict[str, np.ndarray],   # {pid: binary VAD at fps}
    min_overlap_frames: int = 3,    # ~100ms minimum for a real overlap
    fps: float = 30.0,
) -> list[dict]:
    """
    Returns list of overlap events: {onset_s, duration_s, participants}.
    """
    events = []
    pids = list(vad_by_participant.keys())
    n_frames = max(len(v) for v in vad_by_participant.values())

    # Find frames where >= 2 participants are speaking
    overlap_mask = np.zeros(n_frames, dtype=int)
    for pid, vad in vad_by_participant.items():
        overlap_mask[:len(vad)] += vad

    in_overlap = False
    onset_f = 0
    active_pids = []

    for fi in range(n_frames):
        n_speakers = int(overlap_mask[fi])
        if n_speakers >= 2 and not in_overlap:
            in_overlap = True
            onset_f = fi
            active_pids = [p for p, v in vad_by_participant.items()
                          if fi < len(v) and v[fi]]
        elif n_speakers < 2 and in_overlap:
            in_overlap = False
            duration_f = fi - onset_f
            if duration_f >= min_overlap_frames:
                events.append({
                    "onset_s":     onset_f / fps,
                    "duration_s":  duration_f / fps,
                    "participants": active_pids,
                    "n_overlapping": len(active_pids),
                })

    return events
```

### 6.2 Floor-Holding Duration

How long each participant holds the speaking floor per turn and in total per task.
Long floor-holding suggests dominance; very short turns suggest passive participation:

```python
def floor_holding_stats(
    vad_by_participant: dict[str, np.ndarray],
    fps: float = 30.0,
    min_turn_s: float = 0.5,
) -> dict[str, dict]:
    """
    Per-participant turn statistics for one task.
    """
    stats = {}
    min_frames = int(min_turn_s * fps)

    for pid, vad in vad_by_participant.items():
        turns = []
        in_turn = False
        onset_f = 0
        for fi, active in enumerate(vad):
            if active and not in_turn:
                in_turn = True
                onset_f = fi
            elif not active and in_turn:
                in_turn = False
                duration_f = fi - onset_f
                if duration_f >= min_frames:
                    turns.append(duration_f / fps)

        stats[pid] = {
            "n_turns":          len(turns),
            "total_speech_s":   sum(turns),
            "mean_turn_s":      float(np.mean(turns)) if turns else 0.0,
            "max_turn_s":       float(np.max(turns)) if turns else 0.0,
            "speech_proportion": sum(turns) / (len(vad) / fps) if len(vad) > 0 else 0.0,
        }

    return stats
```

### 6.3 Back-Channel Detection

Short vocalizations (< 500ms: "mm-hmm", "yeah", "ok") during another person's
speech floor indicate active listening. These can be detected from the per-mic VAD
combined with the overlap detector:

```python
def detect_backchannels(
    vad_by_participant: dict[str, np.ndarray],
    fps: float = 30.0,
    max_backchannel_s: float = 0.5,
    min_floor_holder_s: float = 1.5,
) -> list[dict]:
    """
    A backchannel is a short vocalization by participant B while participant A
    holds the floor (A speaking >= 1.5s, B speaks < 500ms during this).
    """
    max_bc_frames = int(max_backchannel_s * fps)
    min_floor_frames = int(min_floor_holder_s * fps)
    backchannels = []
    pids = list(vad_by_participant.keys())

    for floor_holder in pids:
        fh_vad = vad_by_participant[floor_holder]
        for listener in pids:
            if listener == floor_holder:
                continue
            l_vad = vad_by_participant[listener]
            n = min(len(fh_vad), len(l_vad))

            # Find stretches where floor_holder is speaking
            fh_active = fh_vad[:n]
            l_active = l_vad[:n]

            # Identify floor-holder turns >= min_floor_frames
            in_fh_turn = False
            turn_start = 0
            for fi in range(n):
                if fh_active[fi] and not in_fh_turn:
                    in_fh_turn = True
                    turn_start = fi
                elif not fh_active[fi] and in_fh_turn:
                    in_fh_turn = False
                    turn_len = fi - turn_start
                    if turn_len >= min_floor_frames:
                        # Scan listener's activity during this turn
                        l_window = l_active[turn_start:fi]
                        bc_on = False
                        bc_start = 0
                        for bi, la in enumerate(l_window):
                            if la and not bc_on:
                                bc_on = True
                                bc_start = bi
                            elif not la and bc_on:
                                bc_on = False
                                bc_len = bi - bc_start
                                if 1 <= bc_len <= max_bc_frames:
                                    backchannels.append({
                                        "floor_holder": floor_holder,
                                        "backchannel_by": listener,
                                        "onset_s": (turn_start + bc_start) / fps,
                                        "duration_s": bc_len / fps,
                                    })
    return backchannels
```

---

## 7. Probe Response Latency

### 7.1 Already Computed — Now Use It

Section §1.2 shows how to compute response latency. The **distribution of latencies**
across the session and their relationship to the content of the response are the
behavioral signal:

| Pattern | Behavioral interpretation |
|---------|--------------------------|
| Long latency + extreme rating (1 or 9) | Strong emotion, took time to process |
| Long latency + mid rating (4-5) | Uncertain, avoided commitment |
| Short latency + extreme rating | Salient, pre-formed opinion |
| Short latency + mid rating | Habitual "neutral" responding |
| Very long latency (> 30s) or no response | Distracted / task-absorbed |

### 7.2 Latency as a Continuous Behavioral Regressor

Use response latency as a predictor in models of behavioral outcomes:

```python
# Example regression features for predicting T1 decision (hidden profile)
LATENCY_FEATURES = {
    "mean_vad_rt_s":         "Mean VAD response time across task",
    "max_vad_rt_s":          "Slowest response (most deliberative moment)",
    "rt_cv":                 "Coefficient of variation (stable vs. variable engagement)",
    "rt_trend":              "Slope of RT over task (speeding up = habituation?)",
    "n_missing_responses":   "VAD prompts with no response (absence = disengagement)",
}
```

---

## 8. Device Connection and Engagement Proxy

### 8.1 Tablet Reconnection Events

Every `device_connect` and `device_disconnect` event is logged to LSL via
`event_logger.py`. Frequent reconnections indicate a tablet that went to sleep
(participant wasn't interacting with it) or Wi-Fi issues:

```python
def tablet_engagement_proxy(
    events_tsv: pd.DataFrame,
    participant: str,
    task_start_lsl: float,
    task_end_lsl: float,
) -> dict:
    """
    Use device connection events as a proxy for tablet engagement.
    """
    task_events = events_tsv[
        (events_tsv["participant"] == participant) &
        (events_tsv["lsl_clock"] >= task_start_lsl) &
        (events_tsv["lsl_clock"] <= task_end_lsl)
    ]

    reconnects = task_events[task_events["event_type"] == "device_connect"]
    disconnects = task_events[task_events["event_type"] == "device_disconnect"]

    return {
        "participant":   participant,
        "n_reconnects":  len(reconnects),
        "n_disconnects": len(disconnects),
        # Long disconnects = tablet went to sleep = not looking at tablet
        "disconnected_flag": len(disconnects) > 0,
    }
```

### 8.2 Wake Lock Success

The tablet JS requests a Screen Wake Lock (`navigator.wakeLock.request('screen')`)
to prevent sleep. If this fails (older Android, battery saver mode), the tablet
may sleep mid-task. Wake lock failures are logged to the browser console but not
currently to the server. Consider adding a server-side log endpoint:

```javascript
// Add to tablet JS after wakeLock acquisition attempt:
if (!wakeLock) {
    fetch('/log_event', {
        method: 'POST',
        body: JSON.stringify({
            event: 'wakelock_failed',
            participant: TABLET_NUM,
            ts: performance.now()
        })
    });
}
```

---

## 9. Cross-Participant Behavioral Synchrony

### 9.1 VAD Synchrony — Emotional Contagion

If two participants' VAD trajectories are positively correlated across a task,
this indicates **emotional contagion or co-regulation**. The lag-correlation
between their VAD series reveals who leads and who follows:

```python
from scipy.signal import correlate

def vad_synchrony(
    vad_a: np.ndarray,   # valence time series for participant A (interpolated to common grid)
    vad_b: np.ndarray,   # valence time series for participant B
    max_lag_samples: int = 3,   # at VAD probe rate ~1/60s, 3 samples = ~3 minutes
) -> dict:
    """
    Compute cross-correlation between two VAD series at multiple lags.
    Positive lag: A leads B. Negative lag: B leads A.
    """
    corr = correlate(vad_a - np.nanmean(vad_a),
                     vad_b - np.nanmean(vad_b), mode='full')
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    center = len(corr) // 2
    corr_at_lags = corr[center - max_lag_samples: center + max_lag_samples + 1]

    peak_lag = int(lags[np.argmax(np.abs(corr_at_lags))])
    peak_corr = float(corr_at_lags[np.argmax(np.abs(corr_at_lags))])

    return {
        "zero_lag_correlation": float(corr[center]),
        "peak_lag_samples":     peak_lag,
        "peak_correlation":     peak_corr,
        "synchrony_index":      float(corr[center]),  # 0-lag is cleanest for small N
        "a_leads_b":            peak_lag > 0,
    }
```

### 9.2 Motion Synchrony — Entrainment

Head movement (from Tobii IMU) and body movement (from 3D skeleton wrist velocity)
can be compared across participants to detect **behavioral entrainment** — people
unconsciously synchronizing their movement patterns during interaction:

```python
def motion_entrainment(
    wrist_velocities: dict[str, np.ndarray],   # {pid: velocity at fps}
    pairs: list[tuple[str, str]] | None = None,
    fps: float = 30.0,
    window_s: float = 10.0,
) -> list[dict]:
    """
    Compute windowed cross-correlation of movement velocity between all pairs.
    High correlation = behavioral entrainment.
    """
    if pairs is None:
        pids = list(wrist_velocities.keys())
        pairs = [(pids[i], pids[j]) for i in range(len(pids))
                 for j in range(i+1, len(pids))]

    window_frames = int(window_s * fps)
    results = []

    for pid_a, pid_b in pairs:
        va = wrist_velocities.get(pid_a, np.array([]))
        vb = wrist_velocities.get(pid_b, np.array([]))
        n = min(len(va), len(vb))
        if n < window_frames:
            continue

        correlations = []
        for start in range(0, n - window_frames, window_frames // 2):
            wa = va[start: start + window_frames]
            wb = vb[start: start + window_frames]
            if np.std(wa) > 0 and np.std(wb) > 0:
                r = float(np.corrcoef(wa, wb)[0, 1])
                correlations.append({
                    "onset_s":     start / fps,
                    "correlation": r,
                    "pair":        (pid_a, pid_b),
                })
        results.extend(correlations)

    return results
```

---

## 10. Task-Phase Behavioral Profiles

### 10.1 Phase-Aligned Feature Extraction

Each task has defined phases (from the moderator `push_task_content` LSL events).
Compute behavioral features **per phase** rather than per task, enabling
comparison of behavior during structured vs. unstructured phases:

```python
PHASE_BEHAVIORAL_PROFILE_FEATURES = {
    # Phase-level features (computed over each task phase window)
    "speaking_proportion":         "fraction of frames with speech",
    "n_interruptions":             "count of overlap events",
    "n_backchannels":              "count of backchannels received",
    "mean_wrist_velocity":         "mean wrist velocity (m/s)",
    "gaze_on_screen_proportion":   "fraction of gaze on big_screen ROI",
    "gaze_on_desk_proportion":     "fraction of gaze on desk ROI",
    "gaze_on_tablet_proportion":   "fraction of gaze on own tablet ROI",
    "mutual_gaze_proportion":      "fraction of frames in mutual gaze",
    "joint_attention_proportion":  "fraction of frames in joint attention",
    "mean_vad_valence":            "mean valence during phase",
    "mean_vad_arousal":            "mean arousal during phase",
    "head_nod_rate":               "nods per minute",
    "head_shake_rate":             "shakes per minute",
    "lean_forward_proportion":     "proportion of frames leaning forward",
}
```

### 10.2 Structured vs. Unstructured Phase Comparison

T3 has a unique structure with a **silent phase** (idea generation: no speech
allowed) and a **verbal phase** (round-robin + discussion). Comparing behavioral
features across these phases provides a within-session baseline:

```python
T3_PHASE_CONTRASTS = {
    "idea_generation vs. discussion": {
        "prediction": "speech_proportion dramatically increases",
        "behavior": ["speaking_proportion", "n_interruptions", "wrist_velocity"],
        "expected_direction": [+1, +1, +1],  # all increase in discussion
    },
    "round_robin vs. free_discussion": {
        "prediction": "turn-taking becomes more competitive",
        "behavior": ["n_interruptions", "backchannel_rate", "mutual_gaze_proportion"],
        "expected_direction": [+1, +1, -1],  # mutual gaze decreases as ideas compete
    },
}
```

---

## 11. Integration with 3D Pipeline

### 11.1 Cross-Modal Validation Table

Every behavioral signal in this document can be validated against or correlated
with signals from the 3D gaze/skeleton pipeline:

| Behavioral signal (this doc) | 3D pipeline signal (gaze/ROI plans) | Validation hypothesis |
|------------------------------|-------------------------------------|----------------------|
| Tablet dwell time (§1.4) | Gaze-on-tablet proportion (ROI plan §4) | Long dwell → high gaze-on-tablet |
| VAD arousal trajectory (§2.1) | IMU motion score (fusion plan §3.4) | High arousal → more IMU motion |
| Dominance rating (§3.1) | Speaking proportion (§6.2) | High perceived dominance → more speech |
| T3 idea selected (§4.2) | Gaze received proportion (gaze plan §7) | Idea winner → more gaze from others |
| VAD response latency (§1.2) | Tablet ROI confirmed by logs (fusion §2.2) | Fast RT = fewer occlusion gaps |
| Interruptions (§6.1) | Gesture suppression (fusion §5.3) | Overlap frames → suppress beat gesture |
| Motion synchrony (§9.2) | Head nod from IMU (fusion §4.2) | Synchrony bursts co-occur with agreement |

### 11.2 Behavioral Ground Truth for ML Training

Combine tablet logs with 3D signals to build labeled datasets:

```python
# Weak supervision labels from tablet data:
WEAK_SUPERVISION_LABELS = {
    # Label name           : source                 : target signal to predict
    "gaze_on_tablet":      "touch_start/end log",   "3D gaze ROI classification",
    "high_engagement":     "VAD arousal > 6",       "head movement / wrist velocity",
    "speaker_identity":    "per-mic VAD",            "3D head orientation",
    "dominant_participant":"postblock ratings",      "speaking proportion / gaze received",
    "cooperative":         "T4 contribution > 5",   "lean-forward / eye contact gestures",
    "head_nod":            "IMU + skeleton agree",  "audio prosody (pitch rise before nod)",
}
```

---

## 12. Priority Order

| Priority | Section | Signal | Effort | Value |
|----------|---------|--------|--------|-------|
| 🔴 Critical | **§3.1** | Dominance sociometric matrix from postblock | Low | 4×4 peer perception network per task — unique social measure |
| 🔴 Critical | **§1.5** | T4 contribution as behavioral cooperativeness | Low | Cleanest behavioral measure in the dataset |
| 🔴 Critical | **§5.1** | Tobii calibration event as gaze validation baseline | Low | Per-session, per-task accuracy check — free from existing LSL events |
| 🟠 High | **§2.1** | VAD trajectory features (slope, delta, std) | Low | Converts 3 ratings into 10+ trajectory-level features |
| 🟠 High | **§1.2** | Response latency per participant per task | Low | Engagement proxy, already computable from existing timestamps |
| 🟠 High | **§3.2** | Familiarity network from T1 postblock | Low | Controls for pre-existing relationships in all subsequent analysis |
| 🟠 High | **§3.3** | Trust network from T2 and T4 postblock | Low | Directed trust graph with known seating topology |
| 🟠 High | **§4.2** | T3 idea authorship → influence index | Low | Connects behavioral outcome to multimodal signals |
| 🟡 Medium | **§6.1** | Interruption detection from VAD overlap | Medium | Group dynamics, turn competition |
| 🟡 Medium | **§6.2** | Floor-holding and speech proportion | Medium | Speaking dominance index |
| 🟡 Medium | **§2.2** | Group emotional convergence across VAD ratings | Low | Group-level affect measure |
| 🟡 Medium | **§5.2** | Orienting response latency to screen pushes | Medium | Attention tracking from existing gaze + LSL events |
| 🟡 Medium | **§4.1** | T2 outcome quality (integrative vs distributive) | Low | Labels negotiation success for behavioral analysis |
| 🟡 Medium | **§9.1** | VAD synchrony between participant pairs | Low | Emotional contagion / co-regulation measure |
| 🟢 Lower | **§6.3** | Back-channel detection | High | Subtle social signal, requires good VAD |
| 🟢 Lower | **§9.2** | Motion entrainment from IMU/skeleton | High | Requires reliable 3D signals first |
| 🟢 Lower | **§10.1** | Phase-aligned behavioral profiles | Medium | Structured comparison but needs all other signals first |
| 🟢 Lower | **§7.2** | Latency as regression feature | Low | Useful when building ML models, not standalone |
| 🟢 Lower | **§1.6** | T3 idea text NLP features | Medium | Adds creativity/specificity dimension |

---

*Document generated 2026-04-19. Companion to:*
- *`3d_gaze_improvement_plan.md`*
- *`3d_roi_gesture_skeleton_plan.md`*
- *`multimodal_fusion_3d_enhancement_plan.md`*

*All signals here use data already collected and stored in the BIDS/JSONL/TSV outputs.
No additional hardware or re-collection is required.*
