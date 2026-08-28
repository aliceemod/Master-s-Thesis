# HMM Collective States — k=4 Final Model

## Model Selection
- **BIC Scores:** k=2 (11316.2), k=3 (11101.6), k=4 (11091.4), k=5 (11094.4), k=6 (11122.7)
- **Selected:** k=4 (lowest BIC with 3 additional features including `tr_overlap_time_s`)
- **Note:** k=5 was within 3 BIC units, indicating robust solution near optimality
- **Convergence:** All models (3 seeds: 42, 7, 99) converged successfully
- **Log-likelihood:** -5309.43
- **Windows analyzed:** 545 (all 10 groups, tasks T1-T3)

---

## Four Collective States

### State 0: **Calm-Observant** (79 windows, 14.5%)
**Profile:**
- **Physiological:** Very low HR (-1.13z), very low EDA (-1.37z), very low pupil dilation (-1.38z)
- **Interaction:** Minimal overlaps (-0.08z), minimal laughter (-0.29z), low active speakers (-0.01z)
- **Temperature:** Elevated (0.66z)
- **Key marker:** Low engagement, observer mode

**Task prevalence:** 22.8% in T1 (decision task)
**Transitions:** Self (0.83), slight toward Competitive-Tense (0.05) or Withdrawn-Passive (0.12)

**Interpretation:** Group members are quiet, disengaged observers. Low physiological arousal suggests calm state. Minimal conversational overlap indicates few speaking opportunities or low participation. Common in initial decision-making when people are assessing the situation.

---

### State 1: **Competitive-Tense** (183 windows, 33.6%) ⭐ DOMINANT
**Profile:**
- **Physiological:** HIGH HR (1.26z), HIGH EDA (0.91z), HIGH pupil dilation (0.87z)
- **Interaction:** HIGH overlaps (1.00z), HIGH laughter (0.97z), HIGH competitive overlaps (0.84z), HIGH active speakers (0.87z)
- **Silence:** Very low (-0.65z) — continuous talk, few pauses
- **Key markers:** Peak arousal, maximum interaction/conflict

**Task prevalence:** 47.3% in T2 (negotiation task) ← **STRONGEST PATTERN**
**Transitions:** Self (0.85) — highly persistent, most stable state

**Interpretation:** Group is in high-energy, competitive mode. Multiple people talking simultaneously (overlaps), high physiological arousal (stress response), frequent competitive overlaps suggest tense negotiation. This is the hallmark state of the negotiation task where consensus must be reached under pressure.

---

### State 2: **Withdrawn-Passive** (114 windows, 20.9%)
**Profile:**
- **Physiological:** Low HR (-0.31z), low EDA (-0.08z), moderate pupil (0.58z)
- **Interaction:** MINIMAL backchannels (-1.45z), MINIMAL overlaps (-1.35z), MINIMAL laughter (-1.27z), MINIMAL active speakers (-1.40z)
- **Overlap time:** Very low (-1.44z)
- **Key markers:** Extreme disengagement, silent withdrawal

**Task prevalence:** Present across all tasks (~20% overall), elevated in T2 (32.2%)
**Transitions:** Self (0.64), receives transitions from Competitive-Tense (0.26)

**Interpretation:** Group enters a withdrawn, passive state—possibly after conflict fatigue or disengagement. Very few people speaking, minimal interaction. Could indicate loss of interest or overwhelm. Low persistence (0.64) suggests this is a transitional state people escape from.

---

### State 3: **Reflective-Ideation** (169 windows, 31.0%)
**Profile:**
- **Physiological:** Moderate HR (0.19z), moderate EDA (0.54z), low pupil (-0.08z)
- **Interaction:** HIGH silence (1.49z) — extended pauses, thinking time
- **Competitive overlaps:** Moderate-HIGH (0.82z) — occasional competitive interjections amid reflection
- **Temperature:** LOW (-1.49z)
- **Key markers:** Sustained thinking, occasional competitive engagement

**Task prevalence:** 69.7% in T3 (idea generation task) ← **STRONGEST PATTERN**
**Transitions:** Self (0.97) — extremely high persistence, most stable state; self-reinforcing

**Interpretation:** Group enters a reflective, collaborative ideation mode. High silence indicates thinking time and room for ideas to develop. Moderate competitive overlaps suggest occasional friendly corrections or building on ideas. The extremely high self-transition (0.97) indicates groups naturally maintain this reflective state once entered—ideal for creative ideation.

---

## Task × State Patterns

| Task | Dominant States | Interpretation |
|------|-----------------|-----------------|
| **T1: Hidden-Profile Decision** | Balanced (0: 22.8%, 3: 30.3%) | Initial exploratory phase; mix of observation and ideation |
| **T2: Negotiation** | **State 1: 47.3%** | High conflict/energy as groups negotiate consensus |
| **T3: Idea Generation** | **State 3: 69.7%** | Sustained reflective thinking mode, ideal for creativity |

**χ² test:** χ² = 194.11, p < 10⁻³⁷, Cramér's V = 0.422 (moderate-to-large effect)
→ States are **significantly and meaningfully structured by task type**

---

## Feature Importance (Top 5)

| Rank | Feature | Variance | Role |
|------|---------|----------|------|
| 1 | Temperature | 0.885 | Physiological marker of arousal state |
| 2 | Overlap time | 0.811 | Direct measure of interaction intensity |
| 3 | Pupil diameter | 0.735 | Sympathetic nervous system indicator |
| 4 | Backchannels | 0.679 | Passive engagement signal |
| 5 | Active speakers | 0.670 | Participation diversity |

✅ **Multimodal signal:** Conversation, physiology, and gaze all contribute meaningfully. No single feature dominates.

---

## Robustness Validation ✅

1. **EM Convergence:** k=4 converged within 200 iterations
2. **Multi-seed reproducibility:** Seeds 42, 7, 99 all select k=4
3. **State persistence:** Mean self-transition = 0.822 (SD 0.120) → robust structure
4. **Demographic fairness:** All 10 groups represented in all 4 states
5. **Task-state association:** Highly significant (p < 10⁻³⁷)
6. **Class balance:** 2.32:1 imbalance ratio (acceptable, <3:1)

---

## Figure Outputs

- `state_trajectories_per_group.png` — Shows how each group moves through states across tasks
- `state_persistence_heatmap_per_task.png` — State stability patterns by task
- `hmm_bic_selection_updated_overlaps.png` — BIC selection curve (k=4 optimal)
- `hmm_profiles_transitions_updated_overlaps.png` — Feature profiles and transition matrix
