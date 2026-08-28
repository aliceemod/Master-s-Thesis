# Effectiveness Target Investigation Log

Record of the diagnostic work done to find a defensible, predictable outcome
target for the multimodal (HMM latent-state / raw physio-audio-transcript)
prediction models. Covers the full session from initial notebook setup
through to the final recommended target and its latent-vs-raw-feature
validation. All temp scripts referenced below were run via
`.\audio_venv\Scripts\python.exe`, printed their results, and were deleted
immediately after (per repo convention) — none persist in the repo. Results
quoted here are the authoritative record of what was found, cross-checked
against the actual session transcript.

## 1. Starting point

`analysis/effectiveness_prediction_models.ipynb` sets up the modeling
pipeline: HMM state-% features (`S0_pct`–`S3_pct`, from
`icmi_paper/results/hmm_cluster_assignments_k5_updated_overlaps.tsv` +
`hmm_input_features_final.tsv`) and 4 audio features, aggregated to
group×task level (n=27, 10 independent groups, tasks T1-T3). Original
target: `perceived_effectiveness_z`, a composite built from **different**
single-item Likert questions per task (T1: `team_coordination` +
`decision_confidence`; T2: `cooperative` + `manipcheck_t2` + `satisfaction`;
T3: `team_coordination` + `satisfaction` + `idea_quality`).

## 2. Original target (`perceived_effectiveness_z`) — diagnosed as broken

- **Variance decomposition:** 76% of variance is within group-task
  (idiosyncratic to the individual rater); only 23.8% is shared group×task
  variance. Group alone explains ~6%; task alone ~0%. This behaves like an
  individual trait/mood measure, not a property of the shared task context
  — so group-level behavioral/physio signals have a low predictive ceiling
  no matter the model.
- **Univariate correlations (pairwise, uncorrected):** only 2 of 17 features
  hit nominal p<0.05 — `tr_overlap_time_s` (ρ=0.22) and `audio_pitch_mean`
  (ρ=-0.21). Neither survives Bonferroni correction (α≈0.003 for 17 tests).
  Everything else is noise-level (|ρ|<0.2, p>0.1).
- **Distribution:** left-skewed (skew=-0.90) — most ratings cluster high.
- **Cross-task (same-participant) stability:** ~0 (T1 vs T2 ρ=-0.01, T2 vs T3
  ρ=0.05, T1 vs T3 ρ=0.22, none significant) — not even trait-stable. This
  directly contradicted the initial "stable individual trait" hypothesis
  suggested by the variance decomposition above.
- **Correlates strongly with its own components** (expected, not
  informative, since it's built from them): `team_coordination` ρ=0.81,
  `manipcheck_t2` ρ=0.81, `satisfaction` ρ=0.77, `decision_confidence`
  ρ=0.67.
- **Model-type check:** tried multiple model families at the participant
  level — all ≈ -0.01 to -0.09. Confirms the ceiling is the data/target
  relationship, not model choice.
- **Root cause identified:** the composite stitches together *different*
  single-item Likert (1-7) questions per task — each is a single raw answer,
  not an averaged multi-item scale, so there's no way to compute reliability
  (no repeated items to cancel out response noise). This is a
  construct-validity flaw, not a modeling problem.
- **Confirmed target problem, not a feature problem:** tested full 18-feature
  Ridge (R²=-0.175) and PCA(2-5)+Ridge (R²=-0.018 to -0.121) at group×task
  level — all null or negative. Curating fewer features was not the issue;
  `perceived_effectiveness_z` isn't recoverable regardless of method.

## 3. Personality / demographic traits — two rounds, both null

**Privacy issue found and fixed first:** `metadata/participants.tsv`,
`metadata/high_level_group_inventory.csv`, and
`metadata/high_level_session_inventory.csv` were found to contain **real
full participant names in tracked repo files**, mapped directly to `sub-XXX`
and `grp-XX` — a violation of this repo's own privacy rules (real names
must live only in the gitignored `.private/registration_ledger.jsonl`).
Rather than modify those files, a new anonymised table was built:
[analysis/build_participant_traits.py](build_participant_traits.py) →
[analysis/results/participant_traits.tsv](results/participant_traits.tsv)
(40 rows, `group_id` + `participant` P1-P4 only, zero name columns; P1-P4
assignment inferred from name order in `high_level_session_inventory.csv`,
assumed to equal tablet order — 39/40 exact name matches, 1 fuzzy; no
ground-truth registration ledger available in this workspace to confirm
the ordering assumption). Traits available: BFI-44 `E/A/C/N/O`, age, sex,
handedness, English proficiency, education.

**Round 1 — participant-level, vs. `perceived_effectiveness_z` (n=40):**

| Feature set | CV R² |
|---|---|
| Behavioral (7 feats) | -0.019 |
| Traits only (age + BFI-44, 6 feats) | -0.028 |
| Combined (13 feats) | -0.099 (worse — more overfitting) |

**Round 2 — group-level, vs. `team_coordination`** (the trait-consistent
single-item target found later, §4; T1+T3 pooled, n=63-80):

| Feature set | CV R² |
|---|---|
| Behavioral | -0.125 |
| Traits only | -0.057 |
| Combined | -0.312 (worse — more overfitting) |

**Conclusion: personality/demographic traits do not predict either target,
alone or combined with behavioral features, at either level of analysis.**
All done in ad-hoc temp scripts (not the notebook).

## 4. Single-item alternative targets — first pass

Before building the final multi-task composites (§8), each candidate
single-item target was explored individually:

- **`team_coordination` (T1+T3 only, same item both times):** cross-task
  same-participant correlation ρ=0.42, p=0.007, n=40 — a real, reliable,
  trait-consistent signal (unlike the original composite). Group-level
  (T1+T3 pooled, n=17 group×task obs): `S1_pct` alone → **CV R²=0.451**,
  the first genuinely predictive result of the whole investigation. All 4
  state-% together → R²=-0.165 (overfit, too many features for n=17).
  Univariate: `S0_pct` ρ=-0.62 (p=0.008), `S1_pct` ρ=+0.79 (p<0.001).
  Caveat raised: this excludes T2 entirely (not asked), shrinking n from
  27 to 17 — undesirable for an already-small sample.
- **`voice_inclusion` (asked in all of T1-T3, no exclusion needed):**
  first-pass result n=27, `S1_pct` alone → CV R²=0.115, `S1_pct`+`S2_pct` →
  R²=0.140. Weaker than `team_coordination` but uses the full sample.
  (Re-tested later with the finalized within-task-centering pipeline, §8,
  giving R²=0.093 — same conclusion: weak but positive, different construct.)
- **`mental_demand` (asked T1-T3, no exclusion):** cross-task same-person
  stability: T2 vs T3 ρ=0.61 (p<0.001), T1 vs T3 ρ=0.45 (p=0.004), T1 vs T2
  ρ=0.30 (p=0.066) — trait-consistent. Group-level: `S2_pct`
  ("Aroused-Engaged" state) ρ=+0.43 (p=0.025); CV R² (S1+S2 pct, n=27) =
  **0.10**.
- **`engagement`:** all negative (R²=-0.02 to -0.20). No signal.
- **`overall_valence`:** best R²=0.18 (S1+S2 pct). Modest.

No "mental load" scored item exists in the questionnaire (only incidental
free-text workload mentions).

## 5. Formative vs. reflective measurement-theory justification

Raised because the user asked whether the original questionnaire design
could be defended at all, given the construct-validity flaw in §2.
`perceived_effectiveness_z` can be defended, but **not** as a "trait-like
reflective scale" (which the near-zero cross-task correlation disproves):

- A **reflective** scale assumes all items are interchangeable manifestations
  of one latent trait — that's what was tested in §2 and it failed.
- A **formative index** only requires that each task-specific item plausibly
  indicates "this task went well," which the original design does satisfy.
- Correct citable justification: Diamantopoulos & Winklhofer (2001) on
  formative indicator construction.
- The counterfactual is clear: if a single *consistent* "how effective was
  this task/group" item had been asked in every task, this would be a
  reflective scale with real cross-task signal (like `team_coordination`/
  `mental_demand` show) and none of this diagnostic work would have been
  necessary. The core problem was never the model or features — it was not
  having one item asked consistently across tasks.

## 6. Methodology corrections established mid-investigation

1. **Sanity check rule:** before trusting any pooled/composite target R²,
   always test task-dummies-ONLY (zero real features) through the same
   GroupKFold/RidgeCV pipeline. A high R² here means task-mean leakage, not
   real signal.
2. **Never assume different means = different scales.** Always check raw
   min/max/value_counts before rescaling anything.
3. **When items share a scale but differ in task-level means for genuine
   content reasons** (e.g. negotiation task rated more competitive), use
   **within-task centering** (`groupby("task").transform(x - x.mean())`),
   **not z-scoring** — z-scoring forces equal distributions and can erase
   real between-task signal.
4. **Cross-task + cross-construct correlation tests are confounded** (e.g.
   `cooperative`@T2 vs `team_coordination`@T1) — they conflate construct
   difference with task/time difference and are not valid evidence of
   substitutability either way.
5. **Correct test for construct relatedness = same-task correlation**
   (holds occasion constant). This showed `team_coordination`, `cooperative`,
   `info_sharing`, `voice_inclusion`, `decision_confidence`,
   `equality_of_contribution`, `perceived_control` form a coherent
   "team-functioning" construct family (ρ=0.31-0.56, mostly p<0.05,
   within-task).
6. Formative vs. reflective index framing (Diamantopoulos & Winklhofer 2001)
   is the correct citable justification for a composite built from
   different-but-task-appropriate items.
7. R² must be judged against social-science benchmarks (0.10-0.30 = a real
   effect), not ML benchmarks, given n=27 / 10 independent groups.

## 7. Proxy/substitution testing and same-task correlation evidence

Prompted by the user's question: "is there not anything I can use in place
of team coordination for T2, or of satisfaction for T1, like confidence in
the decision?"

**Cross-task + cross-construct substitution tests (later shown confounded,
§6 point 4 — kept here as the empirical record):**

| Candidate substitute | For gap | Result |
|---|---|---|
| `cooperative` (T2) as `team_coordination` proxy | T2 coordination | ρ=0.04-0.23, p>0.17 — not viable |
| `perceived_control` (T2) as `team_coordination` proxy | T2 coordination | non-significant — not viable |
| `decision_confidence` (T1) as `satisfaction` proxy | T1 satisfaction | ρ=0.10, p=0.54 — not viable |
| `perceived_control` (T1) as `satisfaction` (T2) proxy | T1→T2 bridge | **ρ=+0.40, p=0.013, n=38 — the one substitution that held up** |

These near-zero cross-task/cross-construct correlations initially looked
like evidence the items were unrelated — but the user correctly challenged
this (see §6 point 4): it conflates two different sources of "unrelatedness"
(genuine construct difference AND genuine task-to-task variation).

**Same-task correlation (correct test, holds occasion constant) —
`team_coordination`/`cooperative` and related items form one coherent
"team-functioning" construct family:**

| Pair (same task) | ρ | p |
|---|---|---|
| `team_coordination` vs `info_sharing` @T1 | 0.413 | .008 |
| `team_coordination` vs `equality_of_contribution` @T1 | 0.310 | .051 |
| `team_coordination` vs `voice_inclusion` @T1 | 0.556 | <.001 |
| `team_coordination` vs `decision_confidence` @T1 | 0.312 | .050 |
| `team_coordination` vs `satisfaction` @T3 | 0.367 | .021 |
| `team_coordination` vs `idea_quality` @T3 | 0.107 | ns |
| `team_coordination` vs `voice_inclusion` @T3 | 0.503 | .001 |
| `cooperative` vs `perceived_control` @T2 | 0.507 | .001 |
| `cooperative` vs `voice_inclusion` @T2 | 0.437 | .005 |

This confirms `team_coordination`, `cooperative`, `info_sharing`,
`voice_inclusion`, `decision_confidence`, `equality_of_contribution`, and
`perceived_control` are legitimately part of the same construct family,
justifying `cooperative` as a T2 stand-in for `team_coordination` in §8.

## 8. `team_coordination` + `cooperative` composite — final validated result

Built to cover T1-T3 with one team-functioning item per task
(`team_coordination` T1/T3, `cooperative` T2 as stand-in).

- **First attempt (raw, unscaled):** suspiciously high R²=0.714 →
  task-dummies-only sanity check confirmed it was an **artifact**
  (R²=0.657): `cooperative` T2 mean=3.70 vs `team_coordination` T1
  mean=5.65 / T3 mean=5.85 — the huge gap was leaking through task identity.
- **Second attempt (z-scored):** looked clean at the time (sanity R²=-0.160,
  S1_pct alone R²=0.398/0.40) but was **methodologically wrong** — raw
  value_counts confirmed `cooperative` is on the *same* 1-7 scale as
  `team_coordination` (it asks whether the interaction felt more
  cooperative than competitive, same Likert scale as the other items), so
  z-scoring incorrectly forced distributional equivalence and erased a real
  content difference (negotiation task legitimately rated as less
  cooperative). Caught by the user's pushback, not found independently.
- **Rejected alternative T2 stand-in — `manipcheck_t2`** ("was the outcome
  mutually beneficial for all parties?" item): still scale-mismatched
  (mean 4.47 vs 5.75), sanity R²=0.231 (not clean), worse than `cooperative`.
- **Final, correct approach — within-task centering:** task means:
  T1=5.65 (sd 0.59), T2=3.70 (sd 0.62), T3=5.85 (sd 0.72). Sanity check on
  centered target with task-dummies-only: R²=-0.167 (clean, no leakage).
  - **`S1_pct` alone: CV R² = 0.297 (≈0.30)** ← best result of the entire investigation
  - `S2_pct` alone: R²=-0.331
  - `S1_pct`+`S2_pct`: R²=0.202

**This is the current best, fully sanity-checked, positive result:
R²≈0.30, S1_pct ("Active/Floor-Contested" state %) alone, n=27.** For
context, everything else tried in this session — personality traits, full
multimodal Ridge/PCA, `engagement`, raw `perceived_effectiveness_z` — landed
at R²≈0 or negative, so a genuine positive 0.30 (n=27, 10 independent
groups) is a real, moderate effect for this domain (ρ≈0.55), appropriately
judged against social-science R² benchmarks (0.10-0.30 = a real effect),
not ML benchmarks.

### 8a. Latent-state (HMM) vs. raw-feature comparison for this composite

Per thesis requirement to compare a latent-states approach against a raw-features
approach, the same within-task-centered composite was also tested against the
**raw** 16 features (12 physio/transcript features from `hmm_input_features_final.tsv`
+ 4 curated audio features from `audio_window_features.tsv`, i.e. the pre-HMM
inputs) instead of the derived `S0-S3_pct` states:

| Approach | Best result |
|---|---|
| **Latent states** (`S1_pct` alone) | **R²=0.297** |
| Raw features, full 16-feat Ridge | R²=-1.052 (severe overfit, n=27 vs 16 features) |
| Raw features, PCA(2)+Ridge | R²=0.154 (best raw-feature result) |
| Raw features, PCA(3-5)+Ridge | R²=0.146, 0.110, 0.052 (declining) |
| Raw features, best single feature (`tr_ovl_contested`) | R²=0.084 |

**Conclusion: the latent-state (HMM) approach clearly outperforms the raw-feature
approach for this target** — both because the 4-state HMM compression acts as a
denoising/dimensionality-reduction step well suited to n=27, and because the raw
features individually carry almost no signal (all ≤0.084, most negative).

## 9. `satisfaction` + `decision_confidence` composite — tested, null

Built to cover T1 (via `decision_confidence` stand-in) + T2/T3
(`satisfaction`). Both confirmed same 1-7 scale (satisfaction mean=5.56,
decision_confidence mean=5.83 — much smaller gap than the coordination
case). Within-task centering applied; task means T1=5.83 (sd .76),
T2=5.08 (sd .57), T3=6.09 (sd .53).

- Sanity check (task-dummies-only): R²=-0.274 (clean)
- All state-% features: **negative** (S0=-0.142, S1=-0.154, S2=-0.148,
  S3=-0.230, S1+S2=-0.175)

Conclusion: this composite is not predictable from HMM state features at
all — a genuine null, not an artifact.

## 10. Objective outcome tested: task/discussion duration ("taking longer is worse")

Explored an objective (non-self-report) target: how long a group took, on the
premise that longer = worse. Went through three progressively more accurate
duration measures:

1. **`T1/T2/T3_duration_s`** in [metadata/session_metadata_report.tsv](../metadata/session_metadata_report.tsv)
   — total task block duration. **Rejected**: includes instructions/reading time,
   not just discussion (correctly flagged by the user). Also `grp-10` T1 was a
   corrupted value (69165s / ~19h, clock glitch) — excluded.
2. **`*_task_run_windows.tsv`** (from the downloaded BIDS annotation release,
   `bids_release_no_video/sub-01/ses-*/annot/`) — task-level windows from
   `tobii_calibration` to `finish`. Better, but still includes non-discussion
   phases (calibration, silent reading/idea-gen).
3. **True discussion-only duration (final, correct measure):** parsed raw
   `push_content` marker events from `stimuli_events/grp-*/events.tsv` (`value`
   JSON field, not `description`), using the exact phase boundaries: T1 start
   = `discussion_selection`, T2 start = `role_card`, T3 start =
   `show_ideas_discussion`, end = `finish` (all tasks). This is the
   authoritative discussion-only duration.

**Results (within-task-centered log-duration, true discussion-only measure),
tested against BOTH latent states and raw features:**

| Feature set | Best CV R² |
|---|---|
| Task-dummies-only (sanity) | -0.220 (clean) |
| HMM state-% (`S0-S3_pct`, best single = `S0_pct`) | -0.039 |
| HMM state-% combinations | all negative (-0.04 to -0.28) |
| Raw 16-feature Ridge (full) | -1.490 |
| Raw PCA(2-5)+Ridge | all negative (-0.47 to -0.84) |
| Raw best single feature (`tr_backchannel_count`) | -0.094 |

**Conclusion: task/discussion duration is not predictable from either the
latent-state or the raw-feature approach — a genuine, robust null, confirmed
with the most precise possible measurement of the outcome.** This was
cross-checked with three different duration definitions and two different
feature representations; all converge on no signal.

## 11. Summary table — all targets tried

| Target | Best CV R² | Sanity check | Verdict |
|---|---|---|---|
| `perceived_effectiveness_z` (original) | -0.02 to -0.18 (full feat/PCA) | n/a | Broken — different item per task, not trait-stable, not group-level |
| Personality/demographic traits alone | -0.057 | n/a | No signal |
| `mental_demand` | 0.10 | — | Weak but trait-consistent |
| `engagement` | negative | — | No signal |
| `overall_valence` | 0.18 | — | Modest |
| `team_coordination` (T1+T3 only, no T2 substitute), latent states | 0.451 | — | Strong, but excludes T2 (n=17 not 27) — superseded by the composite below |
| `team_coordination` + `cooperative` (within-task centered), **latent states** | **0.297** | clean (-0.167) | **Best result — recommended target** |
| `team_coordination` + `cooperative`, raw features (best: PCA(2)) | 0.154 | clean (-0.292) | Latent states clearly beat raw features for this target |
| `satisfaction` + `decision_confidence` (within-task centered) | negative | clean (-0.274) | Null |
| `voice_inclusion` (all 3 tasks) | 0.093-0.140 (varies by feature combo tested) | clean (-0.154) | Weak — different construct (participation equality, not coordination/effectiveness) |
| Task/discussion duration (objective, true discussion-only), latent states | -0.039 | clean (-0.220) | Null |
| Task/discussion duration, raw features | -0.094 (best single feat) | clean (-0.220) | Null — confirmed with 3 duration definitions × 2 feature sets |

## 12. Recommendation / open next steps

- **Primary candidate target:** `team_coordination` + `cooperative`
  composite, within-task centered, predicted by `S1_pct`
  ("Active/Floor-Contested" state %). R²≈0.30 is a genuine, moderate effect
  for this sample size and domain (ρ≈0.55). **Confirmed this is a genuine
  latent-state advantage** — raw features top out at R²=0.154 (PCA(2)) on
  the same target (§8a), supporting the thesis's latent-vs-raw comparison.
  **Statistically validated:** group-block permutation test p=0.0015 (§13
  point 2); robust to leave-one-group-out (R² range 0.20-0.44, §13 point 1).
- Not yet done:
  - Latent-vs-raw comparison for `mental_demand`, `overall_valence`, and
    `voice_inclusion` (only tested against HMM state-% so far, §4).
  - Multi-item-per-task version of the same composite (averaging all
    available team-functioning items per task instead of one substitute)
    as a noise-reduction robustness check.
  - Porting the validated composite + within-task-centering pipeline into
    `effectiveness_prediction_models.ipynb` as permanent code (everything
    above was done in throwaway `analysis/_tmp_*.py` scripts).
  - Writing the formative-index / construct-validity justification into the
    thesis methods section, citing Diamantopoulos & Winklhofer (2001) and
    the same-task correlation evidence in §5.
  - Disclosing the target-selection multiplicity openly in the thesis
    methods section (§13 point 3) — the one remaining open item from the
    feedback review.

## 13. Honest assessment — is this strong enough for a masters thesis?

Blunt answer: **this is not a weak analysis — it's actually one of the more
methodologically careful diagnostic processes you could show in a thesis.**
But the *headline number* (R²≈0.30) is modest and needs to be framed
correctly, or it will look weaker than the work behind it deserves. Both
things are true at once.

**Why this is a genuine strength, not a weakness:**

- You caught and fixed a **real construct-validity flaw** in your own
  instrument (different item per task in the original composite) instead of
  reporting a broken number. That diagnostic process (variance
  decomposition → cross-task stability check → root-cause identification →
  formative-index reframing) is itself a defensible methodological
  contribution, independent of any R² value.
- Every positive result was **sanity-checked against task-mean leakage**
  before being trusted (§6 point 1) — this caught a real artifact once
  (R²=0.714 → 0.657, §8) and a wrong "fix" once (z-scoring, §8). Many
  published papers with small group-level samples don't do this.
- The final result (R²≈0.30) is **cross-validated (GroupKFold, held-out
  groups)**, not an in-sample correlation — it survived an out-of-sample
  test, which is a materially stronger claim than "these two variables
  correlate."
- You explicitly tested the **latent-state vs. raw-feature** comparison your
  thesis requires (§8a) and got a clean, interpretable answer (latent states
  win, 0.297 vs 0.154) — this is a real, useable finding on its own,
  regardless of the absolute R².
- You **reported every null result** (personality, satisfaction+confidence,
  duration, engagement) instead of only the one that worked. A thesis
  section built around "we tried N reasonable hypotheses, here's what held
  up and what didn't, and here's why" is a legitimate, common, and
  defensible thesis structure (exploratory/diagnostic study), especially
  when the original instrument had a design flaw discovered partway through.

**Why you should still be cautious, and what would make it airtight:**

1. **Small independent-unit count (n=10 groups). ✅ CHECKED — result is
   robust.** Ran a leave-one-group-out sensitivity check (refit the
   identical `S1_pct`-only GroupKFold pipeline excluding each of the 10
   groups once, n=9 groups per refit): **R² ranged from 0.20 to 0.44 across
   all 10 refits (mean 0.291, sd 0.068), full-sample baseline 0.306 —
   never dropped below 0.20 and never flipped sign.** No single group is
   driving the effect; the strongest single-group influence (`grp-11`, whose
   removal moves R² up to 0.44) still leaves the result solidly positive
   either way. This is meaningfully more reassuring than the single point
   estimate alone.

   *Anticipated objection, addressed:* "but each group contributes several
   minutes of continuous multimodal recording, not one data point." True,
   and important — but that richness reduces **measurement error in the
   feature** (`S1_pct` is a well-estimated summary over many timesteps, not
   a single noisy reading), it does **not** create additional independent
   **replicates of the group×outcome relationship**. The outcome
   (`team_coordination`/`cooperative` rating) exists once per group×task;
   three task-observations from the same group are not independent draws
   either (same people, same room, same baseline physiology/personality) —
   which is exactly why GroupKFold clusters by `group_id`. This is a
   standard clustered/multilevel-data distinction: many within-unit
   measurements shrink noise in *X*, but the number of independent units
   available to generalize the *X→Y relationship across groups* is still
   10. Analogy: measuring one patient's blood pressure 1,000 times makes
   that estimate precise, but it is not equivalent to 1,000 patients' worth
   of evidence that a drug works across people. The genuine upside of your
   point: because feature measurement error is low, a real effect is less
   likely to be masked by noise — which is a reason 10 groups isn't hopeless
   for detecting a true effect, not a reason the effective n is larger than
   10 for inference/generalizability purposes.
2. **No permutation-based significance test. ✅ CHECKED — significant.** Ran
   a **group-block permutation test**: the within-task-centered composite
   values were pivoted to one row per group (columns = task), then the row
   *labels* (i.e. which group's whole target-vector is paired with which
   group's features) were randomly shuffled 2,000 times — preserving
   within-group task structure and the GroupKFold clustering, only breaking
   the group↔behavior link, exactly the right null model for this
   clustered design. Same `S1_pct`-only pipeline refit on each shuffle.
   - Observed CV R² = 0.3057 (matches §8's 0.297, small difference from the
     RidgeCV alpha grid/refit run-to-run, not a discrepancy)
   - Sanity check on this run: task-dummies-only R²=-0.118 (clean, consistent with §8)
   - Null distribution (2,000 group-block permutations): mean R²=-0.160, sd=0.086
   - 95th percentile of null = -0.012; 99th percentile of null = 0.142
   - **Only 3 of 2,000 permutations reached R²≥0.306 → empirical p = 0.0015**
   - **This converts the result from "R²=0.30" to "R²=0.31, p=0.0015 vs. a
     group-block permutation null (n=2,000)" — a materially stronger,
     publication-appropriate claim.** The observed R² sits far outside the
     null distribution's 99th percentile, so this is not "one lucky fold."
3. **Multiplicity / "garden of forking paths."** Roughly 10-15 different
   targets and dozens of feature combinations were tried this session alone
   before landing on the `team_coordination`+`cooperative` composite. CV
   protects against *overfitting the model* to noise, but it does **not**
   protect against *overfitting the choice of target/composite* to noise —
   that selection was informed by seeing results along the way. **Action:**
   be transparent about this in the thesis (a short "target selection
   process" paragraph, exactly like this log, is good practice and expected
   in exploratory work) — don't present R²=0.30 as if it were a single
   pre-registered confirmatory test. Reviewers respect honesty about this
   far more than a hidden multiplicity problem discovered later.
4. **Single-feature model.** The winning result is one feature (`S1_pct`)
   predicting one composite — a strong univariate relationship, not a rich
   multivariate model. That's fine and honest, but describe it as such
   ("a single interpretable behavioral marker of task engagement/floor
   contest predicts group-level team-functioning ratings") rather than
   implying a general multimodal prediction system was fully validated.
5. **Effect size framing.** R²=0.30 (ρ≈0.55) is a *real, moderate-to-strong*
   effect by social-science standards (§6 point 7) — don't undersell it out
   of anxiety, but don't oversell it as a large/robust ML result either.
   Anchor the thesis language to comparable behavioral-science team studies,
   not to ML benchmark expectations.

**Bottom line:** the work is not weak — it is a rigorous, honestly-reported
diagnostic investigation that ends in one real, moderate, cross-validated,
theoretically-grounded, sanity-checked, **statistically significant
(p=0.0015 vs. group-block permutation null)** positive result that is
**robust to any single group being dropped (R² range 0.20-0.44 across
leave-one-group-out refits)**, plus a clean latent-vs-raw comparison. The
remaining risk is no longer statistical fragility — it's *transparency
about multiplicity* (§13 point 3): disclose the target-selection process
openly in the thesis methods section, exactly as logged here, and this
result stands on solid ground for a masters thesis working with n=10
groups of real human behavioral data.
