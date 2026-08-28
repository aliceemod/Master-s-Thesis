# Thesis Research Questions, Sub-Questions & Hypotheses

> **Master's Thesis Topic:** Predicting perceived effectiveness in small group meetings through multimodal features — comparing raw-feature prediction against latent-state discovery via unsupervised HMM.

---

## 0. Study Context

| Aspect | Detail |
|--------|--------|
| **Central question** | Do latent interaction states (discovered via HMM) predict how effective a meeting is perceived to be, better than raw multimodal features? |
| **Design** | 10 groups × 4 participants × 3 tasks (T1 Hidden-Profile, T2 Negotiation, T3 Idea Generation), 30 s windows |
| **Available modalities** | Physiology (HR, EDA, temp — EmotiBit), eye-tracking (pupil, gaze, blink — Tobii), conversation structure (overlaps, laughter, backchannel, silence — transcript), lexical content (14 features — transcript), audio prosody (pitch, energy, HNR — openSMILE, task-level only) |
| **HMM feature set** | 10 features → PCA → 5 latent states (Gaussian HMM, BIC-selected) |
| **Outcome targets** | `team_coordination` (T1/T3), `cooperative` (T2), `voice_inclusion` (T1–T3), `mental_demand` (T1–T3) — within-task-centred |
| **Current best result** | S1_pct alone → team-functioning composite, R² ≈ 0.30 (n=27, 10 groups, LOGO-CV) |
| **Key negative results** | Personality/demographics → null; raw 16-feature Ridge → R² = −1.05 (overfit); PCA(2)+Ridge → R² = 0.15; best single raw feature → R² = 0.08 |
| **The small-n problem** | n=27 observations, 10 independent groups. Raw multi-feature models overfit; dimensionality reduction is necessary. The thesis question is whether the HMM's advantage is purely statistical (better compression for small n) or also substantive (temporal state structure captures emergent group dynamics that linear compression misses) |

---

## 1. Main Research Question

> **MRQ: Can perceived meeting effectiveness be predicted from multimodal interaction signals, and does discovering latent collective states outperform using raw features directly?**

This decomposes into three pillars:

| Pillar | Focus |
|--------|-------|
| **RQ1** | *What predicts perceived effectiveness?* — Which multimodal features (physio, ET, conversation, lexical, audio) relate to how groups perceive their meetings? |
| **RQ2** | *Do latent states help?* — Does unsupervised HMM state discovery improve prediction over raw features, and why? |
| **RQ3** | *What characterises effective interaction?* — What do the states look like, what drives them, and does the pattern change across task types? |

---

## 2. RQ1 — What Predicts Perceived Effectiveness?

> *Which multimodal signals — individually and in combination — predict team coordination, cooperation, voice inclusion, and perceived mental demand?*

### Sub-RQ1.1: Do individual modalities carry predictive signal?

Test each modality in isolation (1–4 features, avoids overfitting) against the four outcome targets.

#### H1.1a: Physiological arousal predicts perceived effectiveness
> Heart rate and EDA phasic rate, aggregated at group level, correlate with within-task-centred team-functioning composite. Direction expected to be *non-monotonic*: moderate arousal → better outcomes than very low (disengaged) or very high (stressed).

**Test:** Single-feature and 2-feature (HR + EDA) Ridge, LOGO-CV; also test quadratic HR term.

#### H1.1b: Conversation-structure features are stronger univariate predictors than physiology
> Silence duration, laughter count, backchannel count, and overlap time capture the *social process* directly, while physiology is an indirect proxy. Conversation features should yield higher single-feature R² than any physio feature.

**Test:** Compare best single-feature R² from each modality (physio vs. conversation vs. ET vs. lexical).

**Literature:** Pentland (2012) showed "honest signals" (turn-taking patterns) predict team performance better than content analysis.

#### H1.1c: Pupil diameter predicts mental demand but not coordination
> Pupil dilation indexes cognitive load (Kahneman, 1973). It should predict `mental_demand` (an individual-experience outcome) but not `team_coordination` (an emergent group property), because cognitive effort and social coordination are different constructs.

**Test:** Separate Ridge models for each target using `group_et_pupil_mean_mean` alone.

#### H1.1d: Lexical features carry modest but real predictive signal
> Language content (agreement, sentiment, hedging) captures social dynamics that timing features miss. But with 14 features and n=27, only 1–2 feature models are viable.

**Test:** Best single lexical feature and best 2-feature lexical model, LOGO-CV, against all four targets.

---

### Sub-RQ1.2: Which outcome dimensions are predictable from group-level signals?

#### H1.2a: Coordination and cooperation are more predictable than voice inclusion and mental demand
> Coordination and cooperation are *emergent group outcomes* — they arise from the interaction itself. Voice inclusion and mental demand are *individual experiences* that happen to be group-averaged. Group-level features should predict group-emergent outcomes better.

**Test:** Compare best-achievable R² across the four targets using the same feature sets.

#### H1.2b: Mental demand is best predicted by physiology; coordination by conversation structure
> Different constructs have different "natural modalities." Cognitive load → pupil/HR; social coordination → overlap patterns, laughter, participation balance.

**Test:** Modality-stratified prediction (physio-only vs. conversation-only vs. ET-only vs. lexical-only) per target; report which modality "wins" for each outcome.

#### H1.2c: Voice inclusion is predicted by participation *balance*, not total activity
> Groups that talk a lot but unevenly will report low voice inclusion. The Gini coefficient or speaker entropy of word counts should predict voice inclusion above total word count.

**Test:** Partial correlation: speaker_entropy → voice_inclusion | total_word_count.

**Literature:** Woolley et al. (2010) — equal turn-taking predicts collective intelligence.

---

### Sub-RQ1.3: Do raw features interact with task type?

#### H1.3a: The same feature has opposite effects in different tasks
> Hedging helps in T2 (negotiation — face-saving) but hurts in T1 (decision — delays convergence). Agreement helps in T1/T3 (builds consensus) but is irrelevant in T2 (can signal premature concession).

**Test:** Task × feature interaction in mixed-effects models; or stratified correlations comparing sign/magnitude across T1, T2, T3.

#### H1.3b: Silence is productive in decision-making but harmful in negotiation
> In T1 (hidden-profile), silence indicates thinking/deliberation → positive. In T2 (negotiation), silence indicates impasse → negative. The sign of `tr_silence_duration_s` → outcome should flip between tasks.

**Test:** Spearman correlation of silence with target, per task.

---

## 3. RQ2 — Do Latent States Improve Prediction?

> *Does compressing multimodal windows into HMM-discovered latent states predict perceived effectiveness better than using raw features directly — and is the advantage substantive or merely statistical?*

### Sub-RQ2.1: Latent states vs. raw features — the core comparison

#### H2.1a: HMM state proportions outperform raw features at matched dimensionality
> The HMM compresses 10 features into k−1 = 4 state proportions. To rule out a pure dimensionality advantage, compare HMM states against the best 4 raw features (selected by univariate correlation). If HMM still wins, the advantage is *structural*, not just compression.

**Test:** Ridge CV with 4 HMM state-% vs. Ridge CV with best-4 raw features (selected on training folds only, to avoid leakage). Report R² difference with bootstrap CI.

#### H2.1b: HMM outperforms PCA at the same dimensionality because it captures temporal structure
> PCA finds linear variance-maximising directions; HMM finds *sequential state patterns*. The HMM's self-transition structure (mean 0.82) encodes state persistence — groups dwell in states for multiple windows, and the *sequence* matters. PCA ignores window order.

**Test:** PCA(4 components) + Ridge vs. HMM(4 state-%) + Ridge. Same n, same k, different inductive bias.

**Current evidence:** PCA(2)+Ridge = R² 0.15; HMM(S1_pct alone) = R² 0.30. Supports the hypothesis.

#### H2.1c: The HMM advantage is largest for group-emergent outcomes
> Latent states capture *patterns of interaction* (e.g., "high overlap + high laughter + moderate HR" = engaged-playful state). These patterns are precisely what drives group-emergent outcomes like coordination, but are less relevant for individual-experience outcomes like mental demand. The R² gap between HMM and raw features should be largest for coordination/cooperation and smallest for mental demand.

**Test:** Compute (R²_HMM − R²_raw) per outcome; report the interaction.

---

### Sub-RQ2.2: What makes the HMM advantage work?

#### H2.2a: The HMM acts as a denoiser — state labels are more stable than raw features
> Raw features fluctuate window-to-window due to measurement noise. The HMM smooths this through its transition matrix (states are "sticky"). The resulting state proportions per task have lower coefficient of variation than task-aggregated raw features.

**Test:** Compare CV of S_k_pct vs. CV of task-mean raw features across groups.

#### H2.2b: States capture *nonlinear feature combinations* that single features miss
> No single feature can encode "high laughter + low overlap + moderate HR simultaneously." But an HMM state can. If the best-performing state (S1) is characterised by a specific *combination* of features (not just one extreme value), the advantage is genuinely multivariate, not just a dimensionality trick.

**Test:** Profile S1 across all 10 features — if it's characterised by ≥ 3 features being in a specific range simultaneously, the combination argument holds. Also: test whether a binary "all 3 conditions met" indicator predicts outcomes as well as S1_pct.

#### H2.2c: Temporal contiguity matters — shuffling window order within a task should degrade HMM prediction
> If the sequential structure is important (not just the marginal distribution of states), then running the HMM on shuffled window sequences should produce different state proportions and worse prediction.

**Test:** Shuffle window order within each group×task, re-run Viterbi decoding, re-compute state proportions, re-run Ridge. Compare R² to original (repeat 100 times for CI).

---

### Sub-RQ2.3: Do additional features (lexical, audio) improve the HMM?

#### H2.3a: Adding lexical features to the HMM sharpens state boundaries
> Lexical content (agreement, sentiment, questions) adds semantic information that timing features lack. If BIC decreases when lexical features are added, the HMM benefits from knowing *what* is said, not just *when*.

**Test:** Compare 10-feature HMM (current) vs. 14-feature HMM (+4 lexical); report ΔBIC, state stability (NMI of assignments).

#### H2.3b: Adding lexical features does not improve *prediction* despite improving state discovery
> Even if BIC improves (H2.3a), prediction R² may not — because the additional state resolution might not align with the outcome variable. This would show that *discoverable* structure ≠ *useful* structure.

**Test:** Compare R² of prediction using 10-feature-HMM states vs. 14-feature-HMM states.

#### H2.3c: Audio prosodic features (pitch, energy) would be the most valuable addition if windowed
> Vocal quality captures arousal and emotional tone independently from physiology and language content. Audio features are currently task-level only; if windowed, they would add a dimension no other modality covers.

**Prediction:** In a leave-one-modality-out ablation, removing audio (if available) degrades prediction more than removing any other single modality.

**Test:** Blocked in this workspace (raw audio not accessible), but the methodological claim can be made and tested in future work.

---

## 4. RQ3 — What Characterises Effective Group Interaction?

> *What do the HMM states look like, what multimodal signals drive them, and do effective groups navigate different state trajectories than ineffective groups?*

### Sub-RQ3.1: State interpretation — what does each state mean?

#### H3.1a: States are interpretable as distinct interaction modes, not statistical artefacts
> Each HMM state should have a distinctive multimodal profile (≥ 2 features with |z| > 0.5) and a plausible psychological interpretation. If states are just noise-clustering, profiles will be flat.

**Test:** Heatmap of mean z-scored features per state; report number of features with |z| > 0.5 per state.

#### H3.1b: Lexical features provide the *interpretive key* to otherwise ambiguous physiological states
> Some states may have similar HR/EDA profiles but differ in what people are saying (e.g., both "moderate arousal" but one with high agreement and one with high certainty). Lexical features disambiguate.

**Test:** For state pairs with |Δz| < 0.3 on all physio features, check whether lexical features show |Δz| > 0.5.

#### H3.1c: Laughter is the single most distinctive feature of the positive-outcome state
> Laughter is a unique social signal — no other feature captures positive social affect directly. The state with the highest β for predicting effectiveness should also have the highest mean laughter count.

**Test:** Rank features by between-state F-ratio; check whether `tr_laughter_count` is top-3 for the positive state.

---

### Sub-RQ3.2: How do physiological, conversational, and lexical signals interact within states?

#### H3.2a: Agreement language co-occurs with higher HRV (parasympathetic activation)
> Polyvagal theory (Porges, 2007): social engagement is linked to vagal regulation. If agreement language and HRV correlate within windows, two independent modalities are measuring the same underlying "social safety" construct.

**Test:** Multilevel model: `hrv_rmssd ~ agreement_count + word_count + (1|group/task)`.

#### H3.2b: Hedging language under conflict has different physiological correlates than hedging under coordination
> The *same* linguistic act (hedging) may signal different things depending on context. In conflict states (high competitive overlap, high EDA), hedging is stress-regulatory face-work. In coordination states, hedging is collaborative exploration.

**Test:** Interaction term: `EDA ~ hedging × state_type + (1|group)`.

#### H3.2c: Pupil dilation tracks question density most strongly in the information-seeking task (T1)
> Pupil dilation indexes cognitive load (Kahneman, 1973). Questions require formulating and processing information gaps. The coupling should be strongest in T1 (hidden-profile), where information discovery is the explicit goal.

**Test:** Task-stratified partial correlation of `lex_question_count` with `group_et_pupil_mean_mean`, controlling for word count.

#### H3.2d: EDA phasic rate spikes during competitive overlaps, not cooperative ones
> The two-layer overlap taxonomy distinguishes competitive (floor fights, 500ms+) from cooperative overlaps (collaborative context labels). If EDA selectively rises with competitive but not cooperative overlaps, it validates the taxonomy via an independent physiological channel.

**Test:** Partial correlations of `eda_phasic_rate_hz` with `tr_ovl_competitive_timing` vs. `tr_ovl_collaboration_index`, controlling for total overlap time.

---

### Sub-RQ3.3: Does participation structure predict how people perceive the meeting?

#### H3.3a: Participation equality predicts voice inclusion above total activity
> Groups where everyone contributes equally to the conversation (low Gini of word counts) will report higher voice inclusion, independent of how much total talking occurs. This operationalises "being heard" as having proportional linguistic presence.

**Test:** Partial Spearman: $\rho(\text{word\_Gini}, \text{voice\_inclusion} \mid \text{total\_words})$.

**Literature:** Woolley et al. (2010) — equal turn-taking → collective intelligence.

#### H3.3b: Being *responded to* predicts voice inclusion better than speaking time
> A participant who talks a lot but whose contributions are followed by topic changes (no lexical overlap with the next speaker's utterance) will feel unheard. Per-speaker "response relevance" (Jaccard between speaker's last utterance and the next turn) should predict that individual's voice inclusion score.

**Test:** Per-speaker within-group regression: `voice_inclusion_i ~ response_relevance_i + word_count_i`.

**Novel feature required:** Per-speaker response relevance from sequential utterance comparison.

#### H3.3c: Receiving agreement directed at your contributions predicts your voice inclusion
> Agreement has a target — the person whose idea is being affirmed. If "yeah, exactly" follows P3's proposal, P3 receives the agreement. The count of agreement markers *received* per speaker should predict that speaker's voice inclusion.

**Test:** Per-speaker within-group regression: `voice_inclusion_i ~ agreement_received_i + speaking_share_i`.

**Novel feature required:** Per-speaker agreement-received count (from sequential turn analysis).

#### H3.3d: Dominance by a single speaker suppresses group-level voice inclusion
> `lex_dominant_share` (max speaker word fraction) should negatively predict group-mean `voice_inclusion` — one person monopolising the floor makes everyone else feel unheard.

**Test:** Spearman: dominant_share → voice_inclusion.

---

### Sub-RQ3.4: Do effective groups follow different temporal trajectories?

#### H3.4a: Effective groups transition *into* positive states earlier in the task
> Groups with higher team-functioning scores reach S1 (coordination state) earlier in the task than low-scoring groups. First-passage time to S1 should negatively correlate with outcome.

**Test:** First window index assigned to S1 per group×task; Spearman with outcome.

#### H3.4b: Effective negotiation groups follow a "soften → close" linguistic trajectory
> In T2, the slope of `certainty − hedging` over window index should be positive in high-cooperation groups (opening with exploration, closing with conviction) and flat/negative in low-cooperation groups.

**Test:** OLS slope per group in T2; Spearman with within-task-centred `cooperative`.

**Literature:** Fisher & Ury (1981), "Getting to Yes" — expanding → claiming progression.

#### H3.4c: Lexical novelty injection precedes transitions to coordination states
> Windows immediately before a transition into S1 should show higher TTR or lower inter-turn vocabulary overlap (someone reframes the conversation). This suggests linguistic innovation can *trigger* state shifts.

**Test:** Pre-transition vs. within-state comparison; Mann-Whitney U.

**Novelty:** Tests a causal-direction-suggestive mechanism for state change.

#### H3.4d: Effective idea-generation groups sustain lexical novelty longer
> In T3 (NGT), per-window "novelty ratio" (words not seen in prior windows / total words) should decay more slowly in groups with higher T3 coordination — they keep introducing new concepts rather than rehashing.

**Test:** Per-group decay slope of novelty ratio over window index; Spearman with T3 outcome.

**Literature:** Nijstad & Stroebe (2006) — cognitive stimulation in brainstorming depends on exposure to diverse ideas.

---

### Sub-RQ3.5: What predicts perceived mental demand?

#### H3.5a: Silence duration positively predicts mental demand — silence is cognitive work
> Silence is not disengagement; it's thinking time. Groups reporting higher mental demand should show more silence, more pauses, and lower speech rate. This explains why S2 ("Silent/Thinking", β = +0.36) has a positive effectiveness coefficient — those pauses represent genuine cognitive engagement.

**Test:** Spearman of `tr_silence_duration_s` with `mental_demand`.

#### H3.5b: Lexical simplification signals high cognitive load
> Under high cognitive demand, working memory constrains language production (Sweller, 1988). Groups in high-load windows should show lower TTR, shorter mean word length, and lower bigram entropy — a "linguistic load signature."

**Test:** Within-group partial correlations of lexical complexity features with pupil diameter (a validated load measure), controlling for word count.

#### H3.5c: Question density correlates with mental demand in T1 but not T2
> In T1 (hidden-profile), asking questions is the *mechanism* for discovering hidden information — it's effortful cognitive work. In T2 (negotiation), questions may be strategic probes (lower cognitive cost). Task-moderated correlation.

**Test:** Task-stratified Spearman of `lex_question_count` with `mental_demand`.

---

### Sub-RQ3.6: Do lexical features carry enough information to approximate full multimodal states?

#### H3.6a: Lexical features alone can classify HMM states above chance (> 40% accuracy)
> If lexical features alone can partially decode the HMM states (trained from physio + conversation-timing + ET), language carries substantial redundant information — enabling a lightweight deployment path (ASR-only → state estimate).

**Test:** Logistic regression on 14 lexical features, 5-class, LOGO-CV. Chance = 20%.

#### H3.6b: Conversation-structure features (overlap, silence, laughter) recover states better than physiology alone
> This tests which modality is the *primary driver* of the HMM. If conversation features are sufficient to reconstruct most of the state structure, the HMM is primarily a "conversation pattern" model rather than a "physiological state" model.

**Test:** Per-modality state classification accuracy: conversation-only vs. physio-only vs. ET-only vs. lexical-only.

---

## 5. Multimodal Feature Reference

### 5.1 Features Currently in the HMM (10 features)

| Feature | Modality | What it captures |
|---------|----------|-----------------|
| `group_hr_mean_bpm_mean` | Physio | Collective physiological arousal |
| `group_eda_phasic_rate_hz_mean` | Physio | Collective emotional reactivity |
| `group_et_pupil_mean_mean` | Eye-tracking | Attention / cognitive load |
| `tr_silence_duration_s` | Conversation | Floor quietness (thinking vs. impasse) |
| `tr_backchannel_count` | Conversation | Listener affiliation |
| `tr_laughter_count` | Conversation | Positive social affect |
| `tr_n_active_speakers` | Conversation | Participation breadth |
| `tr_ovl_time_s` | Overlap | Total simultaneous speech load |
| `tr_ovl_competitive_timing` | Overlap | Floor competition intensity |
| `tr_ovl_collaboration_index` | Overlap | Cooperative intent fraction |

### 5.2 Lexical Features (14 implemented, window-level)

| Feature | Category | What it captures |
|---------|----------|-----------------|
| `lex_word_count` | Verbosity | Total speech output |
| `lex_unique_words` | Complexity | Vocabulary size |
| `lex_ttr` | Diversity | Type-token ratio |
| `lex_bigram_entropy` / `trigram_entropy` | Predictability | Language predictability/formulaicity |
| `lex_agreement_count` | Social | Affiliation signals ("yes", "exactly") |
| `lex_hedging_count` | Uncertainty | Face-saving / exploration ("maybe", "perhaps") |
| `lex_certainty_count` | Confidence | Conviction ("definitely", "clearly") |
| `lex_positive_count` / `negative_count` | Affect | Sentiment word counts |
| `lex_question_count` | Turn-taking | Information-seeking |
| `lex_suggestion_count` | Initiative | Proposals ("let's", "we should") |
| `lex_sentiment_ratio` | Affect | Net positive/negative balance |
| `lex_social_composite` | Social | Agreement + positive + hedging |

### 5.3 Available but Not in HMM (thesis extension candidates)

| Feature | Modality | Why useful | Status |
|---------|----------|-----------|--------|
| `group_hrv_rmssd_ms_mean` | Physio | Vagal tone / stress regulation | Available (88% coverage) |
| `group_hr_mean_bpm_std` | Physio | Physiological synchrony | Available (88%) |
| `group_et_blink_rate_per_min_mean` | ET | Fatigue / disengagement | Available (84%) |
| `group_et_gaze_dispersion_mean` | ET | Visual scanning vs. focused attention | Available (85%) |
| `tr_filled_pause_count` | Conversation | Hesitation / cognitive load | Available (36%) |
| `tr_ovl_smooth` | Overlap | Coordinated turn transitions | Available (100%) |
| Audio prosody (pitch, energy, HNR) | Audio | Vocal arousal/stress | Task-level only — not windowed |
| VAD self-report probes | Self-report | Subjective affect state | Excluded from HMM by design (validation target) |

### 5.4 Proposed New Features

| Feature | Effort | Rationale |
|---------|--------|-----------|
| `lex_word_gini` | Low | Participation inequality (Woolley replication) |
| `lex_speaker_entropy` | Low | Information-theoretic balance |
| `lex_dominant_share` | Low | Single-speaker monopoly |
| `lex_vocab_overlap_slope` | Medium | Lexical convergence over task (CAT) |
| `lex_response_relevance` | Medium | Being responded to (voice inclusion predictor) |
| `lex_agreement_received` | Medium | Per-speaker directed agreement |
| `lex_deliberation_quality` | Low | $z(\text{question}) + z(\text{ttr}) - z(\text{certainty})$ |
| `lex_warmth` / `lex_dominance` | Low | Interpersonal Circumplex axes |
| `lex_novelty_ratio` | Low | Per-window fresh-vocabulary fraction |
| `lex_certainty_slope` | Low | Closing trajectory within task |

---

## 6. Analysis Roadmap

### Phase 1: Raw-Feature Baselines (Per-Modality Prediction) — RQ1
1. ⏳ Single-feature and per-modality R² for all 4 targets
2. ⏳ Best-4 raw features model as dimensionality-matched baseline
3. ⏳ Task × feature interaction tests

### Phase 2: Latent-State Comparison — RQ2
1. ✅ HMM S1_pct → R² ≈ 0.30 (completed)
2. ⏳ PCA(4) vs. HMM(4 states) at matched dimensionality
3. ⏳ Best-4-raw vs. HMM comparison
4. ⏳ Temporal-order ablation (window shuffle test)
5. ⏳ Leave-one-modality-out ablation

### Phase 3: State Characterisation — RQ3.1–3.2
1. ⏳ State profiles (heatmaps, radar plots) across all modalities including lexical
2. ⏳ Cross-modal coupling tests (agreement×HRV, hedging×EDA, questions×pupil)
3. ⏳ Overlap taxonomy validation via EDA

### Phase 4: Participation & Voice Inclusion — RQ3.3
1. ⏳ Implement participation balance features (Gini, entropy, dominant share)
2. ⏳ Response relevance and directed agreement features
3. ⏳ Voice inclusion prediction models

### Phase 5: Temporal Trajectories — RQ3.4
1. ⏳ First-passage time to S1
2. ⏳ Hedging/certainty slope in T2
3. ⏳ Novelty injection before state transitions
4. ⏳ Novelty decay rate in T3

### Phase 6: Mental Demand & Cognitive Load — RQ3.5
1. ⏳ Silence × mental demand
2. ⏳ Lexical simplification under load
3. ⏳ Task-moderated question-demand coupling

### Phase 7: State Recovery & Modality Importance — RQ3.6
1. ⏳ Per-modality state classification
2. ⏳ Lexical-only state approximation

---

## 7. Key References

### Prediction & Collective Intelligence
- Woolley, A. W. et al. (2010). Evidence for a collective intelligence factor. *Science*.
- Pentland, A. (2012). The new science of building great teams. *Harvard Business Review*.

### HMMs & Latent States
- Rabiner, L. R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*.
- Scheffer, M. et al. (2009). Early-warning signals for critical transitions. *Nature*.

### Physiology & Social Engagement
- Porges, S. W. (2007). The polyvagal perspective. *Biological Psychology*.
- Kok, B. E. & Fredrickson, B. L. (2010). Upward spirals of the heart. *Psychological Science*.
- Kahneman, D. (1973). *Attention and Effort*. Prentice-Hall.

### Linguistic Accommodation
- Giles, H. (1973). Accent mobility. *Anthropological Linguistics*.
- Ireland, M. E. et al. (2011). Language style matching predicts relationship initiation and stability. *Psychological Science*.

### Negotiation & Group Decision
- Fisher, R. & Ury, W. (1981). *Getting to Yes*. Houghton Mifflin.
- Stasser, G. & Titus, W. (1985). Pooling of unshared information. *JPSP*.
- Janis, I. L. (1972). *Victims of Groupthink*. Houghton Mifflin.

### Cognitive Load & Language
- Sweller, J. (1988). Cognitive load during problem solving. *Cognitive Science*.
- Nijstad, B. A. & Stroebe, W. (2006). How the group affects the mind. *PSPR*.

### Interpersonal Theory
- Brown, P. & Levinson, S. C. (1987). *Politeness*. Cambridge UP.
- Wiggins, J. S. (1979). A psychological taxonomy of trait-descriptive terms. *JPSP*.

### Group Dynamics
- Tuckman, B. W. (1965). Developmental sequence in small groups. *Psychological Bulletin*.
- Delaherche, E. et al. (2012). Interpersonal synchrony: A survey. *Affective Computing*.

---

## 8. Summary

### The Thesis Argument in One Paragraph

Predicting how effective a small group meeting is perceived to be from raw multimodal features is hampered by the curse of dimensionality (n=27, 10+ features → overfit). Latent-state discovery via HMM provides a principled compression that outperforms both raw features and linear PCA — not merely because it reduces dimensionality, but because it captures *temporal, nonlinear interaction patterns* (state persistence, feature combinations) that linear methods miss. The resulting states are interpretable through their multimodal profiles (physiology, conversation structure, lexical content) and offer practical value: they identify which collective interaction modes (e.g., engaged-playful vs. tense-contested) predict better perceived outcomes, opening the door to real-time meeting facilitation.

### Hypothesis Count

| Research Question | Sub-RQs | Hypotheses |
|-------------------|---------|------------|
| RQ1: What predicts effectiveness? | 3 | 9 (H1.1a–H1.3b) |
| RQ2: Do latent states help? | 3 | 8 (H2.1a–H2.3c) |
| RQ3: What characterises effective interaction? | 6 | 16 (H3.1a–H3.6b) |
| **Total** | **12** | **33** |

### What's Feasible for the Thesis

Not all 33 hypotheses need to be tested. Priority tiers:

| Tier | Hypotheses | Why |
|------|-----------|-----|
| **Must-have** (core thesis) | H2.1a, H2.1b, H2.1c, H1.1b, H3.1a | The raw-vs-latent comparison and basic state interpretation — this *is* the thesis |
| **Should-have** (strong support) | H1.2a, H1.2b, H2.2b, H2.2c, H3.3a, H3.4a, H3.5a | Outcome-specificity, state mechanism, voice inclusion, temporal dynamics |
| **Nice-to-have** (depth/novelty) | H3.2a–d, H3.3b–c, H3.4b–d, H3.5b–c, H2.3a–b | Cross-modal coupling, accommodation, trajectory shapes, HMM extension |
| **Exploratory** (if time permits) | H3.6a–b, H1.3a–b, H3.4c | State recovery from language alone, task interactions |
