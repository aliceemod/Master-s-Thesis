# Lexical Feature Implementation Plan — Master's Thesis

> **Reference:** Lai & Murray (2018) "Predicting group satisfaction in meeting discussions" — ACM Workshop on Modeling Cognitive Processes from Multimodal Data
>
> **Their approach:** Acoustic + Lexical + Turn-taking features → group-level satisfaction prediction on AMI corpus

---

## 1. Current Feature Inventory (Already Implemented)

### 1.1 Conversation Structure (8 features)
- `tr_silence_duration_s`, `tr_backchannel_count`, `tr_laughter_count`
- `tr_overlap_count`, `tr_overlap_duration_s`
- `tr_total_speaking_duration_s`, `tr_num_turns`
- `tr_interjection_count`

### 1.2 Lexical Features (14 features)
- `lex_word_count`, `lex_unique_words`, `lex_ttr`
- `lex_agreement_count`, `lex_hedging_count`, `lex_certainty_count`
- `lex_positive_count`, `lex_negative_count`
- `lex_question_count`, `lex_suggestion_count`
- `lex_dominant_share`, `lex_speaker_entropy`
- `lex_mean_utterance_length`, `lex_pos_neg_ratio`

### 1.3 Physiology (4 features)
- `group_hr_mean_mean`, `group_eda_phasic_rate_hz_mean`
- `group_temp_mean_mean`

### 1.4 Eye-tracking (4 features)
- `group_et_pupil_mean_mean`, `group_et_blink_rate_mean`

---

## 2. Priority 1: Quick Wins (1-2 days)

### 2.1 NRC Emotion Lexicon (Free)
**Source:** https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

| Feature | Description |
|---------|-------------|
| `lex_nrc_anger` | Count of anger-associated words |
| `lex_nrc_fear` | Count of fear-associated words |
| `lex_nrc_joy` | Count of joy-associated words |
| `lex_nrc_sadness` | Count of sadness-associated words |
| `lex_nrc_surprise` | Count of surprise-associated words |
| `lex_nrc_trust` | Count of trust-associated words |
| `lex_nrc_disgust` | Count of disgust-associated words |
| `lex_nrc_anticipation` | Count of anticipation-associated words |
| `lex_nrc_positive` | NRC positive valence |
| `lex_nrc_negative` | NRC negative valence |

**Implementation:**
```python
# Download NRC lexicon (research use)
# Parse TSV format: word \t emotion \t association(0/1)
from collections import defaultdict

def load_nrc_lexicon(path: str) -> dict[str, set[str]]:
    """Load NRC emotion lexicon."""
    emotion_words = defaultdict(set)
    with open(path) as f:
        for line in f:
            word, emotion, assoc = line.strip().split('\t')
            if assoc == '1':
                emotion_words[emotion].add(word.lower())
    return emotion_words
```

### 2.2 Discourse Markers (Free)
**Source:** Custom list based on Schiffrin (1987) and Fraser (1999)

| Feature | Description | Example words |
|---------|-------------|---------------|
| `lex_dm_elaboration` | Elaboration markers | "like", "I mean", "you know" |
| `lex_dm_contrast` | Contrastive markers | "but", "however", "although" |
| `lex_dm_causal` | Causal connectives | "because", "so", "therefore" |
| `lex_dm_additive` | Additive markers | "and", "also", "moreover" |
| `lex_dm_temporal` | Temporal markers | "then", "first", "finally" |

**Implementation:**
```python
DISCOURSE_MARKERS = {
    'elaboration': {'like', 'i mean', 'you know', 'basically', 'actually'},
    'contrast': {'but', 'however', 'although', 'though', 'whereas'},
    'causal': {'because', 'so', 'therefore', 'since', 'thus'},
    'additive': {'and', 'also', 'moreover', 'besides', 'furthermore'},
    'temporal': {'then', 'first', 'finally', 'next', 'meanwhile'},
}
```

### 2.3 Participation Balance Features
**Hypotheses:** H3.3a, H3.3d

| Feature | Description |
|---------|-------------|
| `lex_word_gini` | Gini coefficient of per-speaker word counts (0=equal, 1=monopoly) |
| `lex_turn_gini` | Gini coefficient of per-speaker turn counts |
| `lex_hhi` | Herfindahl-Hirschman Index of speaking share |

**Implementation:**
```python
def gini_coefficient(x: np.ndarray) -> float:
    """Calculate Gini coefficient for participation inequality."""
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n+1) * x))) / (n * np.sum(x)) - (n + 1) / n

def hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index (sum of squared shares)."""
    return np.sum(shares ** 2)
```

---

## 3. Priority 2: Lai & Murray Alignment (2-3 days)

### 3.1 LIWC-style Categories (Without License)
Use open alternatives or build custom dictionaries for key categories:

| Category | Description | Open Source |
|----------|-------------|-------------|
| Cognitive processes | Think, know, believe | Custom list |
| Social processes | We, they, friends | Custom list |
| Affective processes | Happy, sad, angry | NRC lexicon |
| Perceptual | See, hear, feel | Custom list |
| Relativity | Time, space, motion | Custom list |

**Note:** Full LIWC requires license (~$90 for academic use). Alternative: VADER for sentiment, NRC for emotion.

### 3.2 Speaking Rate & Fluency (From Audio)
If audio features are available at window level:

| Feature | Description |
|---------|-------------|
| `lex_words_per_second` | Speaking rate (word count / speaking duration) |
| `lex_filled_pause_rate` | "um", "uh", "er" per minute |
| `lex_repair_count` | Self-corrections and restarts |

### 3.3 Lexical Sophistication
| Feature | Description |
|---------|-------------|
| `lex_avg_word_length` | Mean characters per word |
| `lex_long_word_ratio` | % words > 6 characters |
| `lex_content_word_ratio` | Content words / total words (requires POS) |

---

## 4. Priority 3: Semantic & Embedding Features (3-5 days)

### 4.1 Sentence-BERT Embeddings
**Purpose:** Semantic similarity, topic coherence, accommodation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_features(utterances: list[str]) -> dict:
    """Compute embedding-based features for a window."""
    embeddings = model.encode(utterances)
    
    # Semantic alignment: cosine similarity between consecutive utterances
    from sklearn.metrics.pairwise import cosine_similarity
    if len(embeddings) > 1:
        consecutive_sim = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0,0] 
                          for i in range(len(embeddings)-1)]
        mean_alignment = np.mean(consecutive_sim)
        alignment_std = np.std(consecutive_sim)
    else:
        mean_alignment = alignment_std = 0.0
    
    # Intra-window coherence: mean pairwise similarity
    if len(embeddings) > 1:
        sim_matrix = cosine_similarity(embeddings)
        triu_idx = np.triu_indices(len(embeddings), k=1)
        coherence = np.mean(sim_matrix[triu_idx])
    else:
        coherence = 1.0
    
    return {
        'sem_mean_alignment': mean_alignment,
        'sem_alignment_std': alignment_std,
        'sem_coherence': coherence,
    }
```

**Features derived:**
| Feature | Description | Hypothesis |
|---------|-------------|------------|
| `sem_mean_alignment` | Mean consecutive-turn similarity | H3.4c (accommodation) |
| `sem_alignment_std` | Variability in alignment | Topic shift detection |
| `sem_coherence` | Intra-window topic coherence | H3.4d |

### 4.2 Cross-Speaker Lexical Accommodation
**Hypothesis:** H3.4c — lexical novelty injection precedes state transitions

| Feature | Description |
|---------|-------------|
| `lex_vocab_overlap` | Jaccard similarity of vocabulary between speaker pairs |
| `lex_echo_rate` | % words that appeared in previous speaker's turn |
| `lex_novelty_injection` | New vocabulary items not seen in prior 3 turns |

```python
def lexical_accommodation(prev_tokens: set, curr_tokens: set) -> dict:
    """Measure lexical accommodation between consecutive speakers."""
    if not prev_tokens or not curr_tokens:
        return {'vocab_overlap': 0, 'echo_rate': 0}
    
    overlap = len(prev_tokens & curr_tokens)
    jaccard = overlap / len(prev_tokens | curr_tokens)
    echo_rate = overlap / len(curr_tokens) if curr_tokens else 0
    
    return {
        'vocab_overlap': jaccard,
        'echo_rate': echo_rate,
    }
```

### 4.3 GoEmotions Classification (Optional, Heavy)
**Model:** `monologg/bert-base-cased-goemotions-original` (27 emotions)

Only worth implementing if prediction from NRC + custom features is insufficient.

---

## 5. Priority 4: Per-Speaker Features (for H3.3b, H3.3c)

### 5.1 Response Relevance
**Hypothesis:** H3.3b — being *responded to* predicts voice inclusion

| Feature | Description |
|---------|-------------|
| `per_speaker_response_relevance_P1` | Mean Jaccard of next turn with P1's last utterance |
| `per_speaker_response_relevance_P2` | ... |
| `per_speaker_response_relevance_P3` | ... |
| `per_speaker_response_relevance_P4` | ... |

### 5.2 Agreement Received
**Hypothesis:** H3.3c — receiving agreement predicts voice inclusion

Track who the agreement is *about* (the prior speaker):

| Feature | Description |
|---------|-------------|
| `per_speaker_agreement_received_P1` | Count of agreement markers following P1's turn |
| ... | ... |

---

## 6. Implementation Order

### Week 1: Foundation
1. ✅ **NRC Emotion Lexicon** — download, parse, integrate (1 day)
2. ✅ **Discourse markers** — define lists, count (0.5 day)
3. ✅ **Participation balance** — Gini, HHI (0.5 day)
4. ✅ **Lexical sophistication** — word length, long word ratio (0.5 day)

### Week 2: Semantic
5. ✅ **Sentence-BERT** — alignment, coherence (1 day)
6. ✅ **Lexical accommodation** — echo rate, novelty (1 day)
7. ✅ **Cross-speaker features** — response relevance, agreement received (1.5 days)

### Week 3: Integration & HMM
8. ✅ **Integrate into HMM input** — merge with existing 17 features
9. ✅ **Re-run HMM with BIC selection** — may find different k
10. ✅ **Compare prediction** — do new features improve R²?
11. ✅ **State interpretation** — do new lexical features disambiguate states?

---

## 7. Expected Feature Count Summary

| Category | Current | After Week 1 | After Week 2 | Total |
|----------|---------|--------------|--------------|-------|
| Conversation structure | 8 | 8 | 8 | 8 |
| Lexical (basic) | 14 | 14 | 14 | 14 |
| NRC emotions | 0 | 10 | 10 | 10 |
| Discourse markers | 0 | 5 | 5 | 5 |
| Participation balance | 0 | 3 | 3 | 3 |
| Lexical sophistication | 0 | 3 | 3 | 3 |
| Semantic (embedding) | 0 | 0 | 3 | 3 |
| Accommodation | 0 | 0 | 3 | 3 |
| Per-speaker (× 4) | 0 | 0 | 8 | 8 |
| Physio + ET | 8 | 8 | 8 | 8 |
| **Total** | **30** | **51** | **65** | **65** |

**Note:** With n=27 observations, you cannot use all 65 features directly. Strategy:
1. Use HMM to compress into k states (dimensionality reduction)
2. Use PCA on feature subsets
3. Use single-feature or 2-feature models for hypothesis testing
4. Compare: HMM(65 features) vs HMM(30 features) vs raw best-4

---

## 8. Lai & Murray (2018) vs. Your Thesis

| Aspect | Lai & Murray | Your Thesis |
|--------|--------------|-------------|
| **Corpus** | AMI (meeting transcripts) | AffectAI (group tasks) |
| **N** | ~130 meetings | 27 group×task observations |
| **Outcome** | Satisfaction (3 aspects) | Effectiveness (4 aspects) |
| **Modalities** | Acoustic, lexical, turn-taking | + Physiology, eye-tracking |
| **Approach** | SVM/regression on features | HMM states → regression |
| **Innovation** | Multimodal fusion | Latent state discovery as compression |

**Your thesis adds:**
1. **Latent state discovery** — HMM captures temporal patterns
2. **Physiology + eye-tracking** — richer multimodal signal
3. **Small-n methodology** — comparing HMM vs raw features under constraint

---

## 9. Notebook Structure

Create these notebooks in `icmi_paper/analysis/`:

1. `nrc_feature_extraction.ipynb` — NRC emotion lexicon integration
2. `semantic_feature_extraction.ipynb` — Sentence-BERT embeddings
3. `participation_balance_features.ipynb` — Gini, HHI, per-speaker
4. `hmm_extended_features.ipynb` — Re-run HMM with expanded feature set
5. `feature_comparison.ipynb` — Compare prediction with old vs new features

---

## 10. Next Immediate Action

Run this command to download NRC lexicon (requires manual acceptance of terms):

```bash
# Visit https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
# Download NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
# Place in configs/nrc_emotion_lexicon.txt
```

Then implement `analysis/extract_nrc_features.py`.
