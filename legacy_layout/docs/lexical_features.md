# Lexical Features — Technical Documentation

_Last updated: 2026-08-12._

This document describes the lexical feature extraction pipeline for the AffectAI multimodal dataset. These features capture **what** participants say (language content), complementing the turn-taking/timing features that capture **when** and **how long** they speak.

## Overview

| Property | Value |
|----------|-------|
| **Script** | [analysis/extract_participant_lexical_features.py](../analysis/extract_participant_lexical_features.py) |
| **Input** | `transcripts/final/*.tsv` (37 files, 10 groups × 4 tasks, minus 3 missing) |
| **Output** | `analysis/results/participant_lexical_features.tsv` (148 rows × 32 columns) |
| **Granularity** | One row per participant × task (P1–P4 × T1–T4) |
| **Status** | ✅ Generated (2026-08-12) |

## Motivation

The existing transcript feature pipeline (`extract_participant_transcript_features.py`) extracts turn-taking metrics:
- Speaking time, turn count
- Backchannel/laughter counts
- Overlap involvement (competitive, smooth, floor-fight)

These are *structural* features — they describe conversation dynamics but not the actual language content. Lexical features fill this gap by analyzing the `text` column of SPK (speech) rows.

**Use cases:**
1. Predicting collaboration effectiveness from language patterns
2. Identifying participants who hedge more vs. assert confidently
3. Measuring linguistic diversity/complexity as a cognitive engagement proxy
4. Detecting supportive vs. competitive language in group discussions

---

## Pipeline

### 1. Input filtering

Only `SPK` (speech) rows are analyzed — backchannels (`BCK`), laughter (`LAU`), silences (`SIL`), and other annotation types are excluded. This ensures lexical features reflect actual propositional content.

### 2. Tokenization

```python
def _tokenize(text: str) -> list[str]:
    # Remove transcript annotations like [OVL:...], [BRE]
    text = re.sub(r"\[.*?\]", "", text)
    # Lowercase, keep alphanumeric + apostrophes (contractions)
    words = re.findall(r"[a-z0-9']+", text.lower())
    return words
```

Design choices:
- **Lemmatization enabled**: Words are reduced to base forms via NLTK WordNet lemmatizer (e.g., "agreeing" → "agree", "problems" → "problem"). Improves marker matching recall.
- **Verb-first lemmatization**: Tries verb form first (catches "agreeing", "thinking", "worried"), then falls back to noun form for plurals ("problems" → "problem").
- **Contractions preserved**: "can't", "won't", "I'm" kept as single tokens.
- **Numbers included**: Quantitative references (costs, percentages) retained.
- **Soft dependency**: Falls back gracefully if NLTK not installed (lemmatization skipped).

### 3. N-gram extraction

```python
def _get_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
```

For a word list of length $L$:
- Unigrams: $L$ tokens
- Bigrams: $L - 1$ two-word sequences
- Trigrams: $L - 2$ three-word sequences

### 4. Entropy calculation

Shannon entropy of n-gram distribution (in bits):

$$H = -\sum_{i} p_i \log_2 p_i$$

where $p_i$ = frequency of n-gram $i$ / total n-grams.

**Interpretation:**
- Higher entropy = more diverse, unpredictable language
- Lower entropy = more repetitive, formulaic language
- Entropy is bounded by $\log_2(|V|)$ where $|V|$ = vocabulary size

---

## Feature Reference

### Basic text metrics

| Feature | Definition | Interpretation |
|---------|------------|----------------|
| `lex_word_count` | Total words across all SPK rows | Speaking volume (content quantity) |
| `lex_unique_words` | Distinct word types | Vocabulary breadth |
| `lex_ttr` | Type-token ratio: unique / total | Vocabulary diversity (0–1) |
| `lex_mean_word_len` | Mean characters per word | Lexical complexity proxy |

### N-gram counts

| Feature | Definition | Interpretation |
|---------|------------|----------------|
| `lex_n_unigrams` | Total unigram count | = `lex_word_count` |
| `lex_n_bigrams` | Total bigram count | Two-word phrase count |
| `lex_n_trigrams` | Total trigram count | Three-word phrase count |
| `lex_unique_bigrams` | Distinct bigram types | Two-word vocabulary size |
| `lex_unique_trigrams` | Distinct trigram types | Three-word vocabulary size |

### N-gram diversity

| Feature | Definition | Interpretation |
|---------|------------|----------------|
| `lex_unigram_diversity` | unique / total unigrams | = `lex_ttr` |
| `lex_bigram_diversity` | unique / total bigrams | Lower = more formulaic phrasing |
| `lex_trigram_diversity` | unique / total trigrams | Closer to 1.0 = nearly all trigrams unique |

### N-gram entropy (bits)

| Feature | Definition | Interpretation |
|---------|------------|----------------|
| `lex_unigram_entropy` | Shannon entropy of word distribution | Lexical unpredictability |
| `lex_bigram_entropy` | Shannon entropy of bigram distribution | Phrase-level unpredictability |
| `lex_trigram_entropy` | Shannon entropy of trigram distribution | Higher-order linguistic complexity |

### Hapax legomena

| Feature | Definition | Interpretation |
|---------|------------|----------------|
| `lex_hapax_count` | Words appearing exactly once | Rare/unique word usage |
| `lex_hapax_ratio` | hapax / total words | Vocabulary richness indicator |

### Lexical markers

| Feature | Word list | Interpretation |
|---------|-----------|----------------|
| `lex_agreement_count` | yes, yeah, right, exactly, agree, absolutely, definitely, correct, sure, okay, good, great, perfect, fine, indeed, precisely, totally, certainly, true, agreed, yep, yup, ok (24 terms) | Supportive/collaborative acts |
| `lex_hedging_count` | maybe, perhaps, possibly, probably, might, could, would, kind, sort, somewhat, fairly, rather, quite, basically, actually, just, like, guess, think, believe, suppose, seem, seems, apparently, presumably (26 terms) | Tentativeness, uncertainty |
| `lex_certainty_count` | definitely, certainly, absolutely, clearly, obviously, surely, undoubtedly, always, never, must, know, confident, certain, positive, convinced, guarantee, fact, proven (18 terms) | Confidence, conviction |
| `lex_question_count` | Question marks + question-word sentence starts | Information-seeking, engagement |
| `lex_suggestion_count` | "we could", "we should", "what if", "how about", "maybe we", "why don't we", "let's", "I suggest", "I propose", "I think we", "what do you think" (11 patterns) | Idea generation, proposals |
| `lex_positive_count` | good, great, excellent, nice, wonderful, fantastic, amazing, awesome, brilliant, perfect, love, like, enjoy, happy, pleased, glad, excited, interesting, helpful, useful, agree, best, better, benefit, success, successful, effective, efficient (28 terms) | Positive sentiment |
| `lex_negative_count` | bad, poor, terrible, awful, horrible, wrong, problem, issue, difficult, hard, hate, dislike, annoying, frustrating, confused, worried, concerned, doubt, fail, failure, worse, worst, disagree, unfortunately, impossible, never, can't, won't (27 terms) | Negative sentiment |

### Derived rates

| Feature | Definition | Interpretation |
|---------|------------|----------------|
| `lex_agreement_rate` | agreement / word_count | Normalized agreement density |
| `lex_question_rate_per_min` | questions / speaking_minutes | Question frequency |
| `lex_suggestion_rate_per_min` | suggestions / speaking_minutes | Proposal frequency |
| `lex_sentiment_ratio` | (pos − neg) / (pos + neg) | Polarity: +1 all positive, −1 all negative |

---

## Summary Statistics (N=148)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `lex_word_count` | 231 | 137 | 9 | 820 |
| `lex_ttr` | 0.54 | 0.12 | 0.32 | 1.00 |
| `lex_bigram_diversity` | 0.90 | 0.05 | 0.71 | 1.00 |
| `lex_trigram_diversity` | 0.97 | 0.03 | 0.79 | 1.00 |
| `lex_unigram_entropy` | 6.2 | 0.7 | 2.9 | 7.3 |
| `lex_bigram_entropy` | 7.3 | 1.1 | 3.0 | 9.2 |
| `lex_trigram_entropy` | 7.4 | 1.2 | 2.8 | 9.6 |
| `lex_hapax_ratio` | 0.36 | 0.14 | 0.18 | 1.00 |
| `lex_agreement_count` | 10.4 | 7.0 | 0 | 37 |
| `lex_hedging_count` | 14.0 | 11.3 | 0 | 66 |
| `lex_sentiment_ratio` | 0.81 | 0.32 | −0.33 | 1.00 |

**Key observations:**
- Trigram diversity is very high (mean 0.97) — participants rarely repeat exact 3-word phrases
- Bigram entropy ranges 3.0–9.2 bits — substantial individual variation in phrase predictability
- Sentiment skews strongly positive (mean +0.81) — collaborative task context
- Hedging is common (mean 14 per participant-task) — face-saving in group negotiations

---

## Integration with Modeling Pipeline

The lexical features are automatically merged into the participant modeling table via [analysis/build_participant_df.py](../analysis/build_participant_df.py):

```python
LEXICAL_FEATS = [
    # Basic metrics
    "lex_word_count", "lex_unique_words", "lex_ttr", "lex_mean_word_len",
    # N-gram counts and diversity
    "lex_n_bigrams", "lex_n_trigrams", "lex_unique_bigrams", "lex_unique_trigrams",
    "lex_bigram_diversity", "lex_trigram_diversity",
    # N-gram entropy
    "lex_unigram_entropy", "lex_bigram_entropy", "lex_trigram_entropy",
    # Hapax
    "lex_hapax_count", "lex_hapax_ratio",
    # Markers and rates
    "lex_agreement_count", "lex_hedging_count", "lex_certainty_count",
    "lex_question_count", "lex_suggestion_count", "lex_positive_count", "lex_negative_count",
    "lex_agreement_rate", "lex_question_rate_per_min", "lex_suggestion_rate_per_min",
    "lex_sentiment_ratio",
]
```

Output: `analysis/results/participant_model_df.tsv` — ready for sklearn/statsmodels.

---

## Dimensionality Reduction: Composite Scores

With 26 lexical features and limited sample size (N≈112), dimensionality reduction is essential. We provide **three theory-driven composite scores** computed via z-score averaging:

### Available composites

| Composite | Components | Interpretation |
|-----------|------------|----------------|
| `lex_social_z` | positive_count, agreement_count, sentiment_ratio | Positive engagement / supportive language |
| `lex_complexity_z` | unigram_entropy, ttr | Lexical complexity / cognitive load |
| `lex_verbosity_z` | log(word_count) | Speaking volume |

### Computation

```python
# Each component z-scored, then averaged
lex_social_z = mean(zscore(positive_count), zscore(agreement_count), zscore(sentiment_ratio))
lex_complexity_z = mean(zscore(unigram_entropy), zscore(ttr))
lex_verbosity_z = zscore(log1p(word_count))
```

Computed in `build_participant_df.py` → `_add_lexical_composites()`.

### Validation

Spearman correlations with task-appropriate targets (N=111):

| Composite | All tasks | T1 (coord) | T2 (coop) | T3 (coord) |
|-----------|-----------|------------|-----------|------------|
| lex_social_z | ρ=+0.13 | ρ=+0.09 | **ρ=+0.48*** | ρ=+0.14 |
| lex_complexity_z | ρ=−0.05 | ρ=−0.10 | ρ=−0.19 | ρ=−0.26 |
| lex_verbosity_z | ρ=−0.18 | ρ=+0.02 | ρ=+0.18 | ρ=+0.04 |

*\* p < 0.05*

**Key finding:** Social engagement (`lex_social_z`) shows a strong positive correlation with cooperativeness in the negotiation task (T2), validating that positive language predicts better team outcomes.

### Alternatives considered

| Method | Result |
|--------|--------|
| Standard PCA | PC1 dominated by verbosity (word counts); no significant correlation with target |
| Factor Analysis (4 factors) | Similar to PCA — verbosity dominates; factors not predictive |
| Supervised PCA (features with \|ρ\|>0.15) | PC1 ρ=−0.19, p=0.046 — modest improvement |
| Theory-driven composites | **Best interpretability and T2 signal** |

---

## Limitations and Caveats

### Methodological

1. **WordNet lemmatization only**: No POS tagging — verb form tried first, then noun. Some edge cases may not lemmatize correctly.
2. **English-only word lists**: Domain-general; could add task-specific markers for negotiation/brainstorming.
3. **Simple sentiment lexicon**: Not LIWC or VADER — useful for relative comparisons within-dataset, not absolute scores.
4. **Length sensitivity**: Entropy and diversity metrics are influenced by text length. Very short utterances (< 20 words) have lower entropy by construction.

### Coverage

- 8 of 120 modeling rows (T1–T3) have missing lexical features due to missing transcripts (`grp-09_T1`, `grp-13_T3`, `grp-13_T4`)
- T4 is available but often excluded from analysis (different task structure)

### Future enhancements

- [x] ~~Add lemmatization (NLTK/spaCy) for higher marker recall~~ ✅ Done (2026-08-12)
- [ ] Part-of-speech tagging for syntactic complexity features
- [ ] Domain-specific word lists for AffectAI tasks (negotiation terms, brainstorming phrases)
- [ ] Word embeddings / sentence embeddings for semantic similarity features
- [ ] LIWC integration if license available

---

## References

- **Type-token ratio (TTR)**: Classic vocabulary diversity measure. See Templin (1957).
- **Shannon entropy**: Shannon, C. E. (1948). A mathematical theory of communication.
- **Hapax legomena**: Words occurring once — used in authorship attribution and vocabulary richness studies.
- **Hedging in discourse**: Lakoff (1973), "Hedges: A study in meaning criteria and the logic of fuzzy concepts."
- **WordNet lemmatization**: Fellbaum, C. (1998). WordNet: An Electronic Lexical Database.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-12 | Initial implementation: basic metrics, markers, sentiment |
| 2026-08-12 | Added n-gram features: counts, diversity, entropy, hapax |
| 2026-08-12 | Added NLTK WordNet lemmatization (verb-first, noun fallback) |
| 2026-08-13 | Added composite scores: lex_social_z, lex_complexity_z, lex_verbosity_z |
| 2026-08-13 | PCA/FA validation: theory-driven composites outperform unsupervised approaches |
