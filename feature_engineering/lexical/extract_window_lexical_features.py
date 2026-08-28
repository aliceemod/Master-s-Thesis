"""Window-level (30s) lexical features for HMM integration.

Bins transcript SPK rows into 30-second windows (matching the physio/ET/HMM pipeline),
extracts lexical features per window, and aggregates across P1-P4 to produce group-level
features compatible with the collective feature matrix.

Output granularity: One row per group × task × window_index.

This complements the participant×task lexical features in `extract_participant_lexical_features.py`
by providing the temporal resolution needed for HMM latent state discovery.

Output: analysis/results/lexical_window_30s.tsv

Usage:
    python analysis/extract_window_lexical_features.py
    python analysis/extract_window_lexical_features.py --groups grp-07 grp-10 --tasks T1 T2
    python analysis/extract_window_lexical_features.py --verbose
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Optional NLTK lemmatizer — graceful fallback if not installed
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    _LEMMATIZER = WordNetLemmatizer()
    _LEMMATIZE = True
except ImportError:
    _LEMMATIZER = None
    _LEMMATIZE = False

# Optional VADER sentiment — graceful fallback if not installed
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    _VADER = None
    _VADER_AVAILABLE = False

# Optional spaCy — for TIER 3 syntactic complexity features
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    except OSError:
        # Model not installed — try downloading it
        from spacy.cli import download
        download("en_core_web_sm")
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    _SPACY_AVAILABLE = True
except (ImportError, OSError):
    _NLP = None
    _SPACY_AVAILABLE = False

LOG = logging.getLogger("extract_window_lexical_features")

# ── Configuration ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "transcripts" / "final"
OUT_PATH = REPO_ROOT / "analysis" / "results" / "lexical_window_30s.tsv"
LEXICONS_DIR = REPO_ROOT / "configs" / "lexicons"

DEFAULT_GROUPS = [f"grp-{i:02d}" for i in range(7, 17)]  # grp-07..grp-16
DEFAULT_TASKS = ["T1", "T2", "T3"]
PARTICIPANTS = ("P1", "P2", "P3", "P4")
WINDOW_S = 30.0

_FNAME_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)

# ── Concreteness lexicon (Brysbaert et al. 2014) ───────────────────────────────
# Reference: Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). 
# Concreteness ratings for 40 thousand generally known English word lemmas.
# Behavior Research Methods, 46(3), 904-911.

_CONCRETENESS: dict[str, float] = {}
_CONCRETENESS_AVAILABLE = False


def _load_concreteness_lexicon() -> bool:
    """Load Brysbaert concreteness ratings from local file or download.
    
    Returns True if lexicon loaded successfully, False otherwise.
    Scale: 1 (abstract) to 5 (concrete).
    """
    global _CONCRETENESS, _CONCRETENESS_AVAILABLE
    
    lexicon_path = LEXICONS_DIR / "brysbaert_concreteness.tsv"
    
    if lexicon_path.exists():
        try:
            df = pd.read_csv(lexicon_path, sep="\t")
            _CONCRETENESS = dict(zip(df["Word"].str.lower(), df["Conc.M"]))
            _CONCRETENESS_AVAILABLE = True
            LOG.info("Loaded concreteness lexicon: %d words", len(_CONCRETENESS))
            return True
        except Exception as e:
            LOG.warning("Failed to load concreteness lexicon: %s", e)
            return False
    
    # Try to download from web
    url = "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
    LOG.info("Downloading concreteness lexicon from %s", url)
    
    try:
        import urllib.request
        LEXICONS_DIR.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")
        
        # Parse the TSV content
        lines = content.strip().split("\n")
        header = lines[0].split("\t")
        word_idx = header.index("Word")
        conc_idx = header.index("Conc.M")
        
        with open(lexicon_path, "w", encoding="utf-8") as f:
            f.write("Word\tConc.M\n")
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) > max(word_idx, conc_idx):
                    word = parts[word_idx].strip().lower()
                    try:
                        conc = float(parts[conc_idx])
                        f.write(f"{word}\t{conc}\n")
                        _CONCRETENESS[word] = conc
                    except ValueError:
                        continue
        
        _CONCRETENESS_AVAILABLE = True
        LOG.info("Downloaded and saved concreteness lexicon: %d words", len(_CONCRETENESS))
        return True
        
    except Exception as e:
        LOG.warning("Failed to download concreteness lexicon: %s", e)
        LOG.warning("Concreteness features will be NaN. To fix: manually download from %s", url)
        return False


# ── Word lists (same as participant-level extractor) ──────────────────────────

AGREEMENT_WORDS = {
    "yes", "yeah", "yep", "yup", "right", "exactly", "absolutely", "definitely",
    "agree", "true", "correct", "sure", "okay", "ok", "good", "great",
    "perfect", "fine", "indeed", "precisely", "totally", "certainly",
}

HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could", "would",
    "kind", "sort", "somewhat", "fairly", "rather", "quite", "basically",
    "actually", "just", "like", "guess", "think", "believe", "suppose",
    "seem", "apparently", "presumably",
}

CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "clearly", "obviously", "surely",
    "undoubtedly", "always", "never", "must", "know", "confident", "certain",
    "positive", "convinced", "guarantee", "fact", "proven",
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "nice", "wonderful", "fantastic", "amazing",
    "awesome", "brilliant", "perfect", "love", "like", "enjoy", "happy",
    "pleased", "glad", "excited", "interesting", "helpful", "useful", "agree",
    "best", "better", "benefit", "success", "successful", "effective", "efficient",
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "horrible", "wrong", "problem", "issue",
    "difficult", "hard", "hate", "dislike", "annoying", "frustrating", "confused",
    "worried", "concerned", "doubt", "fail", "failure", "worse", "worst",
    "disagree", "unfortunately", "impossible", "never", "can't", "won't",
}

QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose"}

SUGGESTION_PATTERNS = [
    r"\bwe could\b", r"\bwe should\b", r"\bwhat if\b", r"\bhow about\b",
    r"\bmaybe we\b", r"\bwhy don'?t we\b", r"\blet'?s\b", r"\bi suggest\b",
    r"\bi propose\b", r"\bi think we\b", r"\bwhat do you think\b",
]
_SUGGESTION_RE = re.compile("|".join(SUGGESTION_PATTERNS), re.IGNORECASE)

# ── TIER 1: Discourse Markers (Schiffrin 1987, Fraser 1999) ────────────────────

DISCOURSE_MARKERS = {
    "elaboration": {"like", "basically", "actually", "essentially", "specifically"},
    "contrast": {"but", "however", "although", "though", "whereas", "yet", "instead"},
    "causal": {"because", "so", "therefore", "since", "thus", "hence", "consequently"},
    "additive": {"and", "also", "moreover", "besides", "furthermore", "additionally"},
    "temporal": {"then", "first", "finally", "next", "meanwhile", "afterwards", "before"},
}
# Multi-word discourse markers (require regex)
MULTIWORD_DM = {
    "elaboration": [r"\bi mean\b", r"\byou know\b", r"\bin other words\b"],
    "contrast": [r"\bon the other hand\b", r"\bin contrast\b"],
    "causal": [r"\bas a result\b", r"\bdue to\b"],
    "additive": [r"\bin addition\b", r"\bas well\b"],
    "temporal": [r"\bat first\b", r"\bin the end\b", r"\bafter that\b"],
}
_MULTIWORD_DM_RE = {
    cat: re.compile("|".join(patterns), re.IGNORECASE)
    for cat, patterns in MULTIWORD_DM.items()
}

# ── TIER 1: Pronouns for We/I ratio ────────────────────────────────────────────

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}

# ── Helper functions ───────────────────────────────────────────────────────────


def _normalize_group_id(raw: str) -> str:
    """Normalize e.g. 'grp8'/'grp-08' to 'grp-08'."""
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"


def _lemmatize_word(word: str) -> str:
    """Lemmatize a word to its base form using WordNet."""
    if not _LEMMATIZE or _LEMMATIZER is None:
        return word
    lemma = _LEMMATIZER.lemmatize(word, "v")
    if lemma != word:
        return lemma
    return _LEMMATIZER.lemmatize(word, "n")


def _tokenize(text: str, lemmatize: bool = True) -> list[str]:
    """Word tokenization with optional lemmatization."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r"\[.*?\]", "", text)
    words = re.findall(r"[a-z0-9']+", text.lower())
    words = [w for w in words if len(w) > 0]
    if lemmatize and _LEMMATIZE:
        words = [_lemmatize_word(w) for w in words]
    return words


def _get_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from a word list."""
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def _ngram_entropy(ngrams: list[tuple[str, ...]]) -> float:
    """Calculate Shannon entropy of n-gram distribution (bits)."""
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total = len(ngrams)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _count_markers(words: list[str], marker_set: set[str]) -> int:
    """Count how many words appear in a marker set."""
    return sum(1 for w in words if w in marker_set)


def _count_questions(text: str) -> int:
    """Count question marks."""
    if not isinstance(text, str):
        return 0
    return text.count("?")


def _count_suggestions(text: str) -> int:
    """Count suggestion/proposal phrases."""
    if not isinstance(text, str):
        return 0
    return len(_SUGGESTION_RE.findall(text))


def _compute_concreteness(words: list[str]) -> dict[str, float]:
    """Compute concreteness features using Brysbaert et al. (2014) norms.
    
    Returns dict with:
    - lex_concreteness_mean: Mean concreteness (1=abstract, 5=concrete)
    - lex_concreteness_std: Standard deviation of concreteness
    - lex_concreteness_coverage: Proportion of words found in lexicon
    
    Reference: Lai & Murray (2018) used concreteness from MRC psycholinguistic database.
    """
    if not _CONCRETENESS_AVAILABLE or not words:
        return {
            "lex_concreteness_mean": float("nan"),
            "lex_concreteness_std": float("nan"),
            "lex_concreteness_coverage": float("nan"),
        }
    
    # Look up concreteness for each word
    concreteness_vals = []
    for word in words:
        word_lower = word.lower()
        if word_lower in _CONCRETENESS:
            concreteness_vals.append(_CONCRETENESS[word_lower])
    
    if not concreteness_vals:
        return {
            "lex_concreteness_mean": float("nan"),
            "lex_concreteness_std": float("nan"),
            "lex_concreteness_coverage": 0.0,
        }
    
    import numpy as np
    return {
        "lex_concreteness_mean": float(np.mean(concreteness_vals)),
        "lex_concreteness_std": float(np.std(concreteness_vals)),
        "lex_concreteness_coverage": len(concreteness_vals) / len(words),
    }


def _compute_vader_sentiment(text: str) -> dict[str, float]:
    """Compute VADER sentiment scores for text.
    
    Returns dict with:
    - vader_compound: Overall sentiment (-1 to +1)
    - vader_pos: Positive proportion (0 to 1)
    - vader_neg: Negative proportion (0 to 1)
    - vader_neu: Neutral proportion (0 to 1)
    
    Falls back to NaN if VADER not available or text is empty.
    """
    if not _VADER_AVAILABLE or not isinstance(text, str) or not text.strip():
        return {
            "lex_vader_compound": float("nan"),
            "lex_vader_pos": float("nan"),
            "lex_vader_neg": float("nan"),
            "lex_vader_neu": float("nan"),
        }
    
    scores = _VADER.polarity_scores(text)
    return {
        "lex_vader_compound": scores["compound"],
        "lex_vader_pos": scores["pos"],
        "lex_vader_neg": scores["neg"],
        "lex_vader_neu": scores["neu"],
    }


# ── TIER 1: New helper functions ───────────────────────────────────────────────


def _count_discourse_markers(words: list[str], text: str) -> dict[str, int]:
    """Count discourse markers by category (single-word + multi-word).
    
    Returns dict with keys: dm_elaboration, dm_contrast, dm_causal, dm_additive, dm_temporal
    """
    counts = {}
    for category, word_set in DISCOURSE_MARKERS.items():
        # Single-word markers
        single_count = sum(1 for w in words if w in word_set)
        # Multi-word markers (regex)
        multi_count = len(_MULTIWORD_DM_RE[category].findall(text)) if text else 0
        counts[f"lex_dm_{category}"] = single_count + multi_count
    return counts


def _compute_pronoun_features(words: list[str]) -> dict[str, float]:
    """Compute We/I pronoun ratio and counts.
    
    Higher we_i_ratio suggests collective orientation; lower suggests individualism.
    """
    i_count = sum(1 for w in words if w in FIRST_PERSON_SINGULAR)
    we_count = sum(1 for w in words if w in FIRST_PERSON_PLURAL)
    
    # Ratio: we / (we + i), bounded [0, 1], NaN if no pronouns
    total_pronouns = i_count + we_count
    if total_pronouns == 0:
        we_i_ratio = float("nan")
    else:
        we_i_ratio = we_count / total_pronouns
    
    return {
        "lex_i_count": i_count,
        "lex_we_count": we_count,
        "lex_we_i_ratio": we_i_ratio,
    }


# ── TIER 3: Syntactic Complexity ───────────────────────────────────────────────

def _compute_syntactic_complexity(text: str) -> dict[str, float]:
    """Compute syntactic complexity features using spaCy dependency parsing.
    
    Features:
    - parse_depth_mean: Mean depth of dependency trees (deeper = more embedded clauses)
    - clause_density: Clauses per sentence (subordinate + main)
    - subordination_ratio: Proportion of subordinate clauses
    
    Gracefully returns NaN if spaCy is unavailable.
    
    Reference: Lu, X. (2010). Automatic analysis of syntactic complexity in 
    second language writing. International journal of corpus linguistics, 15(4), 474-496.
    """
    if not _SPACY_AVAILABLE or _NLP is None:
        return {
            "lex_parse_depth_mean": float("nan"),
            "lex_clause_density": float("nan"),
            "lex_subordination_ratio": float("nan"),
        }
    
    if not isinstance(text, str) or not text.strip():
        return {
            "lex_parse_depth_mean": float("nan"),
            "lex_clause_density": float("nan"),
            "lex_subordination_ratio": float("nan"),
        }
    
    doc = _NLP(text)
    
    # Parse depth: max distance from any token to root in each sentence
    depths = []
    for sent in doc.sents:
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            depths.append(depth)
    
    parse_depth_mean = sum(depths) / len(depths) if depths else float("nan")
    
    # Clause detection via clausal subjects/complements and subordinate markers
    # SBAR-like markers: ccomp, advcl, acl, relcl (clausal complements, adverbial clauses, relative clauses)
    subordinate_deps = {"ccomp", "advcl", "acl", "relcl", "xcomp"}
    main_clause_deps = {"ROOT"}
    
    subordinate_count = sum(1 for token in doc if token.dep_ in subordinate_deps)
    main_clause_count = sum(1 for sent in doc.sents for _ in [1])  # One main clause per sentence
    
    total_clauses = subordinate_count + main_clause_count
    n_sentences = sum(1 for _ in doc.sents)
    
    clause_density = total_clauses / n_sentences if n_sentences > 0 else float("nan")
    subordination_ratio = subordinate_count / total_clauses if total_clauses > 0 else 0.0
    
    return {
        "lex_parse_depth_mean": parse_depth_mean,
        "lex_clause_density": clause_density,
        "lex_subordination_ratio": subordination_ratio,
    }


def _gini_coefficient(values: list[float]) -> float:
    """Calculate Gini coefficient for participation inequality.
    
    0 = perfect equality (everyone speaks equally)
    1 = perfect inequality (one person speaks all)
    """
    import numpy as np
    
    if not values or sum(values) == 0:
        return float("nan")
    
    x = np.array(sorted(values), dtype=float)
    n = len(x)
    if n == 0:
        return float("nan")
    
    cumsum = np.cumsum(x)
    return (2 * np.sum((np.arange(1, n + 1) * x)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def _hhi(shares: list[float]) -> float:
    """Herfindahl-Hirschman Index (sum of squared shares).
    
    Measures concentration: 0.25 = perfectly equal (4 speakers), 1 = monopoly
    """
    import numpy as np
    
    if not shares or sum(shares) == 0:
        return float("nan")
    
    total = sum(shares)
    proportions = np.array(shares) / total
    return float(np.sum(proportions ** 2))


def _lexical_sophistication(words: list[str]) -> dict[str, float]:
    """Compute lexical sophistication features.
    
    - long_word_ratio: % words > 6 characters (proxy for vocabulary sophistication)
    - avg_syllables: estimated syllables per word (simple heuristic)
    """
    if not words:
        return {
            "lex_long_word_ratio": float("nan"),
        }
    
    long_words = sum(1 for w in words if len(w) > 6)
    long_word_ratio = long_words / len(words)
    
    return {
        "lex_long_word_ratio": long_word_ratio,
    }


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _extract_window_features(words: list[str], all_text: str) -> dict:
    """Extract lexical features from a set of words and raw text.
    
    Returns a dict of feature name -> value for a single window.
    """
    word_count = len(words)
    unique_words = len(set(words))
    
    if word_count == 0:
        # Return zeros/NaNs for empty window
        return {
            "lex_word_count": 0,
            "lex_unique_words": 0,
            "lex_ttr": float("nan"),
            "lex_mean_word_len": float("nan"),
            "lex_bigram_entropy": float("nan"),
            "lex_trigram_entropy": float("nan"),
            "lex_agreement_count": 0,
            "lex_hedging_count": 0,
            "lex_certainty_count": 0,
            "lex_positive_count": 0,
            "lex_negative_count": 0,
            "lex_question_count": 0,
            "lex_suggestion_count": 0,
            "lex_sentiment_ratio": float("nan"),
            "lex_social_composite": float("nan"),
            # TIER 1: Discourse markers
            "lex_dm_elaboration": 0,
            "lex_dm_contrast": 0,
            "lex_dm_causal": 0,
            "lex_dm_additive": 0,
            "lex_dm_temporal": 0,
            # TIER 1: Pronouns
            "lex_i_count": 0,
            "lex_we_count": 0,
            "lex_we_i_ratio": float("nan"),
            # TIER 1: Lexical sophistication
            "lex_long_word_ratio": float("nan"),
            # TIER 2: VADER sentiment (replaces simple word-list sentiment)
            "lex_vader_compound": float("nan"),
            "lex_vader_pos": float("nan"),
            "lex_vader_neg": float("nan"),
            "lex_vader_neu": float("nan"),
            # TIER 2: Concreteness (Brysbaert et al. 2014, used by Lai & Murray)
            "lex_concreteness_mean": float("nan"),
            "lex_concreteness_std": float("nan"),
            "lex_concreteness_coverage": float("nan"),
            # TIER 3: Syntactic complexity (spaCy dependency parsing)
            "lex_parse_depth_mean": float("nan"),
            "lex_clause_density": float("nan"),
            "lex_subordination_ratio": float("nan"),
        }
    
    # Basic metrics
    ttr = unique_words / word_count
    mean_word_len = sum(len(w) for w in words) / word_count
    
    # N-gram entropy
    bigrams = _get_ngrams(words, 2)
    trigrams = _get_ngrams(words, 3)
    bigram_entropy = _ngram_entropy(bigrams)
    trigram_entropy = _ngram_entropy(trigrams)
    
    # Marker counts
    agreement_count = _count_markers(words, AGREEMENT_WORDS)
    hedging_count = _count_markers(words, HEDGING_WORDS)
    certainty_count = _count_markers(words, CERTAINTY_WORDS)
    positive_count = _count_markers(words, POSITIVE_WORDS)
    negative_count = _count_markers(words, NEGATIVE_WORDS)
    question_count = _count_questions(all_text)
    suggestion_count = _count_suggestions(all_text)
    
    # Derived
    sentiment_ratio = (positive_count - negative_count) / (positive_count + negative_count + 1)
    
    # Social composite (agreement + positive normalized by word count)
    social_composite = (agreement_count + positive_count) / word_count if word_count > 0 else float("nan")
    
    # TIER 1: Discourse markers
    dm_counts = _count_discourse_markers(words, all_text)
    
    # TIER 1: Pronoun features (We/I ratio)
    pronoun_feats = _compute_pronoun_features(words)
    
    # TIER 1: Lexical sophistication
    sophistication_feats = _lexical_sophistication(words)
    
    # TIER 2: VADER sentiment (handles negation, intensifiers)
    vader_feats = _compute_vader_sentiment(all_text)
    
    # TIER 2: Concreteness (Brysbaert et al. 2014 - used by Lai & Murray 2018)
    concreteness_feats = _compute_concreteness(words)
    
    # TIER 3: Syntactic complexity (spaCy dependency parsing)
    syntactic_feats = _compute_syntactic_complexity(all_text)
    
    result = {
        "lex_word_count": word_count,
        "lex_unique_words": unique_words,
        "lex_ttr": ttr,
        "lex_mean_word_len": mean_word_len,
        "lex_bigram_entropy": bigram_entropy,
        "lex_trigram_entropy": trigram_entropy,
        "lex_agreement_count": agreement_count,
        "lex_hedging_count": hedging_count,
        "lex_certainty_count": certainty_count,
        "lex_positive_count": positive_count,
        "lex_negative_count": negative_count,
        "lex_question_count": question_count,
        "lex_suggestion_count": suggestion_count,
        "lex_sentiment_ratio": sentiment_ratio,
        "lex_social_composite": social_composite,
    }
    
    # Merge TIER 1 features
    result.update(dm_counts)
    result.update(pronoun_feats)
    result.update(sophistication_feats)
    
    # Merge TIER 2 features (VADER sentiment)
    result.update(vader_feats)
    
    # Merge TIER 2 features (Concreteness)
    result.update(concreteness_feats)
    
    # Merge TIER 3 features (Syntactic complexity)
    result.update(syntactic_feats)
    
    return result


def _bin_lexical_features(
    events: pd.DataFrame,
    window_s: float = WINDOW_S,
) -> pd.DataFrame:
    """Bin transcript SPK events into 30s windows and extract lexical features.
    
    Returns one row per (group_id, task_id, window_index) with group-aggregated
    lexical features (all P1-P4 speech combined per window).
    
    Window anchoring matches the physio/ET pipeline: window_index = floor(onset / 30).
    
    TIER 1 additions:
    - Participation balance (Gini coefficient, HHI)
    - Turn-to-turn lexical cohesion (mean Jaccard between consecutive turns)
    """
    if events.empty:
        return pd.DataFrame()
    
    rows: list[dict] = []
    
    for (grp, task), grp_df in events.groupby(["group_id", "task_id"]):
        # Filter to SPK rows from participants only
        spk_df = grp_df[
            (grp_df["type"] == "SPK") & 
            (grp_df["speaker"].isin(PARTICIPANTS))
        ].copy()
        
        if spk_df.empty:
            continue
        
        # Assign window indices (anchored to task start, matching physio/ET)
        spk_df["window_index"] = (spk_df["onset"] / window_s).apply(
            lambda x: int(x) if pd.notna(x) and x >= 0 else -1
        )
        spk_df = spk_df[spk_df["window_index"] >= 0]
        
        # Get window range for this task
        max_window = int(spk_df["window_index"].max()) if not spk_df.empty else 0
        
        # Extract features per window
        for win_idx in range(max_window + 1):
            win_df = spk_df[spk_df["window_index"] == win_idx]
            
            # Aggregate all text in this window (group-level)
            all_text = " ".join(win_df["text"].dropna().astype(str))
            words = _tokenize(all_text)
            
            # Extract features
            feats = _extract_window_features(words, all_text)
            
            # ── TIER 1: Participation balance (Gini, HHI) ──────────────────
            # Compute per-speaker word counts within this window
            speaker_word_counts = []
            for spk in PARTICIPANTS:
                spk_text = " ".join(
                    win_df[win_df["speaker"] == spk]["text"].dropna().astype(str)
                )
                spk_words = _tokenize(spk_text, lemmatize=False)
                speaker_word_counts.append(len(spk_words))
            
            feats["lex_word_gini"] = _gini_coefficient(speaker_word_counts)
            feats["lex_word_hhi"] = _hhi(speaker_word_counts)
            
            # ── TIER 1: Turn-to-turn lexical cohesion (mean Jaccard) ───────
            # Compute Jaccard similarity between consecutive turns
            if len(win_df) >= 2:
                sorted_turns = win_df.sort_values("onset")
                jaccard_scores = []
                prev_vocab = None
                for _, row in sorted_turns.iterrows():
                    turn_words = _tokenize(str(row.get("text", "")), lemmatize=False)
                    curr_vocab = set(turn_words)
                    if prev_vocab is not None and curr_vocab:
                        jaccard_scores.append(_jaccard_similarity(prev_vocab, curr_vocab))
                    prev_vocab = curr_vocab if curr_vocab else prev_vocab
                
                feats["lex_turn_cohesion"] = (
                    sum(jaccard_scores) / len(jaccard_scores)
                    if jaccard_scores else float("nan")
                )
            else:
                feats["lex_turn_cohesion"] = float("nan")
            
            # Add metadata
            feats["group_id"] = grp
            feats["task_id"] = task
            feats["window_index"] = win_idx
            feats["window_start_s"] = win_idx * window_s
            feats["n_spk_rows"] = len(win_df)
            feats["n_speakers"] = win_df["speaker"].nunique()
            
            rows.append(feats)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Reorder columns: metadata first, then features
    meta_cols = ["group_id", "task_id", "window_index", "window_start_s", "n_spk_rows", "n_speakers"]
    feat_cols = [c for c in df.columns if c.startswith("lex_")]
    df = df[meta_cols + sorted(feat_cols)]
    
    return df


def load_transcript(path: Path) -> pd.DataFrame | None:
    """Load a transcript TSV file."""
    try:
        df = pd.read_csv(path, sep="\t")
        # Ensure required columns
        required = {"onset", "duration", "speaker", "text", "type"}
        if not required.issubset(df.columns):
            LOG.warning("Missing columns in %s: %s", path.name, required - set(df.columns))
            return None
        return df
    except Exception as e:
        LOG.warning("Failed to load %s: %s", path.name, e)
        return None


def main(
    groups: list[str] | None = None,
    tasks: list[str] | None = None,
    verbose: bool = False,
) -> None:
    """Extract window-level lexical features from transcripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    
    if _LEMMATIZE:
        LOG.info("NLTK lemmatizer enabled")
    else:
        LOG.warning("NLTK not available — lemmatization disabled")
    
    if _VADER_AVAILABLE:
        LOG.info("VADER sentiment analyzer enabled")
    else:
        LOG.warning("VADER not available — sentiment features will be NaN (pip install vaderSentiment)")
    
    # Load concreteness lexicon (downloads on first use)
    _load_concreteness_lexicon()
    if _CONCRETENESS_AVAILABLE:
        LOG.info("Concreteness lexicon enabled (%d words)", len(_CONCRETENESS))
    else:
        LOG.warning("Concreteness lexicon not available — features will be NaN")
    
    groups = groups or DEFAULT_GROUPS
    tasks = tasks or DEFAULT_TASKS
    
    LOG.info("Groups: %s", groups)
    LOG.info("Tasks: %s", tasks)
    
    # Find and load transcripts
    all_events: list[pd.DataFrame] = []
    
    for tfile in sorted(TRANSCRIPT_DIR.glob("*.tsv")):
        match = _FNAME_RE.search(tfile.name)
        if not match:
            continue
        
        grp_raw, task = match.groups()
        grp = _normalize_group_id(grp_raw)
        
        if grp not in groups or task not in tasks:
            continue
        
        df = load_transcript(tfile)
        if df is None:
            continue
        
        df["group_id"] = grp
        df["task_id"] = task
        all_events.append(df)
        LOG.debug("Loaded %s: %d rows", tfile.name, len(df))
    
    if not all_events:
        LOG.error("No transcripts found in %s", TRANSCRIPT_DIR)
        return
    
    events = pd.concat(all_events, ignore_index=True)
    LOG.info("Loaded %d transcript rows from %d files", len(events), len(all_events))
    
    # Extract window-level features
    window_df = _bin_lexical_features(events, window_s=WINDOW_S)
    
    if window_df.empty:
        LOG.error("No windows extracted")
        return
    
    LOG.info("Extracted %d windows across %d groups × %d tasks",
             len(window_df), window_df["group_id"].nunique(), window_df["task_id"].nunique())
    
    # Summary stats
    LOG.info("Window range: %d–%d (mean %.1f)",
             window_df["window_index"].min(),
             window_df["window_index"].max(),
             window_df["window_index"].mean())
    LOG.info("Mean word count per window: %.1f (std %.1f)",
             window_df["lex_word_count"].mean(),
             window_df["lex_word_count"].std())
    
    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(OUT_PATH, sep="\t", index=False)
    LOG.info("Wrote %d rows → %s", len(window_df), OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="*", help="Groups to process (default: all)")
    parser.add_argument("--tasks", nargs="*", help="Tasks to process (default: T1 T2 T3)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    main(groups=args.groups, tasks=args.tasks, verbose=args.verbose)
