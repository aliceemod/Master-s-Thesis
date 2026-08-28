"""Per-participant lexical features from `transcripts/final/*.tsv`.

Extracts text-based features from actual speech content (the `text` column), complementing
the turn-taking/timing features in `extract_participant_transcript_features.py`. These
capture *what* participants say, not just *when* and *how long* they speak.

Feature categories:
1. Basic text metrics: word count, vocabulary richness (type-token ratio), mean word length
2. N-gram features: unigram/bigram/trigram counts, diversity (unique/total), entropy
3. Collaboration markers: agreement, hedging, certainty language
4. Question frequency: interrogatives per turn/minute
5. Idea/proposal language: suggestion phrases ("we could", "what if", etc.)
6. Sentiment proxies: positive/negative word ratios (simple lexicon-based)

Lemmatization: Words are lemmatized to base forms (via NLTK WordNet) to improve marker
matching recall (e.g., "agreeing" → "agree", "problems" → "problem").

Output: analysis/results/participant_lexical_features.tsv
"""
from __future__ import annotations

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
    # Ensure wordnet data is available
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if _LEMMATIZE:
    logger.info("NLTK lemmatizer enabled")
else:
    logger.warning("NLTK not available — lemmatization disabled (install with: pip install nltk)")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "transcripts" / "final"
OUT_PATH = REPO_ROOT / "analysis" / "results" / "participant_lexical_features.tsv"

_FNAME_RE = re.compile(r"transcript_(grp[_-]?\d+)_(T\d+)_", re.IGNORECASE)
PARTICIPANTS = {"P1", "P2", "P3", "P4"}

# --- Lexical marker word lists (base forms for lemmatized matching) ---
# Agreement/support markers (collaborative conversation signals)
AGREEMENT_WORDS = {
    "yes", "yeah", "yep", "yup", "right", "exactly", "absolutely", "definitely",
    "agree", "true", "correct", "sure", "okay", "ok", "good", "great",
    "perfect", "fine", "indeed", "precisely", "totally", "certainly",
}

# Hedging language (tentativeness, face-saving)
HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could", "would",
    "kind", "sort", "somewhat", "fairly", "rather", "quite", "basically",
    "actually", "just", "like", "guess", "think", "believe", "suppose",
    "seem", "apparently", "presumably",
}

# Certainty/confidence language
CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "clearly", "obviously", "surely",
    "undoubtedly", "always", "never", "must", "know", "confident", "certain",
    "positive", "convinced", "guarantee", "fact", "proven",
}

# Suggestion/proposal phrases (captured via regex patterns below)
SUGGESTION_PATTERNS = [
    r"\bwe could\b",
    r"\bwe should\b",
    r"\bwhat if\b",
    r"\bhow about\b",
    r"\bmaybe we\b",
    r"\bwhy don'?t we\b",
    r"\blet'?s\b",
    r"\bi suggest\b",
    r"\bi propose\b",
    r"\bi think we\b",
    r"\bwhat do you think\b",
]
_SUGGESTION_RE = re.compile("|".join(SUGGESTION_PATTERNS), re.IGNORECASE)

# Simple positive/negative sentiment lexicons (domain-neutral subset)
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

# Question markers
QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose"}


def _normalize_group_id(raw: str) -> str:
    """Normalize e.g. 'grp8'/'grp-08' to 'grp-08'."""
    digits = re.sub(r"^grp[_-]?", "", raw.lower())
    return f"grp-{int(digits):02d}"


def _lemmatize_word(word: str) -> str:
    """Lemmatize a word to its base form using WordNet.
    
    Tries verb lemmatization first (catches "agreeing" → "agree"), then noun.
    Falls back to original word if NLTK not available.
    """
    if not _LEMMATIZE or _LEMMATIZER is None:
        return word
    # Try verb form first (catches more variations like "agreeing", "thinking")
    lemma = _LEMMATIZER.lemmatize(word, "v")
    if lemma != word:
        return lemma
    # Fall back to noun form
    return _LEMMATIZER.lemmatize(word, "n")


def _tokenize(text: str, lemmatize: bool = True) -> list[str]:
    """Word tokenization with optional lemmatization.
    
    Args:
        text: Input text string
        lemmatize: If True (default), words are lemmatized to base forms
        
    Returns:
        List of lowercase tokens, optionally lemmatized
    """
    if not isinstance(text, str) or not text.strip():
        return []
    # Remove common transcript annotations like [OVL:...], [BRE], etc.
    text = re.sub(r"\[.*?\]", "", text)
    # Keep only alphanumeric and apostrophes (for contractions)
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
    """Calculate Shannon entropy of n-gram distribution (bits).
    
    Higher entropy = more diverse/unpredictable language.
    Lower entropy = more repetitive/formulaic language.
    """
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
    """Count question marks and question-word-initiated sentences."""
    if not isinstance(text, str):
        return 0
    q_marks = text.count("?")
    # Also count sentence starts with question words (rough heuristic)
    q_word_starts = len(re.findall(r"(?:^|[.!?]\s*)(" + "|".join(QUESTION_WORDS) + r")\b", text, re.IGNORECASE))
    return max(q_marks, q_word_starts)


def _count_suggestions(text: str) -> int:
    """Count suggestion/proposal phrases."""
    if not isinstance(text, str):
        return 0
    return len(_SUGGESTION_RE.findall(text))


def _extract_participant_features(df: pd.DataFrame, participant: str) -> dict:
    """Extract lexical features for one participant from their SPK rows."""
    # Filter to this participant's speech rows only
    p_df = df[(df["speaker"] == participant) & (df["type"] == "SPK")]
    
    # Concatenate all text
    all_text = " ".join(p_df["text"].dropna().astype(str))
    words = _tokenize(all_text)
    
    # Basic metrics (unigrams)
    word_count = len(words)  # total unigrams
    unique_words = len(set(words))  # unique unigrams
    ttr = unique_words / word_count if word_count > 0 else 0.0  # type-token ratio
    mean_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    
    # N-gram features
    unigrams = words  # already have these
    bigrams = _get_ngrams(words, 2)
    trigrams = _get_ngrams(words, 3)
    
    # N-gram counts
    n_unigrams = len(unigrams)
    n_bigrams = len(bigrams)
    n_trigrams = len(trigrams)
    
    # Unique n-grams (vocabulary at each level)
    unique_unigrams = len(set(unigrams))
    unique_bigrams = len(set(bigrams))
    unique_trigrams = len(set(trigrams))
    
    # N-gram diversity (unique / total) - vocabulary richness at each level
    unigram_diversity = unique_unigrams / n_unigrams if n_unigrams > 0 else 0.0
    bigram_diversity = unique_bigrams / n_bigrams if n_bigrams > 0 else 0.0
    trigram_diversity = unique_trigrams / n_trigrams if n_trigrams > 0 else 0.0
    
    # N-gram entropy (bits) - higher = more diverse/unpredictable language
    unigram_entropy = _ngram_entropy([(w,) for w in unigrams])
    bigram_entropy = _ngram_entropy(bigrams)
    trigram_entropy = _ngram_entropy(trigrams)
    
    # Hapax legomena (words appearing exactly once) - linguistic richness indicator
    word_counts = Counter(words)
    hapax_count = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = hapax_count / word_count if word_count > 0 else 0.0
    
    # Speaking time for rate calculations
    speaking_time_s = p_df["duration"].sum() if "duration" in p_df.columns else 0.0
    speaking_time_min = speaking_time_s / 60.0 if speaking_time_s > 0 else 0.0
    
    # Marker counts
    agreement_count = _count_markers(words, AGREEMENT_WORDS)
    hedging_count = _count_markers(words, HEDGING_WORDS)
    certainty_count = _count_markers(words, CERTAINTY_WORDS)
    positive_count = _count_markers(words, POSITIVE_WORDS)
    negative_count = _count_markers(words, NEGATIVE_WORDS)
    
    # Text-based counts (need original text, not just words)
    question_count = sum(_count_questions(t) for t in p_df["text"].dropna())
    suggestion_count = sum(_count_suggestions(t) for t in p_df["text"].dropna())
    
    # Derived rates
    agreement_rate = agreement_count / word_count if word_count > 0 else 0.0
    question_rate_per_min = question_count / speaking_time_min if speaking_time_min > 0 else 0.0
    suggestion_rate_per_min = suggestion_count / speaking_time_min if speaking_time_min > 0 else 0.0
    
    # Sentiment ratio: (positive - negative) / (positive + negative)
    sentiment_total = positive_count + negative_count
    sentiment_ratio = (positive_count - negative_count) / sentiment_total if sentiment_total > 0 else 0.0
    
    return {
        # Basic metrics
        "lex_word_count": word_count,
        "lex_unique_words": unique_words,
        "lex_ttr": round(ttr, 4),
        "lex_mean_word_len": round(mean_word_len, 2),
        # N-gram counts
        "lex_n_unigrams": n_unigrams,
        "lex_n_bigrams": n_bigrams,
        "lex_n_trigrams": n_trigrams,
        "lex_unique_bigrams": unique_bigrams,
        "lex_unique_trigrams": unique_trigrams,
        # N-gram diversity (type-token ratio at each level)
        "lex_unigram_diversity": round(unigram_diversity, 4),
        "lex_bigram_diversity": round(bigram_diversity, 4),
        "lex_trigram_diversity": round(trigram_diversity, 4),
        # N-gram entropy (bits) - linguistic unpredictability
        "lex_unigram_entropy": round(unigram_entropy, 3),
        "lex_bigram_entropy": round(bigram_entropy, 3),
        "lex_trigram_entropy": round(trigram_entropy, 3),
        # Hapax legomena (once-occurring words)
        "lex_hapax_count": hapax_count,
        "lex_hapax_ratio": round(hapax_ratio, 4),
        # Marker counts
        "lex_agreement_count": agreement_count,
        "lex_hedging_count": hedging_count,
        "lex_certainty_count": certainty_count,
        "lex_question_count": question_count,
        "lex_suggestion_count": suggestion_count,
        "lex_positive_count": positive_count,
        "lex_negative_count": negative_count,
        # Derived rates
        "lex_agreement_rate": round(agreement_rate, 4),
        "lex_question_rate_per_min": round(question_rate_per_min, 2),
        "lex_suggestion_rate_per_min": round(suggestion_rate_per_min, 2),
        "lex_sentiment_ratio": round(sentiment_ratio, 3),
        "lex_speaking_time_s": round(speaking_time_s, 3),
    }


def _extract_one_file(fpath: Path) -> pd.DataFrame:
    """Return one row per participant with their lexical features."""
    m = _FNAME_RE.search(fpath.name)
    if m is None:
        logger.warning("Skipping unrecognized filename: %s", fpath.name)
        return pd.DataFrame()

    group_id = _normalize_group_id(m.group(1))
    task = m.group(2)

    df = pd.read_csv(fpath, sep="\t", dtype=str)
    df["duration"] = pd.to_numeric(df.get("duration"), errors="coerce").fillna(0.0)

    rows = []
    for participant in sorted(PARTICIPANTS):
        feats = _extract_participant_features(df, participant)
        row = {"group_id": group_id, "task": task, "participant": participant, **feats}
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    transcript_files = [
        f for f in sorted(TRANSCRIPT_DIR.glob("transcript_grp*.tsv"))
        if not f.name.endswith(".bak.tsv") and _FNAME_RE.search(f.name)
    ]
    logger.info("Found %d transcript files", len(transcript_files))

    all_rows = [_extract_one_file(f) for f in transcript_files]
    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["group_id", "task", "participant"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, sep="\t", index=False)
    logger.info("Wrote %d rows (%d groups x %d tasks x 4 participants) -> %s",
                len(result), result["group_id"].nunique(), result["task"].nunique(), OUT_PATH)


if __name__ == "__main__":
    main()
