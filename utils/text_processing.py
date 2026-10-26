"""Text normalisation, synonyms and keyword extraction.

No external dependencies. Everything here is pure text work so the
rule engine can be tested without Flask.
"""
import re
import string

SYNONYMS = {
    "hi": "hello", "hey": "hello", "hiya": "hello", "yo": "hello",
    "bye": "goodbye", "good bye": "goodbye", "see ya": "goodbye",
    "thanks": "thank you", "thx": "thank you", "thankyou": "thank you",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cs": "computer science",
    "calc": "calculate", "calculation": "calculate",
    "solve": "calculate",
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "who", "how",
    "can", "could", "please", "tell", "me", "about", "do", "does", "i",
    "you", "your", "to", "of", "in", "on", "for", "and", "or",
}


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip most punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Keep characters needed for maths in a separate path; for rule
    # matching we remove punctuation.
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_synonyms(text: str) -> str:
    """Replace known synonyms / abbreviations with canonical forms."""
    if not text:
        return text
    out = f" {text} "
    # Longer phrases first so "thankyou" etc. are handled correctly.
    for old in sorted(SYNONYMS, key=len, reverse=True):
        new = SYNONYMS[old]
        out = out.replace(f" {old} ", f" {new} ")
    return out.strip()


def normalize_for_matching(text: str) -> str:
    return apply_synonyms(normalize_text(text))


def extract_keywords(text: str):
    """Return content words from normalised text."""
    words = normalize_for_matching(text).split()
    return [w for w in words if w not in STOPWORDS]


def contains_any(text: str, words) -> bool:
    norm = f" {normalize_for_matching(text)} "
    return any(f" {w} " in norm for w in words)


# ---- Text processing for better web answers (2026-10-27) ----
def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, trim."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str):
    """Split cleaned text into sentences, keeping each sentence whole."""
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for part in parts:
        part = part.strip()
        if part:
            if part[-1] not in ".!?":
                part += "."
            out.append(part)
    return out


def sentence_score(sentence: str, query_keywords) -> int:
    """Score a sentence by how many query keywords it covers."""
    low = sentence.lower()
    score = 0
    for kw in query_keywords:
        if kw and f" {kw} " in f" {low} " or kw in low.split():
            score += 2
        elif kw and kw in low:
            score += 1
    # Prefer readable, medium-length sentences; penalise fragments.
    if 40 <= len(sentence) <= 220:
        score += 1
    if len(sentence) < 20:
        score -= 1
    return score


def extract_key_sentences(texts, query: str, max_sentences: int = 3):
    """Pick the most query-relevant, non-duplicate sentences from texts.

    texts: list of snippets / paragraphs. Returns sentences in
    relevance order, deduplicated by normalised lowercased text.
    """
    keywords = [w for w in extract_keywords(query) if len(w) > 2]
    candidates = []  # (score, order, sentence)
    order = 0
    seen = set()
    for text in texts or []:
        for sent in split_into_sentences(text):
            key = sent.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((sentence_score(sent, keywords), order, sent))
            order += 1
    # Highest score first; original order breaks ties.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [sent for _score, _order, sent in candidates[:max_sentences]]
