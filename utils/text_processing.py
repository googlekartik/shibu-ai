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
