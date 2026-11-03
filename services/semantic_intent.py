"""Semantic intent understanding (stdlib only: difflib / re).

Different wordings of the same question must reach the same intent,
e.g. "Calculate 25% of 800", "Find 25 percent from 800" and
"What is one-fourth of 800?" are all the math_calculation intent
with percentage=25, number=800.

What this layer does, in order:
  1. normalize: lowercase, punctuation handling (keeps % and
     decimals), whitespace collapse.
  2. spelling correction: a small curated dictionary plus difflib
     close matches against known intent words. Numbers and entity
     words are NEVER "corrected".
  3. synonym expansion (utils/text_processing.SYNONYMS, extended
     with percent / fraction / calculate wording).
  4. entity extraction for percentages, including word fractions
     (one-fourth, quarter, half, third, three-quarters).
  5. semantic similarity: token Jaccard + difflib ratio against
     rule examples (derived from the existing RULES dict — its
     shape is unchanged) and approved user aliases, with
     confidence scoring and (confidence, priority) conflict
     resolution. Below SEMANTIC_CONFIDENCE_THRESHOLD (default
     0.70) the result is UNKNOWN and the caller must clarify
     instead of guessing.

Rule records follow the spec's shape in code: each derived
example carries rule_id, intent, examples, synonyms, priority,
confidence_threshold and action — derived from RULES, not stored
separately. This module is pure text work: it never calls a
model, never stores anything, and never sees a user id. Approved
aliases are passed in by the caller (request_router), which
loads them per-user from SQLiteStore.
"""
from __future__ import annotations

import difflib
import os
import re

# -- normalisation / spelling -------------------------------------------

_SPELLING = {
    "precent": "percent", "percantage": "percent", "percentag": "percent",
    "percentange": "percent", "calculte": "calculate", "calcuate": "calculate",
    "calcualte": "calculate", "wht": "what", "waht": "what", "teh": "the",
    "fotosintesis": "photosynthesis", "photosyntesis": "photosynthesis",
    "summarise": "summarize", "summarizee": "summarize",
    "quater": "quarter", "quorter": "quarter", "halph": "half",
    "thrid": "third", "findd": "find", "calculat": "calculate",
}

# Words that may be spelling-corrected (intent vocabulary). Anything
# else — names, numbers, entities — is left untouched.
_KNOWN_WORDS = {
    "calculate", "compute", "find", "percent", "percentage", "quarter",
    "half", "third", "what", "photosynthesis", "python", "black", "hole",
    "machine", "learning", "summarize", "summary", "explain", "search",
    "latest", "news", "weather", "code", "factorial", "fibonacci",
    "derivative", "integral", "solve", "factor", "convert", "mean",
    "median", "sqrt", "hello", "goodbye", "thanks", "document",
}

_FRACTION_WORDS = {
    "one-fourth": 25.0, "one fourth": 25.0, "a quarter": 25.0,
    "quarter": 25.0, "one-quarter": 25.0,
    "half": 50.0, "one-half": 50.0, "one half": 50.0,
    "third": 100.0 / 3.0, "one-third": 100.0 / 3.0, "one third": 100.0 / 3.0,
    "three-quarters": 75.0, "three quarters": 75.0,
    "three-fourths": 75.0, "three fourths": 75.0,
}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000,
}


def normalize_query(text: str) -> str:
    """Lowercase, keep % and decimal numbers, drop other punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace("%", " % ")
    text = re.sub(r"[^a-z0-9%.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def correct_spelling(text: str) -> str:
    """Correct known misspellings and close matches of intent words.

    Numbers (digits or number words) are never changed, and words
    already in the known vocabulary are left alone.
    """
    words = normalize_query(text).split()
    out = []
    for word in words:
        if word in _NUMBER_WORDS or re.fullmatch(r"[\d.]+%?", word):
            out.append(word)
            continue
        if word in _SPELLING:
            out.append(_SPELLING[word])
            continue
        if word not in _KNOWN_WORDS and len(word) >= 5:
            matches = difflib.get_close_matches(word, _KNOWN_WORDS, n=1, cutoff=0.8)
            if matches:
                out.append(matches[0])
                continue
        out.append(word)
    return " ".join(out)


def expand_synonyms(text: str) -> str:
    """Apply the shared synonym table to already-normalised text."""
    from utils.text_processing import apply_synonyms
    return apply_synonyms(text)


def normalize_semantic(text: str) -> str:
    """Full normalisation pipeline: normalize -> spelling -> synonyms
    -> intent canonicalisation (% becomes "percent"; "find" becomes
    "calculate", since both request a computed answer)."""
    normalized = expand_synonyms(correct_spelling(normalize_query(text)))
    tokens = []
    for token in normalized.split():
        if token == "%":
            token = "percent"
        elif token == "find":
            token = "calculate"
        tokens.append(token)
    return " ".join(tokens)


# -- entity extraction ---------------------------------------------------

_PERCENT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s*cent)\s*(?:of|from|out\s+of)?\s*(\d+(?:\.\d+)?)")
_FRACTION_RE = re.compile(
    r"(three\s+quarters|three\s+fourths|one\s+fourth|one\s+quarter|"
    r"one\s+half|one\s+third|a\s+quarter|quarter|half|third)"
    r"\s*(?:of|from|out\s+of)?\s*(\d+(?:\.\d+)?)")


def extract_percentage_entities(text: str):
    """Return {"percentage": float, "number": float} or None.

    Accepts numeric percentages ("25% of 800", "25 percent from
    800", "25 per cent of 800") and word fractions ("one-fourth
    of 800", "a quarter of 800", "half of 800"). The percentage
    is a number (25.0), never the fraction wording, so every
    paraphrase feeds the same calculation.
    """
    low = normalize_query(text)
    m = _PERCENT_RE.search(low)
    if m:
        return {"percentage": float(m.group(1)), "number": float(m.group(2))}
    m = _FRACTION_RE.search(low)
    if m:
        fraction = re.sub(r"\s+", " ", m.group(1)).strip()
        pct = _FRACTION_WORDS.get(fraction)
        if pct is not None:
            return {"percentage": pct, "number": float(m.group(2))}
    return None


# -- semantic similarity -------------------------------------------------

def semantic_confidence_threshold() -> float:
    try:
        value = float(os.environ.get("SEMANTIC_CONFIDENCE_THRESHOLD", "0.70"))
    except (TypeError, ValueError):
        value = 0.70
    return max(0.0, min(1.0, value))


def _token_jaccard(a: str, b: str) -> float:
    filler = {"of", "from", "what", "is", "the", "a", "an", "please"}
    set_a = {w for w in a.split() if w not in filler}
    set_b = {w for w in b.split() if w not in filler}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def similarity(a: str, b: str) -> float:
    """Token Jaccard + difflib ratio, equally weighted, in [0, 1]."""
    na, nb = normalize_semantic(a), normalize_semantic(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    jaccard = _token_jaccard(na, nb)
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return round(0.5 * jaccard + 0.5 * ratio, 4)


# Curated examples for the intents where paraphrase matters most.
# Each record follows the spec's rule shape (rule_id, intent,
# examples, synonyms, priority, confidence_threshold, action).
_CURATED_RULES = [
    {"rule_id": "percentage_calculation", "intent": "math_calculation",
     "examples": ["calculate 25% of 800", "find 25 percent of 800",
                  "what is one-fourth of 800", "calculate 25 percent from 800",
                  "what is 25 per cent of 800", "find a quarter of 800",
                  "what is half of 800"],
     "synonyms": ["percentage", "percent", "per cent", "portion",
                  "quarter", "half", "third"],
     "priority": 100, "confidence_threshold": 0.70,
     "action": "safe_math_calculator", "version": 1},
    {"rule_id": "greeting", "intent": "GREETING",
     "examples": ["hello", "hi", "hey", "good morning", "good evening"],
     "synonyms": ["hello", "hi", "hey"],
     "priority": 50, "confidence_threshold": 0.70,
     "action": "rule_engine", "version": 1},
    {"rule_id": "web_search", "intent": "WEB_SEARCH",
     "examples": ["search latest news", "search the web for ai news",
                  "what is happening today", "latest updates"],
     "synonyms": ["search", "latest", "news"],
     "priority": 60, "confidence_threshold": 0.70,
     "action": "chrome_search", "version": 1},
]


def _derived_rule_records():
    """Rule records derived from the existing RULES dict (shape
    unchanged): one example per rule built from its keywords/topic,
    synonyms from its synonyms dict, priority from its priority.
    Curated records win ties by priority/confidence (see below)."""
    try:
        from chatbot.patterns import RULES
    except Exception:
        return []
    records = []
    for rule in RULES:
        if rule.get("id") == "unknown":
            continue
        keywords = [str(k) for k in (rule.get("keywords") or [])]
        topic = str(rule.get("topic") or "").strip()
        examples = []
        if keywords:
            examples.append(" ".join(keywords[:4]))
        if topic:
            examples.append(topic)
            examples.append(f"what is {topic}")
        if not examples:
            continue
        records.append({
            "rule_id": rule["id"], "intent": rule["intent"],
            "examples": examples,
            "synonyms": list((rule.get("synonyms") or {}).keys()),
            "priority": int(rule.get("priority") or 0),
            "confidence_threshold": semantic_confidence_threshold(),
            "action": "rule_engine", "version": 1,
        })
    return records


def classify_semantic(text: str, aliases=None):
    """Classify one message semantically.

    Returns a dict with normalized_query, intent, confidence,
    entities, similar_query_found, rule_id, priority and
    clarification_needed. `aliases` is a list of approved-alias
    dicts {alias_text, canonical_intent, confidence} loaded
    per-user by the caller; an alias match sets
    similar_query_found=True. Below threshold the intent is
    UNKNOWN and clarification_needed is True — never a guess.
    """
    normalized = normalize_semantic(text)
    threshold = semantic_confidence_threshold()

    # Deterministic maths wins: a percentage entity + a maths-ish
    # wording is the same intent in every paraphrase.
    entities = extract_percentage_entities(text) or {}
    math_words = ("percent", "per cent", "%", "quarter", "half", "third",
                  "calculate", "compute", "find", "what is")
    if entities and any(w in normalized for w in math_words):
        return {"normalized_query": normalized, "intent": "math_calculation",
                "confidence": 0.98, "entities": entities,
                "similar_query_found": True, "rule_id": "percentage_calculation",
                "priority": 100, "clarification_needed": False,
                "matched_example": "percentage calculation"}

    # Approved aliases (per-user, caller-supplied).
    best_alias = None
    for alias in aliases or []:
        score = similarity(normalized, str(alias.get("alias_text", "")))
        if best_alias is None or score > best_alias[0]:
            best_alias = (score, alias)
    if best_alias and best_alias[0] >= 0.90:
        alias = best_alias[1]
        return {"normalized_query": normalized,
                "intent": alias.get("canonical_intent") or "UNKNOWN",
                "confidence": round(float(alias.get("confidence") or 0.95), 4),
                "entities": entities, "similar_query_found": True,
                "rule_id": alias.get("rule_id") or "approved_alias",
                "priority": 90, "clarification_needed": False,
                "matched_example": str(alias.get("alias_text", ""))}

    # Rule examples: highest (confidence, priority) wins.
    best = None
    for record in _CURATED_RULES + _derived_rule_records():
        for example in record["examples"]:
            score = similarity(normalized, example)
            key = (score, record["priority"])
            if best is None or key > best[0]:
                best = (key, record, example)
    if best is not None:
        (score, _priority), record, example = best
        if score >= max(threshold, record["confidence_threshold"]):
            return {"normalized_query": normalized, "intent": record["intent"],
                    "confidence": round(score, 4), "entities": entities,
                    "similar_query_found": bool(best_alias and best_alias[0] >= threshold),
                    "rule_id": record["rule_id"], "priority": record["priority"],
                    "clarification_needed": False, "matched_example": example}
    return {"normalized_query": normalized, "intent": "UNKNOWN",
            "confidence": round(best[0][0], 4) if best else 0.0,
            "entities": entities, "similar_query_found": False,
            "rule_id": "unknown", "priority": 0,
            "clarification_needed": True, "matched_example": None}
