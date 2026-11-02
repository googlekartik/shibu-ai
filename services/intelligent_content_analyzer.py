"""IntelligentContentAnalyzer: facts, definitions, examples, stats.

Works only on passages already retrieved. It classifies sentences
that are already in the text; it does not generate new claims, so
an analysis can never contain a fact that was not retrieved.
"""
from __future__ import annotations

import re

from services.content_extraction_engine import is_quiz_fragment
from utils.text_processing import extract_keywords

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(%|percent|million|billion|thousand|km|kg|mb|gb|years?|days?|hours?|minutes?|seconds?|°C|°F)?\b", re.I)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_UNCERTAIN = ("may ", "might ", "could ", "possibly", "unclear", "unknown", "estimated", "approximately", "roughly")


def _sentences(passages):
    return [p["text"] for p in passages if p.get("text")]


def _with_source(passages, sentence):
    for passage in passages:
        if passage["text"] == sentence:
            return passage
    return {}


def analyze(passages: list, query: str, content_type: str = "explain") -> dict:
    """Return facts/definitions/examples/statistics/conclusions, plus
    contradictions, uncertainties and gaps detected in the text."""
    sentences = _sentences(passages)

    def relevant(sentence):
        # Passages are already retrieved for this query, so every
        # sentence in them is in scope; keyword overlap only ranks
        # key points, it must not silently drop statistics or
        # definitions that use different wording.
        return True

    facts, definitions, examples, statistics, conclusions = [], [], [], [], []
    for sentence in sentences:
        if is_quiz_fragment(sentence):
            continue
        if not relevant(sentence):
            continue
        low = sentence.lower()
        entry = {"text": sentence, **_with_source(passages, sentence)}
        if re.match(r"^[A-Z][\w' -]{1,50}\s+(is|are|refers to|means)\b", sentence):
            definitions.append(entry)
        if any(phrase in low for phrase in ("for example", "such as", "e.g.", "for instance")):
            examples.append(entry)
        if _NUMBER_RE.search(sentence):
            statistics.append(entry)
        if any(phrase in low for phrase in ("in conclusion", "overall,", "in summary", "therefore,", "as a result")):
            conclusions.append(entry)
        facts.append(entry)

    # Contradictions: same numeric pattern, different values, in
    # different sources. Flagged, never silently merged.
    contradictions = []
    by_pattern = {}
    for stat in statistics:
        match = _NUMBER_RE.search(stat["text"])
        if not match:
            continue
        words = re.findall(r"[a-z]+", stat["text"].lower())[:3]
        unit = (match.group(1) or "").lower()
        key = " ".join(words) + "|" + unit
        by_pattern.setdefault(key, []).append(stat)
    for pattern, group in by_pattern.items():
        values = {re.sub(r"\s+", " ", _NUMBER_RE.search(g["text"]).group(0)) for g in group if _NUMBER_RE.search(g["text"])}
        sources = {g.get("source_name") for g in group}
        if len(values) > 1 and len(sources) > 1:
            contradictions.append({
                "pattern": pattern,
                "values": sorted(values),
                "sources": sorted(s for s in sources if s),
                "note": "Sources disagree on this figure; both values are shown, neither is chosen silently.",
            })

    uncertainties = [s for s in sentences if any(w in s.lower() for w in _UNCERTAIN) and relevant(s)]
    missing = []
    if content_type in ("explain", "document_summary") and not definitions and sentences:
        missing.append("No definition was found in the retrieved text.")
    if content_type == "statistics" and not statistics:
        missing.append("No statistics were found in the retrieved text.")
    if not sentences:
        missing.append("No retrieved passages were available to analyse.")
    return {
        "facts": facts[:12],
        "definitions": definitions[:3],
        "examples": examples[:4],
        "statistics": statistics[:8],
        "conclusions": conclusions[:3],
        "key_points": facts[:6],
        "contradictions": contradictions,
        "uncertainties": uncertainties[:4],
        "missing": missing,
    }
