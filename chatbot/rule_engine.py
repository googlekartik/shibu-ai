"""Rule engine: regex -> keywords -> synonyms, with priority + confidence."""
import re

from chatbot.confidence import calculate_confidence
from chatbot.patterns import RULES
from utils.text_processing import normalize_for_matching


def match_rule(user_input: str):
    """Return {rule_id, intent, confidence, response, matched_patterns, topic}."""
    normalized = normalize_for_matching(user_input)
    best = None
    for rule in RULES:
        if rule["id"] == "unknown":
            continue
        regex_matches = []
        for pat in rule.get("patterns", []):
            try:
                if re.search(pat, normalized):
                    regex_matches.append(pat)
            except re.error:
                continue
        keywords = rule.get("keywords", [])
        keyword_hits = sum(1 for k in keywords if f" {k} " in f" {normalized} ")
        synonym_hits = 0
        for syn in rule.get("synonyms", {}):
            if f" {syn} " in f" {normalized} ":
                synonym_hits += 1
        exact = normalized in [p.strip("^$") for p in rule.get("patterns", [])] or \
            any(normalized == pat.strip("^$") for pat in rule.get("patterns", []))
        # Exact short greetings should score highest.
        if normalized in ("hello", "hi", "hey", "bye", "goodbye") and regex_matches:
            exact = True
        if not regex_matches and keyword_hits == 0 and synonym_hits == 0:
            continue
        confidence = calculate_confidence(
            exact=exact, regex_matches=len(regex_matches),
            keyword_hits=keyword_hits, keyword_total=len(keywords) or 1,
            synonym_hits=synonym_hits)
        score = confidence * 100 + rule.get("priority", 0)
        if best is None or score > best["_score"]:
            response = rule["responses"][0]  # deterministic: first response
            best = {
                "rule_id": rule["id"],
                "intent": rule["intent"],
                "confidence": confidence,
                "response": response,
                "matched_patterns": regex_matches,
                "topic": rule.get("topic", ""),
                "_score": score,
            }
    if best is None:
        unknown = next(r for r in RULES if r["id"] == "unknown")
        return {
            "rule_id": "unknown", "intent": "UNKNOWN", "confidence": 0.15,
            "response": unknown["responses"][0], "matched_patterns": [],
            "topic": "",
        }
    best.pop("_score", None)
    return best
