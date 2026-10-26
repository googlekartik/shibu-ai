"""Confidence calculation, normalised to 0.0-1.0."""


def classify(confidence: float) -> str:
    if confidence >= 0.75:
        return "HIGH"
    if confidence >= 0.45:
        return "MEDIUM"
    return "LOW"


def calculate_confidence(*, exact=False, regex_matches=0, keyword_hits=0,
                         keyword_total=0, synonym_hits=0) -> float:
    """Combine evidence into a single 0-1 score.

    Exact phrase  -> 0.98
    Regex match   -> 0.90 (plus a little per extra match)
    Keywords      -> proportional to coverage, capped at 0.85
    Synonyms      -> up to 0.65
    """
    if exact:
        return 0.98
    score = 0.0
    if regex_matches:
        score = max(score, 0.88 + min(0.07, 0.02 * (regex_matches - 1)))
    if keyword_total and keyword_hits:
        coverage = keyword_hits / keyword_total
        score = max(score, min(0.85, 0.35 + 0.50 * coverage))
    if synonym_hits:
        score = max(score, min(0.65, 0.40 + 0.10 * synonym_hits))
    if score == 0.0:
        score = 0.15
    return round(max(0.0, min(1.0, score)), 2)
