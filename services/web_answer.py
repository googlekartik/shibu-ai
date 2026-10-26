"""Text-processed web answers for Shibu AI.

Raw DuckDuckGo snippets are often fragments, duplicates, or only
loosely related to the question. This module turns them into a real
answer instead of a raw list:

1. Clean every snippet (HTML, extra spaces).
2. Split snippets into sentences.
3. Score each sentence by query-keyword coverage (see
   utils/text_processing.extract_key_sentences).
4. Keep the top sentences, in relevance order, without duplicates.
5. List every source, so the web answer is never confused with a
   rule-database answer.

No external dependencies, and no LLM is needed for this step.
"""

from utils.text_processing import clean_text, extract_key_sentences


def summarise_web_results(query: str, results, max_points: int = 3):
    """Return {'summary': str, 'key_points': [str]} for search results."""
    snippets = [clean_text(r.get("snippet", "")) for r in results if r.get("snippet")]
    titles = [clean_text(r.get("title", "")) for r in results if r.get("title")]
    points = extract_key_sentences(snippets, query, max_sentences=max_points)
    if not points and titles:
        # Fall back to titles when snippets are empty fragments.
        points = titles[:max_points]
    summary = " ".join(points)
    return {"summary": summary, "key_points": points}


def format_web_answer(query: str, results) -> str:
    """Full phrased web answer, with summary, key points and sources."""
    summary = summarise_web_results(query, results)
    parts = [f"Here's what I found on the web about '{query}':"]
    if summary["summary"]:
        parts.append("")
        parts.append("In short, according to the latest web results:")
        parts.append(summary["summary"])
    if summary["key_points"]:
        parts.append("")
        parts.append("Key points:")
        for point in summary["key_points"]:
            parts.append(f"- {point}")
    parts.append("")
    parts.append("Sources (from the web, not from my rule database):")
    for i, r in enumerate(results, 1):
        parts.append(f"{i}. {r.get('title', '')}")
        if r.get("snippet"):
            parts.append(f"   {clean_text(r['snippet'])}")
        if r.get("url"):
            parts.append(f"   {r['url']}")
    parts.append("")
    parts.append("This answer is processed from web search results, not from my rule database — please check the sources above for the latest details.")
    parts.append("Want me to go deeper on any of these sources? Just tell me which one.")
    return "\n".join(parts)


def clean_search_topic(text: str) -> str:
    """Turn a 'what is X / define X / tell me about X' question into a search topic."""
    import re as _re
    t = text.strip()
    t = _re.sub(r"(?i)^(what is|whats|what is the|define|meaning of|tell me about|explain|search|find|look up|browse)\s+", "", t)
    t = _re.sub(r"(?i)\s+(in simple terms|simply|in hindi|in english)\s*$", "", t)
    return t.strip(" ?.")
