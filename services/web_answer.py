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
import re

_GENERIC_WORDS = frozenset({
    "latest", "recent", "news", "today", "update", "updates", "breaking",
    "search", "find", "current", "trending", "information", "info", "top",
    "best", "new", "please", "tell", "me", "about", "what", "whats", "is",
    "the", "a", "an", "how", "to", "do", "i", "can", "you", "give", "for",
    "of", "in", "on", "and", "or", "are", "was", "were", "be", "by", "with",
    "from", "that", "this", "it", "its", "my", "your", "when", "where",
    "which", "who", "why", "does", "did", "should", "would", "could",
    "explain", "define", "meaning", "look", "up", "browse", "step", "steps",
})

# Title/snippet patterns that mark a result as media/budget content
# rather than instructions. Only applied to how-to queries, where a
# film or book title starting "How to..." otherwise pollutes the steps.
_POLLUTION_PATTERNS = (
    re.compile(r"\bfilm\b", re.I),
    re.compile(r"\bmovie\b", re.I),
    re.compile(r"\bbox office\b", re.I),
    re.compile(r"\bstarring\b", re.I),
    re.compile(r"\bdirected by\b", re.I),
    re.compile(r"\bnovel\b", re.I),
    re.compile(r"\bbestseller\b", re.I),
    re.compile(r"\bpublished\b.{0,20}\bby\b", re.I),
    re.compile(r"\bbudget\b", re.I),
    re.compile(r"\$\s?\d", re.I),
    re.compile(r"\((film|movie|novel|book|tv series)\)", re.I),
)


def _substantive_terms(query: str) -> list:
    """Query words that a relevant result must mention. Generic
    words like 'news' or 'latest' never qualify a result on their
    own — 'AI' in 'latest AI news' must match the result text."""
    words = re.findall(r"[a-z0-9]+", (query or "").lower())
    terms = [w for w in words if w not in _GENERIC_WORDS and len(w) >= 2]
    low = (query or "").lower()
    if "artificial intelligence" in low and "artificial intelligence" not in terms:
        terms.append("artificial intelligence")
    return terms


def _mentions(text: str, term: str) -> bool:
    if " " in term:
        return term in text.lower()
    return re.search(r"\b" + re.escape(term) + r"\b", text.lower()) is not None


def _is_polluted(result: dict) -> bool:
    hay = f"{result.get('title', '')} {result.get('snippet', '')}"
    return any(p.search(hay) for p in _POLLUTION_PATTERNS)


def filter_results(query: str, results) -> list:
    """Drop irrelevant and non-instructional results before an
    answer is built from them.

    - Relevance: a result must mention at least one substantive
      query term in its title or snippet. A query with no
      substantive terms keeps all results, and if every result
      would be dropped, all are kept (show something rather than
      nothing).
    - How-to pollution: for how-to queries, results about films,
      books, budgets or box-office figures are dropped, since
      their titles masquerade as instructions.
    The original order and result dicts are preserved.
    """
    results = list(results or [])
    if not results:
        return results
    terms = _substantive_terms(query)
    kept = results
    if terms:
        relevant = [r for r in results
                    if any(_mentions(f"{r.get('title', '')} {r.get('snippet', '')}", t)
                           for t in terms)]
        if relevant:
            kept = relevant
    if _is_howto_query(query):
        clean = [r for r in kept if not _is_polluted(r)]
        if clean:
            kept = clean
    return kept


_IMPERATIVE_VERBS = ("boil", "pour", "add", "stir", "heat", "place", "put", "fill",
                    "wait", "let", "use", "take", "remove", "cover", "steep",
                    "brew", "mix", "chop", "bake", "cook", "rinse", "dry")
_META_WORDS = ("wikihow", "article", "will teach", "explains how", "guide explains",
              "with pictures", "read more", "learn more")


def _is_howto_query(query: str) -> bool:
    low = (query or "").lower()
    return any(w in low for w in ("how to", "how do i", "how can i", "step by step", "steps to"))


def _howto_score(sentence: str) -> int:
    """Bonus for instructional sentences, penalty for article
    descriptions: a how-to answer must contain steps to follow,
    not titles of articles about the task."""
    low = sentence.lower().strip()
    score = 0
    first_word = low.split(" ", 1)[0].strip(".,:;") if low else ""
    if first_word in _IMPERATIVE_VERBS:
        score += 3
    elif any(f" {v} " in f" {low} " for v in _IMPERATIVE_VERBS):
        score += 1
    if any(w in low for w in _META_WORDS):
        score -= 3
    return score


def summarise_web_results(query: str, results, max_points: int = 3):
    """Return {'summary': str, 'key_points': [str]} for search results."""
    results = filter_results(query, results)
    snippets = [clean_text(r.get("snippet", "")) for r in results if r.get("snippet")]
    titles = [clean_text(r.get("title", "")) for r in results if r.get("title")]
    points = extract_key_sentences(snippets, query, max_sentences=max_points)
    if _is_howto_query(query):
        # Re-rank a wider pool so real steps beat article blurbs.
        pool = extract_key_sentences(snippets, query, max_sentences=8)
        pool.sort(key=lambda s: _howto_score(s), reverse=True)
        points = pool[:max_points]
    if not points and titles:
        # Fall back to titles when snippets are empty fragments.
        points = titles[:max_points]
    summary = " ".join(points)
    return {"summary": summary, "key_points": points}


def format_web_answer(query: str, results) -> str:
    """Full phrased web answer, with summary, key points and sources."""
    results = filter_results(query, results)
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
