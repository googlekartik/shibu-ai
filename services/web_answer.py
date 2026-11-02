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

from utils.text_processing import clean_text, extract_key_sentences, split_into_sentences
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
    re.compile(r"\bthriller\b", re.I),
    re.compile(r"\bIMDb\b"),
    re.compile(r"\bbox office\b", re.I),
    re.compile(r"\bstarring\b", re.I),
    re.compile(r"\bdirected by\b", re.I),
    re.compile(r"\bnovel\b", re.I),
    re.compile(r"\bbook\b", re.I),
    re.compile(r"\bhardcover\b|\bpaperback\b|\bISBN\b", re.I),
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


# Verbs that name the ACTION of a how-to query, not its topic:
# "make" in "how to make tea" must never qualify a result on its
# own, or results about the Make automation platform (which
# mention "make" but never "tea") hijack the answer.
_ACTION_VERBS = frozenset({
    "make", "do", "does", "create", "build", "cook", "prepare", "write",
    "get", "take", "use", "using", "learn", "start", "grow", "fix",
    "repair", "change", "open", "close", "remove", "install", "set",
    "turn", "keep", "improve", "bake", "brew", "tie", "draw", "paint",
    "play", "run", "say", "speak", "lose", "gain", "stop", "avoid",
    "choose", "find", "give", "send", "move", "clean", "wash", "cut",
    "fold", "wear", "drive", "buy", "sell", "grow", "plant", "boil",
    "pour", "add", "stir", "heat", "place", "put", "fill", "wait",
    "become", "being", "be",
})


def _content_nouns(query: str) -> list:
    """The query's core content noun(s): substantive terms minus
    the action verb(s). "how to make tea" -> ["tea"]; a query
    with no noun left (e.g. "how to make it") yields []."""
    return [t for t in _substantive_terms(query) if t not in _ACTION_VERBS]


def _result_mentions_nouns(result: dict, nouns: list) -> bool:
    """True when a result's title or snippet mentions at least one
    content noun."""
    hay = f"{result.get('title', '')} {result.get('snippet', '')}"
    return any(_mentions(hay, noun) for noun in (nouns or []))


def _is_polluted_text(text: str) -> bool:
    """Media/budget content in a single string (title or sentence),
    by the same patterns _is_polluted applies to whole results."""
    return any(p.search(text or "") for p in _POLLUTION_PATTERNS)


def _is_polluted(result: dict) -> bool:
    hay = f"{result.get('title', '')} {result.get('snippet', '')}"
    return _is_polluted_text(hay)


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
    - Content-noun relevance (how-to queries only): a result must
      mention the query's core content noun(s) — "tea" in "how
      to make tea", not just the verb "make". Results with zero
      content-noun overlap are dropped outright (no keep-all
      fallback), so a platform named after the verb can never
      hijack the answer. The caller may retry with an expanded
      query before giving up.
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
        # A polluted result survives ONLY if its own snippet
        # contains a real imperative instruction (e.g. a
        # filmmaking guide that happens to say "film"): media
        # titles with no steps are dropped outright. When every
        # result is polluted and step-less, nothing is kept — the
        # caller then says honestly that no steps came back,
        # instead of dressing headlines up as instructions.
        clean = [r for r in kept
                 if not _is_polluted(r) or _howto_steps([clean_text(r.get("snippet", ""))], max_points=1)]
        kept = clean
        # Content-noun gate: the verb alone ("make") never
        # qualifies a result; the topic noun ("tea") must appear
        # in the title or snippet. Zero overlap anywhere means
        # the whole set is about something else — drop it all
        # (the caller retries with an expanded query first).
        nouns = _content_nouns(query)
        if nouns:
            kept = [r for r in kept if _result_mentions_nouns(r, nouns)]
    return kept


_IMPERATIVE_VERBS = ("boil", "pour", "add", "stir", "heat", "place", "put", "fill",
                    "wait", "let", "use", "take", "remove", "cover", "steep",
                    "brew", "mix", "chop", "bake", "cook", "rinse", "dry",
                    "turn", "set", "bring", "strain", "serve", "drink", "enjoy",
                    "slice", "whisk", "measure", "preheat", "season", "peel",
                    "grate", "crush", "shake", "squeeze", "spread", "toast",
                    "freeze", "thaw", "wash", "scrub", "cut", "fold", "knead",
                    "roll", "sprinkle", "drizzle", "garnish", "check", "ensure",
                    "avoid", "start", "begin", "continue", "repeat", "allow",
                    "leave", "keep", "store", "open", "close", "press", "tap",
                    "click", "type", "enter", "select", "choose", "follow",
                    "create", "make", "write", "read", "watch", "listen")
# Sequence openers that precede the real imperative verb.
_SEQUENCE_WORDS = ("first", "then", "next", "after", "finally", "now", "once")


def _imperative_verb(sentence: str):
    """The imperative verb a step starts with, or None.

    "Boil water." -> boil. "First, pour..." -> pour. A sentence
    starting with anything else (a headline, a description) is
    not an instruction step, however relevant its words are.
    """
    words = re.findall(r"[a-z']+", (sentence or "").lower())
    if not words:
        return None
    first = words[0]
    if first in _SEQUENCE_WORDS and len(words) > 1:
        first = words[1]
    if first in _IMPERATIVE_VERBS:
        return first
    return None


def _is_title_like(sentence: str) -> bool:
    """Headlines and article titles masquerading as steps: a
    trailing '!', a site/brand suffix after ' - ', or a colon
    headline that restates the query ('How to Make Tea: ...')."""
    text = (sentence or "").strip()
    if not text:
        return True
    if text.endswith("!"):
        return True
    # A fragment that is only the tail of a previous headline
    # ("- Brew Buch.") is a title suffix, never a step.
    if text[0] in "-–•":
        return True
    if text.lower().startswith("how to"):
        return True
    if " - " in text:
        suffix = text.rsplit(" - ", 1)[1].strip()
        if len(suffix.split()) <= 3 and not _imperative_verb(suffix):
            return True
    if ":" in text:
        head = text.split(":", 1)[0].strip()
        if len(head.split()) <= 7 and _imperative_verb(head) is None:
            return True
    return False


def _howto_steps(snippets, max_points: int = 3) -> list:
    """Steps built ONLY from imperative instruction sentences in
    the result bodies. Titles are never steps: when no imperative
    sentence exists, the result is empty rather than a headline.
    A sentence that merely DEFINES the action verb ("Make is an
    AI automation platform...") is a description, not an
    instruction, and is dropped too."""
    steps, seen = [], set()
    for snippet in snippets or []:
        for sentence in split_into_sentences(snippet):
            key = sentence.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            if _imperative_verb(sentence) is None:
                continue
            if _is_title_like(sentence):
                continue
            if _is_verb_definition(sentence):
                continue
            if len(sentence.split()) < 3:
                # A two-word fragment ("Brew Buch.") is a title tail,
                # not an instruction a user can follow.
                continue
            steps.append(sentence)
            if len(steps) >= max_points:
                return steps
    return steps


def _is_verb_definition(sentence: str) -> bool:
    """"<verb> is/are ..." — the sentence defines the action word
    ("Make is an AI automation platform") instead of instructing
    the reader. Genuine instructions put an object after the verb
    ("Make tea", "Boil water"), never a copula."""
    words = re.findall(r"[a-z']+", (sentence or "").lower())
    if not words:
        return False
    first = words[0]
    if first in _SEQUENCE_WORDS and len(words) > 1:
        first, words = words[1], words[1:]
    return first in _IMPERATIVE_VERBS and len(words) > 1 and words[1] in ("is", "are", "was", "were")
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
    if _is_howto_query(query):
        # Steps come ONLY from imperative sentences in the bodies.
        # Titles are never steps, and there is no title fallback:
        # an answer with no real steps says so by being empty here
        # rather than dressing a headline up as an instruction.
        points = _howto_steps(snippets, max_points)
    else:
        points = extract_key_sentences(snippets, query, max_sentences=max_points)
    if not points and titles and not _is_howto_query(query):
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


# Built-in last-resort how-to steps: used ONLY when the model and the
# web both yield nothing usable for a how-to question. A handful of
# very common tasks, each a short list of imperative steps, labelled
# as Shibu AI's own knowledge wherever they are shown. Nothing
# outside this table gets steps invented for it.
_BUILTIN_HOWTO_STEPS = {
    "tea": ("How to make tea", [
        "Boil fresh water in a kettle.",
        "Warm the cup or teapot with a little of the hot water, then pour it away.",
        "Put a tea bag, or a teaspoon of loose tea leaves, in the cup or pot.",
        "Pour the boiling water over the tea.",
        "Leave it to steep for 3 to 5 minutes.",
        "Remove the tea bag, or strain out the leaves.",
        "Add milk or sugar if you like, stir, and serve while it is hot.",
    ]),
    "coffee": ("How to make coffee", [
        "Boil fresh water in a kettle.",
        "Put ground coffee in a mug, filter, or French press — about one heaped tablespoon per cup.",
        "Pour the hot water over the coffee grounds.",
        "Leave it to brew for about 4 minutes, or let it drip through the filter.",
        "Strain the coffee, or press the plunger down slowly.",
        "Add milk or sugar if you like, stir, and serve while it is hot.",
    ]),
}


def builtin_howto_steps(query: str):
    """(title, steps) for a very common how-to task, or None.

    Matching uses the query's content nouns ("tea" in "how to make
    tea"), so "a cup of tea" and "brew coffee" match too. Anything
    outside the small built-in table returns None — no steps are
    invented for tasks Shibu does not know."""
    low = (query or "").lower()
    nouns = _content_nouns(query or "")
    for key, (title, steps) in _BUILTIN_HOWTO_STEPS.items():
        if key in nouns or re.search(r"\b" + re.escape(key) + r"\b", low):
            return title, list(steps)
    return None