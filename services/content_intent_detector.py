"""ContentIntentDetector: what kind of answer does this question need?

Deterministic keyword/regex detection only. It never calls a model:
the router and the existing preferences already decide *where* an
answer comes from; this decides *how* it should be presented.

Content types follow the feature spec's table: explain, compare,
algorithm, maths, code, process steps, web research, document
summary, statistics, roadmap, historical timeline, pros/cons, quiz,
and simple (a short answer that must stay short).
"""
from __future__ import annotations

import re

_CONTENT_TYPES = (
    "explain", "compare", "process_steps", "algorithm", "maths", "code",
    "web_search", "document_summary", "statistics", "roadmap",
    "historical", "pros_cons", "quiz", "simple",
)

_INTERACTIVE_WORDS = ("quiz", "mcq", "flashcard", "flashcards", "practice questions",
                      "test me", "practice problems", "revise me")


def _level_from_text(low: str, prefs: dict) -> str:
    if any(w in low for w in ("exam", "for my exam", "board exam", "important questions")):
        return "exam"
    if any(w in low for w in ("beginner", "simple language", "easy words", "explain like i'm 5", "explain like i am 5")):
        return "beginner"
    if any(w in low for w in ("advanced", "in depth", "in-depth", "deep dive", "technical details")):
        return "advanced"
    if any(w in low for w in ("professional", "at work", "industry", "production")):
        return "professional"
    if any(w in low for w in ("intermediate", "more detail", "detailed")):
        return "intermediate"
    pref_level = (prefs or {}).get("level", "")
    if pref_level == "beginner":
        return "beginner"
    if pref_level == "expert":
        return "advanced"
    return "intermediate"


def _compare_entities(text: str) -> list:
    def _trim(entity: str) -> str:
        # "Java for beginners" -> "Java": trailing purpose phrases
        # are not part of the compared entity.
        cut = re.split(r"\s+(?:for|to|in|with|as|from)\b", entity, maxsplit=1)[0]
        return cut.strip(" ?")
    m = re.search(r"compare\s+(.+?)\s+(?:vs\.?|versus|and|with)\s+(.+)", text, re.I)
    if m:
        return [_trim(m.group(1)), _trim(m.group(2))]
    m = re.search(r"(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+)", text, re.I)
    if m:
        return [_trim(m.group(1)), _trim(m.group(2))]
    m = re.search(r"(.+?)\s+vs\.?\s+(.+)", text, re.I)
    if m:
        return [_trim(m.group(1)), _trim(m.group(2))]
    return []


def detect_intent(message: str, prefs: dict = None, rule_intent: str = "") -> dict:
    """Return {content_type, level, entities, topic, needs_interactive, raw_request}."""
    text = (message or "").strip()
    low = text.lower()
    prefs = prefs or {}
    entities = _compare_entities(text)
    topic = text
    for prefix in ("what is ", "what are ", "explain ", "tell me about ", "search "):
        if low.startswith(prefix):
            topic = text[len(prefix):].strip(" ?")
            break
    needs_interactive = any(w in low for w in _INTERACTIVE_WORDS)
    level = _level_from_text(low, prefs)

    if needs_interactive:
        content_type = "quiz"
    elif entities:
        content_type = "compare"
    elif any(w in low for w in ("advantages and disadvantages", "pros and cons", "pros & cons")):
        content_type = "pros_cons"
    elif any(w in low for w in ("roadmap", "learning path", "study plan")):
        content_type = "roadmap"
    elif any(w in low for w in ("timeline", "history of", "historical events", "when did")) or re.search(r"\b(19|20)\d{2}\b", low) and "history" in low:
        content_type = "historical"
    elif any(w in low for w in ("statistics", "statistical", "show me data", "chart", "graph of")):
        content_type = "statistics"
    elif any(w in low for w in ("summarize my document", "summarise my document", "summary of my document",
                                "summarize the document", "summarize this pdf", "summarize this file")):
        content_type = "document_summary"
    elif any(w in low for w in ("algorithm", "how does the algorithm", "complexity of", "time complexity")):
        content_type = "algorithm"
    elif any(w in low for w in ("how to", "how do i", "how can i", "steps to", "step by step", "procedure", "process of", "workflow")):
        content_type = "process_steps"
    elif any(w in low for w in ("write code", "code for", "python code", "debug", "fix this code", "explain this code")):
        content_type = "code"
    elif re.search(r"\d\s*[-+*/%^]\s*\d|% of|calculate|derivative|integral|solve for", low):
        content_type = "maths"
    elif any(w in low for w in ("latest", "news", "search the web", "search for", "current", "today's")):
        content_type = "web_search"
    elif low.startswith(("what is ", "what are ", "who is ", "define ", "explain ", "tell me about ")):
        content_type = "explain"
    elif len(text.split()) <= 2:
        content_type = "simple"
    elif len(text.split()) <= 8 and "?" in text and not any(w in low for w in ("explain", "compare", "steps")):
        content_type = "simple"
    else:
        content_type = "explain"
    return {
        "content_type": content_type,
        "level": level,
        "entities": entities[:2],
        "topic": topic[:120],
        "needs_interactive": needs_interactive,
        "raw_request": bool(any(w in low for w in ("raw", "unformatted", "plain text only", "no formatting"))),
    }


def is_simple_request(intent: dict, response: str = "") -> bool:
    """Simple questions must stay simple: no visual components."""
    if intent.get("raw_request"):
        return True
    if intent.get("content_type") in ("simple", "maths", "code"):
        return True
    return bool(response) and len(response) < 200 and intent.get("content_type") not in (
        "compare", "process_steps", "historical", "roadmap", "statistics", "pros_cons", "quiz")
