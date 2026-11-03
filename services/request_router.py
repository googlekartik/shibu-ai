"""Low-cost intent planner for cloud tools.

The planner does not answer questions. It only decides which
existing tools are worth calling, preventing accidental web/LLM
calls for ordinary rule or math questions, and it returns the
internal routing structure for server logs / dev diagnostics.
That structure is INTERNAL ONLY: it is never added to a normal
/api/chat response and is never shown to users (no internal
metadata or hidden reasoning leaks into chat).

Processing order enforced downstream (app.py / chatbot/router.py):
  validate/normalize -> conversation context -> intent (semantic
  layer) -> deterministic rules -> maths -> private RAG -> Chrome
  search -> Gemini -> labelled fallback.
"""
from __future__ import annotations

import re

from services.semantic_intent import classify_semantic

DOCUMENT_HINTS = (
    "my document", "my file", "uploaded", "in my document", "from my document",
    "my notes", "my pdf", "my spreadsheet",
)
WEB_HINTS = (
    "search the web", "search web", "web search", "look online", "browse the web",
    "latest", "current", "today", "recent news", "live price",
)


def plan_request(message: str, aliases=None) -> dict:
    """Plan one request. `aliases` (optional) are the caller's
    approved aliases, loaded per-user; they feed the semantic
    layer so a previously approved paraphrase is recognised
    (similar_query_found=True)."""
    text = (message or "").strip().lower()
    semantic = classify_semantic(message or "", aliases=aliases)
    document = any(hint in text for hint in DOCUMENT_HINTS)
    web = any(hint in text for hint in WEB_HINTS) or bool(
        re.match(r"^(search|find|look up|browse)\b", text))
    comparison = any(phrase in text for phrase in (
        "compare my document", "document and web", "file and web", "uploaded and web",
    ))
    if semantic["intent"] == "math_calculation" and semantic["confidence"] >= 0.70:
        route = "deterministic_math"
    elif document and (web or comparison):
        route = "rag_and_web"
    elif document:
        route = "rag"
    elif web:
        route = "web"
    else:
        route = "standard"
    return {
        # Internal routing structure (logs / dev diagnostics only).
        "normalized_query": semantic["normalized_query"],
        "intent": semantic["intent"],
        "confidence": semantic["confidence"],
        "entities": semantic["entities"],
        "similar_query_found": semantic["similar_query_found"],
        "route": route,
        "gemini_used": False,
        "fallback_used": False,
        "sources": [],
        "learning_candidate_created": False,
        # Backwards-compatible planning flags used by app.py.
        "use_rag": route in ("rag", "rag_and_web"),
        "use_web": route in ("web", "rag_and_web"),
    }
