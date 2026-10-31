"""Low-cost intent planner for cloud tools.

The planner does not answer questions.  It only decides which existing tools
are worth calling, preventing accidental web/LLM calls for ordinary rule or
math questions.
"""
from __future__ import annotations

import re


DOCUMENT_HINTS = (
    "my document", "my file", "uploaded", "in my document", "from my document",
    "my notes", "my pdf", "my spreadsheet",
)
WEB_HINTS = (
    "search the web", "search web", "web search", "look online", "browse the web",
    "latest", "current", "today", "recent news", "live price",
)


def plan_request(message: str) -> dict:
    text = (message or "").strip().lower()
    document = any(hint in text for hint in DOCUMENT_HINTS)
    web = any(hint in text for hint in WEB_HINTS) or bool(
        re.match(r"^(search|find|look up|browse)\b", text))
    comparison = any(phrase in text for phrase in (
        "compare my document", "document and web", "file and web", "uploaded and web",
    ))
    if document and (web or comparison):
        route = "rag_and_web"
    elif document:
        route = "rag"
    elif web:
        route = "web"
    else:
        route = "standard"
    return {"route": route, "use_rag": route in ("rag", "rag_and_web"),
            "use_web": route in ("web", "rag_and_web")}
