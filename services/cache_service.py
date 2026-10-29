"""Small in-memory cache for repeated web questions (5 minutes,
max 100 entries, cleared on restart/redeploy).

Only short, non-personal summaries are cached, never a full page
and never a user's own message or history. Callers must decide
before caching that a query is not personal — see app.py's
cache gate.

Normalization for cache keys: lowercase, collapse whitespace,
strip trailing punctuation. "What is Python?" and "what is python"
are the same cache key; "What is Python used for?" is not.
"""

from __future__ import annotations

import re
import time

MEMORY_TTL_SECONDS = 300
MEMORY_MAX_ENTRIES = 100

_memory: dict = {}


def normalize_query(query: str) -> str:
    text = (query or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?.!]+$", "", text).strip()
    return text


def looks_personal(query: str) -> bool:
    """Heuristic gate: never cache queries that look personal.
    This is a safety net, not a guarantee — callers should also
    only cache answers that came from the public web, not from
    a user's document, history, or account."""
    lowered = (query or "").lower()
    personal_words = (
        "my ", "i am", "i'm", "my name", "my password", "my email",
        "my phone", "my address", "my account", "my age", "my salary",
        "my bank", "my card",
    )
    return any(w in lowered for w in personal_words)


def memory_get(normalized: str):
    entry = _memory.get(normalized)
    if not entry:
        return None
    value, expires_at = entry
    if time.time() > expires_at:
        _memory.pop(normalized, None)
        return None
    return value


def memory_put(normalized: str, value: dict) -> None:
    if len(_memory) >= MEMORY_MAX_ENTRIES:
        # Drop the oldest entry (dicts preserve insertion order).
        _memory.pop(next(iter(_memory)), None)
    _memory[normalized] = (value, time.time() + MEMORY_TTL_SECONDS)


def memory_clear() -> None:
    _memory.clear()


def cached_answer_lookup(normalized: str):
    """Memory only. Returns a dict or None."""
    return memory_get(normalized)


def cached_answer_store(normalized: str, summary: str, sources: list,
                        tool: str = "web") -> None:
    """Store in memory only. Never raises — caching must never
    break a chat answer."""
    result = {"summary": summary, "sources": sources, "tool": tool, "cached": True}
    memory_put(normalized, result)
