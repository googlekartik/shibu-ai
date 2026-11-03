"""Preference / learning-memory guards (per-user, privacy-first).

Only useful, non-sensitive preferences are ever stored: language,
answer style, academic level, response length, output format, and
explicitly saved topics. Sensitive traits are NEVER inferred or
stored — health, medical conditions, religion, political beliefs,
sexual orientation, intelligence, personality, financial status,
or private identity traits. A phrase matching the blocklist is
refused (not stored, not used to personalise), whether it arrives
as a preference, an explicit memory, an alias, or feedback text.

Silence is never positive feedback, and one interaction never
auto-applies anything: learning candidates need repeated
evidence (>=3 similar occurrences) or one explicit user approval
before they become approved aliases, via an explicit endpoint.

This module is pure: no store import, no network, no secrets.
"""
from __future__ import annotations

import hashlib
import re

ALLOWED_PREFERENCE_KEYS = frozenset({
    "language", "answer_style", "tone", "level", "response_length",
    "explanation_level", "preferred_format", "topic",
    "personalization_enabled",
})

# Sensitive-trait blocklist: phrases in these areas are refused
# for storage. The patterns are deliberately lexical (words the
# user would have to type), not inferred from behaviour.
_SENSITIVE_PATTERNS = (
    re.compile(r"\b(depress\w*|anxiet\w*|anxiety|mental health|therapy|therapist|"
               r"suicid\w*|self[- ]harm|bipolar|schizophrenia|autism|adhd|"
               r"diabetes|cancer|asthma|hiv|aids|illness|disease|disorder|"
               r"medication|prescription|diagnos\w*)\b", re.I),
    re.compile(r"\b(christian|muslim|islam|hindu|buddhist|jewish|atheist|"
               r"religion|religious|god|allah|church|mosque|temple|pray\w*)\b", re.I),
    re.compile(r"\b(conservative|liberal|left[- ]wing|right[- ]wing|socialist|"
               r"communist|fascist|republican|democrat|bjp|congress|vote[rd]?|"
               r"election|political|politics)\b", re.I),
    re.compile(r"\b(gay|lesbian|bisexual|transgender|queer|homosexual|"
               r"sexual orientation|straight|heterosexual)\b", re.I),
    re.compile(r"\b(iq\b|intelligence|genius|stupid|dumb|smartness)\b", re.I),
    re.compile(r"\b(introvert|extrovert|personality|mbti|enneagram)\b", re.I),
    re.compile(r"\b(salary|income|bank balance|debt|loan|net worth|rich|poor|"
               r"financial|credit card|bank account)\b", re.I),
    re.compile(r"\b(password|api key|secret|access token|otp|one-time code)\b", re.I),
)


def is_sensitive_text(text: str) -> bool:
    """True when text names a sensitive trait that must not be stored."""
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in _SENSITIVE_PATTERNS)


def validate_preference(key: str, value: str):
    """Return (ok, cleaned_value_or_error). Unknown keys and
    sensitive values are refused, never stored."""
    if key not in ALLOWED_PREFERENCE_KEYS:
        return False, f"Unknown preference key: {key}"
    if not isinstance(value, str):
        return False, "Preference value must be text."
    cleaned = " ".join(value.strip().split())[:200]
    if not cleaned:
        return False, "Preference value cannot be empty."
    if is_sensitive_text(cleaned):
        return False, "That preference names a sensitive personal trait, so it is not stored."
    return True, cleaned


def hash_question(text: str) -> str:
    """Stable sha256 hash of a question — learning candidates store
    the hash, never the raw question text."""
    return hashlib.sha256((text or "").strip().lower().encode("utf-8")).hexdigest()


def redact_pii(text: str) -> str:
    """Redact email addresses and phone-like numbers from text
    stored for analytics (query_events / dev panel)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted]", text)
    text = re.sub(r"\+?\d[\d\s\-()]{7,}\d", "[redacted]", text)
    return text
