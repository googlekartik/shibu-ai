"""Gemini-only provider routing for Shibu AI (policy version 2).

Google Gemini is the ONLY hosted AI provider. classify() always
returns provider "gemini"; the task-kind / length heuristic is kept
purely as diagnostics (task_kind + a stable reason code) so
/api/llm/status can explain what kind of request was made. There
is no second provider, no fallback leg, and at most ONE Gemini
call per request — when Gemini fails, callers use the
deterministic fallback, never another provider.

Env override ROUTER_PREFERENCE: "auto" (default) or "gemini".
Anything else (including a value naming a removed provider) is ignored and treated
as "auto", so a stale variable from a removed provider can never
route a request anywhere else.

No secrets are read here. No prompt text is stored: classify()
returns only the provider, a stable reason code, and the task kind.
"""

import os
import re

POLICY_VERSION = 2

GEMINI = "gemini"

LONG_PROMPT_CHARS = 600

# Task kinds callers may pass (diagnostics only — every kind maps
# to Gemini). Kept as a closed set so an unknown hint cannot be
# mistaken for a provider name.
_TASK_KINDS = frozenset({
    "plan", "write", "email", "essay", "letter", "story",
    "quiz", "code", "summary", "rewrite", "howto", "explain",
})

_CODE_RE = re.compile(
    r"\b(code|python|javascript|typescript|function|debug|debugging|"
    r"compile|traceback|sql|regex)\b", re.IGNORECASE)
_CODE_PHRASES = ("fix the code", "fix my code", "fix this code", "write code",
                 "generate code", "explain the code", "explain this code",
                 "explain my code", "debug the code", "debug my code")

_STRUCTURED_RE = re.compile(
    r"\b(summarise|summarize|summary|shorten|shorter|simpler|simplify|"
    r"rewrite|quiz|flashcard)\b", re.IGNORECASE)

_DRAFT_RE = re.compile(
    r"\b(write|email|letter|essay|story|cover letter|draft|plan|"
    r"resume|apology|complaint|speech|poem)\b", re.IGNORECASE)


def router_preference(env=None):
    """Validated ROUTER_PREFERENCE: "auto" or "gemini".

    Invalid, empty, missing, or removed-provider values fall back
    to "auto". `env` is an injectable mapping for tests; defaults
    to os.environ.
    """
    source = os.environ if env is None else env
    raw = str(source.get("ROUTER_PREFERENCE", "") or "").strip().lower()
    if raw in (GEMINI, "auto"):
        return raw
    return "auto"


def _last_user_text(messages):
    """Text of the last user message, or "" when there is none."""
    if not messages:
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content") or ""
            return content if isinstance(content, str) else ""
    last = messages[-1]
    if isinstance(last, dict):
        content = last.get("content") or ""
        return content if isinstance(content, str) else ""
    return ""


def _is_multi_part(text):
    """Simple deterministic multi-part test: two or more questions
    joined by "and", or three or more sentences."""
    low = text.lower()
    if low.count("?") >= 2 and " and " in low:
        return True
    sentences = [s for s in re.split(r"[.?!]+", text) if s.strip()]
    return len(sentences) >= 3


def classify(task_kind_hint=None, messages=None, preference=None):
    """Decide the provider for one LLM request: always "gemini".

    Returns {"provider", "reason", "task_kind", "preference"}.
    Pure and deterministic: no network, no randomness, no secrets.
    `preference` overrides ROUTER_PREFERENCE when given (tests and
    callers that already resolved it); otherwise the env var is read.
    The reason code records WHY the request was classified the way
    it was (task kind / keyword / length) for diagnostics only.
    """
    pref = preference if preference is not None else router_preference()
    if pref not in (GEMINI, "auto"):
        pref = "auto"
    kind = str(task_kind_hint or "").strip().lower() or None
    if kind not in _TASK_KINDS:
        kind = None
    text = _last_user_text(messages or [])

    if pref == GEMINI:
        return {"provider": GEMINI, "reason": "forced_preference",
                "task_kind": kind or "heuristic", "preference": pref}

    if kind:
        return {"provider": GEMINI,
                "reason": f"task_kind:{kind}",
                "task_kind": kind, "preference": "auto"}

    low = text.lower()
    if "```" in text:
        reason = "code_block"
    elif _CODE_RE.search(text) or any(p in low for p in _CODE_PHRASES):
        reason = "keyword_code"
    elif _STRUCTURED_RE.search(text):
        reason = "keyword_structured"
    elif len(text) >= LONG_PROMPT_CHARS:
        reason = "long_prompt"
    elif _DRAFT_RE.search(text):
        reason = "keyword_draft"
    elif _is_multi_part(text):
        reason = "multi_part"
    else:
        reason = "short_prompt"
    return {"provider": GEMINI, "reason": reason,
            "task_kind": kind or "heuristic", "preference": "auto"}
