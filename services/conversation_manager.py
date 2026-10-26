"""Conversation manager: history, context window, and language for the LLM.

This adapts the existing ConversationContext (chatbot/context.py), which
already stores the conversation and preferences, instead of duplicating
it. The LLM only ever sees a trimmed window: the last few turns, cut to
a character budget, so a long chat cannot overflow the model's context
limit and cannot silently lose the newest question.

Language handling: if the user writes in Devanagari, or in Hinglish
(common Hindi words written in Latin script), the model is instructed
to answer in that same language. If detection is unsure, English is
used and the instruction says "reply in the user's language".
"""

import re

SYSTEM_PROMPT = (
    "You are Shibu AI, a friendly, helpful study assistant. "
    "Answer clearly and concisely, and explain difficult ideas in simple "
    "language, step by step when the user asks. If you do not know an "
    "answer, say so plainly instead of inventing facts. Do not repeat "
    "the user's question back as your answer."
)

# Common Hinglish words (Latin script). A single word is not enough on
# its own in code that calls detect_language — see the threshold below.
HINGLISH_WORDS = {
    "kya", "kaise", "kyun", "kab", "kahan", "kaun", "mera", "meri", "tum",
    "tumhara", "tumhari", "aap", "hai", "hain", "ho", "batao", "bataiye",
    "samajh", "samjhao", "karo", "karke", "liye", "aur", "bhi", "nahi",
    "matlab", "kaisa", "kitna", "kaise", "chahiye", "zaroor", "accha",
}

MAX_TURNS = 6          # last 6 exchanges (user + assistant pairs)
MAX_CHARS = 6000       # rough context budget for the prompt text


def detect_language(text):
    """Return 'hindi' (Devanagari), 'hinglish', or 'english'."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if len(words & HINGLISH_WORDS) >= 2:
        return "hinglish"
    return "english"


def language_instruction(language):
    if language == "hindi":
        return "Reply in Hindi (Devanagari script), matching the user's language."
    if language == "hinglish":
        return "Reply in Hinglish (Hindi words written in Latin script), matching the user's language."
    return "Reply in English."


def build_messages(context, user_message, max_turns=MAX_TURNS, max_chars=MAX_CHARS):
    """Build the message list for the LLM from stored history.

    `context` is a ConversationContext. History entries are dicts with
    role/text (older entries) or, from the router, plain strings — both
    are handled. Newest turns win when trimming: we walk backwards and
    stop at the character budget.
    """
    language = detect_language(user_message)
    system = SYSTEM_PROMPT + " " + language_instruction(language)
    messages = [{"role": "system", "content": system}]

    history = []
    for entry in getattr(context, "conversation_history", []) or []:
        if isinstance(entry, dict):
            history.append(entry)
        elif isinstance(entry, str):
            history.append({"role": "user", "text": entry})
    # Keep only the most recent turns, and drop the turn that is the
    # current question itself if the caller already recorded it.
    history = history[-max_turns * 2:]
    chosen, total = [], 0
    for entry in reversed(history):
        text = str(entry.get("text", ""))
        if total + len(text) > max_chars:
            break
        # The stored history uses role "bot" for assistant turns.
        role = entry.get("role", "user")
        if role == "bot":
            role = "assistant"
        chosen.append({"role": role, "content": text})
        total += len(text)
    messages.extend(reversed(chosen))
    messages.append({"role": "user", "content": user_message})
    return messages, language


def trim_for_display(text, limit=2000):
    """Cap a generated answer so one runaway generation cannot flood the UI."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"
