"""Simple conversation context memory, built on ConversationContext.

The project already kept per-session history (chatbot/context.py,
keyed by Flask session id, cleared by /api/clear), but follow-ups
only covered "what is it used for?". This module adds the missing
follow-up language, on top of that same history, without a database
and without storing any secret (history holds only the messages
already exchanged in this session):

  - "Explain it more simply." / "Make it shorter." -> a simplified /
    shortened version of the previous answer.
  - "Give me another example." -> the previous topic, retained, with
    a different sentence from its answer as the new example.
  - "What about the second one?" / "the second example" -> the second
    numbered/bulleted example from the previous answer, if there is
    one; otherwise a short clarification question, never a guess.
  - "Continue from where you stopped." -> the later part of the
    previous answer for the same topic.
  - "it" / "that topic" / "the previous question" -> resolved to the
    previous topic/question for rule matching.
  - A genuinely new topic (a rule match on a different topic, in a
    message that is not a follow-up) is answered on its own merits:
    the old topic is never injected into it. History is kept, not
    mixed in.

resolve_followup() returns:
  {"kind": ..., "answer": str}            -> answer directly, or
  {"kind": "clarify", "question": str}    -> ask, do not guess, or
  None                                    -> not a follow-up; use the
                                             normal pipeline (which
                                             already ignores old
                                             context for new topics).
"""

import re


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _examples(text):
    """Numbered / bulleted / 'Example N:' items from an answer."""
    items = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*(?:\d+[.)]|[-*•])\s+(.+)", line)
        if m:
            items.append(m.group(1).strip())
        else:
            m = re.match(r"(?i)\s*example\s*\d*\s*:\s*(.+)", line)
            if m:
                items.append(m.group(1).strip())
    return items


def _last_pair(context):
    """(last user question, last real bot answer) or (None, None).

    Prefer last_main_response (the answer before any follow-up
    wrappers) so follow-up chains do not degrade into wrappers of
    wrappers.
    """
    user = getattr(context, "last_user_message", "") or ""
    bot = getattr(context, "last_main_response", "") or getattr(context, "last_bot_response", "") or ""
    if not user and not bot:
        return None, None
    return user, bot


def resolve_followup(text, context):
    low = text.lower().strip()
    user_q, bot_a = _last_pair(context)
    topic = getattr(context, "last_topic", "") or ""

    def need_context(what):
        return {"kind": "clarify",
                "question": (f"Which {what} do you mean? I don't have a previous "
                             "answer in this chat to refer back to — please ask the "
                             "question again, or tell me the topic.")}

    # "Explain it more simply." / "Explain that more simply."
    if re.search(r"\bexplain (it|that|this)\b.*\b(simpl|easier)\b|\bmore simply\b|\bsimpler\b", low):
        if not bot_a:
            return need_context("answer")
        first = _sentences(bot_a)
        return {"kind": "simpler",
                "answer": f"In simple terms: {first[0]}" if first else bot_a}
    # "Make it shorter."
    if re.search(r"\b(make it|make that) shorter\b|\bshorter version\b|\bshort version\b", low):
        if not bot_a:
            return need_context("answer")
        first = _sentences(bot_a)
        return {"kind": "shorter",
                "answer": f"Short version: {first[0]}" if first else bot_a}
    # "Give me another example." — topic must be retained, not re-asked.
    # But "another example of <new topic>" names its own topic; let the
    # normal pipeline answer that topic instead of forcing old context.
    _of = re.search(r"\banother example of ([a-z]+)\b", low)
    if _of and _of.group(1) != topic:
        return None
    if re.search(r"\banother example\b|\bmore examples\b|\bother example\b", low):
        if not bot_a:
            return need_context("topic")
        sents = _sentences(bot_a)
        example = sents[1] if len(sents) > 1 else (sents[0] if sents else bot_a)
        label = f" for {topic}" if topic else ""
        return {"kind": "another_example",
                "answer": f"Another example{label}: {example}"}
    # "What about the second one?" / "the second example"
    # A named topic ("second example of photosynthesis") is not a
    # bare follow-up; let the normal pipeline answer that topic.
    if re.search(r"\b(second|third|fourth)\s+(example|one)\s+of\s+", low):
        return None
    m = re.search(r"\b(second|third|fourth)\b.*\b(one|example|point)\b|\bthe (second|third|fourth)\b", low)
    if m:
        if not bot_a:
            return need_context("example")
        word = m.group(1) or m.group(3)
        if word not in ("second", "third", "fourth"):
            return {"kind": "clarify",
                    "question": "Which one do you mean? Please name the topic or the example you are referring to, so I don't guess."}
        idx = {"second": 1, "third": 2, "fourth": 3}[word]
        items = _examples(bot_a)
        if len(items) > idx:
            return {"kind": "nth_example",
                    "answer": f"The {word} example{' for ' + topic if topic else ''}: {items[idx]}"}
        return {"kind": "clarify",
                "question": (f"I want to get this right instead of guessing — which "
                             f"{word} one do you mean? My previous answer"
                             + (f" about {topic} " if topic else " ")
                             + (f"only listed {len(items)} example(s), so I can't "
                                "tell which one you mean. Please name the topic or "
                                "paste the list again."))}
    # "Continue from where you stopped."
    if re.search(r"\bcontinue\b|\bgo on\b|\bcarry on\b", low) and re.search(r"\b(where you stopped|from there|from where)\b|\bcontinue\b", low):
        if not bot_a:
            return need_context("answer")
        sents = _sentences(bot_a)
        if len(sents) > 1:
            return {"kind": "continue",
                    "answer": f"Continuing{' ' + topic if topic else ''}: " + " ".join(sents[1:])}
        return {"kind": "continue",
                "answer": (f"That was the full answer{' on ' + topic if topic else ''}, "
                           "so there is nothing left to continue — say 'explain it more "
                           "simply' or 'give me another example' if you want more on it.")}
    # Generic "it / that topic / previous question" references.
    if re.search(r"\b(it|that|this)\b", low) and topic:
        if re.search(r"\b(tell me more|more about|explain|what about|describe)\b", low):
            return {"kind": "reference", "topic": topic}
    return None


def is_topic_change(text, context, new_topic):
    """True when a non-follow-up message is about a different topic.

    The router uses this only as documentation of the rule it already
    follows: a new-topic message is matched on its own words, and the
    previous topic is never injected into the answer.
    """
    if resolve_followup(text, context) is not None:
        return False
    old = getattr(context, "last_topic", "") or ""
    return bool(old and new_topic and new_topic != old)
