"""Response generator: validate and format an LLM answer before showing it.

Rules enforced here, not left to the model:
- An empty or whitespace-only answer is a failure, not a blank reply.
- An answer identical to the bot's previous answer is not sent as-is
  (that is the "repeating the same answer" failure mode); the caller
  should regenerate or fall back, so this function flags it.
- Uncertainty is preserved: if the model says it does not know, that is
  kept and labelled, never rewritten into a confident-sounding fact.
- AI answers are always labelled, so a user can tell a model-generated
  answer from a rule-based answer or a web result.
"""

UNCERTAIN_PHRASES = (
    "i don't know", "i'm not sure", "i am not sure", "i don't have",
    "cannot say for certain", "not certain", "unsure",
)


def validate_llm_answer(text, previous_bot_answer=None):
    """Return (ok, cleaned_text, note). ok=False means 'do not show this'."""
    if text is None:
        return False, "", "The language model returned nothing."
    cleaned = text.strip()
    if not cleaned:
        return False, "", "The language model returned an empty answer."
    if previous_bot_answer and cleaned == previous_bot_answer.strip():
        return False, cleaned, "The model repeated the previous answer."
    return True, cleaned, ""


def is_uncertain(text):
    low = text.lower()
    return any(p in low for p in UNCERTAIN_PHRASES)


def format_llm_answer(text, model_id=None, sources=None):
    """Format a validated LLM answer with its label and any web sources."""
    out = text.strip()
    label = "[AI-generated answer"
    if model_id:
        label += f" — {model_id}"
    label += "]"
    out += f"\n\n{label}"
    if is_uncertain(text):
        out += "\nNote: Shibu AI is not sure about this one — please double-check it from a reliable source."
    if sources:
        out += "\n\nSources:"
        for i, s in enumerate(sources, 1):
            title = s.get("title", "Source")
            url = s.get("url", "")
            out += f"\n{i}. {title} — {url}" if url else f"\n{i}. {title}"
    return out
