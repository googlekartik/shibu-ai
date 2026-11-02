"""Daily-task service: LLM-first answers for plans, writing and how-tos.

Design B (chosen 2026-11-01): when a message is a daily task, the
language model answers first, with a task-specific prompt, and the
rule engine is skipped. Maths, greetings and explicit web searches
keep their existing priority in the router and are never treated
as daily tasks here.

When no model is enabled or generation fails, the answer is an
honest, labelled template (plan/email) or the existing web fallback
(how-to) — never the generic "I'm not sure" dead end, and never a
hidden error. Details the user did not give are placeholders
([Name], [Date]) or marked as suggested times, never invented
as personal facts.
"""
from __future__ import annotations

import re

from chatbot.response_manager import format_response
from services.model_manager import ModelManager

_PLAN_RE = re.compile(
    r"\b(plan|schedule|organise|organize|timetable)\b.*\b(day|today|tomorrow|week|morning|routine)\b"
    r"|\bmake (me )?a (daily )?plan\b|\bplan my (day|week|morning)\b", re.I)
_WRITE_RE = re.compile(
    r"\b(write|draft|compose)\b.*\b(email|letter|message|application|cover letter|leave application|resignation)\b"
    r"|\bemail (to|for)\b|\bwrite to (my )?(teacher|boss|manager|professor)\b", re.I)
_HOWTO_RE = re.compile(
    r"^(how to|how do i|how can i|how do you)\b|\bsteps (to|for)\b|\bhow to make\b", re.I)
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-|–)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
_SINGLE_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)

_SYSTEM = {
    "plan": ("You are Shibu AI, a helpful daily assistant. Create a clear daily timetable "
             "from ONLY the activities and times the user gave. If a time is missing, choose a "
             "sensible time and mark it as suggested. Use a heading and one line per time slot. "
             "Do not invent personal details, names, or places the user did not give."),
    "write": ("You are Shibu AI, a helpful daily assistant. Draft ONLY the requested email or "
              "letter, ready to send: a subject line for emails, a greeting, the body, and a "
              "closing. Use [Name] and [Date] placeholders for details the user did not give. "
              "Do not add explanations before or after the draft, and do not invent facts."),
    "howto": ("You are Shibu AI, a helpful daily assistant. Give clear numbered steps for the "
              "task the user asked about. List ingredients or materials first when needed, "
              "keep each step to one or two sentences, and add a short safety note only when "
              "the task genuinely needs one. Do not add anything unrelated."),
}

_KIND_LABEL = {"plan": "Daily plan", "write": "Writing", "howto": "How-to steps"}


def detect_daily_task(message: str) -> str | None:
    """Return 'plan', 'write' or 'howto', or None.

    Exclusions keep the existing priorities intact: greetings
    ("how are you"), maths, and file/PDF requests are not daily
    tasks, and a bare "how" with no task is not a how-to.
    """
    text = (message or "").strip()
    low = text.lower()
    if re.search(r"\bhow are you\b|\bhow is it going\b", low):
        return None
    if re.search(r"\d\s*[-+*/%^]\s*\d|% of\b|calculate\b", low):
        return None
    if any(word in low for word in ("create a pdf", "create a powerpoint", "create a word document", "create an excel")):
        return None
    if _PLAN_RE.search(text):
        return "plan"
    if _WRITE_RE.search(text):
        return "write"
    if _HOWTO_RE.search(text):
        return "howto"
    return None


def _context_lines(context) -> list:
    try:
        from services.conversation_manager import context_system_lines
        return context_system_lines(context)
    except Exception:
        return []


def _friendly_error(err: str) -> str:
    """Provider errors are internal detail: users get one short,
    neutral line. Raw provider text (model names, payload shapes,
    API messages, URLs) is never shown to the user."""
    text = (err or "").strip().lower()
    if "429" in text or "no credits" in text or "insufficient_quota" in text or "quota" in text:
        return "the language model's account has no credits left"
    return "the language model could not draft this"


def _llm_draft(kind: str, message: str, context) -> tuple:
    """Return (text, error, model_id). error is 'off' when no model
    is enabled, so the caller can choose the labelled fallback."""
    manager = ModelManager()
    if manager.mode == "off":
        return None, "off", None
    lines = _context_lines(context)
    system = _SYSTEM[kind]
    if lines:
        system += " Context: " + " ".join(lines)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": message}]
    try:
        raw, err = manager.generate(messages, max_new_tokens=400, temperature=0.5)
    except Exception:
        # A provider surprise must never crash the chat or vanish
        # without a trace: record the failure (structure-only, so
        # /api/llm/status can show it) and fall through to the
        # template, exactly as a returned error does.
        try:
            from services.model_manager import _record_provider_diagnostic
            _record_provider_diagnostic(
                getattr(manager, "mode", "unknown"),
                {"status": None, "response_keys": []})
        except Exception:
            pass
        return None, "error", None
    if err or not raw:
        return None, _friendly_error(err or "The language model returned nothing."), None
    return raw.strip(), None, (getattr(manager, "effective_model", None) or manager.model_id)


def _work_range_hours(message: str):
    """(start_hour_24, end_hour_24) for a stated work range like
    '9 to 5', assuming 9am-5pm when no am/pm is given. None when
    no range is stated."""
    match = _TIME_RE.search(message or "")
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(4))
    start_ampm, end_ampm = (match.group(3) or "").lower(), (match.group(6) or "").lower()
    if not start_ampm and not end_ampm and start <= 12 and end <= 12 and end < start:
        start_ampm, end_ampm = "am", "pm"
    def to24(hour, ampm):
        if ampm == "pm" and hour < 12:
            return hour + 12
        if ampm == "am" and hour == 12:
            return 0
        return hour
    return to24(start, start_ampm), to24(end, end_ampm)


def _fmt_hour_12(hour: int) -> str:
    """24h hour -> '9am' / '5pm' for plan slot labels."""
    if hour == 0:
        return "12am"
    if hour < 12:
        return f"{hour}am"
    if hour == 12:
        return "12pm"
    return f"{hour - 12}pm"


def _template_plan(message: str) -> str:
    """Timetable from the activities the user named, in the order
    they appear. Times the user gave are kept; the rest are
    suggested slots, labelled as such."""
    text = message
    activities = []
    for chunk in re.split(r"[:,;]|\band\b|\bthen\b", text, flags=re.I):
        chunk = chunk.strip(" .")
        if not chunk or len(chunk) < 3:
            continue
        low = chunk.lower()
        if any(w in low for w in ("plan my day", "plan my", "i work", "i want", "i have")):
            chunk = re.sub(r"(?i)^(plan my \w+|i work|i want|i have)\s*", "", chunk).strip()
        if chunk:
            activities.append(chunk)
    if not activities:
        activities = ["Work", "Break", "Study"]
    lines = ["Your day plan:", ""]
    work = _work_range_hours(message)
    # Suggested slots must never clash with a stated work range:
    # activities without a given time go before work or after it.
    pool = ["07:00", "12:00", "18:00", "19:00", "20:00"]
    if work:
        def _hour(slot):
            return int(slot.split(":")[0])
        pool = [s for s in pool if _hour(s) < work[0] or _hour(s) >= work[1]] or ["07:00", "20:00"]
    slots = []
    for chunk in activities:
        slot_match = _TIME_RE.search(chunk)
        if slot_match:
            slot = slot_match.group(0)
            chunk = _TIME_RE.sub("", chunk).strip(" .")
            if not chunk:
                chunk = "Work"
        else:
            single = _SINGLE_TIME_RE.search(chunk)
            slot = single.group(0) if single else None
            if single:
                chunk = _SINGLE_TIME_RE.sub("", chunk).strip(" .")
        chunk = re.sub(r"(?i)^(plan my \w+|i work|want|i have)\s*", "", chunk).strip()
        slots.append((slot or pool[len(slots) % len(pool)] + " (suggested)", chunk[:80] or "Activity"))
    for slot, activity in slots[:6]:
        lines.append(f"- {slot}: {activity}")
    return "\n".join(lines)


def _template_write(message: str) -> str:
    """A ready-to-send skeleton using only details in the request."""
    low = message.lower()
    recipient = "Teacher"
    if "boss" in low or "manager" in low:
        recipient = "Manager"
    elif "professor" in low:
        recipient = "Professor"
    reason = "leave"
    if "sick" in low:
        reason = "sick leave"
    when = "[Date]"
    if "tomorrow" in low:
        when = "tomorrow"
    elif "today" in low:
        when = "today"
    # Natural phrasing: "leave tomorrow", never "leave on tomorrow",
    # and no date is ever hardcoded — "tomorrow" stays "tomorrow".
    when_phrase = when if when in ("tomorrow", "today") else f"on {when}"
    subject = f"Subject: Request for {reason} {when_phrase}"
    body = (
        f"Dear {recipient},\n"
        f"\n"
        f"I am writing to request {reason} {when_phrase}. "
        f"[Add a short reason here if you wish.]\n"
        f"\n"
        f"I will make sure my work is covered and catch up on anything I miss.\n"
        f"\n"
        f"Thank you for your understanding.\n"
        f"\n"
        f"Yours sincerely,\n"
        f"[Your name]")
    return ("Fill in the [bracketed] details:\n\n"
            + subject + "\n\n" + body)


def _template_week_plan(message: str) -> str:
    """Day-by-day week timetable from the activities the user named.

    This is the fallback used when no drafted plan is available, so
    it must stand on its own: gym is assigned to three specific
    days, study is spread over weekday evenings, work keeps its
    stated hours Monday to Friday, and the family dinner is on
    Sunday evening. Times the user did not give are marked
    suggested; nothing here mentions how the plan was produced.
    """
    low = (message or "").lower()
    work = _work_range_hours(message)
    work_label = (f"{_fmt_hour_12(work[0])} to {_fmt_hour_12(work[1])}"
                  if work else "Work")
    has_gym = "gym" in low
    has_study = "study" in low
    study_label = "study for an exam" if "exam" in low else "study"
    has_dinner = "dinner" in low
    if not (work or has_gym or has_study or has_dinner):
        # Nothing recognisable to distribute: reuse the day-plan
        # template rather than inventing a week.
        return _template_plan(message).replace("Your day plan:", "Your week plan:", 1)
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    gym_days = {"Monday", "Wednesday", "Friday"}
    lines = ["Your week plan:", ""]
    for day in ("Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"):
        day_lines = []
        if day in weekdays and work:
            day_lines.append(f"- {work_label}: Work")
        if has_gym and day in gym_days:
            day_lines.append("- 18:00 (suggested): gym")
        if has_study and day in weekdays:
            day_lines.append(f"- 19:00 (suggested): {study_label} (1 hour)")
        if day == "Sunday" and has_dinner:
            day_lines.append("- 20:00 (suggested): family dinner")
        lines.append(f"{day}:")
        if day_lines:
            lines.extend(day_lines)
        else:
            lines.append("- Rest day — no fixed plans.")
        lines.append("")
    return "\n".join(lines).rstrip()


def unavailable_howto_payload(message: str, context, processing_ms: float = 0.0):
    """Honest last resort for a how-to when neither the model nor
    the web could answer. Returning this — instead of falling
    through to the rule engine — is what stops a how-to question
    being answered with an unrelated greeting rule."""
    text = ("I could not look up steps for that right now, and no web results came back. "
            "Please try again in a moment, or rephrase with the exact task, "
            "for example: 'How to make a cup of tea'.")
    payload = format_response(
        text=text, intent="DAILY_TASK", rule_id="daily_howto_unavailable", confidence=0.90,
        tool="Daily tasks", action="How-to unavailable (no model, no web results)",
        processing_ms=processing_ms)
    context.add_exchange(message, text, "DAILY_TASK", "daily_howto_unavailable", 0.90)
    return payload


def build_daily_task_payload(message: str, context, processing_ms: float = 0.0):
    """LLM-first daily-task answer. Returns (payload, needs_web_fallback).

    needs_web_fallback is True only for a how-to when no model is
    available or it failed, so the router can try the web fallback
    before giving up. All other kinds always return a payload.
    """
    kind = detect_daily_task(message)
    if kind is None:
        return None, False
    rule_id = f"daily_{kind}"
    draft, error, model_id = _llm_draft(kind, message, context)
    if draft:
        payload = format_response(
            text=draft, intent="DAILY_TASK", rule_id=rule_id, confidence=0.80,
            tool=f"Language model ({model_id})" if model_id else "Language model",
            action=f"Draft {_KIND_LABEL[kind].lower()} with the language model",
            processing_ms=processing_ms)
        context.add_exchange(message, draft, "DAILY_TASK", rule_id, 0.80)
        return payload, False
    if kind == "howto":
        # The web fallback in the router is the honest next step for
        # a how-to; returning None here would lose that chance.
        return None, True
    if kind == "plan" and "week" in message.lower():
        template = _template_week_plan(message)
    else:
        template = _template_plan(message) if kind == "plan" else _template_write(message)
    if error and "no credits left" in error:
        # Genuinely quota-labelled failure only: the one friendly
        # line the quota behaviour requires, with no provider name,
        # URL, or raw detail. Every other failure delivers the
        # template cleanly, with no mention of the failure at all.
        template += ("\n\nNote: the language model's account has no credits "
                     "left, so this is a fill-in version you can complete.")
    # The fallback is otherwise delivered cleanly: no mention of
    # the language model, providers, or errors — the user only
    # ever sees the plan/email itself. The internal error category
    # stays in logs.
    payload = format_response(
        text=template, intent="DAILY_TASK", rule_id=rule_id, confidence=0.70,
        tool="Template", action="Draft from a template",
        processing_ms=processing_ms)
    context.add_exchange(message, payload["response"], "DAILY_TASK", rule_id, 0.70)
    return payload, False
