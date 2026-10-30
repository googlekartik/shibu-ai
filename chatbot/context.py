"""Session-based conversation context (in-memory, no database)."""
import re
import threading


def _fmt_num(value):
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return ("%.6f" % value).rstrip("0").rstrip(".")


def _extract_numeric_answer(bot_response):
    """Last numeric result in an answer, or None.

    Maths answers are "… = 49933\nAnswer: 49933" and series answers
    are "Next term: 32", so those two markers are trusted. Other
    answers (a year, a count inside prose) are NOT treated as a
    result to calculate with — guessing there would be wrong.
    """
    # The number must end the answer (or the line): "Answer: 3*x**2"
    # for a derivative must NOT be read as the result 3.
    m = re.search(r"Answer:\s*(-?[\d,]+(?:\.\d+)?)\s*(?:\n|$)", bot_response or "")
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"Next term:\s*(-?[\d,]+(?:\.\d+)?)\s*\.?\s*(?:\n|$)", bot_response or "")
    if m:
        return float(m.group(1).replace(",", ""))
    return None


class ConversationContext:
    def __init__(self):
        self.conversation_history = []  # list of {"role": ..., "text": ...}
        self.last_user_message = ""
        self.last_bot_response = ""
        # Last answer that was NOT itself a follow-up wrapper, so a
        # chain like "explain simply -> another example -> shorter"
        # always refers back to the real answer, not to a wrapper.
        self.last_main_response = ""
        self.last_intent = ""
        self.last_rule = ""
        self.last_confidence = 0.0
        self.last_topic = ""  # e.g. "python" for follow-ups like "what is it used for?"
        # Compact rolling summary of this session only (never shared
        # across sessions, never holding secrets — only topics and the
        # first sentence of recent answers, capped at 500 chars).
        self.summary = ""
        # User-approved preferences (saved per session, never assumed).
        # tone: "normal" | "simple" | "concise" | "detailed"
        # level: "normal" | "beginner" | "expert"
        self.preferences = {"tone": "normal", "level": "normal"}
        # What the user is currently doing in this chat, in plain
        # words — e.g. a running calculation — plus the last numeric
        # result, so a follow-up like "add 5 in previous answer"
        # continues the same task instead of starting a web search.
        # This is task memory only: it never holds secrets, and it is
        # replaced (not mixed) when the user starts a new topic.
        self.active_task = ""
        self.last_numeric_value = None
        # Last uploaded document (set by /api/upload).
        self.last_document = ""       # filename
        self.last_document_text = ""  # extracted plain text
        self.last_document_info = {}  # e.g. {"type": "xlsx", "sheets": [...], "analysis": {...}}
        # Explicit long-term memories: only facts the user asked Shibu
        # to remember ("remember that ..."). Never inferred, never a
        # secret. Persisted per-user in SQLite by app.py; in this
        # session object they are just a capped list of plain strings.
        self.display_name = ""
        self.memories = []
        self.restored_from_sqlite = False

    def remember(self, fact: str) -> bool:
        """Store one explicit memory. False when empty/duplicate/too long."""
        fact = " ".join((fact or "").strip().split())
        if not fact or len(fact) > 200:
            return False
        if any(f.lower() == fact.lower() for f in self.memories):
            return False
        self.memories.append(fact)
        self.memories = self.memories[-20:]
        return True

    def forget_matching(self, query: str) -> int:
        """Remove memories containing `query` (case-insensitive)."""
        q = " ".join((query or "").strip().split()).lower()
        if not q:
            return 0
        before = len(self.memories)
        self.memories = [f for f in self.memories if q not in f.lower()]
        return before - len(self.memories)

    def forget_all_memories(self) -> int:
        count = len(self.memories)
        self.memories = []
        return count

    def restore_from_messages(self, messages):
        """Restore recent history saved in SQLite into this session.

        `messages` are dicts {role, text, ...} in chronological order.
        Only the last 12 entries (6 pairs) are restored, so a long
        old conversation cannot flood the LLM context. This only
        fills the in-memory context used for follow-ups and LLM
        prompts; it does not re-save the messages to SQLite.
        """
        rows = [m for m in (messages or []) if m.get("text")]
        rows = rows[-12:]
        if not rows:
            return False
        self.conversation_history = [
            {"role": ("bot" if m.get("role") == "bot" else "user"),
             "text": m["text"]} for m in rows]
        users = [m["text"] for m in rows if m.get("role") == "user"]
        bots = [m["text"] for m in rows if m.get("role") == "bot"]
        if users:
            self.last_user_message = users[-1]
        if bots:
            self.last_bot_response = bots[-1]
            self.last_main_response = bots[-1]
            value = _extract_numeric_answer(bots[-1])
            if value is not None:
                self.last_numeric_value = value
                self.active_task = f"Running calculation — last result {_fmt_num(value)}"
        self.restored_from_sqlite = True
        return True

    def add_exchange(self, user_message, bot_response, intent="", rule_id="",
                     confidence=0.0, topic=""):
        self.last_user_message = user_message
        self.last_bot_response = bot_response
        if intent not in ("FOLLOW_UP", "CLARIFY"):
            self.last_main_response = bot_response
        self.last_intent = intent
        self.last_rule = rule_id
        self.last_confidence = confidence
        if topic:
            self.last_topic = topic
        # Task memory: a calculation sets (and a calculation follow-up
        # updates) the running total; a new topic replaces the task.
        # Follow-up wrappers (explain simply, etc.) leave it alone.
        if intent in ("MATHEMATICS", "MATHEMATICS_STEP_BY_STEP",
                      "REASONING_SERIES", "MATH_FOLLOWUP"):
            value = _extract_numeric_answer(bot_response)
            if value is not None:
                self.last_numeric_value = value
                self.active_task = (f"Running calculation — last result "
                                    f"{_fmt_num(value)}")
        elif intent not in ("FOLLOW_UP", "CLARIFY") and topic:
            self.active_task = f"Discussing {topic}"
            self.last_numeric_value = None
        self.conversation_history.append({"role": "user", "text": user_message})
        self.conversation_history.append({"role": "bot", "text": bot_response})
        # Keep memory bounded.
        self.conversation_history = self.conversation_history[-40:]
        # Rolling summary: last few topics + first sentence of each
        # answer, so a long conversation keeps its thread without
        # growing the context window. Updated on every exchange.
        first = bot_response.split(".")[0].strip() if bot_response else ""
        bit = f"{topic or self.last_topic or user_message[:40]}: {first[:120]}".strip()
        parts = [p for p in (self.summary.split(" | ") if self.summary else [])]
        if bit not in parts:
            parts.append(bit)
        self.summary = " | ".join(parts[-3:])[:500]

    def last_pair(self):
        """(last user question, last bot answer) as a structured pair."""
        return self.last_user_message, self.last_bot_response

    def recent_pairs(self, n=6):
        """Last n user/assistant pairs, as structured dicts."""
        pairs = []
        hist = self.conversation_history
        for i in range(0, len(hist) - 1, 2):
            if hist[i].get("role") == "user" and hist[i + 1].get("role") == "bot":
                pairs.append({"user": hist[i]["text"], "assistant": hist[i + 1]["text"]})
        return pairs[-n:]

    def clear(self):
        # Clear the conversation, but keep explicit memories and the
        # display name: "clear chat" is not "forget everything about
        # me". Memories are removed only via "forget ...".
        memories = list(self.memories)
        display_name = self.display_name
        self.__init__()
        self.memories = memories
        self.display_name = display_name

    def to_dict(self):
        return {
            "conversation_history": self.conversation_history,
            "last_user_message": self.last_user_message,
            "last_bot_response": self.last_bot_response,
            "last_intent": self.last_intent,
            "last_rule": self.last_rule,
            "last_confidence": self.last_confidence,
            "last_topic": self.last_topic,
            "preferences": self.preferences,
            "last_document": self.last_document,
            "summary": self.summary,
            "recent_pairs": self.recent_pairs(),
            "active_task": self.active_task,
            "display_name": self.display_name,
            "memories": list(self.memories),
            "restored_from_sqlite": self.restored_from_sqlite,
            "last_numeric_answer": (_fmt_num(self.last_numeric_value)
                                     if self.last_numeric_value is not None else None),
        }


class ContextStore:
    """Thread-safe store of contexts keyed by a session id."""

    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> ConversationContext:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ConversationContext()
            return self._store[session_id]

    def clear(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)
