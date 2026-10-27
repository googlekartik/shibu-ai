"""Session-based conversation context (in-memory, no database)."""
import threading


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
        # Last uploaded document (set by /api/upload).
        self.last_document = ""       # filename
        self.last_document_text = ""  # extracted plain text
        self.last_document_info = {}  # e.g. {"type": "xlsx", "sheets": [...], "analysis": {...}}

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
        self.__init__()

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
