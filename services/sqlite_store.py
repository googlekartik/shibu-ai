"""Persistent SQLite store for Shibu AI — users, documents,
conversations, messages. This replaces the removed DynamoDB layer
with a local, file-based store (stdlib sqlite3, no server, no AWS).

Important limits, stated plainly:
- On a PC this file persists. On Render's free plan the filesystem
  is ephemeral, so a redeploy can wipe it — the same limit the old
  uploads/ and generated/ folders had. Pinecone vectors, once
  upserted, survive a redeploy; SQLite metadata does not on free
  Render. Re-indexing from an uploaded file restores it.
- The user key is always google#<sub> from the session, never a
  display name or email, and every query is filtered by user_id.
  Cross-user reads are impossible through these methods because
  the user_id is a WHERE clause, not a request parameter.

Migration rule: existing localStorage / session history is NEVER
migrated automatically. Only migrate_session_history() with
confirm=True does it, and it is called only from the explicit
POST /api/history/migrate endpoint. Tests cover both branches.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid


def default_sqlite_path() -> str:
    env = os.environ.get("SHIBU_SQLITE_PATH", "").strip()
    if env:
        return env
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "shibu.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT DEFAULT '',
    avatar_id TEXT,
    onboarding_completed INTEGER DEFAULT 0,
    created_at REAL, updated_at REAL
);
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_type TEXT DEFAULT '',
    status TEXT DEFAULT 'indexed',
    chunk_count INTEGER DEFAULT 0,
    error TEXT DEFAULT '',
    full_text TEXT DEFAULT '',
    source_key TEXT DEFAULT '',
    content_sha256 TEXT DEFAULT '',
    embedding_provider TEXT DEFAULT '',
    embedding_model TEXT DEFAULT '',
    embedding_version TEXT DEFAULT '',
    created_at REAL, updated_at REAL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    text TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON document_chunks(user_id, document_id);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    created_at REAL, updated_at REAL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    intent TEXT DEFAULT '',
    rule_id TEXT DEFAULT '',
    sources_json TEXT DEFAULT '[]',
    created_at REAL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS user_memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact_text TEXT NOT NULL,
    created_at REAL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON user_memories(user_id, created_at);
CREATE TABLE IF NOT EXISTS query_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    question TEXT DEFAULT '',
    intent TEXT DEFAULT '',
    rule_id TEXT DEFAULT '',
    confidence REAL DEFAULT 0,
    is_fallback INTEGER DEFAULT 0,
    processing_ms REAL DEFAULT 0,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_query_events_created ON query_events(created_at);
CREATE TABLE IF NOT EXISTS login_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_login_events_created ON login_events(created_at);
CREATE TABLE IF NOT EXISTS dev_settings (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT DEFAULT '',
    updated_at REAL
);
"""


class SQLiteStore:
    def __init__(self, path: str = None):
        self.path = path or default_sqlite_path()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate_document_columns()
        self._conn.commit()

    def _migrate_document_columns(self):
        """Add RAG provenance columns to databases created by older releases."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(documents)").fetchall()}
        migrations = {
            "source_key": "TEXT DEFAULT ''",
            "content_sha256": "TEXT DEFAULT ''",
            "embedding_provider": "TEXT DEFAULT ''",
            "embedding_model": "TEXT DEFAULT ''",
            "embedding_version": "TEXT DEFAULT ''",
        }
        for name, definition in migrations.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_source ON documents(user_id, source_key)")

    def _now(self):
        return time.time()

    def _ensure_user(self, user_id: str):
        self._conn.execute("INSERT OR IGNORE INTO users (user_id, display_name, created_at, updated_at) VALUES (?,?,?,?)",
                           (user_id, "", self._now(), self._now()))

    # -- users ------------------------------------------------------
    def get_or_create_user(self, user_id: str, display_name: str = "", avatar_id=None, onboarding_completed=False):
        cur = self._conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        if row:
            return dict(row)
        now = self._now()
        self._conn.execute(
            "INSERT INTO users (user_id, display_name, avatar_id, onboarding_completed, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (user_id, display_name or "", avatar_id, 1 if onboarding_completed else 0, now, now))
        self._conn.commit()
        return {"user_id": user_id, "display_name": display_name or "", "avatar_id": avatar_id,
                "onboarding_completed": bool(onboarding_completed)}

    def update_user(self, user_id: str, display_name=None, avatar_id=None, onboarding_completed=None):
        self.get_or_create_user(user_id)
        sets, vals = [], []
        if display_name is not None:
            sets.append("display_name=?"); vals.append(display_name)
        if avatar_id is not None:
            sets.append("avatar_id=?"); vals.append(avatar_id)
        if onboarding_completed is not None:
            sets.append("onboarding_completed=?"); vals.append(1 if onboarding_completed else 0)
        if sets:
            sets.append("updated_at=?"); vals.append(self._now()); vals.append(user_id)
            self._conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE user_id=?", vals)
            self._conn.commit()
        return dict(self._conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone())

    # -- documents --------------------------------------------------
    def save_document(self, user_id: str, document_id: str, source_name: str, source_type: str,
                      full_text: str, chunks: list, status: str = "indexed", error: str = "",
                      provenance: dict = None):
        self._ensure_user(user_id)
        now = self._now()
        provenance = provenance or {}
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (document_id, user_id, source_name, source_type, status, chunk_count, error, full_text, source_key, content_sha256, embedding_provider, embedding_model, embedding_version, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (document_id, user_id, source_name, source_type, status, len(chunks), error, full_text or "",
             provenance.get("source_key", ""), provenance.get("content_sha256", ""),
             provenance.get("embedding_provider", ""), provenance.get("embedding_model", ""),
             provenance.get("embedding_version", ""), now, now))
        self._conn.execute("DELETE FROM document_chunks WHERE document_id=? AND user_id=?", (document_id, user_id))
        for c in chunks:
            self._conn.execute(
                "INSERT OR REPLACE INTO document_chunks (chunk_id, document_id, user_id, chunk_index, text, metadata_json) VALUES (?,?,?,?,?,?)",
                (c["id"], document_id, user_id, c["chunk_index"], c["text"],
                 json.dumps({"source_name": source_name, "chunk_index": c["chunk_index"],
                             "char_start": c.get("char_start", 0), "char_end": c.get("char_end", 0)})))
        self._conn.commit()
        return self.get_document(user_id, document_id)

    def get_document(self, user_id: str, document_id: str):
        row = self._conn.execute(
            "SELECT document_id, user_id, source_name, source_type, status, chunk_count, error, source_key, content_sha256, embedding_provider, embedding_model, embedding_version, created_at, updated_at FROM documents WHERE document_id=? AND user_id=?",
            (document_id, user_id)).fetchone()
        return dict(row) if row else None

    def get_document_with_text(self, user_id: str, document_id: str):
        row = self._conn.execute(
            "SELECT * FROM documents WHERE document_id=? AND user_id=?", (document_id, user_id)).fetchone()
        return dict(row) if row else None

    def get_document_by_source_key(self, user_id: str, source_key: str):
        row = self._conn.execute(
            "SELECT * FROM documents WHERE user_id=? AND source_key=? ORDER BY updated_at DESC LIMIT 1",
            (user_id, source_key)).fetchone()
        return dict(row) if row else None

    def list_documents(self, user_id: str):
        rows = self._conn.execute(
            "SELECT document_id, user_id, source_name, source_type, status, chunk_count, error, source_key, content_sha256, embedding_provider, embedding_model, embedding_version, created_at, updated_at FROM documents WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, user_id: str, document_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM document_chunks WHERE document_id=? AND user_id=?", (document_id, user_id))
        cur2 = self._conn.execute("DELETE FROM documents WHERE document_id=? AND user_id=?", (document_id, user_id))
        self._conn.commit()
        return cur2.rowcount > 0

    def search_chunks_fallback(self, user_id: str, query: str, top_k: int = 5,
                               document_id: str = None, source_name: str = None) -> list:
        """SQLite fallback retrieval — token overlap, NOT embeddings,
        and labelled as such by callers. Used only when the backend
        is sqlite (development / automated tests). Pinecone queries
        never use this function."""
        import re
        qwords = set(re.findall(r"[a-z]{3,}", (query or "").lower()))
        if not qwords:
            return []
        conditions = ["c.user_id=?"]
        values = [user_id]
        if document_id:
            conditions.append("c.document_id=?")
            values.append(document_id)
        if source_name:
            conditions.append("d.source_name=?")
            values.append(source_name)
        rows = self._conn.execute(
            "SELECT c.chunk_id, c.document_id, c.text, c.metadata_json, d.source_name FROM document_chunks c "
            "JOIN documents d ON d.document_id=c.document_id WHERE " + " AND ".join(conditions), values).fetchall()
        scored = []
        for r in rows:
            words = set(re.findall(r"[a-z]{3,}", r["text"].lower()))
            overlap = len(qwords & words)
            if overlap:
                metadata = json.loads(r["metadata_json"] or "{}")
                scored.append({"id": r["chunk_id"], "document_id": r["document_id"],
                               "score": overlap / max(1, len(qwords)),
                               "text": r["text"], "source_name": r["source_name"],
                               "metadata": metadata,
                               "chunk_index": metadata.get("chunk_index")})
        scored.sort(key=lambda x: (-x["score"], x["document_id"]))
        return scored[:top_k]

    # -- conversations / messages -----------------------------------
    def create_conversation(self, user_id: str, title: str = ""):
        self._ensure_user(user_id)
        cid = uuid.uuid4().hex
        now = self._now()
        self._conn.execute("INSERT INTO conversations (conversation_id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                           (cid, user_id, title or "", now, now))
        self._conn.commit()
        return cid

    def list_conversations(self, user_id: str, limit: int = 20):
        rows = self._conn.execute(
            "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def save_message(self, user_id: str, conversation_id: str, role: str, text: str,
                     intent: str = "", rule_id: str = "", sources=None):
        self._ensure_user(user_id)
        mid = uuid.uuid4().hex
        now = self._now()
        self._conn.execute(
            "INSERT INTO messages (message_id, conversation_id, user_id, role, text, intent, rule_id, sources_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, conversation_id, user_id, role, text, intent, rule_id, json.dumps(sources or []), now))
        self._conn.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=? AND user_id=?", (now, conversation_id, user_id))
        self._conn.commit()
        return mid

    def list_messages(self, user_id: str, conversation_id: str, limit: int = 50):
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? AND user_id=? ORDER BY created_at ASC LIMIT ?",
            (conversation_id, user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- explicit memories (only facts the user asked to remember) --
    def list_memories(self, user_id: str):
        rows = self._conn.execute(
            "SELECT fact_text FROM user_memories WHERE user_id=? ORDER BY created_at ASC",
            (user_id,)).fetchall()
        return [r["fact_text"] for r in rows]

    def replace_memories(self, user_id: str, facts: list):
        """Replace the user's stored facts with `facts` (deduped, capped).

        Called by app.py after a chat, once the session context has
        been restored, so it never wipes memories that were only not
        yet loaded. Every query is scoped by user_id.
        """
        self._ensure_user(user_id)
        seen, clean = set(), []
        for fact in facts or []:
            fact = " ".join(str(fact).strip().split())
            if fact and len(fact) <= 200 and fact.lower() not in seen:
                seen.add(fact.lower())
                clean.append(fact)
        clean = clean[-20:]
        self._conn.execute("DELETE FROM user_memories WHERE user_id=?", (user_id,))
        now = self._now()
        for fact in clean:
            self._conn.execute(
                "INSERT INTO user_memories (memory_id, user_id, fact_text, created_at) VALUES (?,?,?,?)",
                (uuid.uuid4().hex, user_id, fact, now))
        self._conn.commit()
        return list(clean)

    def latest_conversation_messages(self, user_id: str, limit: int = 12):
        """Messages of the user's most recent conversation, oldest first."""
        convs = self.list_conversations(user_id, limit=1)
        if not convs:
            return []
        return self.list_messages(user_id, convs[0]["conversation_id"], limit=limit)

    # -- developer panel: query/login events + settings -----------
    # Best-effort analytics for the owner-only panel. No email is
    # ever stored or returned: the users table has no email column.
    def record_query_event(self, user_id: str, question: str, intent: str = "",
                           rule_id: str = "", confidence: float = 0.0,
                           is_fallback: bool = False, processing_ms: float = 0.0):
        self._conn.execute(
            "INSERT INTO query_events (event_id, user_id, question, intent, rule_id, confidence, is_fallback, processing_ms, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, user_id, (question or "")[:500], intent or "",
             rule_id or "", float(confidence or 0), 1 if is_fallback else 0,
             float(processing_ms or 0), self._now()))
        self._conn.commit()

    def record_login_event(self, user_id: str):
        self._conn.execute(
            "INSERT INTO login_events (event_id, user_id, created_at) VALUES (?,?,?)",
            (uuid.uuid4().hex, user_id, self._now()))
        self._conn.commit()

    def get_setting(self, key: str, default=None):
        row = self._conn.execute(
            "SELECT setting_value FROM dev_settings WHERE setting_key=?", (key,)).fetchone()
        return row["setting_value"] if row else default

    def set_setting(self, key: str, value):
        if isinstance(value, bool):
            value = "1" if value else "0"
        self._conn.execute(
            "INSERT INTO dev_settings (setting_key, setting_value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value, updated_at=excluded.updated_at",
            (key, str(value), self._now()))
        self._conn.commit()

    def analytics_overview(self, days: int = 7):
        now = self._now()
        window_start = now - max(1, days) * 86400
        today_start = now - (now % 86400)
        total_users = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        logins_today = self._conn.execute(
            "SELECT COUNT(*) FROM login_events WHERE created_at>=?", (today_start,)).fetchone()[0]
        logins_7d = self._conn.execute(
            "SELECT COUNT(*) FROM login_events WHERE created_at>=?", (window_start,)).fetchone()[0]
        questions_today = self._conn.execute(
            "SELECT COUNT(*) FROM query_events WHERE created_at>=?", (today_start,)).fetchone()[0]
        questions_7d = self._conn.execute(
            "SELECT COUNT(*) FROM query_events WHERE created_at>=?", (window_start,)).fetchone()[0]
        fallback_count = self._conn.execute(
            "SELECT COUNT(*) FROM query_events WHERE created_at>=? AND is_fallback=1", (window_start,)).fetchone()[0]
        documents_total = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {
            "total_users": total_users,
            "logins_today": logins_today,
            "logins_7d": logins_7d,
            "questions_today": questions_today,
            "questions_7d": questions_7d,
            "fallback_count_7d": fallback_count,
            "fallback_rate_7d": (fallback_count / questions_7d) if questions_7d else 0.0,
            "documents_total": documents_total,
        }

    def top_questions(self, days: int = 7, limit: int = 20):
        cutoff = self._now() - max(1, days) * 86400
        rows = self._conn.execute(
            "SELECT question, COUNT(*) AS c FROM query_events "
            "WHERE created_at>=? AND question != '' GROUP BY question ORDER BY c DESC, question ASC LIMIT ?",
            (cutoff, limit)).fetchall()
        return [{"question": r["question"], "count": r["c"]} for r in rows]

    def failed_questions(self, days: int = 7, limit: int = 20):
        cutoff = self._now() - max(1, days) * 86400
        rows = self._conn.execute(
            "SELECT question, COUNT(*) AS c FROM query_events "
            "WHERE created_at>=? AND is_fallback=1 AND question != '' GROUP BY question ORDER BY c DESC, question ASC LIMIT ?",
            (cutoff, limit)).fetchall()
        return [{"question": r["question"], "count": r["c"]} for r in rows]

    def recent_users(self, limit: int = 20):
        rows = self._conn.execute(
            "SELECT u.user_id, u.display_name, u.avatar_id, u.created_at, u.updated_at, "
            "COUNT(m.message_id) AS message_count, MAX(m.created_at) AS last_msg "
            "FROM users u LEFT JOIN messages m ON m.user_id = u.user_id "
            "GROUP BY u.user_id ORDER BY COALESCE(MAX(m.created_at), u.updated_at) DESC LIMIT ?",
            (limit,)).fetchall()
        return [{"user_id": r["user_id"], "display_name": r["display_name"],
                 "avatar_id": r["avatar_id"], "created_at": r["created_at"],
                 "updated_at": r["updated_at"], "message_count": r["message_count"],
                 "last_active": r["last_msg"] if r["last_msg"] is not None else r["updated_at"]}
                for r in rows]

    def prune_events(self, days: int = 30):
        cutoff = self._now() - max(1, days) * 86400
        q = self._conn.execute("DELETE FROM query_events WHERE created_at<?", (cutoff,))
        l = self._conn.execute("DELETE FROM login_events WHERE created_at<?", (cutoff,))
        self._conn.commit()
        return {"query_events": q.rowcount, "login_events": l.rowcount}

    def migrate_session_history(self, user_id: str, exchanges: list, confirm: bool = False):
        """Explicit migration only. confirm=False does nothing and
        returns migrated=0. confirm=True creates one conversation
        from the given exchanges [{role, text}] and returns its id
        and count. It is never called automatically on login."""
        if not confirm or not exchanges:
            return {"migrated": 0, "conversation_id": None}
        title = ""
        for ex in exchanges:
            if ex.get("role") == "user" and ex.get("text"):
                title = ex["text"][:80]
                break
        cid = self.create_conversation(user_id, title=title)
        count = 0
        for ex in exchanges:
            if ex.get("text"):
                self.save_message(user_id, cid, ex.get("role", "user"), ex["text"])
                count += 1
        return {"migrated": count, "conversation_id": cid}
