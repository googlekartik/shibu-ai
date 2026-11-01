"""Postgres store for the developer panel (Phase 1).

Same tables/columns as services/sqlite_store.py, reached through
Supabase Postgres. The schema must already exist — it is created by
infra/supabase_schema.sql run in the Supabase SQL Editor. This module
ASSUMES the schema; it never creates tables.

Rules, kept strict:
- `psycopg` is imported lazily inside _connect(): the app must run
  without it installed whenever DATABASE_URL is unset.
- One new connection per operation, connect_timeout=3, %s
  placeholders only. No cached connection, no cached failure state.
- On any error the caller (services/dev_store) falls back to the
  SQLite store. This class itself raises; it never hides failures
  from the selection logic.
- Epoch timestamps are DOUBLE PRECISION, matching the SQLite REAL
  columns. No email is ever selected or returned: the users table
  has no email column.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager


class PostgresStore:
    def __init__(self, dsn: str = None, connect_fn=None):
        self.dsn = dsn or os.environ.get("DATABASE_URL", "").strip()
        self._connect_fn = connect_fn

    @contextmanager
    def _connect(self):
        if self._connect_fn is not None:
            conn = self._connect_fn(self.dsn)
            try:
                yield conn
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            return
        import psycopg  # lazy: only needed when DATABASE_URL is set
        conn = psycopg.connect(self.dsn, connect_timeout=3)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _now(self):
        return time.time()

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            return False

    # -- events -----------------------------------------------------
    def record_query_event(self, user_id: str, question: str, intent: str = "",
                           rule_id: str = "", confidence: float = 0.0,
                           is_fallback: bool = False, processing_ms: float = 0.0):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO query_events (event_id, user_id, question, intent, rule_id, confidence, is_fallback, processing_ms, created_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                    (uuid.uuid4().hex, user_id, (question or "")[:500], intent or "",
                     rule_id or "", float(confidence or 0), 1 if is_fallback else 0,
                     float(processing_ms or 0), self._now()))

    def record_login_event(self, user_id: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO login_events (event_id, user_id, created_at) "
                    "VALUES (%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                    (uuid.uuid4().hex, user_id, self._now()))

    # -- settings ---------------------------------------------------
    def get_setting(self, key: str, default=None):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT setting_value FROM dev_settings WHERE setting_key=%s", (key,))
                row = cur.fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value):
        if isinstance(value, bool):
            value = "1" if value else "0"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO dev_settings (setting_key, setting_value, updated_at) VALUES (%s,%s,%s) "
                    "ON CONFLICT (setting_key) DO UPDATE SET setting_value=EXCLUDED.setting_value, updated_at=EXCLUDED.updated_at",
                    (key, str(value), self._now()))

    # -- analytics --------------------------------------------------
    def analytics_overview(self, days: int = 7):
        now = self._now()
        window_start = now - max(1, days) * 86400
        today_start = now - (now % 86400)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM users")
                total_users = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM login_events WHERE created_at>=%s", (today_start,))
                logins_today = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM login_events WHERE created_at>=%s", (window_start,))
                logins_7d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM query_events WHERE created_at>=%s", (today_start,))
                questions_today = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM query_events WHERE created_at>=%s", (window_start,))
                questions_7d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM query_events WHERE created_at>=%s AND is_fallback=1", (window_start,))
                fallback_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM documents")
                documents_total = cur.fetchone()[0]
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
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT question, COUNT(*) FROM query_events "
                    "WHERE created_at>=%s AND question <> '' GROUP BY question ORDER BY COUNT(*) DESC, question ASC LIMIT %s",
                    (cutoff, limit))
                rows = cur.fetchall()
        return [{"question": r[0], "count": r[1]} for r in rows]

    def failed_questions(self, days: int = 7, limit: int = 20):
        cutoff = self._now() - max(1, days) * 86400
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT question, COUNT(*) FROM query_events "
                    "WHERE created_at>=%s AND is_fallback=1 AND question <> '' GROUP BY question ORDER BY COUNT(*) DESC, question ASC LIMIT %s",
                    (cutoff, limit))
                rows = cur.fetchall()
        return [{"question": r[0], "count": r[1]} for r in rows]

    def recent_users(self, limit: int = 20):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT u.user_id, u.display_name, u.avatar_id, u.created_at, u.updated_at, "
                    "COUNT(m.message_id), MAX(m.created_at) "
                    "FROM users u LEFT JOIN messages m ON m.user_id = u.user_id "
                    "GROUP BY u.user_id, u.display_name, u.avatar_id, u.created_at, u.updated_at "
                    "ORDER BY COALESCE(MAX(m.created_at), u.updated_at) DESC LIMIT %s",
                    (limit,))
                rows = cur.fetchall()
        return [{"user_id": r[0], "display_name": r[1], "avatar_id": r[2],
                 "created_at": r[3], "updated_at": r[4], "message_count": r[5],
                 "last_active": r[6] if r[6] is not None else r[4]} for r in rows]

    def prune_events(self, days: int = 30):
        cutoff = self._now() - max(1, days) * 86400
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM query_events WHERE created_at<%s", (cutoff,))
                deleted_q = cur.rowcount
                cur.execute("DELETE FROM login_events WHERE created_at<%s", (cutoff,))
                deleted_l = cur.rowcount
        return {"query_events": deleted_q, "login_events": deleted_l}
