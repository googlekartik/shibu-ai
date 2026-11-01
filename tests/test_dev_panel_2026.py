"""Developer panel Phase 1 tests (2026): SQLite/Postgres event and
settings stores, owner gate (fail-closed), DEV_PANEL mode,
announcement endpoint, web-search setting, and settings validation.
No live Postgres, no network, no real Google login: the Flask
session is set directly, exactly as the app stores a profile, and
Postgres is exercised through a fake connection."""
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import app as app_module
from services.dev_store import (DEFAULT_SETTINGS, get_dev_setting,
                                get_dev_store, set_dev_setting)
from services.postgres_store import PostgresStore
from services.sqlite_store import SQLiteStore


@pytest.fixture()
def dev_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIBU_SQLITE_PATH", str(tmp_path / "dev-panel.db"))
    monkeypatch.setenv("RAG_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.delenv("DEV_PANEL", raising=False)
    monkeypatch.setenv("DEV_OWNER_EMAIL", "owner@example.com")
    monkeypatch.delenv("DEV_OWNER_EMAILS", raising=False)
    monkeypatch.delenv("DEV_OWNER_SUB", raising=False)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _login(client, user_id="google#owner1", email="owner@example.com", name="Owner"):
    with client.session_transaction() as sess:
        sess["google_user_id"] = user_id
        sess["google_email"] = email
        sess["google_display_name"] = name
        # Completed onboarding, so /api/chat reaches the normal
        # pipeline instead of the welcome flow (which is not recorded).
        sess["onboarding_completed"] = True
        sess["profile_avatar_id"] = "maya"


def _store_for(dev_client):
    return app_module._sqlite_store()


# -- SQLite schema / settings ---------------------------------------

def test_new_tables_exist_on_fresh_db(tmp_path):
    store = SQLiteStore(path=str(tmp_path / "fresh.db"))
    tables = {r[0] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"query_events", "login_events", "dev_settings"} <= tables


def test_new_tables_added_to_old_db(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE users (user_id TEXT PRIMARY KEY, display_name TEXT DEFAULT '', "
        "avatar_id TEXT, onboarding_completed INTEGER DEFAULT 0, created_at REAL, updated_at REAL);")
    conn.commit()
    conn.close()
    store = SQLiteStore(path=path)
    tables = {r[0] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"query_events", "login_events", "dev_settings"} <= tables
    store.record_query_event("google#x", "hello", intent="UNKNOWN", is_fallback=True)


def test_settings_roundtrip_and_bool_encoding(dev_client):
    store = _store_for(dev_client)
    assert store.get_setting("web_search_enabled", None) is None
    store.set_setting("web_search_enabled", False)
    store.set_setting("announcement", "Hello")
    assert store.get_setting("web_search_enabled") == "0"
    assert store.get_setting("announcement") == "Hello"
    assert get_dev_setting(store, "web_search_enabled") is False
    assert get_dev_setting(store, "announcement") == "Hello"
    assert get_dev_setting(store, "ensemble_enabled") is False
    store.set_setting("web_search_enabled", True)
    assert get_dev_setting(store, "web_search_enabled") is True


def test_events_and_overview_aggregation(dev_client):
    store = _store_for(dev_client)
    store.get_or_create_user("google#u1", display_name="User One")
    store.get_or_create_user("google#u2", display_name="User Two")
    store.record_login_event("google#u1")
    store.record_query_event("google#u1", "what is python", intent="DEFINITION", rule_id="r1", confidence=0.9)
    store.record_query_event("google#u1", "what is python", intent="DEFINITION", rule_id="r1", confidence=0.9)
    store.record_query_event("google#u2", "zxqv gibberish", intent="UNKNOWN", rule_id="unknown", confidence=0.1, is_fallback=True)
    overview = store.analytics_overview(days=7)
    assert overview["total_users"] == 2
    assert overview["logins_7d"] == 1
    assert overview["questions_7d"] == 3
    assert overview["fallback_count_7d"] == 1
    assert overview["fallback_rate_7d"] == pytest.approx(1 / 3)
    assert store.top_questions(days=7)[0] == {"question": "what is python", "count": 2}
    assert store.failed_questions(days=7) == [{"question": "zxqv gibberish", "count": 1}]
    users = store.recent_users()
    assert {u["user_id"] for u in users} == {"google#u1", "google#u2"}
    assert all("email" not in u for u in users)


def test_prune_deletes_only_old_events(dev_client):
    store = _store_for(dev_client)
    store.record_query_event("google#u1", "recent question")
    store.record_login_event("google#u1")
    old = time.time() - 40 * 86400
    store._conn.execute("UPDATE query_events SET created_at=?", (old,))
    store._conn.execute("UPDATE login_events SET created_at=?", (old,))
    store._conn.commit()
    store.record_query_event("google#u1", "new question")
    pruned = store.prune_events(days=30)
    assert pruned == {"query_events": 1, "login_events": 1}
    assert [q["question"] for q in store.top_questions(days=90)] == ["new question"]


# -- owner gate ------------------------------------------------------

def test_dev_redirects_anonymous_to_login(dev_client):
    res = dev_client.get("/dev")
    assert res.status_code == 302
    assert res.headers["Location"].endswith("/login")


def test_dev_api_anonymous_401_and_non_owner_403(dev_client):
    assert dev_client.get("/api/dev/overview").status_code == 401
    _login(dev_client, user_id="google#other", email="someone@else.com", name="Someone")
    assert dev_client.get("/api/dev/overview").status_code == 403
    res = dev_client.get("/dev")
    assert res.status_code == 403
    assert b"restricted to the owner" in res.data


def test_dev_owner_200_and_overview_shape(dev_client):
    store = _store_for(dev_client)
    store.get_or_create_user("google#owner1", display_name="Owner")
    store.record_login_event("google#owner1")
    _login(dev_client)
    assert dev_client.get("/dev").status_code == 200
    data = dev_client.get("/api/dev/overview").get_json()
    assert data["total_users"] == 1
    assert data["store"] == "sqlite"
    assert data["settings"]["web_search_enabled"] is True
    assert data["settings"]["ensemble_enabled"] is False
    assert data["settings"]["announcement"] == ""
    assert dev_client.get("/api/dev/users").get_json()["users"][0]["display_name"] == "Owner"


def test_panel_locked_when_no_owner_env_set(dev_client, monkeypatch):
    monkeypatch.delenv("DEV_OWNER_EMAIL", raising=False)
    monkeypatch.delenv("DEV_OWNER_EMAILS", raising=False)
    monkeypatch.delenv("DEV_OWNER_SUB", raising=False)
    _login(dev_client)
    assert dev_client.get("/dev").status_code == 403
    assert dev_client.get("/api/dev/overview").status_code == 403


def test_owner_match_by_email_case_insensitive(dev_client):
    _login(dev_client, email="OWNER@Example.com")
    assert dev_client.get("/api/dev/overview").status_code == 200


# -- DEV_PANEL mode ---------------------------------------------------

def test_dev_panel_mode_redirects_and_disables_chat(dev_client, monkeypatch):
    monkeypatch.setenv("DEV_PANEL", "1")
    _login(dev_client)
    res = dev_client.get("/")
    assert res.status_code == 302
    assert res.headers["Location"].endswith("/dev")
    res = dev_client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 403
    assert "Developer panel mode" in res.get_json()["error"]


# -- announcement ------------------------------------------------------

def test_public_settings_returns_saved_announcement(dev_client):
    assert dev_client.get("/api/public/settings").get_json() == {"announcement": ""}
    _store_for(dev_client).set_setting("announcement", "Maintenance tonight")
    assert dev_client.get("/api/public/settings").get_json() == {"announcement": "Maintenance tonight"}


# -- settings validation ------------------------------------------------

def test_settings_post_validation(dev_client):
    _login(dev_client)
    assert dev_client.post("/api/dev/settings", json={"key": "nope", "value": True}).status_code == 400
    assert dev_client.post("/api/dev/settings",
                            json={"key": "web_search_enabled", "value": "false"}).status_code == 400
    assert dev_client.post("/api/dev/settings",
                            json={"key": "announcement", "value": "x" * 501}).status_code == 400
    res = dev_client.post("/api/dev/settings", json={"key": "announcement", "value": "Hi there"})
    assert res.status_code == 200
    assert res.get_json()["settings"]["announcement"] == "Hi there"


# -- web search setting --------------------------------------------------

def test_web_search_disabled_returns_graceful_fallback_without_web(dev_client):
    _store_for(dev_client).set_setting("web_search_enabled", False)
    res = dev_client.post("/api/chat", json={"message": "zxqvkj asdfghjkl qwertyuiop nonsense"})
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["intent"] == "UNKNOWN"
    assert "not sure about that one" in payload["response"]
    assert payload.get("search_results") in (None, [])
    # Re-enable and confirm the setting round-trips through the API.
    _store_for(dev_client).set_setting("web_search_enabled", True)
    assert get_dev_setting(_store_for(dev_client), "web_search_enabled") is True


def test_chat_records_query_event(dev_client):
    _login(dev_client, name="Recorder")
    dev_client.post("/api/chat", json={"message": "What is Python?"})
    store = _store_for(dev_client)
    assert store.analytics_overview(days=7)["questions_7d"] == 1
    assert store.top_questions(days=7)[0]["question"] == "What is Python?"


# -- Postgres store with a fake connection -------------------------------

class _FakeCursor:
    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._fake.sql.append(sql)
        self._fake.params.append(params)
        return self

    def fetchone(self):
        return self._fake.results.pop(0) if self._fake.results else None

    def fetchall(self):
        return self._fake.results.pop(0) if self._fake.results else []

    @property
    def rowcount(self):
        return 1


class _FakeConnection:
    def __init__(self, results):
        self.results = list(results)
        self.sql = []
        self.params = []

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


def test_postgres_store_uses_placeholders_conflict_and_mapping():
    fake = _FakeConnection(results=[[("q1", 3), ("q2", 1)]])
    store = PostgresStore(dsn="postgresql://fake/db",
                          connect_fn=lambda dsn: fake)
    store.record_query_event("google#u1", "q1", intent="UNKNOWN", is_fallback=True)
    store.set_setting("web_search_enabled", False)
    top = store.top_questions(days=7)
    assert top == [{"question": "q1", "count": 3}, {"question": "q2", "count": 1}]
    assert fake.sql, "no SQL was recorded"
    assert all("%s" in sql for sql in fake.sql), fake.sql
    assert not any("?" in sql for sql in fake.sql), fake.sql
    joined = " ".join(fake.sql)
    assert "ON CONFLICT" in joined
    assert "query_events" in joined and "dev_settings" in joined


def test_postgres_ping_failure_falls_back_to_sqlite(dev_client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://bad:bad@localhost:5432/nope")
    sqlite = _store_for(dev_client)
    assert get_dev_store(sqlite) is sqlite
    assert get_dev_setting(sqlite, "web_search_enabled") is True


def test_dev_setting_never_raises_on_broken_store():
    class _Broken:
        def get_setting(self, key, default=None):
            raise RuntimeError("boom")

    assert get_dev_setting(_Broken(), "web_search_enabled") is True
    assert get_dev_setting(_Broken(), "announcement") == ""
    with pytest.raises(ValueError):
        set_dev_setting(_Broken(), "not_a_setting", True)
    assert DEFAULT_SETTINGS["ensemble_enabled"] is False
