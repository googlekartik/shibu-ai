"""Developer-panel store selection + settings (Phase 1).

Selection: Postgres (Supabase) when DATABASE_URL is set AND a ping
succeeds, otherwise the SQLite store passed in. Every selection
re-pings: there is no cached failure state, so a Postgres outage
falls back to SQLite and recovers on the next call without a
restart. Settings reads never raise: on failure they return the
default value.

Settings keys are allowlisted to exactly DEFAULT_SETTINGS
everywhere: web_search_enabled, ensemble_enabled, announcement.
ensemble_enabled is stored and shown only; Phase 3 gives it
behaviour. Booleans are stored as "1"/"0" in both backends.
"""
from __future__ import annotations

import os

from services.postgres_store import PostgresStore

DEFAULT_SETTINGS = {"web_search_enabled": True, "ensemble_enabled": False, "announcement": ""}
SETTING_KEYS = frozenset(DEFAULT_SETTINGS)


def store_kind(store) -> str:
    return "postgres" if isinstance(store, PostgresStore) else "sqlite"


def get_dev_store(sqlite_store):
    """Postgres when configured and reachable, else the SQLite store."""
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        return sqlite_store
    try:
        pg = PostgresStore(dsn=dsn)
        if pg.ping():
            return pg
    except Exception:
        pass
    return sqlite_store


def _coerce(key, raw):
    default = DEFAULT_SETTINGS[key]
    if raw is None:
        return default
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    return str(raw)


def get_dev_setting(store, key, fallback_store=None):
    """Read one setting; never raises. Unknown keys return None."""
    if key not in SETTING_KEYS:
        return None
    try:
        return _coerce(key, store.get_setting(key, None))
    except Exception:
        pass
    if fallback_store is not None and fallback_store is not store:
        try:
            return _coerce(key, fallback_store.get_setting(key, None))
        except Exception:
            pass
    return DEFAULT_SETTINGS[key]


def get_dev_settings_all(store, fallback_store=None):
    return {key: get_dev_setting(store, key, fallback_store) for key in DEFAULT_SETTINGS}


def set_dev_setting(store, key, value, fallback_store=None):
    """Write one setting. Unknown keys raise ValueError; bool keys
    require a real bool; the announcement is capped at 500 chars."""
    if key not in SETTING_KEYS:
        raise ValueError(f"Unknown setting: {key}")
    if isinstance(DEFAULT_SETTINGS[key], bool) and not isinstance(value, bool):
        raise ValueError(f"Setting {key} requires true/false.")
    if key == "announcement":
        if not isinstance(value, str):
            raise ValueError("Setting announcement requires a string.")
        value = value[:500]
    try:
        store.set_setting(key, value)
        return
    except Exception:
        pass
    if fallback_store is not None and fallback_store is not store:
        fallback_store.set_setting(key, value)
        return
    raise RuntimeError("Could not save the setting to any store.")
