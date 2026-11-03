"""Durable diagnostic sidecars: structure-only failure notes that
survive a process restart within one deployment.

Process memory (module globals) is wiped when the host restarts the
service — on Render's free tier that happens routinely (spin-down,
redeploys) — so a provider/Pinecone failure recorded in one process
was invisible to the next /api/llm/status or /api/rag/status call,
which then reported "no error" after a template/fallback answer.
These small JSON files live beside the SQLite database (wiped on
redeploy, like the database itself), so a diagnostic survives
restarts but never goes stale across deployments.

Only structure-only payloads may be written here (endpoint names,
HTTP statuses, field names, exception class names). Callers are
responsible for never passing values, keys, or payload text.
Writes are best-effort and atomic (temp file + replace); reads
return None when the file is absent or unreadable. Nothing here
ever raises.
"""
from __future__ import annotations

import json
import os


def diagnostic_path(name: str) -> str:
    """File for one diagnostic. Overridable for tests via
    SHIBU_DIAGNOSTIC_DIR; otherwise beside SHIBU_SQLITE_PATH when
    set, else in the project's data/ directory."""
    override = os.environ.get("SHIBU_DIAGNOSTIC_DIR", "").strip()
    if override:
        base = override
    else:
        sqlite_path = os.environ.get("SHIBU_SQLITE_PATH", "").strip()
        if sqlite_path:
            base = os.path.dirname(os.path.abspath(sqlite_path))
        else:
            base = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    return os.path.join(base, f"{name}_diagnostic.json")


def write_diagnostic(name: str, payload: dict) -> None:
    """Persist one structure-only diagnostic. Never raises."""
    try:
        if not isinstance(payload, dict):
            return
        path = diagnostic_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except Exception:
        pass


def read_diagnostic(name: str) -> dict | None:
    """The persisted diagnostic, or None when absent/unreadable."""
    try:
        with open(diagnostic_path(name), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def delete_diagnostic(name: str) -> None:
    """Remove a persisted diagnostic (e.g. a failure note that a
    later success has superseded). Never raises."""
    try:
        os.remove(diagnostic_path(name))
    except Exception:
        pass
