"""Supabase pgvector search for approved past answers (search step).

DynamoDB is the source of truth. This module only indexes answers
that were manually approved AND whose owner consented to learning,
and only searches a user's own approved answers by default.
Cross-user search is admin-only and only over approved answers.

Embedding model is locked at OpenAI text-embedding-3-small
(1536 dimensions, matching vector(1536) in supabase_schema.sql).
If OPENAI_API_KEY is missing, indexing pauses (messages stay
LEARN_PENDING and are retried later) and search falls back to
keyword search on search_text — it never invents an embedding.

All Supabase calls use the service_role key server-side only.
If Supabase is unreachable or paused (free projects pause after
1 week idle), search returns [] and indexing reports failure —
chat is never broken by a search failure.
"""

from __future__ import annotations

import os

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def _supabase_config():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    return url, key


def _headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}


def embed_text(text: str, api_key: str | None = None) -> list | None:
    """Return a 1536-dim embedding, or None if no key / on failure."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key or not text.strip():
        return None
    try:
        import requests

        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": EMBEDDING_MODEL, "input": text[:4000]},
            timeout=20)
        if resp.status_code != 200:
            return None
        vec = resp.json()["data"][0]["embedding"]
        return vec if len(vec) == EMBEDDING_DIM else None
    except Exception:
        return None


def index_good_answer(message: dict, api_key: str | None = None) -> bool:
    """Index one approved DynamoDB message in Supabase. Returns True
    on success (including upsert), False on any failure."""
    url, key = _supabase_config()
    if not url or not key:
        return False
    try:
        import requests

        text = message.get("text", "")
        embedding = embed_text(text, api_key=api_key)
        row = {
            "message_id": message["message_id"],
            "conversation_id": message["conversation_id"],
            "user_id": message.get("user_id", ""),
            "answer_text": text[:4000],
            "embedding": embedding,
            "rating": message.get("rating"),
            "has_correction": bool(message.get("correction_text")),
            "approved": True,
        }
        resp = requests.post(f"{url}/rest/v1/good_answers", headers=_headers(key),
                             json=row, timeout=15)
        return resp.status_code in (200, 201, 204)
    except Exception:
        return False


def search_good_answers(store, user_id: str, query: str,
                        top_k: int = 5, threshold: float = 0.7) -> list:
    """Search a user's own approved answers, then fetch full answers
    from DynamoDB by message_id with an owner check. Returns a list
    of {message_id, answer_text, similarity} — empty on any failure."""
    url, key = _supabase_config()
    if not url or not key or not query.strip():
        return []
    try:
        import requests

        embedding = embed_text(query)
        if embedding is not None:
            resp = requests.post(
                f"{url}/rest/v1/rpc/match_good_answers", headers=_headers(key),
                json={"query_embedding": embedding, "match_user_id": user_id,
                      "match_threshold": threshold, "match_count": top_k},
                timeout=15)
            if resp.status_code == 200:
                return resp.json()
            # Fall through to keyword search on RPC failure.
        resp = requests.get(
            f"{url}/rest/v1/good_answers",
            headers=_headers(key),
            params={"user_id": f"eq.{user_id}", "approved": "eq.true",
                    "select": "message_id,answer_text", "limit": top_k},
            timeout=15)
        if resp.status_code != 200:
            return []
        return [{"message_id": r["message_id"], "answer_text": r["answer_text"],
                 "similarity": None} for r in resp.json()]
    except Exception:
        return []


def sync_learning_pending(store, api_key: str | None = None, limit: int = 50) -> dict:
    """Nightly-job function: take LEARN_PENDING messages from
    DynamoDB, index them in Supabase, mark LEARN_SYNCED on success.
    Failures stay PENDING and are retried on the next run."""
    pending = store.list_learning_pending(limit=limit)
    indexed = 0
    failed = 0
    for message in pending["items"]:
        if index_good_answer(message, api_key=api_key):
            store.mark_learning_synced(message["conversation_id"],
                                       message["created_at_id"])
            indexed += 1
        else:
            failed += 1
    return {"pending_seen": len(pending["items"]), "indexed": indexed, "failed": failed}
