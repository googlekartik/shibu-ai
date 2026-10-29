"""Pinecone RAG configuration — env only, never printed secrets.

Inspection result (2026-10-30, this workspace): PINECONE_API_KEY and
PINECONE_INDEX_NAME are both absent from the environment, so live
Pinecone validation is BLOCKED here. That is not a failure of the
code: the code refuses to invent an index, create one, or change
one. Set the variables below in .env (local) or in Render
Environment Variables (hosted) — never in chat, never in git, never
in the frontend — and /api/rag/status will validate the real index.

Index-type rule (do not change an existing index):
- "dense": a normal vector index. The app generates embeddings with
  EMBEDDING_PROVIDER and upserts vectors with index.upsert(), and
  queries with index.query(vector=...). The embedding dimension
  MUST equal the index dimension, or validation fails and RAG is
  disabled. Fixing a mismatch means creating a NEW index with the
  right dimension and re-indexing — never an in-place change, and
  never without the owner's explicit approval.
- "integrated": a Pinecone integrated-embedding index. The app
  upserts raw text with index.upsert_records() and queries with
  index.search(inputs={"text": ...}). No local embedding model is
  used, and EMBEDDING_MODEL is ignored for this type.

Namespace rule: one namespace per authenticated Google user, derived
only from the Google sub (never a display name or email). A query
always runs in the caller's own namespace, so cross-user retrieval
is impossible at the Pinecone layer as well as in SQLite.
"""
from __future__ import annotations

import os
import re

KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "multilingual-e5-large": 1024,
    "llama-text-embed-v2": 1024,
    "all-MiniLM-L6-v2": 384,
}


def namespace_for_user(user_id: str) -> str:
    """google#123 -> user-123. Any other id is sanitised to
    letters/digits/dash only, so it is always a valid namespace."""
    sub = ""
    if user_id:
        sub = user_id.split("#", 1)[-1] if "#" in user_id else user_id
    sub = re.sub(r"[^A-Za-z0-9_-]", "-", sub)[:64] or "anonymous"
    return f"user-{sub}"


def embedding_dimension_for_model(model: str):
    return KNOWN_DIMENSIONS.get((model or "").strip())


def get_rag_config() -> dict:
    """Safe to return from /api/rag/status: presence booleans only,
    never a key value, never a key length, never a key prefix."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "openai").strip().lower()
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    api_key_present = bool(os.environ.get("PINECONE_API_KEY", "").strip())
    index_type = os.environ.get("PINECONE_INDEX_TYPE", "dense").strip().lower()
    if index_type not in ("dense", "integrated"):
        index_type = "dense"
    backend = os.environ.get("RAG_BACKEND", "auto").strip().lower()
    if backend not in ("auto", "pinecone", "sqlite"):
        backend = "auto"
    if backend == "auto":
        backend = "pinecone" if (api_key_present and index_name) else "sqlite"
    try:
        top_k = int(os.environ.get("RAG_TOP_K", "5"))
    except ValueError:
        top_k = 5
    try:
        min_score = float(os.environ.get("RAG_MIN_SCORE", "0.30"))
    except ValueError:
        min_score = 0.30
    try:
        chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", "800"))
    except ValueError:
        chunk_size = 800
    try:
        chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))
    except ValueError:
        chunk_overlap = 100
    return {
        "pinecone_configured": bool(api_key_present and index_name),
        "pinecone_api_key_present": api_key_present,
        "index_name": index_name,
        "index_type": index_type,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimension": embedding_dimension_for_model(model),
        "backend": backend,
        "top_k": max(1, min(top_k, 20)),
        "min_score": min_score,
        "chunk_size": max(200, chunk_size),
        "chunk_overlap": max(0, min(chunk_overlap, chunk_size // 2)),
        "sqlite_path": os.environ.get("SHIBU_SQLITE_PATH", "") or None,
    }


def validate_index_info(index_info: dict, config: dict):
    """Validate a described index against the configured embedding.

    index_info: {"dimension": int|None, "metric": str|None,
                 "index_type": "dense"|"integrated"|None}
    Returns (ok, errors list). Never creates or changes an index.
    A dimension mismatch is a migration requirement, not something
    to fix silently: the owner must approve a new index + re-index.
    """
    errors = []
    if not config["pinecone_configured"]:
        return False, ["Pinecone is not configured: set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env / Render Environment, never in chat."]
    info_type = (index_info or {}).get("index_type") or config["index_type"]
    if info_type != config["index_type"]:
        errors.append(
            f"Index type mismatch: configured PINECONE_INDEX_TYPE={config['index_type']} but the index reports {info_type}. "
            "Set PINECONE_INDEX_TYPE to the index's real type; do not change the index itself.")
    if config["index_type"] == "dense":
        dim = (index_info or {}).get("dimension")
        expected = config["embedding_dimension"]
        if dim is None:
            errors.append("Could not read the index dimension from Pinecone; RAG is disabled until the index can be described.")
        elif expected is None:
            errors.append(f"Unknown embedding dimension for model {config['embedding_model']}; set EMBEDDING_MODEL to a known model or record its dimension before enabling RAG.")
        elif int(dim) != int(expected):
            errors.append(
                f"Dimension mismatch: index dimension is {dim} but embedding model {config['embedding_model']} produces {expected}. "
                "Migration required: create a NEW index with dimension "
                f"{expected} (or switch EMBEDDING_MODEL to a {dim}-dimension model), then re-index all documents. "
                "Do not change the existing index in place, and do not upsert vectors of the wrong dimension.")
    metric = (index_info or {}).get("metric")
    if metric and str(metric).lower() not in ("cosine", "dotproduct", "euclidean"):
        errors.append(f"Unexpected similarity metric: {metric}. Expected cosine, dotproduct, or euclidean.")
    return (len(errors) == 0), errors
