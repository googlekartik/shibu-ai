"""Embedding providers for dense Pinecone indexes.

No fake vectors, ever: if no provider/key is configured, embed()
raises, and the caller must disable Pinecone RAG and report it as
blocked — not invent a vector and upsert it. The SQLite fallback
does not use embeddings at all (it uses token-overlap retrieval,
labelled as such), so tests can run without any embedding key.

Providers:
- openai: POST https://api.openai.com/v1/embeddings, model from
  EMBEDDING_MODEL (default text-embedding-3-small, 1536 dims).
- pinecone: POST https://api.pinecone.io/embed, model from
  EMBEDDING_MODEL, input_type "passage" for documents and "query"
  for questions. This is Pinecone Inference, not a local model.
"""
from __future__ import annotations

import os


class EmbeddingError(Exception):
    pass


def _post_json(url: str, headers: dict, payload: dict):
    import requests
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {})
            msg = msg.get("message", "") if isinstance(msg, dict) else str(msg)
        except Exception:
            msg = resp.text[:300]
        raise EmbeddingError(f"Embedding API returned HTTP {resp.status_code}: {msg}"[:300])
    return resp.json()


class OpenAIEmbedder:
    def __init__(self, api_key=None, model=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
        self.model = (model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")).strip()

    def embed(self, texts: list) -> list:
        if not self.api_key:
            raise EmbeddingError("OPENAI_API_KEY is not set; cannot generate embeddings.")
        if not texts:
            return []
        data = _post_json(
            "https://api.openai.com/v1/embeddings",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {"model": self.model, "input": texts})
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        if len(items) != len(texts):
            raise EmbeddingError("Embedding API returned a different number of vectors than requested.")
        return [item["embedding"] for item in items]


class PineconeInferenceEmbedder:
    def __init__(self, api_key=None, model=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("PINECONE_API_KEY", "")).strip()
        self.model = (model or os.environ.get("EMBEDDING_MODEL", "multilingual-e5-large")).strip()

    def embed(self, texts: list, input_type: str = "passage") -> list:
        if not self.api_key:
            raise EmbeddingError("PINECONE_API_KEY is not set; cannot generate embeddings.")
        if not texts:
            return []
        data = _post_json(
            "https://api.pinecone.io/embed",
            {"Api-Key": self.api_key, "Content-Type": "application/json"},
            {"model": self.model, "parameters": {"input_type": input_type, "truncate": "END"},
             "inputs": [{"text": t} for t in texts]})
        items = data.get("data", [])
        if len(items) != len(texts):
            raise EmbeddingError("Pinecone Inference returned a different number of vectors than requested.")
        return [item["values"] for item in items]


def get_embedder(config: dict):
    """Return an embedder for a dense index, or None when the
    provider is not configured. Callers must treat None as
    'Pinecone RAG blocked', never as a reason to fake vectors."""
    provider = (config or {}).get("embedding_provider", "openai")
    if provider == "openai":
        e = OpenAIEmbedder(model=config.get("embedding_model"))
        return e if e.api_key else None
    if provider == "pinecone":
        e = PineconeInferenceEmbedder(model=config.get("embedding_model"))
        return e if e.api_key else None
    return None
