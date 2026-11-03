"""Embedding providers for dense Pinecone indexes.

No fake vectors, ever: if no provider/key is configured, embed()
raises, and the caller must disable Pinecone RAG and report it as
blocked — not invent a vector and upsert it. The SQLite fallback
does not use embeddings at all (it uses token-overlap retrieval,
labelled as such), so tests can run without any embedding key.

Providers:
- gemini: POST {GEMINI_API_BASE}/models/{GEMINI_EMBEDDING_MODEL}:
  embedContent, key via x-goog-api-key header, model from
  GEMINI_EMBEDDING_MODEL (default text-embedding-004, 768 dims).
  Uses the same GEMINI_API_KEY as chat generation — no separate
  embedding key, and no other provider is called.
- pinecone: POST https://api.pinecone.io/embed, model from
  EMBEDDING_MODEL, input_type "passage" for documents and "query"
  for questions. This is Pinecone Inference, not a local model.

Integrated Pinecone indexes (PINECONE_INDEX_TYPE=integrated)
use neither embedder: the server embeds raw text. This module is
only consulted for dense indexes.
"""
from __future__ import annotations

import os

from services.retry import retry_cloud_call


class EmbeddingError(Exception):
    pass


def _post_json(url: str, headers: dict, payload: dict, attempts: int = 3):
    import requests

    def send():
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(5, 60))
        except requests.RequestException as exc:
            raise EmbeddingError(f"Embedding API request failed: {exc}"[:300]) from exc
        if resp.status_code != 200:
            try:
                msg = resp.json().get("error", {})
                msg = msg.get("message", "") if isinstance(msg, dict) else str(msg)
            except Exception:
                msg = resp.text[:300]
            error = EmbeddingError(f"Embedding API returned HTTP {resp.status_code}: {msg}"[:300])
            error.status_code = resp.status_code
            raise error
        return resp.json()

    return retry_cloud_call(send, attempts=attempts)


class GeminiEmbedder:
    """Gemini text embeddings via the same REST base/key as chat."""

    def __init__(self, api_key=None, model=None, api_base=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.environ.get("GEMINI_EMBEDDING_MODEL", "")).strip() or "text-embedding-004"
        self.api_base = (api_base if api_base is not None
                         else os.environ.get("GEMINI_API_BASE", "")).strip() \
            or "https://generativelanguage.googleapis.com/v1beta"

    def embed(self, texts: list, input_type: str = "passage", attempts: int = 3) -> list:
        if not self.api_key:
            raise EmbeddingError("GEMINI_API_KEY is not set; cannot generate embeddings.")
        if not texts:
            return []
        vectors = []
        # embedContent takes one content per call; batch via embedBatch
        # is avoided so a single failure mode stays diagnosable.
        for text in texts:
            task = "retrievalQuery" if input_type == "query" else "retrievalDocument"
            data = _post_json(
                f"{self.api_base.rstrip('/')}/models/{self.model}:embedContent",
                {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                {"model": f"models/{self.model}",
                 "content": {"parts": [{"text": text}]},
                 "taskType": task},
                attempts=attempts)
            values = ((data.get("embedding") or {}).get("values"))
            if not isinstance(values, list) or not values:
                raise EmbeddingError("Gemini embedding API returned no vector.")
            vectors.append(values)
        return vectors


class PineconeInferenceEmbedder:
    def __init__(self, api_key=None, model=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("PINECONE_API_KEY", "")).strip()
        self.model = (model or os.environ.get("EMBEDDING_MODEL", "multilingual-e5-large")).strip()

    def embed(self, texts: list, input_type: str = "passage", attempts: int = 3) -> list:
        if not self.api_key:
            raise EmbeddingError("PINECONE_API_KEY is not set; cannot generate embeddings.")
        if not texts:
            return []
        data = _post_json(
            "https://api.pinecone.io/embed",
            {"Api-Key": self.api_key, "Content-Type": "application/json"},
            {"model": self.model, "parameters": {"input_type": input_type, "truncate": "END"},
             "inputs": [{"text": t} for t in texts]}, attempts=attempts)
        items = data.get("data", [])
        if len(items) != len(texts):
            raise EmbeddingError("Pinecone Inference returned a different number of vectors than requested.")
        return [item["values"] for item in items]


def get_embedder(config: dict):
    """Return an embedder for a dense index, or None when the
    provider is not configured. Callers must treat None as
    'Pinecone RAG blocked', never as a reason to fake vectors."""
    provider = (config or {}).get("embedding_provider", "gemini")
    if provider == "gemini":
        e = GeminiEmbedder(model=config.get("embedding_model"))
        return e if e.api_key else None
    if provider == "pinecone":
        # The Gemini default model name is not a Pinecone Inference
        # model: when EMBEDDING_MODEL was never set for Pinecone,
        # fall through to the Pinecone default instead of sending
        # a Gemini model id to Pinecone.
        model = config.get("embedding_model")
        if model in (None, "", "text-embedding-004"):
            model = None
        e = PineconeInferenceEmbedder(model=model)
        return e if e.api_key else None
    return None
