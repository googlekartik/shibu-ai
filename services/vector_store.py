"""Pinecone vector store — validate-only, never create/change an index.

API selection by index type (pinecone SDK 10.x, verified):
- dense: index.upsert(vectors=[{id, values, metadata}], namespace)
         index.query(vector=..., top_k, namespace, include_metadata=True)
- integrated: index.upsert_records(records=[{id, text, metadata}], namespace)
              index.search(namespace, inputs={"text": query}, top_k, filter)

The client is created lazily, so importing this module (and running
the app without Pinecone) never requires the pinecone package or a
key. describe() is the inspection step: it reads dimension/metric
from pc.describe_index() and lets rag_config.validate_index_info()
decide whether RAG may run. This module never calls create_index,
delete_index, or configure_index.
"""
from __future__ import annotations

import os


class PineconeError(Exception):
    pass


def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    out = {}
    for attr in ("dimension", "metric", "name", "status", "spec"):
        if hasattr(obj, attr):
            try:
                out[attr] = getattr(obj, attr)
            except Exception:
                pass
    # spec may itself be an object with a pod/serverless type
    spec = out.get("spec")
    if spec is not None and not isinstance(spec, dict):
        spec_dict = {}
        for attr in ("pod", "serverless"):
            if hasattr(spec, attr):
                spec_dict[attr] = getattr(spec, attr)
        out["spec"] = spec_dict
    return out


class PineconeVectorStore:
    def __init__(self, api_key=None, index_name=None, index_type="dense"):
        self.api_key = (api_key if api_key is not None else os.environ.get("PINECONE_API_KEY", "")).strip()
        self.index_name = (index_name or os.environ.get("PINECONE_INDEX_NAME", "")).strip()
        self.index_type = index_type
        self._client = None
        self._index = None

    def _connect(self):
        if self._index is not None:
            return self._index
        if not self.api_key or not self.index_name:
            raise PineconeError("Pinecone is not configured: PINECONE_API_KEY and PINECONE_INDEX_NAME are required.")
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise PineconeError("The 'pinecone' package is not installed. Install it with: pip install pinecone") from exc
        self._client = Pinecone(api_key=self.api_key)
        self._index = self._client.Index(self.index_name)
        return self._index

    def describe(self) -> dict:
        """Return {"dimension", "metric", "index_type"} without
        creating or changing anything. index_type is taken from
        the configured type unless the describe response carries an
        embed configuration, in which case it is integrated."""
        try:
            index = self._connect()
            info = _as_dict(self._client.describe_index(self.index_name))
        except Exception as exc:
            raise PineconeError(f"Could not describe Pinecone index '{self.index_name}': {exc}"[:300]) from exc
        result = {
            "dimension": info.get("dimension"),
            "metric": info.get("metric"),
            "index_type": self.index_type,
        }
        # Integrated indexes carry an embed spec in describe output.
        try:
            spec = info.get("spec") or {}
            if isinstance(spec, dict) and ("embed" in spec or "model" in str(spec)):
                result["index_type"] = "integrated"
        except Exception:
            pass
        return result

    def upsert(self, namespace: str, items: list) -> int:
        """items: dense -> {id, values, metadata}; integrated ->
        {id, text, metadata}. Returns the number of items sent.
        Upsert by stable chunk id is idempotent: re-ingesting the
        same document overwrites, it does not duplicate."""
        if not items:
            return 0
        index = self._connect()
        try:
            if self.index_type == "integrated":
                records = [{"id": it["id"], "text": it["text"],
                            **({"metadata": it["metadata"]} if it.get("metadata") else {})}
                           for it in items]
                index.upsert_records(records=records, namespace=namespace)
            else:
                vectors = [{"id": it["id"], "values": it["values"],
                            "metadata": it.get("metadata", {})} for it in items]
                index.upsert(vectors=vectors, namespace=namespace)
        except Exception as exc:
            raise PineconeError(f"Pinecone upsert failed: {exc}"[:300]) from exc
        return len(items)

    def query(self, namespace: str, query_text: str = None, vector: list = None,
              top_k: int = 5, document_id: str = None) -> list:
        """Return [{id, score, text, metadata}]. Dense indexes need
        a vector; integrated indexes need query_text. A filter on
        document_id is applied only when given (per-document search);
        the namespace already isolates the user."""
        index = self._connect()
        filt = {"document_id": {"$eq": document_id}} if document_id else None
        try:
            if self.index_type == "integrated":
                if not query_text:
                    raise PineconeError("Integrated indexes need query text, not a vector.")
                resp = index.search(namespace=namespace, inputs={"text": query_text},
                                    top_k=top_k, filter=filt)
                hits = []
                # SDK shapes differ by version: result.hits, or dict result.
                raw_hits = []
                try:
                    raw_hits = resp.result.hits  # type: ignore[attr-defined]
                except Exception:
                    if isinstance(resp, dict):
                        raw_hits = (resp.get("result") or {}).get("hits", [])
                for h in raw_hits:
                    if isinstance(h, dict):
                        fields = h.get("fields", {}) or {}
                        hits.append({"id": h.get("_id") or h.get("id"),
                                     "score": h.get("_score", 0.0),
                                     "text": fields.get("text", ""),
                                     "metadata": fields})
                    else:
                        fields = getattr(h, "fields", {}) or {}
                        hits.append({"id": getattr(h, "_id", None),
                                     "score": getattr(h, "_score", 0.0),
                                     "text": fields.get("text", "") if isinstance(fields, dict) else "",
                                     "metadata": fields if isinstance(fields, dict) else {}})
                return hits
            if vector is None:
                raise PineconeError("Dense indexes need a query vector; no embedding provider is configured.")
            resp = index.query(vector=vector, top_k=top_k, namespace=namespace,
                               include_metadata=True, filter=filt)
            matches = resp.matches if hasattr(resp, "matches") else resp.get("matches", [])
            out = []
            for m in matches:
                if isinstance(m, dict):
                    out.append({"id": m.get("id"), "score": m.get("score", 0.0),
                                "text": (m.get("metadata") or {}).get("text", ""),
                                "metadata": m.get("metadata", {})})
                else:
                    md = getattr(m, "metadata", {}) or {}
                    out.append({"id": getattr(m, "id", None), "score": getattr(m, "score", 0.0),
                                "text": md.get("text", ""), "metadata": md})
            return out
        except PineconeError:
            raise
        except Exception as exc:
            raise PineconeError(f"Pinecone query failed: {exc}"[:300]) from exc

    def delete_by_document(self, namespace: str, document_id: str) -> None:
        """Delete every vector for one document, in that user's
        namespace only. Never delete_all, never another namespace."""
        index = self._connect()
        try:
            index.delete(filter={"document_id": {"$eq": document_id}}, namespace=namespace)
        except Exception as exc:
            raise PineconeError(f"Pinecone delete failed: {exc}"[:300]) from exc
