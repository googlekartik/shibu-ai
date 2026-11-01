"""Pinecone vector store — validate-only, never create/change an index.

API selection by index type (pinecone SDK 10.x, verified against the
installed SDK):
- dense: index.upsert(vectors=[{id, values, metadata}], namespace)
         index.query(vector=..., top_k, namespace, include_metadata=True)
- integrated (document-schema index): the legacy Records API
  (index.upsert_records / index.search, /records/...) is REFUSED by
  such indexes with HTTP 400 INVALID_ARGUMENT: writes must go
  through the Documents API. So integrated indexes use the
  Documents REST API directly (requests, already a dependency):
      POST https://{index-host}/namespaces/{ns}/documents/upsert
      POST https://{index-host}/namespaces/{ns}/documents/search
      POST https://{index-host}/namespaces/{ns}/documents/delete
  with headers Api-Key + X-Pinecone-Api-Version: 2026-07 (the
  version the live error message itself requires). The SDK is
  kept only for describe()/validation. Document shape is kept
  identical to before: id, the "text" field, and flat metadata
  (document id, source name, chunk index) — metadata fields are
  top-level document fields, never nested under "metadata", and
  the metadata copy of "text" is dropped so it cannot overwrite
  the real text field.

The client is created lazily, so importing this module (and running
the app without Pinecone) never requires the pinecone package or a
key. describe() is the inspection step: it reads dimension/metric
from pc.describe_index() and lets rag_config.validate_index_info()
decide whether RAG may run. This module never calls create_index,
delete_index, or configure_index.
"""
from __future__ import annotations

import os
from urllib.parse import quote

from services.retry import retry_cloud_call


# Required by document-schema indexes (see module docstring): the
# live server error names this version explicitly.
_DOCUMENTS_API_VERSION = "2026-07"


class PineconeError(Exception):
    pass


class _DocumentsHTTPError(Exception):
    """HTTP failure from the Documents API, with a status code so
    retry_cloud_call retries only transient (429/5xx) failures."""

    def __init__(self, status: int, message: str = ""):
        self.status_code = status
        super().__init__(f"HTTP {status}: {message}"[:300])


def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    out = {}
    for attr in ("dimension", "metric", "name", "status", "host", "spec"):
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
    def __init__(self, api_key=None, index_name=None, index_type="dense", index_host=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("PINECONE_API_KEY", "")).strip()
        self.index_name = (index_name or os.environ.get("PINECONE_INDEX_NAME", "")).strip()
        self.index_type = index_type
        # Data-plane host for the Documents API (from describe_index);
        # injectable in tests so no SDK is needed to exercise it.
        self._host = (index_host or "").strip() or None
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
        if info.get("host"):
            self._host = info["host"]
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

    # -- Documents API (integrated / document-schema indexes) -----
    def _documents_host(self) -> str:
        """Data-plane host from describe_index, cached after first use."""
        if self._host:
            return self._host
        if not self.api_key or not self.index_name:
            raise PineconeError("Pinecone is not configured: PINECONE_API_KEY and PINECONE_INDEX_NAME are required.")
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise PineconeError("The 'pinecone' package is not installed. Install it with: pip install pinecone") from exc
        try:
            if self._client is None:
                self._client = Pinecone(api_key=self.api_key)
            info = _as_dict(self._client.describe_index(self.index_name))
        except Exception as exc:
            raise PineconeError(f"Could not describe Pinecone index '{self.index_name}': {exc}"[:300]) from exc
        host = info.get("host")
        if not host:
            raise PineconeError("Could not determine the Pinecone index host from describe_index; "
                                "document-schema writes need the data-plane host.")
        self._host = host
        return host

    def _documents_request(self, namespace: str, operation: str, payload: dict, attempts: int = 3) -> dict:
        """One Documents API call. Header + URL shape are asserted by
        tests; the API key is sent only in the header, never logged."""
        host = self._documents_host()
        url = (f"https://{host}/namespaces/{quote(namespace, safe='')}"
               f"/documents/{operation}")

        def send():
            import requests
            resp = requests.post(
                url,
                headers={"Api-Key": self.api_key,
                         "X-Pinecone-Api-Version": _DOCUMENTS_API_VERSION,
                         "Content-Type": "application/json"},
                json=payload, timeout=60)
            if resp.status_code in (200, 201, 202):
                try:
                    data = resp.json()
                except Exception:
                    return {}
                return data if isinstance(data, dict) else {}
            raise _DocumentsHTTPError(resp.status_code, resp.text[:200])

        try:
            return retry_cloud_call(send, attempts=attempts)
        except Exception as exc:
            raise PineconeError(f"Pinecone documents {operation} failed: {exc}"[:300]) from exc

    def _documents_upsert(self, namespace: str, items: list, attempts: int = 3) -> int:
        documents = []
        for it in items:
            doc = {"_id": it["id"], "text": it["text"]}
            for key, value in (it.get("metadata") or {}).items():
                # Never let the metadata copy of "text" overwrite the
                # real text field; only scalar fields are stored.
                if key == "text" or not isinstance(value, (str, int, float, bool)):
                    continue
                doc[key] = value
            documents.append(doc)
        self._documents_request(namespace, "upsert", {"documents": documents}, attempts=attempts)
        return len(items)

    def _documents_search(self, namespace: str, query_text: str, top_k: int,
                          filt: dict | None, attempts: int = 3) -> list:
        body = {
            "score_by": [{"type": "text", "query": query_text, "fields": ["text"]}],
            "top_k": top_k,
            "include_fields": ["*"],
        }
        if filt:
            body["filter"] = filt
        data = self._documents_request(namespace, "search", body, attempts=attempts)
        result = data.get("result") if isinstance(data.get("result"), dict) else data
        raw_hits = result.get("matches") or result.get("hits") or []
        hits = []
        for h in raw_hits:
            if isinstance(h, dict):
                fields = h.get("fields") or {}
                if not isinstance(fields, dict):
                    fields = {}
                nested = fields.get("metadata") if isinstance(fields.get("metadata"), dict) else {}
                metadata = {k: v for k, v in fields.items() if k != "text"}
                metadata.update(nested)
                hits.append({"id": h.get("_id") or h.get("id"),
                             "score": h.get("score", h.get("_score", 0.0)),
                             "text": fields.get("text", h.get("text", "")),
                             "metadata": metadata})
            else:
                fields = getattr(h, "fields", {}) or {}
                hits.append({"id": getattr(h, "id", None) or getattr(h, "_id", None),
                             "score": getattr(h, "score", 0.0),
                             "text": fields.get("text", "") if isinstance(fields, dict) else "",
                             "metadata": fields if isinstance(fields, dict) else {}})
        return hits

    def _documents_delete(self, namespace: str, document_id: str, attempts: int = 3) -> None:
        self._documents_request(
            namespace, "delete",
            {"filter": {"document_id": {"$eq": document_id}}}, attempts=attempts)

    def upsert(self, namespace: str, items: list, attempts: int = 3) -> int:
        """items: dense -> {id, values, metadata}; integrated ->
        {id, text, metadata}. Returns the number of items sent.
        Upsert by stable chunk id is idempotent: re-ingesting the
        same document overwrites, it does not duplicate."""
        if not items:
            return 0
        if self.index_type == "integrated":
            # Document-schema indexes refuse the legacy Records API;
            # the Documents API (with the 2026-07 version header) is
            # the only write path. No SDK fallback: falling back
            # would silently re-enter the failing path.
            return self._documents_upsert(namespace, items, attempts=attempts)
        index = self._connect()
        def send():
            vectors = [{"id": it["id"], "values": it["values"],
                        "metadata": it.get("metadata", {})} for it in items]
            return index.upsert(vectors=vectors, namespace=namespace)
        try:
            retry_cloud_call(send, attempts=attempts)
        except Exception as exc:
            raise PineconeError(f"Pinecone upsert failed: {exc}"[:300]) from exc
        return len(items)

    def query(self, namespace: str, query_text: str = None, vector: list = None,
              top_k: int = 5, document_id: str = None, metadata_filter: dict = None,
              attempts: int = 3) -> list:
        """Return [{id, score, text, metadata}]. Dense indexes need
        a vector; integrated indexes need query_text. A filter on
        document_id is applied only when given (per-document search);
        the namespace already isolates the user."""
        filt = dict(metadata_filter or {})
        if document_id:
            filt["document_id"] = {"$eq": document_id}
        filt = filt or None
        try:
            if self.index_type == "integrated":
                if not query_text:
                    raise PineconeError("Integrated indexes need query text, not a vector.")
                return self._documents_search(namespace, query_text, top_k, filt, attempts=attempts)
            index = self._connect()
            if vector is None:
                raise PineconeError("Dense indexes need a query vector; no embedding provider is configured.")
            resp = retry_cloud_call(lambda: index.query(
                vector=vector, top_k=top_k, namespace=namespace,
                include_metadata=True, filter=filt), attempts=attempts)
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

    def delete_by_document(self, namespace: str, document_id: str, attempts: int = 3) -> None:
        """Delete every vector for one document, in that user's
        namespace only. Never delete_all, never another namespace."""
        if self.index_type == "integrated":
            self._documents_delete(namespace, document_id, attempts=attempts)
            return
        index = self._connect()
        try:
            retry_cloud_call(lambda: index.delete(
                filter={"document_id": {"$eq": document_id}}, namespace=namespace), attempts=attempts)
        except Exception as exc:
            raise PineconeError(f"Pinecone delete failed: {exc}"[:300]) from exc

    def rerank(self, query: str, documents: list, model: str, top_n: int,
               attempts: int = 3) -> list:
        """Optional Pinecone-hosted reranking.

        It is only called when explicitly enabled in configuration.  The SDK
        response has changed shape between releases, so both mapping and
        object responses are accepted.  A failure is intentionally surfaced
        to the caller, which can retain the already-good vector ranking.
        """
        if not model:
            return []
        self._connect()
        payload = [{"id": str(i), "text": item.get("text", "")} for i, item in enumerate(documents)]
        try:
            response = retry_cloud_call(lambda: self._client.inference.rerank(
                model=model, query=query, documents=payload, rank_fields=["text"],
                top_n=min(max(1, top_n), len(payload)), return_documents=False), attempts=attempts)
            raw = getattr(response, "data", None)
            if raw is None and isinstance(response, dict):
                raw = response.get("data") or response.get("results") or []
            ordered = []
            for item in raw or []:
                if isinstance(item, dict):
                    idx, score = item.get("index"), item.get("score")
                else:
                    idx, score = getattr(item, "index", None), getattr(item, "score", None)
                if idx is None:
                    continue
                ordered.append({"index": int(idx), "score": float(score or 0.0)})
            return ordered
        except Exception as exc:
            raise PineconeError(f"Pinecone rerank failed: {exc}"[:300]) from exc
