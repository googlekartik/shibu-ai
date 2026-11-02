"""Pinecone vector store — validate-only, never create/change an index.

API selection by index type (pinecone SDK 10.x, verified against the
installed SDK 10.2.0 and the official docs):
- dense: index.upsert(vectors=[{id, values, metadata}], namespace)
         index.query(vector=..., top_k, namespace, include_metadata=True)
- integrated (document-schema index): the legacy Records API
  (index.upsert_records / index.search, /records/...) is REFUSED by
  such indexes with HTTP 400 INVALID_ARGUMENT, and the Vectors API
  is refused too. Writes must go through the Documents API:
  index.documents.upsert/search/delete (SDK, pinecone>=10.1.0),
  which sends the X-Pinecone-Api-Version: 2026-07 header itself.
  See https://docs.pinecone.io/guides/index-data/adopt-the-documents-api
  and https://docs.pinecone.io/guides/manage-data/manage-document-chunks
  (delete reference: https://docs.pinecone.io/reference/api/2026-07/data-plane/delete_documents).
  The schema is read from pc.indexes.describe(name).schema: the
  string field declared with `embed` carries the chunk text (the
  field name is the schema's, never assumed to be "text"), and
  search uses an `embed` score_by clause on that field — a `text`
  (BM25) clause only scores full-text fields and must not be used
  for an embed field. The SDK is kept for describe()/validation
  as well. This module never calls create_index, delete_index,
  or configure_index.
"""
from __future__ import annotations

import os

from services.retry import retry_cloud_call


# Last failed Pinecone call in this process, for /api/rag/status
# diagnostics: endpoint name + HTTP status ONLY. No values, no
# payload, no key. Updated on a Documents/describe failure; read
# (never written) by rag_service.status().
_LAST_PINECONE_DIAGNOSTIC: dict = {}


def _record_pinecone_diagnostic(endpoint: str, status) -> None:
    _LAST_PINECONE_DIAGNOSTIC.clear()
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    _LAST_PINECONE_DIAGNOSTIC.update({"endpoint": endpoint, "status": status})


def pinecone_last_diagnostic() -> dict | None:
    """Copy of the last Pinecone failure diagnostic, or None."""
    return dict(_LAST_PINECONE_DIAGNOSTIC) or None


def _failure_status(exc: Exception):
    """HTTP status of an SDK failure, or None when unknown."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


class PineconeError(Exception):
    pass


def _field_attr(spec, name):
    """One attribute of a schema field spec, dict or object form."""
    if isinstance(spec, dict):
        return spec.get(name)
    return getattr(spec, name, None)


def _parse_schema_fields(schema) -> dict:
    """Normalise a describe() schema to {field: {type, embed,
    full_text_search}}. Accepts the SDK dict form
    ({"fields": {...}}) and object forms with a .fields mapping.
    Anything unrecognised yields {} — callers fall back to "text"."""
    fields = None
    if isinstance(schema, dict):
        fields = schema.get("fields")
    else:
        fields = getattr(schema, "fields", None)
    out = {}
    if isinstance(fields, dict):
        for name, spec in fields.items():
            out[str(name)] = {
                "type": _field_attr(spec, "type"),
                "embed": bool(_field_attr(spec, "embed")),
                "sparse_embed": bool(_field_attr(spec, "sparse_embed")),
                "full_text_search": bool(_field_attr(spec, "full_text_search")),
            }
    return out


def _schema_text_field(schema_fields: dict) -> str:
    """The chunk-text field for a document-schema index: the string
    field declared with `embed` first, else a full-text string
    field, else "text". Never invents a field the schema lacks
    when the schema is known."""
    for name, spec in (schema_fields or {}).items():
        if spec.get("type") == "string" and spec.get("embed"):
            return name
    for name, spec in (schema_fields or {}).items():
        if spec.get("type") == "string" and spec.get("full_text_search"):
            return name
    return "text"


def _match_list(resp) -> list:
    """Matches from an SDK SearchDocumentsResponse or a dict-shaped
    response (dicts are what fakes in tests return)."""
    if isinstance(resp, dict):
        result = resp.get("result") if isinstance(resp.get("result"), dict) else resp
        return result.get("matches") or result.get("hits") or []
    matches = getattr(resp, "matches", None)
    if matches is None and isinstance(resp, dict):
        return []
    return list(matches or [])


def _match_fields(match, text_field: str) -> dict:
    """Field dict for one match: SDK Document objects expose fields
    as attributes; dict matches carry them under "fields"."""
    if isinstance(match, dict):
        fields = match.get("fields")
        if isinstance(fields, dict):
            return fields
        return {k: v for k, v in match.items()
                if k not in ("_id", "id", "score", "_score")}
    fields = getattr(match, "fields", None)
    if isinstance(fields, dict):
        return fields
    out = {}
    for key in ("document_id", "source_name", "chunk_index"):
        value = getattr(match, key, None)
        if value is not None:
            out[key] = value
    value = getattr(match, text_field, None)
    if isinstance(value, str):
        out[text_field] = value
    to_dict = getattr(match, "to_dict", None)
    if callable(to_dict):
        try:
            as_dict = to_dict()
            if isinstance(as_dict, dict):
                out.update(as_dict)
        except Exception:
            pass
    return out


def _parse_document_match(match, text_field: str) -> dict:
    """One SDK/dict match -> {id, score, text, metadata}."""
    fields = _match_fields(match, text_field)
    if isinstance(match, dict):
        doc_id = match.get("_id") or match.get("id")
        score = match.get("score", match.get("_score", 0.0))
    else:
        doc_id = getattr(match, "id", None) or getattr(match, "_id", None)
        score = getattr(match, "score", 0.0)
    nested = fields.get("metadata") if isinstance(fields.get("metadata"), dict) else {}
    metadata = {k: v for k, v in fields.items() if k != text_field}
    metadata.update(nested)
    text = fields.get(text_field, "")
    if not isinstance(text, str):
        text = fields.get("text", "") if isinstance(fields.get("text"), str) else ""
    return {"id": doc_id, "score": score, "text": text, "metadata": metadata}


def _as_dict(obj):
    if isinstance(obj, dict):
        return obj
    out = {}
    for attr in ("dimension", "metric", "name", "status", "host", "spec", "schema"):
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
        client = self._client_or_raise()
        self._index = client.index(name=self.index_name)
        return self._index

    def _client_or_raise(self):
        """The Pinecone control-plane client, created lazily."""
        if self._client is not None:
            return self._client
        if not self.api_key or not self.index_name:
            raise PineconeError("Pinecone is not configured: PINECONE_API_KEY and PINECONE_INDEX_NAME are required.")
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise PineconeError("The 'pinecone' package is not installed. Install it with: pip install pinecone") from exc
        self._client = Pinecone(api_key=self.api_key)
        return self._client

    def describe(self) -> dict:
        """Return {"dimension", "metric", "index_type", "schema_fields",
        "text_field"} without creating or changing anything.
        index_type is taken from the configured type unless the
        describe response carries a schema or an embed
        configuration, in which case it is integrated. The schema
        (when the index has one) names the chunk-text field; it is
        cached on the store so upsert/search use the schema's field
        name instead of assuming "text"."""
        try:
            client = self._client_or_raise()
            info = None
            try:
                info = _as_dict(client.indexes.describe(self.index_name))
            except Exception:
                info = None
            if info is None:
                info = _as_dict(client.describe_index(self.index_name))
        except Exception as exc:
            _record_pinecone_diagnostic("describe", _failure_status(exc))
            raise PineconeError(f"Could not describe Pinecone index '{self.index_name}': {exc}"[:300]) from exc
        if info.get("host"):
            self._host = info["host"]
        schema_fields = _parse_schema_fields(info.get("schema"))
        self._schema_fields = schema_fields
        self._text_field = _schema_text_field(schema_fields)
        result = {
            "dimension": info.get("dimension"),
            "metric": info.get("metric"),
            "index_type": self.index_type,
            "schema_fields": schema_fields,
            "text_field": self._text_field,
        }
        # Integrated indexes carry a schema or an embed spec.
        try:
            if schema_fields:
                result["index_type"] = "integrated"
            else:
                spec = info.get("spec") or {}
                if isinstance(spec, dict) and ("embed" in spec or "model" in str(spec)):
                    result["index_type"] = "integrated"
        except Exception:
            pass
        return result

    def _text_field_name(self) -> str:
        """Chunk-text field from the cached schema, resolving the
        schema once (via describe) when it has not been read yet."""
        if getattr(self, "_text_field", None):
            return self._text_field
        try:
            self.describe()
        except Exception:
            pass
        return getattr(self, "_text_field", None) or "text"

    # -- Documents API (integrated / document-schema indexes) -----
    # Via the SDK: index.documents.upsert/search/delete (pinecone
    # >= 10.1.0). The SDK sends the X-Pinecone-Api-Version: 2026-07
    # header itself; no raw REST, no Records API fallback.
    def _documents_index(self):
        """Data-plane index client for documents.* calls."""
        if self._index is not None:
            return self._index
        return self._connect()

    def _documents_call(self, endpoint: str, operation, attempts: int = 3):
        """Run one Documents API operation with transient retries.
        A failure records the endpoint + HTTP status diagnostic
        (safe for /api/rag/status) and is re-raised as PineconeError
        with the server message truncated — never raw payload."""
        try:
            return retry_cloud_call(operation, attempts=attempts)
        except Exception as exc:
            _record_pinecone_diagnostic(endpoint, _failure_status(exc))
            raise PineconeError(f"Pinecone documents {endpoint.split('/')[-1]} failed: {exc}"[:300]) from exc

    def _documents_upsert(self, namespace: str, items: list, attempts: int = 3) -> int:
        index = self._documents_index()
        text_field = self._text_field_name()
        documents = []
        for it in items:
            doc = {"_id": it["id"], text_field: it["text"]}
            for key, value in (it.get("metadata") or {}).items():
                # The text field itself and non-scalar values are
                # never stored as metadata: a metadata copy of the
                # text field would overwrite the real text.
                if key == text_field or key == "text" or not isinstance(value, (str, int, float, bool)):
                    continue
                doc[key] = value
            documents.append(doc)
        # The server caps one upsert at 96 documents on an index
        # with an embed field; 50 stays inside it with margin.
        for start in range(0, len(documents), 50):
            batch = documents[start:start + 50]
            self._documents_call(
                "documents/upsert",
                lambda batch=batch: index.documents.upsert(namespace=namespace, documents=batch),
                attempts=attempts)
        return len(items)

    def _documents_search(self, namespace: str, query_text: str, top_k: int,
                          filt: dict | None, attempts: int = 3) -> list:
        index = self._documents_index()
        text_field = self._text_field_name()
        schema_fields = getattr(self, "_schema_fields", None) or {}
        spec = schema_fields.get(text_field) or {}
        if spec.get("embed"):
            # Semantic search: Pinecone embeds the query with the
            # field's model. An `embed` clause must stand alone.
            from pinecone.models.documents.score_by import EmbedQuery
            score_by = [EmbedQuery(field=text_field, query=query_text)]
        else:
            # Full-text (BM25) search over the schema's text fields.
            from pinecone.models.documents.score_by import TextQuery
            fields = [name for name, s in schema_fields.items()
                      if s.get("type") == "string" and s.get("full_text_search")] or [text_field]
            score_by = [TextQuery(query=query_text, fields=fields)]
        kwargs = {"namespace": namespace, "score_by": score_by, "top_k": top_k,
                  "include_fields": ["*"]}
        if filt:
            kwargs["filter"] = filt
        resp = self._documents_call(
            "documents/search",
            lambda: index.documents.search(**kwargs), attempts=attempts)
        return [_parse_document_match(m, text_field) for m in _match_list(resp)]

    def _documents_delete(self, namespace: str, document_id: str, attempts: int = 3) -> None:
        index = self._documents_index()
        self._documents_call(
            "documents/delete",
            lambda: index.documents.delete(
                namespace=namespace, filter={"document_id": {"$eq": document_id}}),
            attempts=attempts)
    def upsert(self, namespace: str, items: list, attempts: int = 3) -> int:
        """items: dense -> {id, values, metadata}; integrated ->
        {id, text, metadata}. Returns the number of items sent.
        Upsert by stable chunk id is idempotent: re-ingesting the
        same document overwrites, it does not duplicate."""
        if not items:
            return 0
        if self.index_type == "integrated":
            # Document-schema indexes refuse the legacy Records API
            # and the Vectors API; the SDK Documents API is the only
            # write path. No fallback: falling back would silently
            # re-enter a failing path.
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
