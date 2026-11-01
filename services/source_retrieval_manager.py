"""SourceRetrievalManager: organise already-retrieved material.

This manager does not itself call the web, Pinecone, or a model.
Retrieval stays in the existing pipeline (RAG in ``app.py``, web
search in the router, documents in ``document_service``), so adding
this layer cannot change what is retrieved or add new API costs.
Its job is to collect those results, in the order they arrived,
with their source metadata intact, and to deduplicate them.
"""
from __future__ import annotations

import hashlib

from services.citation_manager import normalize_sources, parse_sources_from_text
from services.content_extraction_engine import clean_passage


def _item_id(source_name: str, text: str) -> str:
    return hashlib.sha256(f"{source_name}\0{text[:200]}".encode("utf-8")).hexdigest()[:16]


def collect_from_payload(payload: dict, context=None) -> list:
    """Build retrieved items from a finished /api/chat payload.

    Sources already on the payload win. A formatted web answer
    without a ``sources`` array is parsed back into sources. The
    uploaded document in the session context is added only when the
    payload is a document answer, so a document is never silently
    mixed into an unrelated web answer.
    """
    payload = payload or {}
    items = []
    sources = normalize_sources(payload.get("sources"))
    source_by_citation = {s["citation"]: s for s in sources}
    if not sources:
        sources = normalize_sources(parse_sources_from_text(payload.get("response", "")))
        source_by_citation = {s["citation"]: s for s in sources}
    response = payload.get("response", "") or ""
    kind = "internal"
    if payload.get("intent") in ("RAG_ANSWER", "DOCUMENT_SUMMARY", "DOCUMENT_QA"):
        kind = "document"
    elif payload.get("intent") in ("WEB_SEARCH",):
        kind = "web"
    if sources:
        for source in sources:
            items.append({
                "text": "", "source_name": source["source_name"], "url": source.get("url", ""),
                "type": kind, "chunk_index": source.get("chunk_index"),
                "citation": source["citation"], "score": source.get("score"),
            })
    # The body text itself is a retrieved item when there are no
    # structured sources (rule answers, LLM answers): label it
    # internal, so verification never calls it web-verified.
    if not items and response:
        items.append({"text": clean_passage(response), "source_name": "Shibu AI knowledge base",
                      "url": "", "type": "internal", "chunk_index": None, "citation": None})
    if context is not None and payload.get("intent") in ("DOCUMENT_SUMMARY", "DOCUMENT_QA", "DOCUMENT_ANALYSIS"):
        doc_text = getattr(context, "last_document_text", "") or ""
        if doc_text:
            items.append({"text": clean_passage(doc_text[:4000]),
                          "source_name": getattr(context, "last_document", "") or "Uploaded document",
                          "url": "", "type": "document", "chunk_index": None, "citation": None})
    # Fill empty item text from the response for source-carrying
    # payloads (RAG answers embed passages in the text), keeping the
    # source metadata alongside the text it came from.
    if items and any(not item.get("text") for item in items) and response:
        body = clean_passage(response)
        for item in items:
            if not item.get("text"):
                item["text"] = body
    return deduplicate(items)


def collect_from_text(text: str, source_name: str, source_type: str = "document", url: str = "") -> list:
    cleaned = clean_passage(text)
    if not cleaned:
        return []
    return [{"text": cleaned, "source_name": source_name, "url": url,
             "type": source_type, "chunk_index": None, "citation": None}]


def deduplicate(items: list) -> list:
    seen, out = set(), []
    for item in items or []:
        key = _item_id(item.get("source_name", ""), item.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
