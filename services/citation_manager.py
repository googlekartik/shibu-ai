"""CitationManager: one place for source citations.

Citations are only ever carried over from real sources (RAG passages,
web results, or the document the user uploaded). This module never
invents a reference: a citation that does not exist in ``sources``
is dropped by :func:`validate_citations`, never fabricated.
"""
from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[(S|W|D|I)(\d+)\]")


def normalize_sources(sources) -> list:
    """Return sources as clean dicts with a stable citation label.

    Accepts the shapes already produced by RAG (``S1``), the web
    combiner (``W1``) and the cache (plain dicts). Missing labels are
    assigned in order, preserving the caller's ordering.
    """
    out = []
    seen = set()
    for i, raw in enumerate(sources or [], 1):
        if not isinstance(raw, dict):
            continue
        citation = str(raw.get("citation") or "").strip()
        if not citation:
            kind = str(raw.get("type") or "").lower()
            # Existing RAG sources are S1.. and web sources W1..;
            # documents keep the S prefix for consistency.
            prefix = "W" if kind == "web" else "S"
            citation = f"{prefix}{i}"
        if citation in seen:
            continue
        seen.add(citation)
        item = {
            "citation": citation,
            "source_name": str(raw.get("source_name") or raw.get("title") or "Source"),
            "url": str(raw.get("url") or ""),
            "type": str(raw.get("type") or ("web" if raw.get("url") else "document")),
        }
        for key in ("chunk_index", "score", "document_id", "page", "section"):
            if raw.get(key) is not None:
                item[key] = raw[key]
        out.append(item)
    return out


def parse_sources_from_text(text: str) -> list:
    """Recover web sources from a formatted web answer.

    ``format_web_answer`` lists sources as ``1. Title`` followed by an
    optional snippet line and a URL line. Only lines that actually
    contain a URL become sources; snippets are never treated as URLs.
    Returns [] for any other text shape.
    """
    if not text or "Sources" not in text:
        return []
    tail = text.split("Sources", 1)[1]
    sources, current = [], None
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        numbered = re.match(r"^(\d+)\.\s+(.+)$", line)
        if numbered and not line.startswith("http"):
            if current:
                sources.append(current)
            current = {"citation": f"W{numbered.group(1)}",
                       "source_name": numbered.group(2), "url": "", "type": "web"}
            continue
        if current and re.match(r"^https?://", line):
            current["url"] = line
    if current:
        sources.append(current)
    return [s for s in sources if s.get("source_name")]


def citation_labels(text: str) -> list:
    return [f"{m.group(1)}{m.group(2)}" for m in _CITATION_RE.finditer(text or "")]


def validate_citations(blocks: list, sources: list) -> list:
    """Drop citation labels that have no matching source.

    Works on the JSON-safe block dicts produced by the structuring
    engine. A block whose only content was an invented citation is
    emptied, and the quality controller drops empty blocks.
    """
    valid = {s["citation"] for s in sources}
    cleaned = []
    for block in blocks or []:
        block = dict(block)
        for key in ("text", "title"):
            if isinstance(block.get(key), str):
                block[key] = _CITATION_RE.sub(
                    lambda m: m.group(0) if m.group(0)[1:-1] in valid else "", block[key])
        if "items" in block and isinstance(block["items"], list):
            block["items"] = [
                _CITATION_RE.sub(lambda m: m.group(0) if m.group(0)[1:-1] in valid else "", str(i))
                for i in block["items"]]
        cleaned.append(block)
    return cleaned
