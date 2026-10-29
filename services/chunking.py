"""Configurable chunking for RAG ingestion.

Stable chunk IDs are the duplicate-prevention mechanism: the ID is
a hash of (document_id, chunk_index, chunk text). Re-ingesting the
same document with the same text produces the same IDs, so a
Pinecone upsert overwrites instead of duplicating, and a SQLite
INSERT OR REPLACE does the same. If the text changes, the IDs
change — which is why document replacement first deletes all chunks
for that document_id, then ingests the new text.
"""
from __future__ import annotations

import hashlib
import re


def _stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = f"{document_id}:{chunk_index}:{text_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def chunk_text(text: str, document_id: str, chunk_size: int = 800,
               chunk_overlap: int = 100) -> list:
    """Split text into overlapping chunks. Returns a list of dicts
    with id, chunk_index, text, char_start, char_end. Empty or
    whitespace-only text returns []."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return []
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 2))
    chunks = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            # Prefer a sentence / word boundary near the end.
            cut = clean.rfind(". ", start, end)
            if cut < start + chunk_size // 2:
                cut = clean.rfind(" ", start, end)
            if cut > start + chunk_size // 2:
                end = cut + 1
        piece = clean[start:end].strip()
        if piece:
            chunks.append({
                "id": _stable_chunk_id(document_id, index, piece),
                "chunk_index": index,
                "text": piece,
                "char_start": start,
                "char_end": end,
            })
            index += 1
        if end >= len(clean):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
