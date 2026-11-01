"""ResponseQualityController: the last gate before presentation.

Rules, in order:
1. Simple requests and raw requests return no blocks (plain text wins).
2. Empty, duplicate, or over-long blocks are dropped/trimmed.
3. Citations in blocks must exist in the source list.
4. If structuring produced almost nothing for a long answer, the
   plain-text answer is kept and blocks are dropped: a broken
   presentation must never replace a working answer.
5. The original response text is never rewritten here.
"""
from __future__ import annotations

from services.citation_manager import validate_citations

_MAX_BLOCKS = 12
_MAX_TEXT = 6000


def validate(blocks: list, original_response: str, intent: dict, sources: list,
             verification: dict | None = None) -> list | None:
    if intent.get("raw_request"):
        return None
    if intent.get("content_type") in ("simple", "maths"):
        return None
    cleaned = []
    seen = set()
    total = 0
    for block in blocks or []:
        if not isinstance(block, dict) or not block.get("kind"):
            continue
        text = str(block.get("text", "")) or " ".join(str(i) for i in block.get("items", []))
        key = (block["kind"], text.strip().lower()[:120])
        if not text.strip() and block["kind"] not in ("table", "quiz", "flashcards", "chart", "flowchart"):
            continue
        if key in seen:
            continue
        seen.add(key)
        if len(text) > 1200:
            block = dict(block)
            block["text"] = text[:1200]
            text = block["text"]
        total += len(text)
        if total > _MAX_TEXT:
            break
        cleaned.append(block)
        if len(cleaned) >= _MAX_BLOCKS:
            break
    cleaned = validate_citations(cleaned, sources or [])
    if not cleaned:
        return None
    # A long answer that structured into a single short paragraph has
    # gained nothing; keep the original plain answer instead.
    if len(original_response or "") > 800 and len(cleaned) <= 1:
        return None
    return cleaned
