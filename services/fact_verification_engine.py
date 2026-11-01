"""FactVerificationEngine: corroboration labels, never new facts.

A claim is *corroborated* only when the same wording/figure appears
in at least two distinct sources. One source is *single_source*.
Text with no source at all (a bare LLM answer) is *unverified*.
Rule-engine answers are labelled *internal* (from Shibu AI's own
knowledge base), which is honest about their provenance without
pretending they were checked against the web.
"""
from __future__ import annotations

import re

from services.content_extraction_engine import clean_passage


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", clean_passage(text).lower()).strip()


def verify(analysis: dict, sources: list, source_kind: str = "web") -> dict:
    """Return {claims, summary, conflicting} for the analysed facts."""
    claims = []
    by_text = {}
    for fact in analysis.get("facts", []):
        key = _normalise(fact["text"])
        by_text.setdefault(key, set()).add(fact.get("source_name") or "")
    for fact in analysis.get("facts", []):
        key = _normalise(fact["text"])
        distinct = {s for s in by_text.get(key, set()) if s}
        if source_kind == "internal":
            status = "internal"
        elif not sources:
            status = "unverified"
        elif len(distinct) >= 2:
            status = "corroborated"
        elif len(distinct) == 1:
            status = "single_source"
        else:
            status = "unverified"
        claims.append({"text": fact["text"], "status": status,
                       "sources": sorted(distinct)})
    conflicting = bool(analysis.get("contradictions"))
    counts = {}
    for claim in claims:
        counts[claim["status"]] = counts.get(claim["status"], 0) + 1
    summary_parts = []
    if counts.get("corroborated"):
        summary_parts.append(f"{counts['corroborated']} corroborated by more than one source")
    if counts.get("single_source"):
        summary_parts.append(f"{counts['single_source']} from a single source")
    if counts.get("internal"):
        summary_parts.append(f"{counts['internal']} from Shibu AI's knowledge base")
    if counts.get("unverified"):
        summary_parts.append(f"{counts['unverified']} unverified (no source)")
    if conflicting:
        summary_parts.append("sources disagree on at least one figure")
    return {
        "claims": claims,
        "summary": "; ".join(summary_parts) or "No factual claims were checked.",
        "conflicting": conflicting,
        "counts": counts,
    }
