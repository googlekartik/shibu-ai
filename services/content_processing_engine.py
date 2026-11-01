"""Content processing orchestrator: Understand -> Retrieve -> Analyze
-> Verify -> Organize -> Visualize -> Respond.

This is the single entry point the Flask app calls. The pipeline
re-presents an answer the existing router/RAG/web code already
produced; it does not re-retrieve, re-call a model, or rewrite
wording. Any exception inside the pipeline is caught by the
caller, which returns the original payload unchanged.
"""
from __future__ import annotations

from services.citation_manager import normalize_sources
from services.content_extraction_engine import extract_from_items
from services.content_intent_detector import detect_intent, is_simple_request
from services.content_structuring_engine import structure
from services.fact_verification_engine import verify
from services.intelligent_content_analyzer import analyze
from services.interactive_component_renderer import build_flashcards, build_quiz, mark_expandable
from services.response_quality_controller import validate
from services.source_retrieval_manager import collect_from_payload
from services.visual_response_generator import generate_visuals

# Intents that must never gain visual blocks: onboarding, maths,
# file generation, memory commands, preferences, and no-evidence.
_SKIP_INTENTS = {"ONBOARDING", "MATHEMATICS", "RAG_NO_EVIDENCE"}
_SKIP_RULE_IDS = {"onboarding_avatar", "onboarding_name", "onboarding_goal",
                  "memory_remember", "memory_list", "memory_forget",
                  "preferences_set", "preferences_show"}


def should_present(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("intent") in _SKIP_INTENTS:
        return False
    if payload.get("rule_id") in _SKIP_RULE_IDS:
        return False
    if payload.get("file_url") or payload.get("pdf_url"):
        return False
    return bool(payload.get("response"))


def process_presentation(message: str, payload: dict, context=None) -> dict | None:
    """Return presentation data, or None to keep the plain answer.

    Result shape (all JSON-safe):
    {content_type, level, blocks, citations, verification, visual_kinds}
    """
    if not should_present(payload):
        return None
    prefs = getattr(context, "preferences", {}) if context is not None else {}
    rule_intent = str(payload.get("intent", ""))
    intent = detect_intent(message, prefs=prefs, rule_intent=rule_intent)
    response = payload.get("response", "")
    if is_simple_request(intent, response):
        return None

    items = collect_from_payload(payload, context=context)
    sources = normalize_sources(payload.get("sources"))
    # Passages: from retrieved items when they carry text, otherwise
    # from the answer text itself (internal knowledge base).
    passages = extract_from_items(items, message)
    if not passages and response:
        passages = [{"text": s, "source_name": "Shibu AI knowledge base",
                     "url": "", "type": "internal", "chunk_index": None}
                    for s in _split_sentences(response)]
    source_kind = "internal" if not sources else ("document" if rule_intent in ("RAG_ANSWER", "DOCUMENT_SUMMARY", "DOCUMENT_QA") else "web")
    analysis = analyze(passages, message, content_type=intent["content_type"])
    verification = verify(analysis, sources, source_kind=source_kind)
    blocks = structure(response, analysis, intent, sources=sources)
    blocks.extend(generate_visuals(blocks, intent, sources=sources))
    quiz = build_quiz(analysis, intent)
    if quiz:
        blocks.append(quiz)
    flashcards = build_flashcards(analysis, intent)
    if flashcards:
        blocks.append(flashcards)
    blocks = mark_expandable(blocks)
    blocks = validate(blocks, response, intent, sources, verification=verification)
    if blocks is None:
        return None
    return {
        "content_type": intent["content_type"],
        "level": intent["level"],
        "blocks": blocks,
        "citations": sources,
        "verification": {"summary": verification["summary"],
                         "conflicting": verification["conflicting"],
                         "counts": verification["counts"]},
        "visual_kinds": sorted({b["kind"] for b in blocks if b["kind"] in
                                ("chart", "flowchart", "timeline", "table", "quiz", "flashcards", "gallery")}),
    }


def _split_sentences(text: str) -> list:
    import re
    from services.content_extraction_engine import clean_passage
    parts = re.split(r"(?<=[.!?])\s+", clean_passage(text))
    return [p.strip() for p in parts if p.strip()]
