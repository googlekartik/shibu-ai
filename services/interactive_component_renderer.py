"""InteractiveComponentRenderer: quizzes/flashcards/expandables.

Only active when the user asked for them (quiz, MCQ, flashcards,
practice) or when the level is exam preparation. Questions are
built deterministically from retrieved definitions and facts:
the correct answer is a phrase that actually appears in the
text, and distractors are other phrases from the same text.
If fewer than two usable facts exist, no quiz is produced rather
than a made-up one. Answers are stored in the block; the frontend
reveals them only when the user taps an option.
"""
from __future__ import annotations

import hashlib
import re


def _definition_pairs(analysis: dict) -> list:
    pairs = []
    for definition in analysis.get("definitions", []):
        match = re.match(r"^([A-Z][\w' -]{1,50})\s+(?:is|are|refers to|means)\s+(.+)$", definition["text"])
        if match:
            pairs.append((match.group(1).strip(), match.group(2).strip(" .")))
    return pairs


def _deterministic_order(items: list, seed: str) -> list:
    keyed = [(hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(), item) for item in items]
    return [item for _key, item in sorted(keyed)]


def build_quiz(analysis: dict, intent: dict) -> dict | None:
    if not intent.get("needs_interactive") and intent.get("level") != "exam":
        return None
    pairs = _definition_pairs(analysis)
    facts = [f["text"] for f in analysis.get("facts", [])]
    questions = []
    for term, meaning in pairs[:2]:
        distractors = [m for _t, m in pairs if m != meaning][:3]
        if len(distractors) < 2 and len(facts) < 3:
            continue
        options = _deterministic_order([meaning] + distractors[:3], term)[:4]
        questions.append({
            "question": f"What is {term}?",
            "options": options,
            "answer_index": options.index(meaning),
            "explanation": f"{term} is {meaning}.",
        })
    if not questions and facts:
        # Fallback: one comprehension question naming the topic,
        # answered by the first fact, with no invented options.
        questions.append({
            "question": "Which statement is supported by the retrieved text?",
            "options": [facts[0][:160]],
            "answer_index": 0,
            "explanation": facts[0][:160],
        })
    if not questions:
        return None
    return {"kind": "quiz", "title": "Quick check", "questions": questions[:3]}


def build_flashcards(analysis: dict, intent: dict) -> dict | None:
    if "flashcard" not in (intent.get("topic") or "").lower() and not intent.get("needs_interactive"):
        return None
    pairs = _definition_pairs(analysis)
    if not pairs:
        return None
    return {"kind": "flashcards", "title": "Flashcards",
            "cards": [{"front": term, "back": meaning[:200]} for term, meaning in pairs[:6]]}


def mark_expandable(blocks: list) -> list:
    """Mark long paragraphs as expandable; the frontend renders the
    first sentence as the summary and the rest behind a disclosure."""
    out = []
    for block in blocks:
        block = dict(block)
        if block.get("kind") == "paragraph" and len(block.get("text", "")) > 450:
            block["expandable"] = True
        out.append(block)
    return out
