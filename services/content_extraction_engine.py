"""ContentExtractionEngine: turn retrieved text into clean passages.

Reuses the existing text helpers (cleaning, sentence splitting,
keyword scoring). It removes the boilerplate the spec calls out
(navigation, adverts, cookie/subscribe prompts, duplicated lines)
from *retrieved* text only. It never edits the user's message and
never invents content: every passage keeps its source reference.
"""
from __future__ import annotations

import re

from utils.text_processing import clean_text, extract_key_sentences, split_into_sentences

_BOILERPLATE = (
    "subscribe", "sign up for", "sign in to", "cookie policy", "accept cookies",
    "advertisement", "click here to", "menu", "navigation", "all rights reserved",
    "follow us on", "share this", "back to top",
)


def _is_boilerplate(line: str) -> bool:
    low = line.lower().strip()
    return any(phrase in low for phrase in _BOILERPLATE)


def is_quiz_fragment(sentence: str) -> bool:
    """True for quiz-block artefacts ("Quiz on the comparison: 1.")
    that must never leak into facts/key points: a quiz intro is a
    prompt, not a retrieved fact, whatever follows it."""
    return bool(re.match(r"^quiz\b", (sentence or "").strip(), re.I))


def clean_passage(text: str) -> str:
    """Strip HTML/boilerplate lines, collapse whitespace, keep sentences."""
    if not text:
        return ""
    lines = [clean_text(line) for line in str(text).replace("\r", "\n").split("\n")]
    kept = [line for line in lines if line and not _is_boilerplate(line)]
    joined = " ".join(kept)
    # A stray URL inside body text belongs in the sources list, not the prose.
    joined = re.sub(r"https?://\S+", "", joined)
    return re.sub(r"\s+", " ", joined).strip()


def split_sections(text: str) -> list:
    """Split structured text into {heading, body} sections.

    Understands Markdown headings, ``Heading:`` lines, bullet lists
    and numbered lists. Unstructured text becomes one untitled section,
    so callers never lose content by expecting structure that is absent.
    """
    cleaned_lines = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if line and not _is_boilerplate(line):
            cleaned_lines.append(line)
    sections, current = [], {"heading": "", "lines": []}
    for line in cleaned_lines:
        heading = None
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        elif re.match(r"^[A-Z][A-Za-z0-9 ,&'/-]{2,60}:$", line):
            heading = line.rstrip(":")
        elif re.match(r"^\*\*[^*]{2,60}\*\*:?$", line):
            heading = line.strip("*").rstrip(":")
        if heading is not None:
            if current["lines"]:
                sections.append(current)
            current = {"heading": heading, "lines": []}
        else:
            current["lines"].append(line)
    if current["lines"]:
        sections.append(current)
    out = []
    for section in sections:
        body = " ".join(section["lines"]).strip()
        if body:
            out.append({"heading": section["heading"], "body": body, "lines": section["lines"]})
    return out or ([{"heading": "", "body": clean_passage(text), "lines": [clean_passage(text)]}]
                   if clean_passage(text) else [])


def extract_passages(text: str, query: str, max_sentences: int = 8) -> list:
    """Key sentences for a query, deduplicated, in relevance order."""
    cleaned = clean_passage(text)
    if not cleaned:
        return []
    sentences = extract_key_sentences([cleaned], query, max_sentences=max_sentences)
    return [{"text": s} for s in sentences if not is_quiz_fragment(s)]


def extract_from_items(items: list, query: str) -> list:
    """Flatten retrieved items into {text, source_name, url, type, chunk_index} passages."""
    passages = []
    for item in items or []:
        raw = item.get("text") or item.get("snippet") or ""
        cleaned = clean_passage(raw)
        if not cleaned:
            continue
        for sentence in split_into_sentences(cleaned):
            if is_quiz_fragment(sentence):
                continue
            passages.append({
                "text": sentence,
                "source_name": item.get("source_name") or item.get("title") or "Source",
                "url": item.get("url") or "",
                "type": item.get("type") or "document",
                "chunk_index": item.get("chunk_index"),
                "score": item.get("score"),
            })
    # Deduplicate identical sentences, keeping the first source.
    seen, out = set(), []
    for passage in passages:
        key = passage["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(passage)
    return out
