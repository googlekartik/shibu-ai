"""ContentStructuringEngine: choose sections, never force them all.

Input is analysed, already-retrieved text. Output is a list of
plain-data blocks (JSON-safe dicts). Section choice follows the
detected content type; a simple question yields one paragraph,
an explanation yields intro/definition/key points/examples/
takeaways, a comparison yields a table. No section is invented:
a table cell or a step that has no source sentence says so.
"""
from __future__ import annotations

import re

from services.content_extraction_engine import clean_passage, is_quiz_fragment, split_sections

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.S)


def _text(blocks):
    return " ".join(str(b.get("text", "")) for b in blocks)


def _sentences_from(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", clean_passage(text))
    return [p.strip() for p in parts if p.strip() and not is_quiz_fragment(p)]


def _list_items(text: str) -> list:
    items = []
    for line in str(text or "").splitlines():
        line = line.strip()
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if m:
            items.append(m.group(1).strip())
    return items


def _code_blocks(text: str) -> list:
    blocks = []
    for match in _CODE_FENCE_RE.finditer(text or ""):
        code = match.group(2).strip()
        if code:
            blocks.append({"kind": "code", "language": match.group(1) or "text", "text": code})
    return blocks


def _equations(text: str) -> list:
    equations = []
    for match in re.finditer(r"\$\$(.+?)\$\$|\$(.+?)\$", text or "", re.S):
        formula = (match.group(1) or match.group(2) or "").strip()
        if formula and len(formula) < 200:
            equations.append({"kind": "equation", "text": formula})
    return equations


def _timeline_events(analysis: dict) -> list:
    events = []
    for fact in analysis.get("facts", []):
        years = _YEAR_RE.findall(fact["text"])
        if years:
            events.append({"year": years[0], "text": fact["text"]})
    events.sort(key=lambda e: e["year"])
    return events[:10]


def _compare_table(analysis: dict, entities: list) -> dict | None:
    if len(entities) != 2:
        return None
    left, right = entities
    left_sents = [f["text"] for f in analysis.get("facts", []) if left.lower() in f["text"].lower()]
    right_sents = [f["text"] for f in analysis.get("facts", []) if right.lower() in f["text"].lower()]
    if not left_sents and not right_sents:
        return None
    return {
        "kind": "table",
        "title": f"{left} vs {right}",
        "headers": ["Aspect", left, right],
        "rows": [
            ["What it is", left_sents[0] if left_sents else "Not stated in the sources.",
             right_sents[0] if right_sents else "Not stated in the sources."],
            ["Key point", left_sents[1] if len(left_sents) > 1 else "Not stated in the sources.",
             right_sents[1] if len(right_sents) > 1 else "Not stated in the sources."],
        ],
    }


def _pros_cons(analysis: dict, text: str) -> dict | None:
    low = text.lower()
    pros, cons = [], []
    for sentence in _sentences_from(text):
        s_low = sentence.lower()
        if any(w in s_low for w in ("advantage", "benefit", "pro:", "strength")):
            pros.append(sentence)
        elif any(w in s_low for w in ("disadvantage", "drawback", "limitation", "con:", "weakness")):
            cons.append(sentence)
    if not pros and not cons:
        return None
    return {"kind": "pros_cons", "pros": pros[:4], "cons": cons[:4]}


def _strip_sources_section(text: str) -> str:
    """Cut everything from the "Sources ..." section heading onward.

    Web answers append a Sources list whose title lines match the
    list-item pattern; without this cut those titles were
    re-presented as steps. The body before Sources carries the
    real instructions."""
    match = re.search(r"(?im)^sources\s*[\(:].*$", text or "")
    return text[:match.start()].rstrip() if match else text


# Sentences that report a failed lookup, never instructions: they
# must never be built into Steps, in any formatter or block. A
# how-to answer with no real steps has no Steps block at all.
_LOOKUP_FAILURE_PATTERNS = (
    re.compile(r"could not look up", re.I),
    re.compile(r"no web results came back", re.I),
    re.compile(r"try again in a moment", re.I),
    re.compile(r"no results (came back|were found|found)", re.I),
)


def _is_lookup_failure(sentence: str) -> bool:
    """True when a sentence reports that a lookup failed, rather
    than instructing the reader."""
    return any(p.search(sentence or "") for p in _LOOKUP_FAILURE_PATTERNS)


def _drop_source_titles(items: list, sources: list) -> list:
    """Drop any item that IS a source title: steps must never be
    built from result titles, in any formatter. Titles come from
    the payload's own source list, so the comparison is exact."""
    titles = set()
    for source in sources or []:
        name = source.get("source_name") if isinstance(source, dict) else None
        if name:
            titles.add(_norm_key(str(name)))
    if not titles:
        return items
    return [it for it in items
            if _norm_key(it) not in titles
            and not any(title and len(title) >= 10 and _norm_key(it).startswith(title)
                        for title in titles)]


def _norm_key(text: str) -> str:
    """First 50 alphanumeric characters: enough to spot the same
    sentence repeated in the intro, a callout and the key points."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())[:50]


def structure(response_text: str, analysis: dict, intent: dict, sources: list = None) -> list:
    """Build blocks for one answer. ``response_text`` is the answer
    the existing pipeline already produced; structuring re-presents
    it and never rewrites its wording."""
    text = response_text or ""
    content_type = intent.get("content_type", "explain")
    topic = intent.get("topic") or ""
    if content_type == "simple" or not text.strip():
        return [{"kind": "paragraph", "text": clean_passage(text)}] if text.strip() else []

    blocks = []
    sections = split_sections(text)
    first_section = sections[0] if sections else {"heading": "", "body": clean_passage(text), "lines": []}
    intro_text = first_section["body"]
    used = {_norm_key(s) for s in _sentences_from(intro_text)}
    if topic:
        blocks.append({"kind": "heading", "level": 2, "text": topic[:80].capitalize() if topic else ""})
    if intro_text:
        blocks.append({"kind": "paragraph", "text": intro_text[:1200]})
    if analysis.get("definitions"):
        definition = analysis["definitions"][0]["text"]
        if _norm_key(definition) not in used:
            blocks.append({"kind": "callout", "title": "Definition", "text": definition})
            used.add(_norm_key(definition))

    if content_type == "compare":
        table = _compare_table(analysis, intent.get("entities", []))
        if table:
            blocks.append(table)
        pros_cons = _pros_cons(analysis, text)
        if pros_cons:
            blocks.append(pros_cons)
    elif content_type in ("process_steps", "algorithm", "roadmap"):
        # Steps are built ONLY from the answer body before the
        # Sources section, and never from a source title: the raw
        # answer text also carries a Sources section whose title
        # lines ("1. How to Make a Killing") match the list-item
        # pattern — without this cut and the title check they were
        # re-presented as steps. Headlines and media/budget lines
        # are dropped; genuine steps (whatever verb they start
        # with) survive.
        steps_text = _strip_sources_section(text)
        items = _list_items(steps_text)
        from services.web_answer import _is_polluted_text, _is_title_like
        items = [it for it in items
                 if not _is_title_like(it) and not _is_polluted_text(it)
                 and not _is_lookup_failure(it)]
        items = _drop_source_titles(items, sources)
        if not items:
            items = [s for s in _sentences_from(steps_text)[:8]
                     if not _is_title_like(s) and not _is_polluted_text(s)
                     and not _is_lookup_failure(s)]
            items = _drop_source_titles(items, sources)
        if items:
            blocks.append({"kind": "steps", "title": "Steps", "items": items[:10]})
        for code in _code_blocks(text):
            blocks.append(code)
    elif content_type == "historical":
        events = _timeline_events(analysis)
        if events:
            blocks.append({"kind": "timeline", "title": "Timeline", "events": events})
    elif content_type == "document_summary":
        for section in sections[1:6]:
            heading = section["heading"] or "Section"
            blocks.append({"kind": "heading", "level": 3, "text": heading[:80]})
            blocks.append({"kind": "paragraph", "text": section["body"][:800]})
    elif content_type == "statistics":
        stats = analysis.get("statistics", [])
        if stats:
            blocks.append({"kind": "table", "title": "Key figures",
                           "headers": ["Figure", "Source"],
                           "rows": [[s["text"][:160], s.get("source_name", "Source")] for s in stats[:6]]})
    else:
        # Explain / web research / default: key points + examples,
        # skipping sentences already shown in the intro/definition.
        key_points = [f["text"] for f in analysis.get("key_points", [])
                      if _norm_key(f["text"]) not in used]
        if key_points:
            blocks.append({"kind": "list", "title": "Key points", "items": key_points[:6]})
            used.update(_norm_key(p) for p in key_points)
        elif not any(b.get("kind") == "callout" for b in blocks):
            # A short rule/RAG paragraph with no analysed key points
            # and no definition callout already shown still deserves
            # structure: split it into a definition callout and key
            # points instead of one flat paragraph.
            sentences = _sentences_from(intro_text)
            if len(sentences) >= 2 and blocks and blocks[-1].get("kind") == "paragraph":
                blocks.pop()
                first = sentences[0]
                title = "Definition" if re.search(r"\bis (a|an|the)\b|\bare\b", first, re.I) else "Overview"
                blocks.append({"kind": "callout", "title": title, "text": first})
                blocks.append({"kind": "list", "title": "Key points", "items": sentences[1:6]})
                used.update(_norm_key(s) for s in sentences)
        if analysis.get("examples"):
            blocks.append({"kind": "list", "title": "Examples",
                           "items": [e["text"] for e in analysis["examples"][:4]]})

    for code in _code_blocks(text):
        if code not in blocks:
            blocks.append(code)
    for equation in _equations(text):
        blocks.append(equation)

    if analysis.get("contradictions"):
        blocks.append({"kind": "callout", "title": "Sources disagree",
                       "text": analysis["contradictions"][0]["note"]})
    if analysis.get("uncertainties"):
        blocks.append({"kind": "callout", "title": "Uncertain in the sources",
                       "text": analysis["uncertainties"][0][:300]})
    conclusions = analysis.get("conclusions", [])
    takeaway = None
    for candidate in [c["text"] for c in conclusions] + [f["text"] for f in analysis.get("key_points", [])]:
        if _norm_key(candidate) not in used:
            takeaway = candidate
            break
    if takeaway:
        blocks.append({"kind": "paragraph", "text": "Key takeaway: " + takeaway})
    return blocks
