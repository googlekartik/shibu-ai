"""Document intelligence: read PDF, DOCX and XLSX, summarise, answer
questions from the document, make notes/study guides, analyse spreadsheets.

No LLM is used: summarisation is extractive (pick the most informative
sentences by keyword frequency) and Q&A is passage retrieval (sentences
that share the most words with the question). The bot says this honestly.
"""
import os
import re
import statistics
from collections import Counter


def extract_text(path: str, ext: str):
    """Return (text, info_dict). Raises ValueError on unsupported/empty files."""
    ext = ext.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)
        return text, {"type": "pdf", "pages": len(reader.pages)}
    if ext == ".docx":
        from docx import Document
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, {"type": "docx", "paragraphs": len(doc.paragraphs)}
    if ext == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(path, data_only=True)
        lines = []
        analysis = {}
        for ws in wb.worksheets:
            rows = list(ws.iter_rows(values_only=True))
            lines.append(f"Sheet: {ws.title}")
            for row in rows:
                lines.append(" | ".join("" if v is None else str(v) for v in row))
            # Numeric trend analysis per column
            if len(rows) > 1:
                headers = rows[0]
                for col_idx, header in enumerate(headers):
                    nums = [r[col_idx] for r in rows[1:] if isinstance(r[col_idx], (int, float))]
                    if len(nums) >= 2:
                        trend = "increasing" if nums[-1] > nums[0] else ("decreasing" if nums[-1] < nums[0] else "flat")
                        analysis[f"{ws.title} / {header}"] = {
                            "count": len(nums), "sum": round(sum(nums), 2),
                            "mean": round(statistics.mean(nums), 2),
                            "min": min(nums), "max": max(nums), "trend": trend,
                        }
        return "\n".join(lines), {"type": "xlsx", "sheets": wb.sheetnames, "analysis": analysis}
    raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX or XLSX.")


def split_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "").strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def summarize(text: str, max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "I could not extract readable text from this document."
    words = re.findall(r"[a-z]{3,}", text.lower())
    freq = Counter(words)
    scored = []
    for i, s in enumerate(sentences):
        score = sum(freq.get(w, 0) for w in re.findall(r"[a-z]{3,}", s.lower()))
        scored.append((score, i, s))
    top = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)


def answer_from_document(question: str, text: str):
    """Return the 3 most relevant sentences, or '' if nothing matches."""
    qwords = set(re.findall(r"[a-z]{3,}", question.lower()))
    if not qwords:
        return ""
    scored = []
    for s in split_sentences(text):
        words = set(re.findall(r"[a-z]{3,}", s.lower()))
        overlap = len(qwords & words)
        if overlap:
            scored.append((overlap, s))
    scored.sort(reverse=True)
    return " ".join(s for _, s in scored[:3])


def make_notes(text: str, title: str) -> str:
    summary = summarize(text, max_sentences=6)
    sentences = split_sentences(text)
    key_points = sentences[:8]
    lines = [f"Notes: {title}", "", "Summary:", summary, "", "Key points:"]
    for s in key_points:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("Study questions to test yourself:")
    lines.append("- What is the main idea of this document, in one sentence?")
    lines.append("- List three key facts from it without looking.")
    lines.append("- How would you explain it to a friend in simple words?")
    return "\n".join(lines)
