"""Central pipeline: validation -> normalisation -> context -> intent
-> rule matching -> confidence -> decision -> tool -> response -> context update.

Pipeline order (see README):
  message -> maths? -> pdf? -> search? -> follow-up? -> rule engine -> fallback
Maths / PDF / search are checked before generic rules so that e.g.
'Calculate sqrt(144)' is not answered by the generic maths-info rule.
"""
import re

from chatbot.confidence import classify  # noqa: F401  (re-exported for tests)
from chatbot.patterns import FOLLOWUP_USED_FOR
from chatbot.response_manager import format_response
from chatbot.rule_engine import match_rule
from services.math_service import MathService, looks_like_math
from services.pdf_service import PDFService
from services.web_search_service import WebSearchService
from services.code_service import generate_code, debug_code, compare as compare_topics
from services.code_runner import run_python_code
from services.document_service import summarize as summarize_doc, answer_from_document, make_notes as make_doc_notes
from services.file_generation_service import generate_pptx, generate_docx, generate_xlsx
from services.model_manager import ModelManager
from services.conversation_manager import build_messages, detect_language
from services.response_generator import validate_llm_answer, format_llm_answer
import os
from utils.text_processing import normalize_for_matching

math_service = MathService()
pdf_service = PDFService()
search_service = WebSearchService()


def _is_follow_up(text_norm: str) -> bool:
    return bool(re.search(r"\bwhat is it used for\b|\bwhat is it\b|\bit used for\b|\btell me more\b|\bmore about it\b", text_norm))


def _search_query_from(text: str) -> str:
    t = re.sub(r"(?i)^(search|find|look up|browse)\s+(for|the)?\s*", "", text.strip())
    t = re.sub(r"(?i)^(what happened today in|latest|search the latest)\s+", "", t)
    t = re.sub(r"(?i)\s*\?$", "", t)
    return t.strip() or text.strip()


def _is_search_request(low: str) -> bool:
    """Detect questions that need current / external information, not rules."""
    if re.search(r"^(search|find|look up|browse)\b", low):
        return True
    if re.search(r"\b(latest|recent|trending|current|breaking)\b.*\b(news|update|information|info)\b", low):
        return True
    if re.search(r"what happened (today|recently)|what is happening", low):
        return True
    if re.search(r"\bweather (today|now|tomorrow)\b|\bprice of\b|\bstock (price|market)\b|\bwho won\b", low):
        return True
    if re.search(r"\blatest\b", low) and not re.search(r"\bwhat is\b.*\b(python|ai|java)\b", low):
        return True
    return False


def _phrase_search_results(query: str, results) -> str:
    """Turn raw search results into a phrased answer, with sources.

    The first snippets become a short summary in Shibu AI's own words,
    then every source is listed so the user can check it. Web results
    are never presented as if they came from the rule database.
    """
    snippets = [r["snippet"] for r in results if r.get("snippet")]
    parts = [f"Here's what I found on the web about '{query}':"]
    if snippets:
        parts.append("")
        parts.append("In short, according to the latest web results:")
        for s in snippets[:3]:
            clean = re.sub(r"\s+", " ", s).strip()
            if clean and not clean.endswith("."):
                clean += "."
            parts.append(f"- {clean}")
    parts.append("")
    parts.append("Sources (from the web, not from my rule database):")
    for i, r in enumerate(results, 1):
        parts.append(f"{i}. {r['title']}")
        if r.get("snippet"):
            parts.append(f"   {r['snippet']}")
        parts.append(f"   {r['url']}")
    parts.append("")
    parts.append("Want me to go deeper on any of these sources? Just tell me which one.")
    return "\n".join(parts)



def _first_sentence(text: str) -> str:
    import re as _re
    parts = _re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text


def _adapt_to_preferences(text: str, context) -> str:
    prefs = getattr(context, "preferences", {}) or {}
    if prefs.get("tone") in ("simple", "concise") or prefs.get("level") == "beginner":
        return _first_sentence(text) + " (Simple version — say 'be detailed' for the full answer.)"
    return text


VAGUE_INPUTS = {"it", "that", "this", "what", "how", "help me", "thing", "stuff", "?"}


def _file_payload(text, intent, rule_id, tool, action, filename, processing_ms):
    return format_response(
        text=text, intent=intent, rule_id=rule_id, confidence=0.95,
        tool=tool, action=action,
        extra={"pdf_file": filename, "file_url": f"/api/download/{filename}"},
        processing_ms=processing_ms)


def _llm_answer(user_message: str, context, processing_ms=0.0):
    """Try a conversational LLM answer.

    Returns (payload, error): payload is not None on success; error is
    a plain-language string when the model is enabled but generation
    failed. Callers must NOT turn a generation error into "not
    enabled" — that was the bug that hid real API failures. When the
    model is simply off, both are None and the caller falls back to
    the rules (or, for explicit "ai:", to the not-enabled message).
    """
    manager = ModelManager()
    if manager.mode == "off":
        return None, None
    messages, _language = build_messages(context, user_message)
    previous = None
    try:
        previous = context.conversation_history[-1].get("text", "") if context.conversation_history else None
    except Exception:
        previous = None
    raw, err = manager.generate(messages, max_new_tokens=256, temperature=0.7)
    if err or raw is None:
        return None, err or "The language model returned nothing."
    ok, cleaned, _note = validate_llm_answer(raw, previous_bot_answer=previous)
    if not ok:
        return None, "The model repeated the previous answer, so it was not shown."
    return format_response(
        text=format_llm_answer(cleaned, model_id=getattr(manager, "effective_model", None) or manager.model_id),
        intent="LLM_ANSWER", rule_id="llm_answer", confidence=0.65,
        tool="Hugging Face LLM", action="Generate conversational answer",
        processing_ms=processing_ms), None


def process_single_message(message: str, context, processing_ms=0.0):
    """Process one message. Returns the API payload dict."""
    text = message.strip()
    norm = normalize_for_matching(text)
    low = text.lower()

    # Explicit AI mode: "ai: ..." / "use ai ..." bypasses the rules and
    # goes straight to the language model. If no model is enabled, say
    # so plainly (with how to enable it) instead of silently answering
    # from the rules as if that were an AI answer.
    m_ai = re.match(r"^(ai:|use ai\b|ask ai\b|ai answer\b)\s*(.*)$", low)
    if m_ai and m_ai.group(2):
        llm, llm_error = _llm_answer(m_ai.group(2), context, processing_ms)
        if llm is not None:
            context.add_exchange(text, llm["response"], "LLM_ANSWER", "llm_answer", 0.65)
            return llm
        if llm_error:
            # Enabled, but the call itself failed (bad token permission,
            # model not on the router, rate limit, model loading...).
            # Say THAT, with the fix — never "not enabled".
            payload = format_response(
                text=(f"The language model is enabled, but Hugging Face could not answer this time.\n\n{llm_error}\n\n"
                      "Your .env and token are being read (that is why /api/llm/status says 'api'). "
                      "Fix the error above, restart Shibu if you changed the token, and try the 'ai:' message again. "
                      "Until then, ask without 'ai:' and I will answer from my rule engine."),
                intent="LLM_ERROR", rule_id="llm_error", confidence=0.95,
                tool="Hugging Face LLM", action="LLM call failed", processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "LLM_ERROR", "llm_error", 0.95)
            return payload
        payload = format_response(
            text=("No language model is enabled yet, so I cannot give an AI-generated answer. "
                  "Set HF_TOKEN in your .env file to use the Hugging Face API (easiest), or set "
                  "HF_USE_LOCAL=1 and install requirements-llm.txt to run a small model on this PC. "
                  "Until then, ask without 'ai:' and I will answer from my rule engine."),
            intent="LLM_UNAVAILABLE", rule_id="llm_unavailable", confidence=0.95,
            tool="Rule engine", action="LLM unavailable", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "LLM_UNAVAILABLE", "llm_unavailable", 0.95)
        return payload

    # 0) Step-by-step reasoning comes before plain maths, so "solve step
    # by step 25 + 50" is not treated as a plain calculation first.
    if "step by step" in low and re.search(r"\d+\s*[\+\-\*/\^]\s*\d+|\d+%\s*of|sqrt|derivative|integral", low):
        math_text = re.sub(r"(?i)step by step[:\s]*", "", text).strip()
        result = math_service.calculate(math_text)
        if result.get("success"):
            steps = f"Step by step:\n1. Read the problem: {result['expression']}\n2. Work it out: {result['calculation']}\n3. Check the result and write the answer."
            payload = format_response(text=f"{steps}\nAnswer: {result['answer']}",
                intent="MATHEMATICS_STEP_BY_STEP", rule_id="math_steps", confidence=0.99, tool="SymPy maths", action="Solve step by step",
                extra={"calculation": result}, processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "MATHEMATICS_STEP_BY_STEP", "math_steps", 0.99)
            return payload

    # 1) Mathematics (explicit calculate / maths syntax)
    if looks_like_math(text):
        result = math_service.calculate(text)
        if result.get("success"):
            payload = format_response(
                text=f"{result['calculation']}\nAnswer: {result['answer']}",
                intent="MATHEMATICS", rule_id="math_service",
                confidence=0.99, tool="SymPy maths", action="Calculate with SymPy",
                extra={"calculation": result, "expression": result["expression"]},
                processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "MATHEMATICS",
                                 "math_service", 0.99)
            return payload
        # Fall through to rules if maths parsing failed, but report the error
        # clearly rather than pretending it was answered.
        payload = format_response(
            text=f"I couldn't parse that calculation: {result.get('error', 'invalid expression')}. "
                 "Try examples like '25 + 50', '25% of 800', 'sqrt(144)', 'solve x + 5 = 12' or 'derivative of x^3'.",
            intent="MATHEMATICS", rule_id="math_service", confidence=0.40,
            tool="SymPy maths", action="Maths parsing failed",
            processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "MATHEMATICS", "math_service", 0.40)
        return payload

    # 2) PDF: export conversation
    if re.search(r"export (my )?conversation|download (my )?conversation|conversation pdf", low):
        try:
            lines = []
            for item in context.conversation_history:
                lines.append(f"{item['role'].title()}: {item['text']}")
            lines.append(f"User: {text}")
            path, filename = pdf_service.generate_pdf(
                "\n\n".join(lines) or "Empty conversation.", title="Conversation Export")
            payload = format_response(
                text="Your conversation PDF is ready — click Download PDF in the chat or the button below.",
                intent="PDF_GENERATION", rule_id="pdf_conversation", confidence=0.95,
                tool="ReportLab PDF", action="Generate conversation PDF",
                extra={"pdf_file": filename, "pdf_url": f"/api/download/{filename}"},
                processing_ms=processing_ms)
        except Exception as exc:  # never leak a traceback to the user
            payload = format_response(
                text=f"Sorry, PDF generation failed: {exc}", intent="PDF_GENERATION",
                rule_id="pdf_conversation", confidence=0.40, tool="ReportLab PDF",
                action="PDF generation failed", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "PDF_GENERATION", "pdf_conversation", 0.95)
        return payload

    # 3) PDF: topic PDF, e.g. "Create a PDF about Python"
    m = re.search(r"create a pdf (about|on)\s+(.+)", low)
    if m:
        topic = m.group(2).strip().rstrip(".")
        rule = match_rule(topic)
        body = rule["response"] if rule["intent"] != "UNKNOWN" else f"Notes about {topic}."
        try:
            path, filename = pdf_service.generate_pdf(
                body, title=f"About {topic.title()}")
            payload = format_response(
                text=f"PDF about {topic.title()} is ready — click Download PDF to get it.",
                intent="PDF_GENERATION", rule_id="pdf_topic", confidence=0.95,
                tool="ReportLab PDF", action="Generate topic PDF",
                extra={"pdf_file": filename, "pdf_url": f"/api/download/{filename}"},
                processing_ms=processing_ms)
        except Exception as exc:
            payload = format_response(
                text=f"Sorry, PDF generation failed: {exc}", intent="PDF_GENERATION",
                rule_id="pdf_topic", confidence=0.40, tool="ReportLab PDF",
                action="PDF generation failed", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "PDF_GENERATION", "pdf_topic", 0.95)
        return payload

    # 4) Web search — phrased as an answer, with sources, not a raw list
    if _is_search_request(low):
        query = _search_query_from(text)
        results = search_service.search(query)
        if results:
            payload = format_response(
                text=_phrase_search_results(query, results),
                intent="WEB_SEARCH", rule_id="web_search", confidence=0.90,
                tool="DuckDuckGo search", action="Search the web",
                extra={"search_results": results, "search_query": query},
                processing_ms=processing_ms)
        else:
            payload = format_response(
                text=f"I tried a web search for '{query}' but no results came back (check your internet connection and try again). "
                     "This is a web-search result, not an answer from my rule database.",
                intent="WEB_SEARCH", rule_id="web_search", confidence=0.40,
                tool="DuckDuckGo search", action="Web search failed / no results",
                processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "WEB_SEARCH", "web_search", payload["confidence"])
        return payload

    # 5) Follow-up using previous topic ("What is it used for?")
    if _is_follow_up(norm) and context.last_topic:
        answer = FOLLOWUP_USED_FOR.get(context.last_topic)
        if answer:
            payload = format_response(
                text=answer, intent="FOLLOW_UP", rule_id="context_followup",
                confidence=0.85, tool="Conversation context",
                action="Answer follow-up from previous topic",
                processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "FOLLOW_UP",
                                 "context_followup", 0.85, topic=context.last_topic)
            return payload


    # 5b) Preferences (user-approved, saved per session)
    m_pref = re.search(r"(?:set tone to|tone)\s+(simple|concise|detailed|normal)", low)
    m_level = re.search(r"(beginner|expert) level", low)
    if "explain simply" in low or "simple language" in low and "explain" in low:
        context.preferences["tone"] = "simple"
        payload = format_response(text="Got it — I'll keep answers simple from now on. Say 'be detailed' if you want the full version again.",
            intent="PREFERENCE", rule_id="preference_simple", confidence=0.95, tool="Preferences", action="Save preference", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "PREFERENCE", "preference_simple", 0.95)
        return payload
    if m_pref or m_level:
        if m_pref:
            context.preferences["tone"] = m_pref.group(1)
        if m_level:
            context.preferences["level"] = m_level.group(1)
        payload = format_response(text=f"Saved. Your preferences are now: tone={context.preferences['tone']}, level={context.preferences['level']}.",
            intent="PREFERENCE", rule_id="preference_set", confidence=0.95, tool="Preferences", action="Save preference", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "PREFERENCE", "preference_set", 0.95)
        return payload
    if "my preferences" in low or "what are my preferences" in low:
        payload = format_response(text=f"Your saved preferences are: tone={context.preferences.get('tone')}, level={context.preferences.get('level')}. Say e.g. 'set tone to simple' or 'expert level' to change them.",
            intent="PREFERENCE", rule_id="preference_show", confidence=0.95, tool="Preferences", action="Show preferences", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "PREFERENCE", "preference_show", 0.95)
        return payload

    # 5c) Clarifying questions for vague input (do not guess)
    if norm in VAGUE_INPUTS or (len(norm) <= 2 and norm not in ("hi",)):
        payload = format_response(text="I want to get this right instead of guessing — could you tell me a little more? For example, what topic is this about, and what do you want me to do with it (explain it, calculate it, search for it, or make a file from it)?",
            intent="CLARIFY", rule_id="clarify_vague", confidence=0.90, tool="Conversation", action="Ask clarifying question", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "CLARIFY", "clarify_vague", 0.90)
        return payload

    # 5d) Step-by-step reasoning for maths
    if "step by step" in low and re.search(r"\d+\s*[\+\-\*/\^]\s*\d+|\d+%\s*of|sqrt|derivative|integral|solve", low):
        math_text = re.sub(r"(?i)step by step[:\s]*", "", text).strip()
        result = math_service.calculate(math_text)
        if result.get("success"):
            steps = f"Step by step:\n1. Read the problem: {result['expression']}\n2. Work it out: {result['calculation']}\n3. Check the result and write the answer."
            payload = format_response(text=f"{steps}\nAnswer: {result['answer']}",
                intent="MATHEMATICS_STEP_BY_STEP", rule_id="math_steps", confidence=0.99, tool="SymPy maths", action="Solve step by step",
                extra={"calculation": result}, processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "MATHEMATICS_STEP_BY_STEP", "math_steps", 0.99)
            return payload

    # 5e) Code generation, debugging, comparison
    if re.search(r"write (python )?code (to|for)", low) or low.startswith("code for") or low.startswith("code to"):
        code, expl = generate_code(text)
        if code:
            payload = format_response(text=f"{expl}\n\n```python\n{code}\n```",
                intent="CODE_GENERATION", rule_id="code_template", confidence=0.90, tool="Code templates", action="Generate code", processing_ms=processing_ms)
        else:
            payload = format_response(text="I don't have a ready template for that code yet, so I won't guess one. I can generate code for: factorial, fibonacci, reverse string, sort, prime check, even/odd, and palindrome. Try e.g. 'Write Python code for factorial'.",
                intent="CODE_GENERATION", rule_id="code_template", confidence=0.40, tool="Code templates", action="Code template not found", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "CODE_GENERATION", "code_template", payload["confidence"])
        return payload
    if low.startswith("debug") and ("code" in low or ":" in text):
        code_part = text.split(":", 1)[1] if ":" in text else ""
        result_text = debug_code(code_part)
        payload = format_response(text=result_text, intent="CODE_DEBUG", rule_id="code_debug", confidence=0.90, tool="AST checker", action="Debug code", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "CODE_DEBUG", "code_debug", 0.90)
        return payload
    m_cmp = re.search(r"compare (.+?) (and|vs|versus) (.+)", low) or re.search(r"difference between (.+?) and (.+)", low)
    if m_cmp:
        a, b = m_cmp.group(1).strip(), m_cmp.group(3).strip() if m_cmp.lastindex and m_cmp.lastindex >= 3 else m_cmp.group(2).strip()
        # For 'difference between X and Y', groups are (X, Y) in group 1/2; handle both shapes above safely:
        if "difference between" in low:
            a, b = m_cmp.group(1).strip(), m_cmp.group(2).strip()
        result_text = compare_topics(a, b)
        payload = format_response(text=result_text, intent="COMPARE", rule_id="compare", confidence=0.85, tool="Comparison", action="Compare alternatives and trade-offs", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "COMPARE", "compare", 0.85)
        return payload

    # 5f) Safe Python runner (user's own code only)
    if low.startswith("run python code") or low.startswith("run code"):
        code_part = text.split(":", 1)[1] if ":" in text else ""
        result = run_python_code(code_part)
        if result["success"]:
            payload = format_response(text=f"Code ran successfully. Output:\n{result['output'] or '(no output)'}",
                intent="CODE_RUN", rule_id="code_runner", confidence=0.95, tool="Safe Python runner", action="Run Python code", processing_ms=processing_ms)
        else:
            payload = format_response(text=f"Code did not run: {result['error']}",
                intent="CODE_RUN", rule_id="code_runner", confidence=0.40, tool="Safe Python runner", action="Code run failed", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "CODE_RUN", "code_runner", payload["confidence"])
        return payload

    # 5g) Document intelligence (needs an uploaded document first)
    doc = getattr(context, "last_document_text", "") or ""
    if "summarize" in low and ("document" in low or "my file" in low or "uploaded" in low):
        if not doc:
            payload = format_response(text="No document is uploaded yet. Use the paperclip button to upload a PDF, DOCX or XLSX first, then say 'Summarize my document'.",
                intent="DOCUMENT", rule_id="document_missing", confidence=0.90, tool="Documents", action="Document not found", processing_ms=processing_ms)
        else:
            payload = format_response(text=f"Summary of {context.last_document}:\n\n{summarize_doc(doc)}",
                intent="DOCUMENT_SUMMARY", rule_id="document_summary", confidence=0.90, tool="Documents", action="Summarise document", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], payload["intent"], payload["rule_id"], payload["confidence"])
        return payload
    if ("study guide" in low or "make notes" in low or "notes from" in low) and doc:
        notes = make_doc_notes(doc, context.last_document or "Document")
        try:
            _p, filename = pdf_service.generate_pdf(notes, title=f"Notes - {context.last_document}")
            payload = _file_payload(f"Your notes / study guide for {context.last_document} is ready — click Download File to get the PDF.\n\n{notes[:800]}", "DOCUMENT_NOTES", "document_notes", "ReportLab PDF", "Generate notes PDF", filename, processing_ms)
        except Exception as exc:
            payload = format_response(text=f"Notes made, but PDF generation failed: {exc}\n\n{notes}", intent="DOCUMENT_NOTES", rule_id="document_notes", confidence=0.60, tool="Documents", action="Notes failed to export", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "DOCUMENT_NOTES", "document_notes", payload["confidence"])
        return payload
    if ("spreadsheet" in low or "analyze" in low or "analyse" in low or "trend" in low) and doc and context.last_document_info.get("analysis"):
        lines = [f"Spreadsheet analysis for {context.last_document}:"]
        for k, v in context.last_document_info["analysis"].items():
            lines.append(f"- {k}: count={v['count']}, sum={v['sum']}, mean={v['mean']}, min={v['min']}, max={v['max']}, trend={v['trend']}")
        payload = format_response(text="\n".join(lines), intent="DOCUMENT_ANALYSIS", rule_id="document_analysis", confidence=0.90, tool="Documents", action="Analyse spreadsheet", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "DOCUMENT_ANALYSIS", "document_analysis", 0.90)
        return payload
    if doc and ("document" in low or "in the document" in low or "from my file" in low):
        ans = answer_from_document(text, doc)
        if ans:
            payload = format_response(text=f"From your document ({context.last_document}):\n\n{ans}",
                intent="DOCUMENT_QA", rule_id="document_qa", confidence=0.85, tool="Documents", action="Answer from document", processing_ms=processing_ms)
        else:
            payload = format_response(text=f"I searched your document ({context.last_document}) but could not find a passage that answers that. Try rephrasing with words that appear in the document.",
                intent="DOCUMENT_QA", rule_id="document_qa", confidence=0.40, tool="Documents", action="Document Q&A no match", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "DOCUMENT_QA", "document_qa", payload["confidence"])
        return payload

    # 5h) File generation: PowerPoint / Word / Excel (PDF is handled above)
    m_ppt = re.search(r"create (?:a )?(?:powerpoint|presentation|ppt)(?: about| on)?\s+(.+)", low)
    m_docx = re.search(r"create (?:a )?word document(?: about| on)?\s+(.+)", low)
    m_xlsx = re.search(r"create (?:an )?excel (?:workbook|spreadsheet|file)(?: about| on)?\s+(.+)", low)
    if m_ppt or m_docx or m_xlsx:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated")
        if m_ppt:
            topic = m_ppt.group(1).strip().rstrip(".")
            rule = match_rule(topic)
            points = [rule["response"]] if rule["intent"] != "UNKNOWN" else [f"Introduction to {topic}", f"Key ideas about {topic}", f"Why {topic} matters", f"Summary of {topic}"]
            _p, filename = generate_pptx(topic.title(), points, base_dir)
            payload = _file_payload(f"Your PowerPoint about {topic.title()} is ready — click Download File to get it.", "FILE_GENERATION", "file_pptx", "python-pptx", "Generate PowerPoint", filename, processing_ms)
        elif m_docx:
            topic = m_docx.group(1).strip().rstrip(".")
            rule = match_rule(topic)
            body = rule["response"] if rule["intent"] != "UNKNOWN" else f"Notes about {topic}."
            _p, filename = generate_docx(topic.title(), body, base_dir)
            payload = _file_payload(f"Your Word document about {topic.title()} is ready — click Download File to get it.", "FILE_GENERATION", "file_docx", "python-docx", "Generate Word document", filename, processing_ms)
        else:
            topic = m_xlsx.group(1).strip().rstrip(".")
            _p, filename = generate_xlsx(topic.title(), ["Item", "Value"], [["Example A", 10], ["Example B", 25], ["Example C", 15]], base_dir)
            payload = _file_payload(f"Your Excel workbook about {topic.title()} is ready — click Download File to get it. It includes a table, SUM formulas in the Total row, and a bar chart.", "FILE_GENERATION", "file_xlsx", "openpyxl", "Generate Excel workbook", filename, processing_ms)
        context.add_exchange(text, payload["response"], "FILE_GENERATION", payload["rule_id"], 0.95)
        return payload

    # 5i) Explain simply: use the rule answer, shortened
    m_simple = re.search(r"explain (.+?) (simply|in simple language|in simple terms)", low)
    if m_simple:
        rule = match_rule(m_simple.group(1))
        if rule["intent"] != "UNKNOWN":
            payload = format_response(text=f"In simple terms: {_first_sentence(rule['response'])}",
                intent=rule["intent"], rule_id=rule["rule_id"], confidence=rule["confidence"], tool="Rule engine", action="Explain simply", processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], rule["intent"], rule["rule_id"], rule["confidence"], topic=rule.get("topic", ""))
            return payload

    # 6) Rule engine, with the LLM as a labelled fallback for UNKNOWN or
    # low-confidence answers only. High-confidence rule answers, maths,
    # search, documents and files above are never replaced by the model.
    rule = match_rule(text)
    if rule["intent"] == "UNKNOWN" or rule["confidence"] < 0.45:
        llm, _llm_error = _llm_answer(text, context, processing_ms)
        if llm is not None:
            context.add_exchange(text, llm["response"], "LLM_ANSWER", "llm_answer", 0.65)
            return llm
    adapted = _adapt_to_preferences(rule["response"], context)
    if rule["intent"] != "UNKNOWN" and rule["confidence"] < 0.45:
        adapted += "\n\nNote: My confidence here is low, so please double-check this — I am not presenting a guess as a fact."
    payload = format_response(
        text=adapted, intent=rule["intent"], rule_id=rule["rule_id"],
        confidence=rule["confidence"], tool="Rule engine",
        action="Answer from rule base" if rule["intent"] != "UNKNOWN" else "Graceful fallback",
        extra={"matched_patterns": rule["matched_patterns"]},
        processing_ms=processing_ms)
    context.add_exchange(text, payload["response"], rule["intent"],
                         rule["rule_id"], rule["confidence"], topic=rule.get("topic", ""))
    return payload


def process_message(message: str, context, processing_ms=0.0):
    """Wrapper: handle multi-step instructions, then a single message.

    Multi-step example: "What is Python? then calculate 25% of 800"
    is split on " then " and each step is processed in order, so the
    bot can follow a sequence instead of answering only the first part.
    """
    text = (message or "").strip()
    low = text.lower()
    if " then " in low and not low.startswith("run python code"):
        parts = [p.strip() for p in re.split(r"\s+then\s+", text) if p.strip()]
        if len(parts) > 1 and len(parts) <= 5:
            responses = []
            last_payload = None
            for part in parts:
                last_payload = process_single_message(part, context, processing_ms)
                responses.append(last_payload["response"])
            combined = format_response(
                text="\n\n---\n\n".join(responses),
                intent="MULTI_STEP", rule_id="multi_step", confidence=0.90,
                tool="Multi-step pipeline", action="Follow multi-step instructions",
                processing_ms=processing_ms)
            context.add_exchange(text, combined["response"], "MULTI_STEP", "multi_step", 0.90)
            return combined
    return process_single_message(message, context, processing_ms)
