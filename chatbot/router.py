"""Central pipeline: validation -> normalisation -> context -> intent
-> rule matching -> confidence -> decision -> tool -> response -> context update.

Pipeline order (see README):
  message -> maths? -> pdf? -> search? -> follow-up? -> rule engine -> fallback
Maths / PDF / search are checked before generic rules so that e.g.
'Calculate sqrt(144)' is not answered by the generic maths-info rule.
"""
import re

from chatbot.confidence import classify  # noqa: F401  (re-exported for tests)
from chatbot.patterns import FOLLOWUP_USED_FOR, RULES
from chatbot.response_manager import format_response
from chatbot.rule_engine import match_rule
from services.math_service import MathService, looks_like_math
from services.pdf_service import PDFService
from services.web_search_service import WebSearchService
from services.web_answer import (format_web_answer, clean_search_topic, filter_results,
                              _is_howto_query, _content_nouns, _result_mentions_nouns)
from services.code_service import (generate_code, debug_code, compare as compare_topics,
                                  detect_code_request, generate_code_for, explain_code)
from services.code_runner import run_python_code
from services.document_service import summarize as summarize_doc, answer_from_document, make_notes as make_doc_notes
from services.file_generation_service import generate_pptx, generate_docx, generate_xlsx
from services.model_manager import ModelManager
from services.conversation_manager import build_messages, detect_language
from services.environment_analyzer import (
    analyze_environment, format_environment_table, is_explicit_request,
    should_analyze)
from services.context_memory import (
    apply_math_followup, resolve_followup, resolve_math_followup,
    resolve_previous_reference, simplify_sentence)
from services.response_generator import validate_llm_answer, format_llm_answer
import os
from utils.text_processing import normalize_for_matching

math_service = MathService()
pdf_service = PDFService()
search_service = WebSearchService()


def _personalize_identity(text: str, rule_id: str, character_name: str) -> str:
    """Name the selected character as Shibu AI's persona in identity
    answers, without changing any other answer. With no character
    selected the static "Shibu AI" wording is returned unchanged, so
    anonymous chat is identical to before. Never claims to be Muse,
    Meta AI, another assistant, or human."""
    if not character_name:
        return text
    if rule_id == "about_bot":
        prefix = "I'm Shibu AI, a Rule-Based Intelligent Chatbot: "
        if text.startswith(prefix):
            return "I'm " + character_name + ", Shibu AI's character: " + text[len(prefix):]
    elif rule_id == "bot_name":
        prefix = "I'm Shibu AI — your Rule-Based Intelligent Chatbot."
        if text.startswith(prefix):
            return ("I'm " + character_name + " — Shibu AI's character and your "
                    "Rule-Based Intelligent Chatbot." + text[len(prefix):])
    elif rule_id == "conversation_human_robot":
        old = "I'm Shibu AI, a rule-based chatbot program."
        if old in text:
            return text.replace(old, "I'm " + character_name + ", the Shibu AI character, a chatbot program.", 1)
    return text


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
    """Turn raw search results into a phrased, text-processed answer.

    Uses services/web_answer: snippets are cleaned, split into
    sentences, scored by query-keyword coverage, deduplicated, and
    then presented as a short summary + key points + sources. Web
    results are never presented as if they came from the rule database.
    """
    return format_web_answer(query, results)


def _first_sentence(text: str) -> str:
    import re as _re
    parts = _re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text


# ------------------------------------------------------------------
# Explicit-memory recall (D9): a question about a personal fact the
# user asked Shibu to remember must be answered from the stored
# memory — never by a generic rule ("my favourite colour is the
# black background"). Runs before the rule engine.
# ------------------------------------------------------------------
_MEMORY_QUESTION_RE = re.compile(
    r"\bwhat(?:'s| is)\s+my\s+(favourite|favorite)\s+([a-z]+)\b"
    r"|\bmy\s+(favourite|favorite)\s+([a-z]+)\b.*\?$"
    r"|\bwhat\s+do\s+you\s+remember\s+about\s+my\s+([a-z ]+?)\s*\?$",
    re.I)


def _memory_recall(text: str, context):
    """Return an answer for a personal-fact question, or None.

    Stored value wins when a memory mentions the asked fact;
    otherwise say plainly that it is not remembered — the generic
    definition rule must not answer a personal question.
    """
    m = _MEMORY_QUESTION_RE.search(text or "")
    if not m:
        return None
    asked = " ".join(g for g in m.groups() if g).lower().strip()
    asked_words = [w for w in re.findall(r"[a-z]+", asked) if w not in ("favourite", "favorite", "my")]
    memories = list(getattr(context, "memories", None) or [])
    for fact in memories:
        low_fact = fact.lower()
        if any(w in low_fact for w in asked_words) or asked in low_fact:
            return (f"From what you asked me to remember: {fact}. "
                    "Say 'Forget that' if you want me to forget it.")
    label = " ".join(asked_words) or "that"
    return (f"I haven't saved anything about your {label} yet. "
            "Say e.g. 'Remember that my favourite colour is blue' and I'll save it.")


# ------------------------------------------------------------------
# Fuzzy fallback (D15): a typo'd topic ("wht is fotosintesis") must
# still reach its rule. Only applied when no rule matched, with a
# strict cutoff, so unrelated gibberish still falls back.
# ------------------------------------------------------------------
def _fuzzy_correct(text: str):
    """Return a corrected message, or None when nothing is close.

    Each word is matched against the rule vocabulary (keywords and
    topics); a word is replaced only by a close match (difflib
    ratio >= 0.75) that is at least 5 letters long, so a short typo
    like 'wht' cannot trigger a correction and unrelated gibberish
    still falls back.
    """
    import difflib
    vocab = set()
    for rule in RULES:
        for kw in rule.get("keywords", []) or []:
            vocab.add(kw.lower())
        topic = (rule.get("topic") or "").lower()
        vocab.update(re.findall(r"[a-z]+", topic))
        for pat in rule.get("patterns", []) or []:
            vocab.update(w for w in re.findall(r"[a-z]+", pat.lower()) if len(w) >= 5)
    vocab = {w for w in vocab if len(w) >= 5}
    words = (text or "").split()
    changed = False
    out = []
    for word in words:
        core = re.sub(r"[^a-z]", "", word.lower())
        if core and core not in vocab and len(core) >= 5:
            matches = difflib.get_close_matches(core, vocab, n=1, cutoff=0.75)
            if matches:
                word = word.lower().replace(core, matches[0])
                changed = True
        out.append(word)
    if not changed:
        return None
    return " ".join(out)


# ------------------------------------------------------------------
# Fun facts bank (D17): small curated facts, one topic per entry.
# Used for "give N fun facts" and multi-part instructions. Facts are
# about the named topic only — never invented from other content.
# ------------------------------------------------------------------
FUN_FACTS = {
    "black holes": [
        "A black hole's gravity is so strong that not even light can escape once it crosses the edge, called the event horizon.",
        "Time runs slower near a black hole than far away from it, an effect predicted by Einstein's theory of relativity.",
        "The black hole at the centre of our Milky Way galaxy, Sagittarius A*, weighs about 4 million times as much as the Sun.",
        "Black holes can form when a very massive star runs out of fuel and collapses under its own gravity.",
        "Some black holes spin so fast that they drag the space around them along, an effect called frame-dragging.",
    ],
    "photosynthesis": [
        "Photosynthesis is how green plants make food using sunlight, water and carbon dioxide.",
        "A by-product of photosynthesis is oxygen, which is the gas most living things breathe.",
        "The green colour of leaves comes from chlorophyll, the pigment that captures sunlight for photosynthesis.",
        "Photosynthesis in ocean plankton produces a large share of the oxygen in Earth's air.",
    ],
    "machine learning": [
        "Machine learning systems learn patterns from data instead of following only fixed rules written by a programmer.",
        "Spam filters use machine learning to learn which messages look like spam from examples.",
        "Machine learning improves with more good data, much like practice improves a skill.",
        "Recommendation systems use machine learning to guess what you might like next from what you watched or bought before.",
    ],
}


def _previous_fact_topic(context) -> str:
    """The topic a bare 'give me N fun facts' refers to.

    The previous question's own words win over the stored topic
    label: the photosynthesis rule's topic label is 'biology', but
    'give me 2 fun facts' after asking about photosynthesis must
    give photosynthesis facts. Falls back to the stored topic.
    """
    last_user = (getattr(context, "last_user_message", "") or "").lower()
    words = re.findall(r"[a-z]+", last_user)
    for word in words:
        if word in FUN_FACTS:
            return word
    for word in words:
        if len(word) >= 4 and match_rule(word)["intent"] != "UNKNOWN":
            return word
    return (getattr(context, "last_topic", "") or "").strip()


def _fun_facts_answer(topic: str, count: int) -> str:
    """N facts about the topic, from the curated bank when it has
    the topic, else from the rule answer's own sentences (which are
    about the topic by construction)."""
    topic_key = (topic or "").strip().lower()
    facts = FUN_FACTS.get(topic_key)
    if facts:
        picked = facts[:max(1, count)]
        lines = [f"Fun fact {i + 1}: {f}" for i, f in enumerate(picked)]
        return "\n".join(lines)
    rule = match_rule(topic_key)
    if rule["intent"] == "UNKNOWN":
        return ""
    sents = re.split(r"(?<=[.!?])\s+", rule["response"].strip())
    picked = [s.strip() for s in sents if s.strip()][:max(1, count)]
    return "\n".join(f"Fun fact {i + 1}: {s}" for i, s in enumerate(picked))


_FACTS_WORDS = ("fun facts", "fun fact", "interesting facts", "interesting fact",
                "cool facts", "cool fact", "amazing facts", "amazing fact")


def _multipart_instruction(text: str, context, processing_ms=0.0):
    """Parse explain + facts + question in one message, joined by
    commas, "then", "and", or any combination:

        Explain black holes to a 10 year old, then give 3 fun facts,
        then ask me one question about it
        Explain black holes, give me 3 interesting facts about
        black holes, and ask me a question about black holes

    Returns a payload dict or None. All three parts are built from
    the topic's own rule answer and the facts bank — the word
    'give' never reaches the dictionary/web fallback. The output
    is always explanation, then labelled facts, then a closing
    question, in that order, whatever separator was used.
    """
    low = (text or "").lower().strip()
    if "explain" not in low or "question" not in low:
        return None
    if not any(word in low for word in _FACTS_WORDS):
        return None
    # Split into parts on commas and on "then"/"and" before a verb.
    raw_parts = re.split(r"\s*,\s*|\s+then\s+|\s+and\s+(?=(?:give|ask|tell|explain)\b)", low)
    parts = []
    for part in raw_parts:
        part = part.strip()
        part = re.sub(r"^(?:and|then)\s+", "", part)
        part = part.strip(" .")
        if part:
            parts.append(part)
    explain_part = next((p for p in parts if p.startswith("explain")), None)
    facts_part = next((p for p in parts if "fact" in p), None)
    question_part = next((p for p in parts if "question" in p), None)
    if not explain_part or not facts_part or not question_part:
        return None
    topic = re.sub(r"^explain\s+", "", explain_part).strip()
    topic = re.sub(r"\s+to\s+(?:a\s+)?(?:\d+\s+)?(?:year\s+old|kid|child|beginner)\s*$", "", topic)
    topic = re.sub(r"\s+(?:in\s+simple\s+(?:words|terms)|simply|like\s+i'?m\s+\d+|to\s+me)\s*$", "", topic)
    topic = topic.strip(" ,")
    if not topic:
        return None
    # The facts part may name its own topic ("facts about X");
    # otherwise it inherits the explain topic.
    facts_topic = topic
    about = re.search(r"facts?\s+about\s+(.+)$", facts_part)
    if about:
        facts_topic = about.group(1).strip(" ,")
    n_match = re.search(r"(\d+)\s+(?:fun\s+|interesting\s+|cool\s+|amazing\s+)?facts?", facts_part)
    n = int(n_match.group(1)) if n_match else 3
    rule = match_rule(topic)
    if rule["intent"] == "UNKNOWN":
        return None
    first = _first_sentence(rule["response"])
    explanation = simplify_sentence(first)
    facts = _fun_facts_answer(facts_topic, n)
    if not facts:
        return None
    question = f"Here is one question for you: can you tell me one thing you learned about {facts_topic}?"
    answer = (f"{explanation}\n\n{facts}\n\n{question}")
    payload = format_response(text=answer, intent="MULTI_PART_EXPLAIN",
        rule_id="multipart_explain", confidence=0.90,
        tool="Rule engine + fun facts", action="Explain simply, give fun facts, ask a question",
        processing_ms=processing_ms)
    context.add_exchange(text, answer, "MULTI_PART_EXPLAIN", "multipart_explain", 0.90,
                         topic=rule.get("topic", topic))
    return payload


def _comparison_quiz_answer(a: str, b: str, context) -> str | None:
    """Build a quiz about a comparison from its own rows/facts.

    Hand-written questions win; otherwise the comparison prose
    supplies one question per entity. Returns None when the pair
    has no comparison content at all (caller falls through).
    """
    from services.code_service import COMPARISONS, COMPARISON_QUIZZES, _normalise_compare_name
    key = (_normalise_compare_name(a), _normalise_compare_name(b))
    rev = (key[1], key[0])
    pair = key if key in COMPARISON_QUIZZES or key in COMPARISONS else rev
    questions = COMPARISON_QUIZZES.get(pair)
    if questions:
        lines = ["Quiz on the comparison:"]
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q['question']}")
            for opt in q["options"]:
                lines.append(f"   - {opt}")
            lines.append(f"   Answer: {q['answer']}")
        return "\n".join(lines)
    prose = COMPARISONS.get(pair)
    if not prose:
        return None
    a_name, b_name = pair
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if s.strip()]
    a_sents = [s for s in sents if a_name in s.lower()]
    b_sents = [s for s in sents if b_name in s.lower()]
    if not a_sents or not b_sents:
        return None
    lines = ["Quiz on the comparison:",
             f"1. Which of these is true of {a_name.title()}?",
             f"   - {a_sents[0]}",
             f"   - {b_sents[0]}",
             f"   Answer: {a_sents[0]}",
             f"2. Which of these is true of {b_name.title()}?",
             f"   - {b_sents[0]}",
             f"   - {a_sents[0]}",
             f"   Answer: {b_sents[0]}"]
    return "\n".join(lines)


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


_SECRET_MEMORY_PATTERNS = re.compile(
    r"\b(password|api key|secret|access token|otp|one-time code|credit card|bank account)\b", re.I)


def _memory_command(text, context):
    """Explicit-memory commands, or None when this is not one.

    Only facts the user explicitly asks to remember are stored, in
    the session context (persisted per-user by app.py). Secrets are
    refused, never stored. Returns a dict with kind + text details.
    """
    low = text.strip().lower()
    # List what is remembered.
    if re.fullmatch(r"(what do you remember( about me)?|what have you remembered|"
                    r"show( me)?( my)? memories|list( what you remember| my memories)?)\??", low):
        return {"kind": "list"}
    # Forget everything.
    if re.fullmatch(r"(forget everything( you remember( about me)?)?|forget all( my)? memories|"
                    r"clear( my)? memories|forget everything about me)\.?", low):
        return {"kind": "clear"}
    # Forget one fact (by words it contains).
    m_forget = re.match(r"^(?:please\s+)?forget\s+(?:that\s+)?(.+?)\.?\s*$", low)
    if m_forget and m_forget.group(1) not in ("it", "that", "this"):
        return {"kind": "forget", "query": m_forget.group(1).strip()}
    # Remember one fact.
    m_rem = re.match(r"^(?:please\s+)?remember\s+(?:that\s+|this:?\s*)?(.+?)\.?\s*$", text.strip(), re.I)
    if m_rem:
        fact = " ".join(m_rem.group(1).strip().split())
        if not fact or fact.lower() in ("it", "that", "this"):
            return {"kind": "clarify",
                    "question": "What should I remember? For example: 'Remember that I prefer simple explanations.'"}
        if _SECRET_MEMORY_PATTERNS.search(fact):
            return {"kind": "refused",
                    "question": ("I won't store passwords, keys, tokens or card details as a memory — "
                                "those belong only in your .env file or password manager, never in chat. "
                                "Tell me something else to remember, and I'll save that instead.")}
        return {"kind": "save", "fact": fact}
    return None


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



# Small built-in synonym / antonym lists for common school words.
# Honest limit: if a word is not here, say so instead of guessing.
_SYNONYMS = {
    "happy": "joyful", "big": "large", "small": "tiny", "fast": "quick",
    "good": "excellent", "bad": "poor", "smart": "intelligent",
    "brave": "courageous", "rich": "wealthy", "angry": "furious",
    "beautiful": "pretty", "strong": "powerful", "weak": "feeble",
    "begin": "start", "end": "finish", "love": "adore",
}
_ANTONYMS = {
    "happy": "sad", "big": "small", "fast": "slow", "good": "bad",
    "smart": "foolish", "brave": "cowardly", "rich": "poor",
    "strong": "weak", "begin": "end", "love": "hate",
    "hot": "cold", "up": "down", "in": "out", "day": "night",
}


def _synonym_antonym_answer(low: str):
    """Return a payload text for 'synonym of X' / 'antonym of X', or None."""
    m = re.search(r"\b(synonym|antonym) of ([a-z]+)\b", low)
    if not m:
        return None
    kind, word = m.group(1), m.group(2)
    table = _SYNONYMS if kind == "synonym" else _ANTONYMS
    if word in table:
        return f"The {kind} of '{word}' is '{table[word]}'. Example: a synonym is a word with a similar meaning, an antonym is the opposite."
    return f"I don't have '{word}' in my small synonym/antonym list yet, so I won't guess one. Words I do cover include: {', '.join(sorted(table))}. Try one of those, or use web search: 'Search synonym of {word}'."


def _series_answer(low: str):
    """Solve a simple arithmetic / geometric number series, or None."""
    m = re.search(r"next (?:number|term) in (?:the )?series\s+([0-9,\s]+)", low)
    if not m:
        return None
    try:
        nums = [float(x) for x in m.group(1).replace(",", " ").split() if x.strip()]
    except Exception:
        return None
    if len(nums) < 3:
        return None
    diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
    if len(set(diffs)) == 1:
        nxt = nums[-1] + diffs[0]
        return f"Series: {', '.join(str(int(n)) if n.is_integer() else str(n) for n in nums)}. The difference is constant ({diffs[0]:g}), so it is an arithmetic series. Next term: {nxt:g}."
    ratios = []
    try:
        ratios = [nums[i + 1] / nums[i] for i in range(len(nums) - 1) if nums[i] != 0]
    except Exception:
        ratios = []
    if ratios and len(set(ratios)) == 1:
        nxt = nums[-1] * ratios[0]
        return f"Series: {', '.join(str(int(n)) if n.is_integer() else str(n) for n in nums)}. The ratio is constant ({ratios[0]:g}), so it is a geometric series. Next term: {nxt:g}."
    return f"Series: {', '.join(str(int(n)) if n.is_integer() else str(n) for n in nums)}. I could not find a constant difference or ratio, so this is not a simple arithmetic or geometric series. Check for squares, cubes, or alternating patterns, and try again."


def _web_fallback_answer(text: str, context, processing_ms=0.0):
    """Try a text-processed web answer for an unknown/definition question.

    Returns a payload or None (no results). This is what replaces the
    old 'I don't have a specific rule for that term yet' dead end:
    if the web has a sourced answer, show it, labelled as web-derived.
    """
    topic = clean_search_topic(text)
    if not topic:
        return None
    raw_results = search_service.search(topic)
    if _is_howto_query(topic):
        # Query expansion: when the first result set never mentions
        # the task's content noun ("tea" in "how to make tea" — a
        # set about the Make platform mentions only the verb), the
        # first search answered the wrong thing. Retry once with
        # an expanded query before giving up, and use the expanded
        # set only when it actually covers the content noun.
        nouns = _content_nouns(topic)
        if nouns and not any(_result_mentions_nouns(r, nouns) for r in (raw_results or [])):
            expanded = search_service.search(topic + " step by step")
            if any(_result_mentions_nouns(r, nouns) for r in (expanded or [])):
                raw_results = expanded
    results = filter_results(topic, raw_results)
    if not results:
        return None
    payload = format_response(
        text=format_web_answer(topic, results),
        intent="WEB_SEARCH", rule_id="web_fallback", confidence=0.85,
        tool="DuckDuckGo search + text processing", action="Answer unknown topic from web",
        extra={"search_results": results, "search_query": topic,
               "sources": [{"citation": f"W{i}", "source_name": r.get("title", "Web result"),
                            "url": r.get("url", ""), "type": "web"}
                           for i, r in enumerate(results[:5], 1)]},
        processing_ms=processing_ms)
    context.add_exchange(text, payload["response"], "WEB_SEARCH", "web_fallback", 0.85)
    return payload


def process_single_message(message: str, context, processing_ms=0.0, web_search_enabled=True, character_name=""):
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

    # 0b) Arithmetic on the previous answer — BEFORE maths, rules and
    # web search, or a message like "add 967464 in privious answer"
    # falls through to UNKNOWN and then to a web search (the bug in
    # the screenshot: first sum 49933, second message web-searched
    # instead of adding). This step uses the task memory in the
    # context (active_task + last_numeric_value), never the web.
    _math_follow = resolve_math_followup(text, context)
    if _math_follow is not None:
        if _math_follow["kind"] == "clarify":
            payload = format_response(text=_math_follow["question"], intent="CLARIFY",
                rule_id="clarify_math_followup", confidence=0.90,
                tool="Conversation context", action="Ask clarifying question",
                processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "CLARIFY", "clarify_math_followup", 0.90)
            return payload
        payload = format_response(text=apply_math_followup(_math_follow),
            intent="MATH_FOLLOWUP", rule_id="math_followup", confidence=0.99,
            tool="Conversation context + SymPy maths", action="Calculate from previous answer",
            processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "MATH_FOLLOWUP", "math_followup", 0.99)
        return payload
    _prev_ref = resolve_previous_reference(text, context)
    if _prev_ref is not None:
        if _prev_ref["kind"] == "show_previous":
            payload = format_response(text=_prev_ref["answer"], intent="FOLLOW_UP",
                rule_id="context_show_previous", confidence=0.95,
                tool="Conversation context", action="Show previous answer",
                processing_ms=processing_ms)
        else:
            payload = format_response(text=_prev_ref["question"], intent="CLARIFY",
                rule_id="clarify_previous", confidence=0.90,
                tool="Conversation context", action="Ask clarifying question",
                processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], payload["intent"], payload["rule_id"], payload["confidence"])
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
        results = filter_results(query, search_service.search(query))
        if results:
            payload = format_response(
                text=_phrase_search_results(query, results),
                intent="WEB_SEARCH", rule_id="web_search", confidence=0.90,
                tool="DuckDuckGo search", action="Search the web",
                extra={"search_results": results, "search_query": query,
                       "sources": [{"citation": f"W{i}", "source_name": r.get("title", "Web result"),
                                    "url": r.get("url", ""), "type": "web"}
                                   for i, r in enumerate(results[:5], 1)]},
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

    # 5b2) Explicit memory: "remember that ...", "what do you
    # remember?", "forget ...". Stored only when the user asks, in
    # the session context; app.py persists it per-user in SQLite.
    _mem = _memory_command(text, context)
    if _mem is not None:
        if _mem["kind"] == "save":
            saved = context.remember(_mem["fact"])
            _txt = (f"Got it — I'll remember: {_mem['fact']}" if saved
                    else f"I'm already remembering: {_mem['fact']}")
            _rid = "memory_save"
        elif _mem["kind"] == "list":
            if not context.memories:
                _txt = ("I haven't been asked to remember anything about you yet. "
                        "Say e.g. 'Remember that I prefer simple explanations' and I'll save it. "
                        "Say 'Forget that' to remove it later.")
            else:
                _txt = "Here's what you've asked me to remember:\n" + "\n".join(f"- {f}" for f in context.memories)
            _rid = "memory_list"
        elif _mem["kind"] == "clear":
            _n = context.forget_all_memories()
            _txt = ("Done — I've forgotten everything you asked me to remember."
                    if _n else "There was nothing saved to forget.")
            _rid = "memory_forget"
        elif _mem["kind"] == "forget":
            _n = context.forget_matching(_mem["query"])
            _txt = (f"Done — I forgot {_n} saved memory/memories about '{_mem['query']}'."
                    if _n else f"I couldn't find a saved memory about '{_mem['query']}'. Say 'What do you remember about me?' to see what's saved.")
            _rid = "memory_forget"
        else:  # clarify / refused
            _txt = _mem["question"]
            _rid = "memory_refused" if _mem["kind"] == "refused" else "clarify_memory"
        payload = format_response(text=_txt, intent="MEMORY", rule_id=_rid, confidence=0.95,
            tool="Memory", action="Explicit memory", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "MEMORY", _rid, 0.95)
        return payload

    # 5b3) Explicit-memory recall: a question about a personal fact
    # the user asked Shibu to remember is answered from the stored
    # memory — or plainly "not remembered" when nothing is stored.
    # A generic rule (e.g. the bot's own favourite colour) must
    # never answer a personal question.
    _recall = _memory_recall(text, context)
    if _recall is not None:
        payload = format_response(text=_recall, intent="MEMORY", rule_id="memory_recall", confidence=0.95,
            tool="Memory", action="Recall explicit memory", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "MEMORY", "memory_recall", 0.95)
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

    # 5d2) "Give me N fun facts (about <topic>)" — facts about the
    # topic from the curated bank or the topic's own rule answer.
    # The word "give" must never reach the dictionary/web fallback.
    _ff = re.search(r"\bgive\s+(?:me\s+)?(\d+)\s+fun facts?(?:\s+about\s+(.+))?", low)
    if _ff:
        _ff_topic = (_ff.group(2) or "").strip().rstrip(".")
        if not _ff_topic:
            _ff_topic = _previous_fact_topic(context)
        if _ff_topic:
            _ff_text = _fun_facts_answer(_ff_topic, int(_ff.group(1)))
            if _ff_text:
                payload = format_response(text=_ff_text, intent="FUN_FACTS", rule_id="fun_facts",
                    confidence=0.90, tool="Fun facts bank", action="Give fun facts",
                    processing_ms=processing_ms)
                context.add_exchange(text, payload["response"], "FUN_FACTS", "fun_facts", 0.90,
                                     topic=_ff_topic)
                return payload

    # 5e) Code generation, debugging, comparison
    if (re.search(r"write (python |javascript |js )?code (to|for)", low) or low.startswith("code for")
            or low.startswith("code to")
            or re.search(r"write the same|same (code )?in (javascript|python|\bjs\b)|write (it|that) in (javascript|python|\bjs\b)", low)):
        _code_req = detect_code_request(text, getattr(context, "last_code_task", None))
        if _code_req is not None:
            _task, _lang = _code_req
            code, expl = generate_code_for(_task, _lang)
            _fence = "javascript" if _lang == "javascript" else "python"
            _bullets = explain_code(_task, _lang)
            _code_text = f"{expl}\n\n```{_fence}\n{code}\n```"
            if _bullets:
                _code_text += f"\n\nHow it works:\n{_bullets}"
            payload = format_response(text=_code_text,
                intent="CODE_GENERATION", rule_id="code_template", confidence=0.90, tool="Code templates", action="Generate code", processing_ms=processing_ms)
            context.last_code_task = _task
            context.last_code_language = _lang
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
        context.last_comparison = (a.strip(), b.strip())
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

    # 5i) Explain simply: the full rule answer, structured as a
    # definition plus key points — never truncated to one flat
    # sentence, so the presentation layer can show real structure.
    m_simple = re.search(r"explain (.+?) (simply|in simple language|in simple terms)", low)
    if m_simple:
        rule = match_rule(m_simple.group(1))
        if rule["intent"] != "UNKNOWN":
            _sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rule["response"].strip()) if s.strip()]
            if len(_sents) >= 2:
                _explain_text = _sents[0] + "\n\nKey points:\n" + "\n".join(f"- {s}" for s in _sents[1:])
            else:
                _explain_text = rule["response"]
            payload = format_response(text=_explain_text,
                intent=rule["intent"], rule_id=rule["rule_id"], confidence=rule["confidence"], tool="Rule engine", action="Explain simply", processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], rule["intent"], rule["rule_id"], rule["confidence"], topic=rule.get("topic", ""))
            return payload

    # 5j) Synonym / antonym lookup and number-series solver
    _vocab = _synonym_antonym_answer(low)
    if _vocab:
        payload = format_response(text=_vocab, intent="VOCABULARY", rule_id="vocabulary_lookup", confidence=0.90, tool="Vocabulary list", action="Synonym / antonym lookup", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "VOCABULARY", "vocabulary_lookup", 0.90)
        return payload
    _series = _series_answer(low)
    if _series:
        payload = format_response(text=_series, intent="REASONING_SERIES", rule_id="series_solver", confidence=0.95, tool="Reasoning solver", action="Solve number series", processing_ms=processing_ms)
        context.add_exchange(text, payload["response"], "REASONING_SERIES", "series_solver", 0.95)
        return payload

    # 5k) Conversation context memory: follow-ups that refer to the
    # previous answer ("explain it more simply", "another example",
    # "the second one", "continue", "it / that topic"). A bare
    # reference with no previous answer gets a short clarification
    # question, never a guess. A genuinely new topic is not a
    # follow-up, so it falls through to the rule engine on its own.
    _follow = resolve_followup(text, context)
    if _follow is not None:
        if _follow["kind"] == "clarify":
            payload = format_response(text=_follow["question"], intent="CLARIFY",
                rule_id="clarify_followup", confidence=0.90, tool="Conversation context",
                action="Ask clarifying question", processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "CLARIFY", "clarify_followup", 0.90)
            return payload
        if _follow["kind"] == "reference":
            _rule = match_rule(_follow["topic"])
            if _rule["intent"] != "UNKNOWN":
                payload = format_response(text=_rule["response"], intent=_rule["intent"],
                    rule_id=_rule["rule_id"], confidence=_rule["confidence"],
                    tool="Rule engine + conversation context",
                    action="Answer follow-up from previous topic", processing_ms=processing_ms)
                context.add_exchange(text, payload["response"], _rule["intent"],
                                     _rule["rule_id"], _rule["confidence"], topic=_follow["topic"])
                return payload
            # Fall through: no rule for the old topic, let the normal
            # pipeline (LLM / web fallback) handle it below.
        else:
            payload = format_response(text=_follow["answer"], intent="FOLLOW_UP",
                rule_id="context_followup_memory", confidence=0.90,
                tool="Conversation context", action="Answer follow-up from previous answer",
                processing_ms=processing_ms)
            context.add_exchange(text, payload["response"], "FOLLOW_UP",
                                 "context_followup_memory", 0.90, topic=context.last_topic)
            return payload

    # 5l) AI Environment Properties Analyzer.
    # Explicit requests ("Classify a self-driving car environment.")
    # are answered with the table itself. So are scenario +
    # environment questions whose only rule match would be the
    # generic "ecosystem" rule or nothing at all (e.g. "Tell me about
    # a robot navigating a warehouse environment") — otherwise that
    # unrelated ecosystem answer would sit above the table. For a
    # scenario question with a genuinely relevant rule answer, the
    # table is appended after the normal answer by process_message().
    if is_explicit_request(low) or should_analyze(text):
        try:
            _env = analyze_environment(text)
        except Exception:
            _env = None
        if _env is not None:
            _rule_for_env = match_rule(text)
            if is_explicit_request(low) or _rule_for_env["intent"] in ("ECOSYSTEM", "UNKNOWN"):
                payload = format_response(
                    text=format_environment_table(_env),
                    intent="ENVIRONMENT_ANALYSIS", rule_id="environment_analysis",
                    confidence=0.90, tool="Environment analyzer",
                    action="Classify AI environment properties",
                    extra={"environment_analysis": _env}, processing_ms=processing_ms)
                context.add_exchange(text, payload["response"], "ENVIRONMENT_ANALYSIS",
                                     "environment_analysis", 0.90)
                return payload
        # Analyzer failure must not hide the normal answer: fall
        # through to the rule engine / web fallback below.

    # 5m) Daily tasks (Design B, LLM-first): plans, writing and
    # how-tos skip the rule engine, so a generic keyword rule can
    # never answer them with the wrong topic. Maths, search,
    # documents and files above keep priority. If the model is
    # unavailable for a how-to, fall through to the web fallback
    # below instead of giving up.
    from services.daily_task_service import build_daily_task_payload, unavailable_howto_payload
    daily_payload, _needs_web = build_daily_task_payload(text, context, processing_ms)
    if daily_payload is not None:
        return daily_payload
    if _needs_web:
        if web_search_enabled:
            web_payload = _web_fallback_answer(text, context, processing_ms)
            if web_payload is not None:
                return web_payload
        # Built-in last resort for very common tasks (tea, coffee):
        # only when the web yielded nothing usable. Anything else
        # falls through to the honest unavailable answer below.
        from services.daily_task_service import builtin_howto_payload
        builtin_payload = builtin_howto_payload(text, context, processing_ms)
        if builtin_payload is not None:
            return builtin_payload
        return unavailable_howto_payload(text, context, processing_ms)

    # 5n) Quiz on the previous comparison: "give me a quiz on that
    # comparison" must quiz the compared topics from the comparison
    # itself — never fall back to a web search (which returned
    # English-grammar content for a Python/JavaScript comparison).
    if re.search(r"\bquiz\b|\btest me\b|\bmcq\b", low):
        _quiz_pair = None
        _cmp = getattr(context, "last_comparison", None) or ()
        if _cmp and (re.search(r"\b(that|this|the)\b.*\bcomparison\b|\bcomparison\b", low)
                    or any(n.lower() in low for n in _cmp)):
            _quiz_pair = _cmp
        else:
            _m_quiz = re.search(r"quiz (?:me )?on ([a-z]+) (?:and|vs|versus) ([a-z]+)", low)
            if _m_quiz:
                _quiz_pair = (_m_quiz.group(1), _m_quiz.group(2))
        if _quiz_pair:
            _quiz_text = _comparison_quiz_answer(_quiz_pair[0], _quiz_pair[1], context)
            if _quiz_text:
                payload = format_response(text=_quiz_text, intent="QUIZ", rule_id="comparison_quiz",
                    confidence=0.90, tool="Comparison", action="Quiz on the comparison",
                    processing_ms=processing_ms)
                context.add_exchange(text, payload["response"], "QUIZ", "comparison_quiz", 0.90)
                return payload

    # 6) Rule engine, with the LLM as a labelled fallback for UNKNOWN or
    # low-confidence answers only. High-confidence rule answers, maths,
    # search, documents and files above are never replaced by the model.
    rule = match_rule(text)
    if rule["intent"] == "UNKNOWN":
        # Typo tolerance: a close misspelling of a known topic
        # ("wht is fotosintesis") is corrected and re-matched, so it
        # reaches its rule instead of the web fallback. Gibberish
        # with no close match keeps falling back as before.
        _corrected = _fuzzy_correct(text)
        if _corrected:
            _fixed = match_rule(_corrected)
            if _fixed["intent"] != "UNKNOWN" and _fixed["confidence"] >= 0.45:
                rule = _fixed
    if rule["intent"] == "UNKNOWN" or rule["confidence"] < 0.45:
        llm, _llm_error = _llm_answer(text, context, processing_ms)
        if llm is not None:
            context.add_exchange(text, llm["response"], "LLM_ANSWER", "llm_answer", 0.65)
            return llm
    # New: if the best rule is the generic "definition" fallback (or
    # UNKNOWN with no LLM), try a text-processed web answer first.
    # This replaces the old dead-end message for unknown topics.
    if rule["intent"] == "UNKNOWN" or rule["rule_id"] == "definition":
        if web_search_enabled:
            web_payload = _web_fallback_answer(text, context, processing_ms)
            if web_payload is not None:
                return web_payload
    adapted = _adapt_to_preferences(rule["response"], context)
    adapted = _personalize_identity(adapted, rule["rule_id"], character_name)
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


def process_message(message: str, context, processing_ms=0.0, web_search_enabled=True, character_name=""):
    """Wrapper: handle multi-step instructions, then a single message.

    Multi-step example: "What is Python? then calculate 25% of 800"
    is split on " then " and each step is processed in order, so the
    bot can follow a sequence instead of answering only the first part.
    """
    text = (message or "").strip()
    low = text.lower()
    # Multi-part instruction ("explain X to a 10 year old, then give
    # 3 fun facts, then ask me one question"): handled whole, before
    # the " then " split, so no part is answered in isolation (the
    # word "give" must never reach the dictionary/web fallback).
    _multi = _multipart_instruction(text, context, processing_ms)
    if _multi is not None:
        return _multi
    if " then " in low and not low.startswith("run python code"):
        parts = [p.strip() for p in re.split(r"\s+then\s+", text) if p.strip()]
        if len(parts) > 1 and len(parts) <= 5:
            responses = []
            last_payload = None
            for part in parts:
                last_payload = process_single_message(part, context, processing_ms, web_search_enabled=web_search_enabled, character_name=character_name)
                responses.append(last_payload["response"])
            combined = format_response(
                text="\n\n---\n\n".join(responses),
                intent="MULTI_STEP", rule_id="multi_step", confidence=0.90,
                tool="Multi-step pipeline", action="Follow multi-step instructions",
                processing_ms=processing_ms)
            context.add_exchange(text, combined["response"], "MULTI_STEP", "multi_step", 0.90)
            return combined
    payload = process_single_message(message, context, processing_ms, web_search_enabled=web_search_enabled, character_name=character_name)
    # Environment table, appended AFTER a normal answer, only when the
    # message actually describes an environment scenario (see
    # should_analyze). Explicit classify requests were already
    # answered with a table in step 5l, so they are not doubled here.
    # Analyzer failure is swallowed: the normal answer still stands.
    if "environment_analysis" not in payload:
        _skip = {"MATHEMATICS", "MATHEMATICS_STEP_BY_STEP", "PDF_GENERATION",
                 "FILE_GENERATION", "CODE_GENERATION", "CODE_DEBUG", "CODE_RUN",
                 "COMPARE", "DOCUMENT", "DOCUMENT_SUMMARY", "DOCUMENT_NOTES",
                 "DOCUMENT_ANALYSIS", "DOCUMENT_QA", "PREFERENCE", "CLARIFY"}
        if payload.get("intent") not in _skip:
            try:
                _env = analyze_environment(message)
            except Exception:
                _env = None
            if _env is not None:
                payload["response"] = payload["response"] + "\n\n" + format_environment_table(_env)
                payload["environment_analysis"] = _env
                # Keep the saved session history consistent with what
                # the user actually saw.
                try:
                    if context.conversation_history and context.conversation_history[-1].get("role") == "bot":
                        context.conversation_history[-1]["text"] = payload["response"]
                    context.last_bot_response = payload["response"]
                except Exception:
                    pass
    return payload
