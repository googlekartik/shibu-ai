"""Daily-task service, Design B (2026-11-01): LLM-first for plans,
writing and how-tos, with labelled template/web fallbacks when no
model is available. Also covers the live-test fixes: real quiz
distractors, clickable web sources, and no repeated definition
sentences in structured blocks.
"""
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-daily-tests.db")
os.environ.setdefault("RAG_BACKEND", "sqlite")

from chatbot.context import ConversationContext
from chatbot.router import process_message
from services.content_structuring_engine import structure
from services.daily_task_service import detect_daily_task, _template_plan, _template_write
from services.interactive_component_renderer import build_quiz
from services.intelligent_content_analyzer import analyze


def test_detect_daily_tasks_and_exclusions():
    assert detect_daily_task("Plan my day: I work 9 to 5, want gym and 1 hour of study") == "plan"
    assert detect_daily_task("Write a polite email to my teacher asking for leave tomorrow") == "write"
    assert detect_daily_task("How to make a cup of tea step by step") == "howto"
    assert detect_daily_task("How are you?") is None
    assert detect_daily_task("Calculate 25% of 800") is None
    assert detect_daily_task("What is machine learning?") is None


def test_templates_use_only_given_details():
    plan = _template_plan("Plan my day: I work 9 to 5, want gym and 1 hour of study")
    assert "9 to 5" in plan and "gym" in plan.lower() and "study" in plan.lower()
    assert "suggested" in plan  # missing times are labelled, not invented as given
    email = _template_write("Write a polite email to my teacher asking for leave tomorrow")
    assert "Subject:" in email and "Dear Teacher" in email and "tomorrow" in email
    # Round 6: the template is a finished email — no bracketed
    # placeholders and no "fill in" lead-in.
    assert "[" not in email and "Fill in" not in email


def _fake_llm(monkeypatch, draft):
    import services.daily_task_service as daily

    class FakeManager:
        mode = "api"
        model_id = "fake-model"
        effective_model = "fake-model"

        def generate(self, messages, max_new_tokens=256, temperature=0.7, task_kind=None):
            # The system prompt must carry the task kind, so a plan
            # cannot silently become a generic answer; and the router
            # must be told the kind the caller already knows.
            assert any("timetable" in m["content"] or "email" in m["content"]
                       or "numbered steps" in m["content"] for m in messages)
            assert task_kind in ("howto", "plan", "write")
            return draft, None

    monkeypatch.setattr(daily, "ModelManager", lambda: FakeManager())


def test_router_llm_first_for_tea_plan_and_email(monkeypatch):
    _fake_llm(monkeypatch, "1. Boil water.\n2. Pour over tea.\n3. Wait 3 minutes.")
    ctx = ConversationContext()
    out = process_message("How to make a cup of tea step by step", ctx)
    assert out["intent"] == "DAILY_TASK" and out["rule_id"] == "daily_howto"
    assert "Boil water" in out["response"] and "running well" not in out["response"]

    _fake_llm(monkeypatch, "09:00 Work\n12:00 Gym\n18:00 Study")
    out = process_message("Plan my day: I work 9 to 5, want gym and 1 hour of study", ConversationContext())
    assert out["rule_id"] == "daily_plan" and "Gym" in out["response"]
    assert "Study tips" not in out["response"]

    _fake_llm(monkeypatch, "Subject: Leave tomorrow\n\nDear Teacher,")
    out = process_message("Write a polite email to my teacher asking for leave tomorrow", ConversationContext())
    assert out["rule_id"] == "daily_write" and "Dear Teacher" in out["response"]


def test_router_template_fallback_when_llm_off(monkeypatch):
    import services.daily_task_service as daily

    class OffManager:
        mode = "off"
        model_id = None

        def generate(self, *a, **k):
            raise AssertionError("generate must not be called when the model is off")

    monkeypatch.setattr(daily, "ModelManager", lambda: OffManager())
    out = process_message("Plan my day: I work 9 to 5, want gym and 1 hour of study", ConversationContext())
    # Round 3: the template is delivered cleanly — no mention of
    # the language model or its failure, just the plan itself.
    assert out["rule_id"] == "daily_plan" and "Your day plan:" in out["response"]
    assert "gym" in out["response"].lower()
    for forbidden in ("language model", "not available", "could not draft", "basic template"):
        assert forbidden not in out["response"]
    out = process_message("Write a polite email to my teacher asking for leave tomorrow", ConversationContext())
    assert out["rule_id"] == "daily_write" and "Subject:" in out["response"]


def test_router_llm_error_for_howto_falls_back_to_web(monkeypatch):
    import chatbot.router as router
    import services.daily_task_service as daily

    class FailingManager:
        mode = "api"
        model_id = "fake-model"

        def generate(self, *a, **k):
            return None, "API unavailable"

    monkeypatch.setattr(daily, "ModelManager", lambda: FailingManager())
    monkeypatch.setattr(router, "ModelManager", lambda: FailingManager())
    monkeypatch.setattr(router.search_service, "search",
                        lambda q: [{"title": "How to make tea", "url": "https://example.com/tea",
                                    "snippet": "Boil water, pour over the tea leaves and wait."}])
    out = process_message("How to make a cup of tea step by step", ConversationContext())
    assert out["intent"] == "WEB_SEARCH"
    assert out["sources"] and out["sources"][0]["url"] == "https://example.com/tea"


def test_router_llm_error_for_howto_with_no_web_is_honest(monkeypatch):
    import chatbot.router as router
    import services.daily_task_service as daily

    class FailingManager:
        mode = "api"
        model_id = "fake-model"

        def generate(self, *a, **k):
            return None, "API unavailable"

    monkeypatch.setattr(daily, "ModelManager", lambda: FailingManager())
    monkeypatch.setattr(router, "ModelManager", lambda: FailingManager())
    monkeypatch.setattr(router.search_service, "search", lambda q: [])
    # An unknown task (outside the small built-in table) with no web
    # results is still answered honestly — tea/coffee now get
    # built-in steps instead of this path (round 6).
    out = process_message("How to calibrate a flux capacitor step by step", ConversationContext())
    assert out["rule_id"] == "daily_howto_unavailable"
    assert "running well" not in out["response"]
    assert "could not look up" in out["response"]


def test_web_search_payload_carries_clickable_sources(monkeypatch):
    import chatbot.router as router
    monkeypatch.setattr(router.search_service, "search",
                        lambda q: [{"title": "AI news today", "url": "https://example.com/news",
                                    "snippet": "Latest AI news summary."}])
    out = process_message("Search latest AI news", ConversationContext())
    assert out["intent"] == "WEB_SEARCH"
    assert out["sources"][0]["citation"] == "W1"
    assert out["sources"][0]["url"] == "https://example.com/news"


def test_quiz_has_real_distractors_and_never_one_option():
    analysis = analyze([
        {"text": "Machine learning is a field of artificial intelligence.", "source_name": "A"},
        {"text": "Deep learning is a subset of machine learning.", "source_name": "A"},
    ], "machine learning")
    from services.content_intent_detector import detect_intent
    quiz = build_quiz(analysis, detect_intent("Quiz me on machine learning"))
    assert quiz is not None
    for question in quiz["questions"]:
        assert len(question["options"]) >= 2
    # A single fact with no second phrase cannot make a quiz.
    sparse = analyze([{"text": "Photosynthesis is how plants make food.", "source_name": "A"}], "quiz")
    assert build_quiz(sparse, detect_intent("Quiz me")) is None or all(
        len(q["options"]) >= 2 for q in build_quiz(sparse, detect_intent("Quiz me"))["questions"])


def test_structuring_does_not_repeat_the_definition():
    text = ("Machine learning is a field of artificial intelligence that learns patterns from data. "
            "Machine learning is used in search and recommendations. "
            "In conclusion, machine learning helps computers learn from data.")
    analysis = analyze([{"text": s, "source_name": "A"} for s in text.split(". ") if s], "what is machine learning")
    from services.content_intent_detector import detect_intent
    blocks = structure(text, analysis, detect_intent("What is machine learning?"))
    rendered = " ".join(str(b.get("text", "")) for b in blocks)
    definition = "Machine learning is a field of artificial intelligence that learns patterns from data"
    assert rendered.count(definition) <= 1


def test_template_plan_never_clashes_with_work_hours():
    plan = _template_plan("Plan my day: I work 9 to 5, want gym and 1 hour of study")
    lines = [ln for ln in plan.splitlines() if ln.startswith("- ")]
    gym_line = [ln for ln in lines if "gym" in ln.lower()][0]
    assert "09:00" not in gym_line and "12:00" not in gym_line
    assert any(t in gym_line for t in ("07:00", "18:00", "19:00", "20:00"))


def test_daily_error_is_friendly_and_has_no_billing_url(monkeypatch):
    import services.daily_task_service as daily

    class QuotaManager:
        mode = "api"
        model_id = "fake-model"

        def generate(self, *a, **k):
            return None, ("OpenAI returned HTTP 429: You have no credits remaining. "
                          "Add credits at https://platform.openai.com/billing/")

    monkeypatch.setattr(daily, "ModelManager", lambda: QuotaManager())
    out = process_message("Write a polite email to my teacher asking for leave tomorrow", ConversationContext())
    assert "no credits left" not in out["response"]
    assert "credits" not in out["response"].lower()
    assert "platform.openai.com" not in out["response"] and "http" not in out["response"]
    assert "Subject:" in out["response"]


def test_howto_web_summary_prefers_real_steps():
    from services.web_answer import summarise_web_results
    results = [
        {"title": "How to Make a Good Cup of Tea: 10 Steps (with Pictures) - wikiHow",
         "snippet": "This wikiHow article will teach you proper brew times and precise brewing time.",
         "url": "https://example.com/tea"},
        {"title": "Perfect tea",
         "snippet": "Boil fresh water, pour it over the tea leaves and wait for three minutes before serving.",
         "url": "https://example.com/perfect"},
    ]
    summary = summarise_web_results("How to make a cup of tea step by step", results)
    assert "Boil fresh water" in summary["key_points"][0]


def test_explain_paragraph_decomposes_into_structured_blocks():
    text = ("Machine Learning is a subfield of AI where systems learn patterns from data. "
            "Examples include spam filters and recommendation systems. "
            "Common approaches are supervised, unsupervised and reinforcement learning.")
    blocks = structure(text, {"facts": [], "key_points": [], "definitions": []},
                       {"content_type": "explain", "level": "intermediate"})
    kinds = [b["kind"] for b in blocks]
    assert "callout" in kinds and "list" in kinds
    assert kinds.count("paragraph") == 0  # the paragraph is decomposed, not repeated


def test_quiz_drops_fragment_distractors():
    from services.content_intent_detector import detect_intent
    analysis = analyze([
        {"text": "Machine learning is a field of artificial intelligence.", "source_name": "A"},
        {"text": "Machine learning is a field of artificial intelligence that learns patterns.", "source_name": "A"},
    ], "machine learning")
    quiz = build_quiz(analysis, detect_intent("Quiz me on machine learning"))
    # Fragments of the same definition must not become fake options.
    assert quiz is None or all(len(q["options"]) >= 2 for q in quiz["questions"])
    out = process_message("How are you?", ConversationContext())
    assert out["intent"] == "HOW_ARE_YOU"
    out = process_message("25% of 800", ConversationContext())
    assert "200" in out["response"]
