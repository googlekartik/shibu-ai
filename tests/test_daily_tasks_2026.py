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
    assert "[Your name]" in email  # details not given stay placeholders


def _fake_llm(monkeypatch, draft):
    import services.daily_task_service as daily

    class FakeManager:
        mode = "api"
        model_id = "fake-model"
        effective_model = "fake-model"

        def generate(self, messages, max_new_tokens=256, temperature=0.7):
            # The system prompt must carry the task kind, so a plan
            # cannot silently become a generic answer.
            assert any("timetable" in m["content"] or "email" in m["content"]
                       or "numbered steps" in m["content"] for m in messages)
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
    assert out["rule_id"] == "daily_plan" and "template" in out["response"].lower()
    assert "gym" in out["response"].lower()
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
    out = process_message("How to make a cup of tea step by step", ConversationContext())
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


def test_greeting_and_maths_still_win_over_daily_detection():
    out = process_message("How are you?", ConversationContext())
    assert out["intent"] == "HOW_ARE_YOU"
    out = process_message("25% of 800", ConversationContext())
    assert "200" in out["response"]
