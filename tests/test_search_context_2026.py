"""SEARCH CONTEXT in LLM prompts (2026-11-03): web-grounded
drafting (how-to / news / current-info) must include the top
search snippets, capped, clearly labelled as background-only;
pure rule/math answers must never trigger a search or carry
snippets. RAG prompt construction is untouched.
"""
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-context-tests.db")

from chatbot.context import ConversationContext  # noqa: E402
from chatbot.router import process_message  # noqa: E402
from services.web_answer import build_search_context  # noqa: E402

FIXTURE_RESULTS = [
    {"title": "Brew tea", "url": "https://example.com/tea",
     "snippet": "Boil water, pour over leaves, steep three minutes.",
     "source": "Google"},
    {"title": "Tea guide", "url": "https://example.com/guide",
     "snippet": "Use fresh water and do not over-steep.",
     "source": "Google"},
    {"title": "Tea times", "url": "https://example.com/times",
     "snippet": "Black tea needs three to five minutes.",
     "source": "Google"},
    {"title": "Extra", "url": "https://example.com/extra",
     "snippet": "This fourth result must be dropped by the cap.",
     "source": "Google"},
]


def test_build_search_context_caps_results_and_chars():
    context = build_search_context(FIXTURE_RESULTS)
    assert "SEARCH CONTEXT" in context
    assert "Boil water" in context
    assert "must be dropped by the cap" not in context  # max 3 results
    assert len(context) <= 1200
    # Background-only wording: never verified fact, never the user.
    assert "background only" in context
    assert "not verified fact" in context


def test_build_search_context_empty_and_dedupe():
    assert build_search_context([]) == ""
    assert build_search_context(None) == ""
    dupes = [dict(FIXTURE_RESULTS[0]), dict(FIXTURE_RESULTS[0])]
    context = build_search_context(dupes)
    assert context.count("https://example.com/tea") == 1
    no_snippet = [{"title": "Empty", "url": "https://example.com/x", "snippet": ""}]
    assert build_search_context(no_snippet) == ""


def test_howto_llm_prompt_contains_snippets(monkeypatch):
    captured = {}

    def fake_generate(self, messages, max_new_tokens=256, temperature=0.7, task_kind=None):
        captured["messages"] = messages
        return "Here are the steps to make tea.", None

    class _FakeSearch:
        def search(self, query):
            return list(FIXTURE_RESULTS)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("services.model_manager.ModelManager.generate", fake_generate)
    monkeypatch.setattr("chatbot.router.search_service.search",
                        lambda q: list(FIXTURE_RESULTS))
    # _llm_draft builds its own WebSearchService; both search paths
    # must see the same fixture results.
    monkeypatch.setattr("services.web_search_service.WebSearchService", _FakeSearch)
    payload = process_message("how to make tea step by step",
                                 ConversationContext(), web_search_enabled=True)
    assert payload is not None
    system_texts = [m["content"] for m in captured["messages"] if m["role"] == "system"]
    joined = "\n".join(system_texts)
    assert "SEARCH CONTEXT" in joined
    assert "Boil water" in joined
    # Exactly one context block — no duplication.
    assert joined.count("SEARCH CONTEXT") == 1


def test_llm_answer_adds_search_context_directly(monkeypatch):
    from chatbot.router import _llm_answer

    captured = {}

    def fake_generate(self, messages, max_new_tokens=256, temperature=0.7):
        captured["messages"] = messages
        return "Here is the latest AI news summary.", None

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("services.model_manager.ModelManager.generate", fake_generate)
    monkeypatch.setattr("chatbot.router.search_service.search",
                        lambda q: list(FIXTURE_RESULTS))
    payload, err = _llm_answer("latest AI news today", ConversationContext())
    assert err is None and payload is not None
    system_texts = [m["content"] for m in captured["messages"] if m["role"] == "system"]
    joined = "\n".join(system_texts)
    assert "SEARCH CONTEXT" in joined
    assert "Boil water" in joined
    assert joined.count("SEARCH CONTEXT") == 1


def test_rule_answer_triggers_no_search(monkeypatch):
    def boom(q):
        raise AssertionError("search must not run for a rule answer")

    monkeypatch.setattr("chatbot.router.search_service.search", boom)
    payload = process_message("hello", ConversationContext(),
                                 web_search_enabled=True)
    assert payload["rule_id"] == "greeting"


def test_math_answer_triggers_no_search(monkeypatch):
    def boom(q):
        raise AssertionError("search must not run for a maths answer")

    monkeypatch.setattr("chatbot.router.search_service.search", boom)
    payload = process_message("Calculate 25% of 800",
                                 ConversationContext(), web_search_enabled=True)
    assert "200" in payload["response"]


def test_daily_task_howto_draft_contains_snippets(monkeypatch):
    from services.daily_task_service import _llm_draft

    captured = {}

    def fake_generate(self, messages, max_new_tokens=256, temperature=0.7, task_kind=None):
        captured["messages"] = messages
        return "Step 1: boil water.", None

    class _FakeSearch:
        def search(self, query):
            return list(FIXTURE_RESULTS)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("services.model_manager.ModelManager.generate", fake_generate)
    monkeypatch.setattr("services.web_search_service.WebSearchService", _FakeSearch)
    text, _err, _model = _llm_draft("howto", "How to make tea step by step",
                                    ConversationContext())
    assert text
    system_texts = [m["content"] for m in captured["messages"] if m["role"] == "system"]
    joined = "\n".join(system_texts)
    assert "SEARCH CONTEXT" in joined
    assert "Boil water" in joined


def test_daily_task_plan_draft_has_no_search_context(monkeypatch):
    from services.daily_task_service import _llm_draft

    captured = {}

    def fake_generate(self, messages, max_new_tokens=256, temperature=0.7, task_kind=None):
        captured["messages"] = messages
        return "Plan: work, gym, rest.", None

    class _BoomSearch:
        def search(self, query):
            raise AssertionError("plans must not fetch web results")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("services.model_manager.ModelManager.generate", fake_generate)
    monkeypatch.setattr("services.web_search_service.WebSearchService", _BoomSearch)
    text, _err, _model = _llm_draft("plan", "Plan my day",
                                    ConversationContext())
    assert text
    joined = "\n".join(m["content"] for m in captured["messages"])
    assert "SEARCH CONTEXT" not in joined
