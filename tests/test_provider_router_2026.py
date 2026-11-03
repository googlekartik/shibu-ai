"""Gemini-only provider router (policy version 2, 2026-11-03):
classify() always returns provider "gemini" — the task-kind /
length heuristic is diagnostics only (reason + task_kind). There
is no second provider, no fallback leg, and at most ONE Gemini
call per request. When Gemini fails, ModelManager returns the
error and callers use the deterministic fallback.
"""
import json
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-router-tests.db")

from services.model_manager import ModelManager, _LAST_ROUTER_DECISION
from services.provider_router import classify, router_preference


def _msg(text):
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------- classify
def test_kind_hints_all_map_to_gemini():
    for kind in ("plan", "write", "email", "essay", "letter", "story",
                 "quiz", "code", "summary", "rewrite", "howto", "explain"):
        result = classify(kind, _msg("hi"))
        assert result["provider"] == "gemini"
        assert result["task_kind"] == kind


def test_kind_hint_reasons_are_stable():
    assert classify("plan", _msg("hi"))["reason"] == "task_kind:plan"
    assert classify("quiz", _msg("hi"))["reason"] == "task_kind:quiz"
    assert classify("plan", _msg("hi"))["task_kind"] == "plan"


def test_explain_reasons_record_length_diagnostics():
    short = classify(None, _msg("Explain this."))
    assert short["provider"] == "gemini" and short["reason"] == "short_prompt"
    long_text = "Explain this in depth. " + ("Please cover every detail carefully. " * 40)
    long_result = classify(None, _msg(long_text))
    assert long_result["provider"] == "gemini" and long_result["reason"] == "long_prompt"


def test_code_and_structured_signals_still_go_to_gemini_with_reasons():
    assert classify(None, _msg("Write Python code for factorial"))["reason"] == "keyword_code"
    assert classify(None, _msg("```python\nprint(1)\n```"))["reason"] == "code_block"
    assert classify(None, _msg("Debug this function, it throws an error"))["reason"] == "keyword_code"
    assert classify(None, _msg("Summarise the uploaded document"))["reason"] == "keyword_structured"
    assert classify(None, _msg("Give me a quiz on that comparison"))["reason"] == "keyword_structured"
    assert classify(None, _msg("make it shorter"))["reason"] == "keyword_structured"
    for text in ("Write Python code for factorial", "Summarise the uploaded document"):
        assert classify(None, _msg(text))["provider"] == "gemini"


def test_drafting_and_multipart_signals_go_to_gemini():
    assert classify(None, _msg("Write an email to my teacher asking for leave"))["provider"] == "gemini"
    assert classify(None, _msg("Plan my day: I work 9 to 5, want gym and study time"))["provider"] == "gemini"
    assert classify(None, _msg("Explain black holes and give me 3 facts and ask me a question about black holes? "
                                   "Also tell me why they matter and how they form?"))["provider"] == "gemini"


def test_length_reasons_short_and_long():
    assert classify(None, _msg("What is photosynthesis?"))["reason"] == "short_prompt"
    assert classify(None, _msg("Write a long detailed account. " * 40))["reason"] == "long_prompt"


def test_classify_is_deterministic_and_has_no_prompt_text():
    first = classify("plan", _msg("Plan my secret day"))
    second = classify("plan", _msg("Plan my secret day"))
    assert first == second
    assert "secret" not in json.dumps(first)


def test_router_preference_override_and_invalid_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("ROUTER_PREFERENCE", "removed-provider")
    # A removed provider's name is ignored, never routed to.
    assert router_preference() == "auto"
    assert classify("plan", _msg("hi"))["provider"] == "gemini"
    monkeypatch.setenv("ROUTER_PREFERENCE", "gemini")
    assert router_preference() == "gemini"
    assert classify("quiz", _msg("hi"))["provider"] == "gemini"
    assert classify("quiz", _msg("hi"))["reason"] == "forced_preference"
    monkeypatch.setenv("ROUTER_PREFERENCE", "bogus")
    assert router_preference() == "auto"
    assert classify("quiz", _msg("hi"))["provider"] == "gemini"
    monkeypatch.delenv("ROUTER_PREFERENCE", raising=False)
    assert router_preference() == "auto"


# ------------------------------------------------------- routing behaviour
class _CountingGemini:
    calls = 0

    def __init__(self, api_key=None, model=None):
        pass

    def generate(self, messages, max_tokens=256, temperature=0.7):
        type(self).calls += 1
        return "Gemini draft.", None


class _FailingGemini:
    calls = 0

    def __init__(self, api_key=None, model=None):
        pass

    def generate(self, messages, max_tokens=256, temperature=0.7):
        type(self).calls += 1
        return None, "Gemini returned HTTP 429: quota exhausted."


def _gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    monkeypatch.delenv("ROUTER_PREFERENCE", raising=False)
    _CountingGemini.calls = 0
    _FailingGemini.calls = 0
    _LAST_ROUTER_DECISION.clear()


def test_plan_goes_to_gemini_exactly_once_on_success(monkeypatch):
    import services.gemini_service as gemini_service
    _gemini_key(monkeypatch)
    monkeypatch.setattr(gemini_service, "GeminiService", _CountingGemini)
    text, err = ModelManager().generate(_msg("Plan my day"), task_kind="plan")
    assert text == "Gemini draft." and err is None
    assert _CountingGemini.calls == 1


def test_code_goes_to_gemini_exactly_once_on_success(monkeypatch):
    import services.gemini_service as gemini_service
    _gemini_key(monkeypatch)
    monkeypatch.setattr(gemini_service, "GeminiService", _CountingGemini)
    text, err = ModelManager().generate(_msg("Write Python code for factorial"), task_kind="code")
    assert text == "Gemini draft." and err is None
    assert _CountingGemini.calls == 1


def test_gemini_failure_makes_exactly_one_call_and_returns_error(monkeypatch):
    import services.gemini_service as gemini_service
    _gemini_key(monkeypatch)
    monkeypatch.setattr(gemini_service, "GeminiService", _FailingGemini)
    text, err = ModelManager().generate(_msg("Plan my day"), task_kind="plan")
    assert text is None and err is not None
    assert _FailingGemini.calls == 1
    # No second provider exists: the error is the Gemini error, and
    # the caller (not the manager) chooses the deterministic fallback.
    assert "429" in err


def test_no_gemini_key_means_off_with_no_call(monkeypatch):
    import services.gemini_service as gemini_service
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_service, "GeminiService", _CountingGemini)
    _CountingGemini.calls = 0
    manager = ModelManager()
    assert manager.mode == "off"
    text, err = manager.generate(_msg("Plan my day"), task_kind="plan")
    assert text is None and err
    assert _CountingGemini.calls == 0


def test_status_exposes_router_block_without_prompt_text(monkeypatch):
    import services.gemini_service as gemini_service
    _gemini_key(monkeypatch)
    monkeypatch.setattr(gemini_service, "GeminiService", _CountingGemini)
    manager = ModelManager()
    manager.generate(_msg("Plan my secret day"), task_kind="plan")
    status = manager.status()
    router = status["router"]
    assert router["policy_version"] == 2
    assert router["preference"] == "auto"
    assert router["last_decision"]["task_kind"] == "plan"
    assert router["last_decision"]["provider"] == "gemini"
    assert router["last_decision"]["fallback_used"] is False
    assert "secret" not in json.dumps(status)


def test_friendly_error_never_mentions_provider_detail():
    from services.daily_task_service import _friendly_error
    assert _friendly_error("Gemini returned HTTP 429: quota exhausted, see billing") == \
        "the language model could not draft this"
