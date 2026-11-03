"""Gemini-only LLM integration (2026-11-03): Google Gemini is
the only AI provider. ModelManager has modes "gemini" (key set)
and "off" (no key) only — no other provider, no local mode, no
fallback leg. All HTTP is mocked; no keys leave the process.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from services.model_manager import ModelManager
from services.conversation_manager import build_messages, detect_language
from services.response_generator import validate_llm_answer, format_llm_answer, is_uncertain
from chatbot.context import ConversationContext
from chatbot.router import process_message


def test_model_manager_off_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    m = ModelManager()
    assert m.mode == "off"
    assert m.status()["how_to_enable"]
    assert "GEMINI_API_KEY" in m.status()["how_to_enable"]
    # The status must never contain a key value, and there must be
    # no "token" field at all.
    assert "token" not in {k.lower() for k in m.status().keys()}
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert text is None and err


def test_model_manager_gemini_mode_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    m = ModelManager()
    assert m.mode == "gemini"
    assert m.model_id == "gemini-2.5-flash"
    assert "fake-key" not in str(m.status())
    assert m.select()["mode"] == "gemini"


def test_hindi_hinglish_detection():
    assert detect_language("नमस्ते, आप कैसे हैं?") == "hindi"
    assert detect_language("photosynthesis kya hai, mujhe batao") == "hinglish"
    assert detect_language("What is photosynthesis?") == "english"


def test_conversation_trims_history_and_maps_bot_role():
    ctx = ConversationContext()
    for i in range(20):
        ctx.add_exchange(f"question {i}", f"answer {i}", "X", "x", 0.9)
    messages, lang = build_messages(ctx, "What about question 19?")
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "What about question 19?"
    roles = {m["role"] for m in messages}
    assert "bot" not in roles  # stored "bot" turns become "assistant"
    assert lang == "english"


def test_response_validation():
    ok, _, _ = validate_llm_answer("Hello there.")
    assert ok
    ok, _, _ = validate_llm_answer("   ")
    assert not ok
    ok, _, note = validate_llm_answer("Same answer.", previous_bot_answer="Same answer.")
    assert not ok and "repeated" in note
    assert is_uncertain("I don't know, to be honest.")
    out = format_llm_answer("Hello.", model_id="gemini-2.5-flash")
    assert "AI-generated answer" in out and "gemini" in out


def test_hindi_greeting_rule(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = process_message("Namaste", ConversationContext())
    assert out["intent"] == "HINDI_GREETING"


def test_router_falls_back_to_rules_when_llm_off(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Isolate the LLM-off path from the live web: stub search to []
    # so the web fallback cannot fire instead of the honest fallback.
    import chatbot.router as router
    monkeypatch.setattr(router.search_service, "search", lambda q: [])
    out = process_message("some totally unknown topic xyzabc", ConversationContext())
    assert out["intent"] == "UNKNOWN"  # no LLM, so the honest fallback remains


def test_router_uses_mocked_llm_for_unknown(monkeypatch):
    import chatbot.router as router

    class FakeManager:
        mode = "gemini"
        model_id = "fake-model"
        def generate(self, messages, max_new_tokens=256, temperature=0.7):
            return "A mocked conversational answer.", None

    monkeypatch.setattr(router, "ModelManager", lambda: FakeManager())
    out = process_message("some totally unknown topic xyzabc", ConversationContext())
    assert out["intent"] == "LLM_ANSWER"
    assert "AI-generated answer" in out["response"]


def test_router_explicit_ai_when_llm_off(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = process_message("ai: explain something unknown", ConversationContext())
    assert out["intent"] == "LLM_UNAVAILABLE"
    assert "GEMINI_API_KEY" in out["response"]


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
    def json(self):
        return self._payload


def test_gemini_mode_wins_when_key_set(monkeypatch):
    import requests as _requests
    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == ("https://generativelanguage.googleapis.com/v1beta"
                       "/models/gemini-2.5-flash:generateContent")
        assert headers["x-goog-api-key"] == "fake-gemini-key-for-test"
        assert "key=" not in url
        return _FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "Fast Gemini answer."}],
                                                          "role": "model"},
                                              "finishReason": "STOP"}]})
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-test")
    m = ModelManager()
    assert m.mode == "gemini"
    assert m.model_id == "gemini-2.5-flash"
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert err is None and text == "Fast Gemini answer."
    assert "fake-gemini-key" not in str(m.status())


def test_gemini_403_gives_key_hint(monkeypatch):
    import requests as _requests
    from services.gemini_service import GeminiService
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(403, {"error": {"message": "API key not valid."}})
    monkeypatch.setattr(_requests, "post", fake_post)
    text, err = GeminiService(api_key="fake-gemini-key-for-test").generate(
        [{"role": "user", "content": "hi"}])
    assert text is None and "403" in err and "GEMINI_API_KEY" in err


def test_gemini_empty_and_blocked_responses_are_failures(monkeypatch):
    import requests as _requests
    from services.gemini_service import GeminiService
    monkeypatch.setattr(_requests, "post",
                        lambda *a, **k: _FakeResp(200, {"candidates": []}))
    text, err = GeminiService(api_key="k").generate([{"role": "user", "content": "hi"}])
    assert text is None and err
    monkeypatch.setattr(_requests, "post", lambda *a, **k: _FakeResp(
        200, {"candidates": [{"content": {"parts": [{"text": "x"}]},
                              "finishReason": "SAFETY"}],
              "promptFeedback": {"blockReason": "SAFETY"}}))
    text, err = GeminiService(api_key="k").generate([{"role": "user", "content": "hi"}])
    assert text is None and "blocked" in err.lower()


def test_gemini_timeout_is_configurable_and_bounded(monkeypatch):
    import requests as _requests
    from services.gemini_service import GeminiService
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["timeout"] = timeout
        return _FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]},
                                               "finishReason": "STOP"}]})

    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "7")
    text, _ = GeminiService(api_key="k").generate([{"role": "user", "content": "hi"}])
    assert text == "ok" and seen["timeout"] == 7.0
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "9999")
    assert GeminiService(api_key="k").timeout == 120.0


def test_explicit_ai_when_gemini_fails_is_llm_error_not_unavailable(monkeypatch):
    import chatbot.router as router

    class FakeManager:
        mode = "gemini"
        model_id = "fake-model"
        def generate(self, messages, max_new_tokens=256, temperature=0.7):
            return None, "Gemini returned HTTP 401: bad key."
    monkeypatch.setattr(router, "ModelManager", lambda: FakeManager())
    out = process_message("ai: explain something", ConversationContext())
    assert out["intent"] == "LLM_ERROR"
    assert "401" in out["response"]
    assert not out["response"].startswith("No language model is enabled")


def test_llm_endpoints_off(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    import app as app_module
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    assert client.get("/api/llm/status").get_json()["mode"] == "off"
    r = client.post("/api/llm/chat", json={"message": "hello"})
    assert r.status_code == 503
    assert "how_to_enable" in r.get_json()
    health = client.get("/api/health").get_json()
    assert health["llm"]["mode"] == "off"
    assert health["gemini"] is False
    assert "fake-gemini-key" not in str(health)
