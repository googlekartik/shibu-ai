import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from services.model_manager import ModelManager, recommend_model, is_cached
from services.conversation_manager import build_messages, detect_language
from services.response_generator import validate_llm_answer, format_llm_answer, is_uncertain
from chatbot.context import ConversationContext
from chatbot.router import process_message


def test_recommend_model_by_ram():
    assert recommend_model(ram_gb=4)[0] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert recommend_model(ram_gb=8)[0] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert recommend_model(ram_gb=12)[0] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert recommend_model(ram_gb=32, gpu=False)[0] == "Qwen/Qwen2.5-3B-Instruct"
    assert recommend_model(ram_gb=32, gpu=True)[0] == "Qwen/Qwen2.5-7B-Instruct"
    # Unknown RAM (here: whatever this machine reports) must never pick
    # the 7B model unless a GPU is also present.
    assert recommend_model(ram_gb=None, gpu=False)[0] != "Qwen/Qwen2.5-7B-Instruct"


def test_model_manager_off_without_token_or_local(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)
    m = ModelManager()
    assert m.mode == "off"
    assert m.status()["how_to_enable"]
    # The help text may name the variable HF_TOKEN, but it must never
    # contain a token value, and there must be no "token" field at all.
    assert "token" not in {k.lower() for k in m.status().keys()}
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert text is None and err


def test_model_manager_api_mode_reuses_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    m = ModelManager()
    assert m.mode == "api"
    assert m.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert "fake-token" not in str(m.status())


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
    out = format_llm_answer("Hello.", model_id="Qwen/Qwen2.5-0.5B-Instruct")
    assert "AI-generated answer" in out and "Qwen" in out


def test_hindi_greeting_rule(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    out = process_message("Namaste", ConversationContext())
    assert out["intent"] == "HINDI_GREETING"


def test_router_falls_back_to_rules_when_llm_off(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    # Isolate the LLM-off path from the live web: "xyzabc" is now a
    # real TikTok hashtag, so DuckDuckGo returns results and the
    # (intentional) web fallback would otherwise fire instead.
    import chatbot.router as router
    monkeypatch.setattr(router.search_service, "search", lambda q: [])
    out = process_message("some totally unknown topic xyzabc", ConversationContext())
    assert out["intent"] == "UNKNOWN"  # no LLM, so the honest fallback remains


def test_router_uses_mocked_llm_for_unknown(monkeypatch):
    import chatbot.router as router

    class FakeManager:
        mode = "api"
        model_id = "fake-model"
        def generate(self, messages, max_new_tokens=256, temperature=0.7):
            return "A mocked conversational answer.", None

    monkeypatch.setattr(router, "ModelManager", lambda: FakeManager())
    out = process_message("some totally unknown topic xyzabc", ConversationContext())
    assert out["intent"] == "LLM_ANSWER"
    assert "AI-generated answer" in out["response"]


def test_router_explicit_ai_when_llm_off(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    out = process_message("ai: explain something unknown", ConversationContext())
    assert out["intent"] == "LLM_UNAVAILABLE"
    assert "HF_TOKEN" in out["response"]


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
    def json(self):
        return self._payload


def test_openai_mode_wins_when_key_set(monkeypatch):
    import requests as _requests
    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == "https://api.openai.com/v1/chat/completions"
        assert json["model"] == "gpt-4o-mini"
        return _FakeResp(200, {"choices": [{"message": {"content": "Fast ChatGPT answer."}}]})
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-for-test")
    monkeypatch.setenv("HF_TOKEN", "fake-hf-token-for-test")
    m = ModelManager()
    assert m.mode == "openai"
    assert m.model_id == "gpt-4o-mini"
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert err is None and text == "Fast ChatGPT answer."
    assert "fake-openai-key" not in str(m.status())


def test_openai_401_gives_key_hint(monkeypatch):
    import requests as _requests
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(401, {"error": {"message": "Incorrect API key provided."}})
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-for-test")
    m = ModelManager()
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert text is None and "401" in err and "OPENAI_API_KEY" in err


def test_api_uses_router_and_parses_chat_completions(monkeypatch):
    import requests as _requests
    import services.model_manager as mm

    seen = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        return _FakeResp(200, {"choices": [{"message": {"content": "Hello from the router."}}]})
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    m = ModelManager()
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert err is None and text == "Hello from the router."
    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"


def test_api_401_gives_permission_hint_not_generic_failure(monkeypatch):
    import requests as _requests
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(401, {"error": "Invalid username or password."})
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    m = ModelManager()
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "401" in err and "Inference Providers" in err


def test_api_retries_with_auto_provider_after_400(monkeypatch):
    import requests as _requests
    calls = []
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "Qwen/Qwen2.5-3B-Instruct":
            return _FakeResp(400, {"error": "The requested model 'Qwen/Qwen2.5-3B-Instruct' is not supported by any provider you have enabled."})
        return _FakeResp(200, {"choices": [{"message": {"content": "Answered via auto provider."}}]})
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    monkeypatch.delenv("HF_PROVIDER", raising=False)
    m = ModelManager()
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert err is None and text == "Answered via auto provider."
    assert calls[0] == "Qwen/Qwen2.5-3B-Instruct"
    assert calls[1] == "Qwen/Qwen2.5-3B-Instruct:auto"
    assert m.effective_model == "Qwen/Qwen2.5-3B-Instruct:auto"


def test_api_400_after_all_providers_mentions_settings(monkeypatch):
    import requests as _requests
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp(400, {"error": "The requested model 'Qwen/Qwen2.5-3B-Instruct' is not supported by any provider you have enabled."})
    monkeypatch.setattr(_requests, "post", fake_post)
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    monkeypatch.delenv("HF_PROVIDER", raising=False)
    m = ModelManager()
    text, err = m.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "inference-providers" in err and "featherless-ai" in err


def test_explicit_ai_when_api_fails_is_llm_error_not_unavailable(monkeypatch):
    import chatbot.router as router

    class FakeManager:
        mode = "api"
        model_id = "fake-model"
        def generate(self, messages, max_new_tokens=256, temperature=0.7):
            return None, "Hugging Face API returned HTTP 401: bad token."
    monkeypatch.setattr(router, "ModelManager", lambda: FakeManager())
    out = process_message("ai: explain something", ConversationContext())
    assert out["intent"] == "LLM_ERROR"
    assert "401" in out["response"]
    assert not out["response"].startswith("No language model is enabled")


def test_llm_endpoints_off(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_USE_LOCAL", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)
    import app as app_module
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    assert client.get("/api/llm/status").get_json()["mode"] == "off"
    r = client.post("/api/llm/chat", json={"message": "hello"})
    assert r.status_code == 503
    assert "how_to_enable" in r.get_json()
    health = client.get("/api/health").get_json()
    assert health["llm"]["mode"] == "off"
