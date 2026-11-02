"""Google Gemini chat service tests (2026-11-03): REST generateContent
calls, message mapping, blocked/empty/error handling, structure-only
diagnostics, and no key leakage. No network, no real keys."""
import json
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-gemini-tests.db")

from services.gemini_service import GeminiService, build_request_body


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _gemini_body(text="Hello there.", finish_reason="STOP"):
    return {"candidates": [{"content": {"parts": [{"text": text}],
                                        "role": "model"},
                            "finishReason": finish_reason,
                            "index": 0,
                            "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT",
                                               "probability": "NEGLIGIBLE"}]}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3,
                              "totalTokenCount": 8}}


def test_inactive_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    svc = GeminiService()
    assert svc.enabled is False
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "GEMINI_API_KEY" in err


def test_request_uses_header_key_and_model_url(monkeypatch):
    import requests
    captured = {}

    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _Resp(_gemini_body())

    monkeypatch.setattr(requests, "post", _post)
    svc = GeminiService(api_key="test-key-123", model="gemini-2.5-flash")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Hello there." and err is None
    assert captured["url"] == ("https://generativelanguage.googleapis.com/v1beta"
                               "/models/gemini-2.5-flash:generateContent")
    # Key in the header only — never in the URL.
    assert captured["headers"]["x-goog-api-key"] == "test-key-123"
    assert "key=" not in captured["url"] and "test-key-123" not in captured["url"]
    assert captured["json"]["generationConfig"]["maxOutputTokens"] == 256


def test_model_override_and_base_override_honoured(monkeypatch):
    import requests
    captured = {}

    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _Resp(_gemini_body())

    monkeypatch.setattr(requests, "post", _post)
    svc = GeminiService(api_key="k", model="gemini-2.0-flash",
                        api_base="https://example.test/v1")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Hello there."
    assert captured["url"] == "https://example.test/v1/models/gemini-2.0-flash:generateContent"


def test_system_instruction_and_role_mapping():
    body = build_request_body([
        {"role": "system", "content": "You are Shibu AI."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "And then?"},
    ])
    assert body["systemInstruction"] == {"parts": [{"text": "You are Shibu AI."}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][1]["parts"] == [{"text": "Hi there"}]
    # Consecutive same-role turns merge; empty turns drop.
    body2 = build_request_body([
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "answer"},
    ])
    assert len(body2["contents"]) == 2
    assert body2["contents"][0]["parts"] == [{"text": "first\nsecond"}]


def test_text_joins_across_parts(monkeypatch):
    import requests
    body = {"candidates": [{"content": {"parts": [{"text": "Hello "},
                                                  {"text": "there."}],
                                        "role": "model"},
                            "finishReason": "STOP"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    svc = GeminiService(api_key="k")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Hello\nthere."
    assert err is None


def test_blocked_safety_response_is_clean_error(monkeypatch):
    import requests
    body = {"candidates": [{"content": {"parts": [{"text": "SECRET BLOCKED TEXT"}],
                                        "role": "model"},
                            "finishReason": "SAFETY",
                            "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH",
                                               "probability": "HIGH", "blocked": True}]}],
            "promptFeedback": {"blockReason": "SAFETY"}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    svc = GeminiService(api_key="k")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None and err == "Gemini blocked the response."
    struct = svc.last_diagnostic["choice_structure"]
    assert struct["finish_reason"] == "SAFETY"
    assert struct["blocked_present"] is True
    assert "SECRET BLOCKED TEXT" not in str(svc.last_diagnostic)


def test_empty_candidates_and_missing_candidates_fail_cleanly(monkeypatch):
    import requests
    svc = GeminiService(api_key="k")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp({"candidates": []}))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert err == "Gemini returned a response in an unexpected format."
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp({}))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert err == "Gemini returned a response in an unexpected format."


def test_non_200_errors_are_clean_with_actionable_hints(monkeypatch):
    import requests
    svc = GeminiService(api_key="k")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp({"error": {"message": "API key not valid."}}, 403))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "403" in err and "GEMINI_API_KEY" in err
    assert "aistudio.google.com" in err
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp({"error": {"message": "model not found"}}, 404))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "404" in err and "GEMINI_MODEL" in err
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp({"error": {"message": "quota"}}, 429))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None and "429" in err


def test_diagnostic_is_structure_only(monkeypatch):
    import requests
    body = _gemini_body()
    body["candidates"][0]["content"]["parts"] = [{"text": {"unexpected": "shape"}}]
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    svc = GeminiService(api_key="k")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert err == "Gemini returned a response in an unexpected format."
    diag = svc.last_diagnostic
    assert diag["status"] == 200
    assert diag["response_keys"] == ["candidates", "usageMetadata"]
    assert diag["choice_structure"]["content_type"] == "list"
    assert diag["choice_structure"]["finish_reason"] == "STOP"
    assert diag["choice_structure"]["blocked_present"] is False
    assert diag["usage_keys"] == {"keys": ["candidatesTokenCount", "promptTokenCount",
                                           "totalTokenCount"], "nested": {}}
    dumped = json.dumps(diag)
    assert "shape" not in dumped
    # Usage key names appear; their VALUES (5/3/8) never do.
    assert "promptTokenCount" in dumped
    for value in (": 5", ": 3", ": 8"):
        assert value not in dumped


def test_diagnostic_absent_usage_yields_none(monkeypatch):
    import requests
    body = {"candidates": [{"content": {"parts": [{"text": "hi"}],
                                        "role": "model"}, "finishReason": "STOP"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    svc = GeminiService(api_key="k")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "hi"
    assert svc.last_diagnostic["usage_keys"] is None


def test_key_value_never_echoed_in_errors(monkeypatch):
    import requests

    class _Resp500:
        status_code = 500
        text = "boom"

        def json(self):
            raise ValueError("no json")

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp500())
    svc = GeminiService(api_key="super-secret-key-value", model="m")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "super-secret-key-value" not in err
    assert "500" in err
