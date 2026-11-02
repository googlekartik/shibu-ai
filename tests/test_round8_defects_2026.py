"""Round-8 regression tests (2026-11-02): structure-only usage_keys
diagnostic (key names only — never values) for provider replies,
preserved after the Meta provider integration was removed and the
OpenAI chat integration was replaced by Gemini: the diagnostic
machinery is provider-agnostic and now exercised through the
Gemini service, the first provider in the chain."""

import json

import pytest

from services.gemini_service import GeminiService


class _Resp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


def _max_tokens_body():
    """The Gemini analogue of the live shape: HTTP 200, empty parts,
    finishReason "MAX_TOKENS", usageMetadata with token counts that
    must never appear as values in the diagnostic."""
    return {"candidates": [{"content": {"parts": [], "role": "model"},
                            "finishReason": "MAX_TOKENS",
                            "index": 0,
                            "safetyRatings": []}],
            "usageMetadata": {"promptTokenCount": 12345,
                              "candidatesTokenCount": 678,
                              "totalTokenCount": 13023}}


def test_r8_gemini_max_tokens_diagnostic_records_usage_keys(monkeypatch):
    """finishReason "MAX_TOKENS" + empty parts: the diagnostic records
    the finishReason value, blocked_present False, and usage_keys
    with key NAMES only — the token counts (12345/678/13023) must
    never appear in the diagnostic."""
    import requests
    svc = GeminiService(api_key="fake-key", model="gemini-test")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(_max_tokens_body()))
    text, err = svc.generate([{"role": "user", "content": "Plan my day"}])
    assert text is None and err
    struct = svc.last_diagnostic["choice_structure"]
    assert struct["finish_reason"] == "MAX_TOKENS"
    assert struct["blocked_present"] is False
    assert struct["text_present"] is False
    assert svc.last_diagnostic["usage_keys"] == {
        "keys": ["candidatesTokenCount", "promptTokenCount", "totalTokenCount"],
        "nested": {}}
    dumped = json.dumps(svc.last_diagnostic)
    for value in ("12345", "678", "13023"):
        assert value not in dumped


def test_r8_gemini_text_extraction_unchanged(monkeypatch):
    """A normal text part still extracts as the answer, and a
    usageMetadata object does not disturb it."""
    import requests
    svc = GeminiService(api_key="fake-key", model="gemini-test")
    body = {"candidates": [{"content": {"parts": [{"text": "Hello there."}],
                                        "role": "model"},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2,
                              "totalTokenCount": 7}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Hello there." and err is None


def test_r8_gemini_request_carries_caller_budget_unchanged(monkeypatch):
    """The Gemini request carries the caller's budget verbatim in
    generationConfig.maxOutputTokens: the default 256, and an
    explicit small budget, are sent unchanged — no provider-specific
    floor or alias field is applied, and the key travels only in
    the header, never in the URL."""
    import requests
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured.update(json or {})
        return _Resp({"candidates": [{"content": {"parts": [{"text": "ok"}],
                                                "role": "model"},
                                    "finishReason": "STOP"}]})

    svc = GeminiService(api_key="fake-key", model="m")
    monkeypatch.setattr(requests, "post", _fake_post)
    svc.generate([{"role": "user", "content": "hi"}])
    assert captured["generationConfig"]["maxOutputTokens"] == 256
    assert set(captured) <= {"url", "headers", "contents", "generationConfig"}
    assert "key=" not in captured["url"]
    svc.generate([{"role": "user", "content": "hi"}], max_tokens=64)
    assert captured["generationConfig"]["maxOutputTokens"] == 64


def test_r8_manager_gemini_branch_passes_budget_and_records_diagnostic(monkeypatch, tmp_path):
    """A caller passing max_new_tokens=64 through ModelManager sends
    64 to the Gemini API; the diagnostic still flows through to
    status() with the usage_keys field."""
    import requests
    import services.model_manager as mm
    saved = dict(mm._LAST_PROVIDER_DIAGNOSTIC)
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Resp(_max_tokens_body())

    monkeypatch.setattr(requests, "post", _fake_post)
    try:
        manager = mm.ModelManager()
        assert manager.mode == "gemini"
        text, err = manager.generate([{"role": "user", "content": "Plan my day"}],
                                     max_new_tokens=64)
        assert text is None and err
        assert captured["generationConfig"]["maxOutputTokens"] == 64
        status = mm.ModelManager().status()
        error = status["provider_error"]
        assert error["provider"] == "gemini"
        assert error["choice_structure"]["finish_reason"] == "MAX_TOKENS"
        assert error["usage_keys"]["keys"] == [
            "candidatesTokenCount", "promptTokenCount", "totalTokenCount"]
    finally:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        mm._LAST_PROVIDER_DIAGNOSTIC.update(saved)


def test_r8_usage_keys_absent_when_no_usage_object(monkeypatch):
    """A body with no "usageMetadata" object yields usage_keys None
    — never an invented structure."""
    import requests
    svc = GeminiService(api_key="fake-key", model="m")
    body = {"candidates": [{"content": {"parts": [], "role": "model"},
                            "finishReason": "STOP"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert svc.last_diagnostic["usage_keys"] is None


def test_r8_nested_usage_keys_record_names_only(monkeypatch):
    """Nested usageMetadata objects (e.g. a thoughtsTokenCount
    sub-object) are recorded as nested key NAMES only — the
    counts themselves never appear, at any depth."""
    import requests
    svc = GeminiService(api_key="fake-key", model="m")
    body = {"candidates": [{"content": {"parts": [], "role": "model"},
                            "finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"totalTokenCount": 99999,
                              "thoughts": {"tokens": 88888}}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert svc.last_diagnostic["usage_keys"] == {
        "keys": ["thoughts", "totalTokenCount"],
        "nested": {"thoughts": ["tokens"]}}
    dumped = json.dumps(svc.last_diagnostic)
    assert "99999" not in dumped and "88888" not in dumped
