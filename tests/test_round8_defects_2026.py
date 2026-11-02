"""Round-8 regression tests (2026-11-02): Meta token budget raised to
1024 after the live diagnostic showed finish_reason "length" with
null visible content at 256, plus the structure-only usage_keys
diagnostic (key names only — never values). Round 9 superseded the
1024 floor with 8192 plus reasoning_effort; the two request-body
tests below were updated to the new intended values."""

import json

import pytest

from services.meta_service import MetaService


class _Resp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


def _length_body():
    """The exact live shape: HTTP 200, content null, no refusal,
    finish_reason "length", usage with a nested details dict."""
    return {"choices": [{"finish_reason": "length", "index": 0,
                         "logprobs": None,
                         "message": {"content": None, "refusal": None,
                                     "role": "assistant"}}],
            "created": 1, "id": "x", "model": "muse-spark-1.3",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 12345, "completion_tokens": 678,
                      "total_tokens": 13023,
                      "completion_tokens_details": {
                          "reasoning_tokens": 678,
                          "accepted_prediction_tokens": 0}}}


def test_r8_meta_length_diagnostic_records_usage_keys(monkeypatch):
    """finish_reason "length" + null content: the diagnostic records
    the finish_reason value, refusal_present False, and usage_keys
    with key NAMES only — the token counts (12345/678/13023) must
    never appear in the diagnostic."""
    import requests
    svc = MetaService(api_key="fake-key", model="muse-spark-1.3")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(_length_body()))
    text, err = svc.generate([{"role": "user", "content": "Plan my day"}])
    assert text is None and err
    struct = svc.last_diagnostic["choice_structure"]
    assert struct["finish_reason"] == "length"
    assert struct["refusal_present"] is False
    assert struct["logprobs_present"] is False
    assert svc.last_diagnostic["usage_keys"] == {
        "keys": ["completion_tokens", "completion_tokens_details",
                 "prompt_tokens", "total_tokens"],
        "nested": {"completion_tokens_details": [
            "accepted_prediction_tokens", "reasoning_tokens"]}}
    dumped = json.dumps(svc.last_diagnostic)
    for value in ("12345", "678", "13023"):
        assert value not in dumped


def test_r8_meta_text_extraction_unchanged(monkeypatch):
    """A normal string content still extracts as the answer, and a
    usage object does not disturb it."""
    import requests
    svc = MetaService(api_key="fake-key", model="muse-spark-1.3")
    body = {"choices": [{"finish_reason": "stop", "index": 0,
                         "message": {"content": "Hello there.", "role": "assistant"}}],
            "created": 1, "id": "x", "model": "muse-spark-1.3",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Hello there." and err is None


def test_r8_meta_request_carries_8192_tokens(monkeypatch):
    """The default call (no explicit budget) sends max_tokens 8192,
    and an explicit small budget is floored to 8192 at the Meta
    boundary — no Meta call may be starved below the floor again.
    (Round 9 raised the floor from 1024 to 8192; the request now
    also carries max_completion_tokens and reasoning_effort.)"""
    import requests
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Resp({"choices": [{"finish_reason": "stop", "index": 0,
                                   "message": {"content": "ok", "role": "assistant"}}],
                      "created": 1, "id": "x", "model": "m",
                      "object": "chat.completion", "usage": {}})

    svc = MetaService(api_key="fake-key", model="m")
    monkeypatch.setattr(requests, "post", _fake_post)
    monkeypatch.delenv("META_REASONING_EFFORT", raising=False)
    svc.generate([{"role": "user", "content": "hi"}])
    assert captured["max_tokens"] == 8192
    assert captured["max_completion_tokens"] == 8192
    assert captured["reasoning_effort"] == "low"
    svc.generate([{"role": "user", "content": "hi"}], max_tokens=256)
    assert captured["max_tokens"] == 8192
    assert captured["max_completion_tokens"] == 8192


def test_r8_manager_meta_branch_floors_small_budget(monkeypatch, tmp_path):
    """A caller passing max_new_tokens=64 through ModelManager still
    sends 8192 to the Meta API; the diagnostic still flows through
    to status() with the new usage_keys field."""
    import requests
    import services.model_manager as mm
    saved = dict(mm._LAST_PROVIDER_DIAGNOSTIC)
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setenv("META_API_KEY", "fake-key")
    monkeypatch.setenv("META_MODEL", "muse-spark-1.3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Resp(_length_body())

    monkeypatch.setattr(requests, "post", _fake_post)
    try:
        manager = mm.ModelManager()
        assert manager.mode == "meta"
        text, err = manager.generate([{"role": "user", "content": "Plan my day"}],
                                     max_new_tokens=64)
        assert text is None and err
        assert captured["max_tokens"] == 8192
        status = mm.ModelManager().status()
        error = status["provider_error"]
        assert error["choice_structure"]["finish_reason"] == "length"
        assert error["usage_keys"]["keys"] == [
            "completion_tokens", "completion_tokens_details",
            "prompt_tokens", "total_tokens"]
        assert error["usage_keys"]["nested"] == {
            "completion_tokens_details": [
                "accepted_prediction_tokens", "reasoning_tokens"]}
    finally:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        mm._LAST_PROVIDER_DIAGNOSTIC.update(saved)


def test_r8_usage_keys_absent_when_no_usage_object(monkeypatch):
    """A body with no "usage" object yields usage_keys None — never
    an invented structure."""
    import requests
    svc = MetaService(api_key="fake-key", model="m")
    body = {"choices": [{"finish_reason": "stop", "index": 0,
                         "message": {"content": None, "role": "assistant"}}],
            "created": 1, "id": "x", "model": "m",
            "object": "chat.completion"}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert svc.last_diagnostic["usage_keys"] is None


def test_r8_usage_keys_nested_non_dict_ignored(monkeypatch):
    """A nested non-dict usage value (e.g. a string) is a key name
    only; it does not appear under "nested"."""
    import requests
    svc = MetaService(api_key="fake-key", model="m")
    body = {"choices": [{"finish_reason": "stop", "index": 0,
                         "message": {"content": None, "role": "assistant"}}],
            "created": 1, "id": "x", "model": "m",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 3, "service_tier": "default"}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    svc.generate([{"role": "user", "content": "hi"}])
    assert svc.last_diagnostic["usage_keys"] == {
        "keys": ["prompt_tokens", "service_tier"], "nested": {}}
