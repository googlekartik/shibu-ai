"""Round-9 regression tests (2026-11-02): Meta reasoning model fix.

Live evidence (focused re-check, round-8 code proven serving by
usage_keys present): finish_reason "length" with null visible
content STILL at the raised 1024 budget, and usage_keys nested
shows completion_tokens_details ["reasoning_tokens"] — the
provider reports reasoning tokens. muse-spark-1.3 is a reasoning
model whose thinking tokens count toward the output cap, so the
fix sends reasoning_effort "low" (validated env override) and
both output-cap fields at 8192."""

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
    finish_reason "length", usage with reasoning_tokens reported
    in a nested details dict."""
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


def _fake_post_ok(captured):
    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json or {})
        return _Resp({"choices": [{"finish_reason": "stop", "index": 0,
                                   "message": {"content": "ok", "role": "assistant"}}],
                      "created": 1, "id": "x", "model": "m",
                      "object": "chat.completion", "usage": {}})
    return _fake_post


def test_r9_meta_request_sends_effort_and_both_caps(monkeypatch):
    """The default call sends reasoning_effort "low" plus
    max_completion_tokens and max_tokens, both 8192."""
    import requests
    captured = {}
    svc = MetaService(api_key="fake-key", model="m")
    monkeypatch.setattr(requests, "post", _fake_post_ok(captured))
    monkeypatch.delenv("META_REASONING_EFFORT", raising=False)
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "ok" and err is None
    assert captured["reasoning_effort"] == "low"
    assert captured["max_completion_tokens"] == 8192
    assert captured["max_tokens"] == 8192
    assert captured["temperature"] == 0.7


def test_r9_meta_small_budget_floored_in_both_caps(monkeypatch):
    """An explicit small budget (256) is floored to 8192 in BOTH
    cap fields — the alias mapping cannot reintroduce a small cap."""
    import requests
    captured = {}
    svc = MetaService(api_key="fake-key", model="m")
    monkeypatch.setattr(requests, "post", _fake_post_ok(captured))
    monkeypatch.delenv("META_REASONING_EFFORT", raising=False)
    svc.generate([{"role": "user", "content": "hi"}], max_tokens=256)
    assert captured["max_completion_tokens"] == 8192
    assert captured["max_tokens"] == 8192
    svc.generate([{"role": "user", "content": "hi"}], max_tokens=8193)
    assert captured["max_completion_tokens"] == 8193
    assert captured["max_tokens"] == 8193


def test_r9_meta_reasoning_effort_env_honoured(monkeypatch):
    """META_REASONING_EFFORT=medium is sent verbatim."""
    import requests
    captured = {}
    svc = MetaService(api_key="fake-key", model="m")
    monkeypatch.setattr(requests, "post", _fake_post_ok(captured))
    monkeypatch.setenv("META_REASONING_EFFORT", "medium")
    svc.generate([{"role": "user", "content": "hi"}])
    assert captured["reasoning_effort"] == "medium"


def test_r9_meta_reasoning_effort_none_falls_back(monkeypatch):
    """META_REASONING_EFFORT=none is rejected by the provider for
    Muse Spark, so it must never be sent — it falls back to "low"."""
    import requests
    captured = {}
    svc = MetaService(api_key="fake-key", model="m")
    monkeypatch.setattr(requests, "post", _fake_post_ok(captured))
    monkeypatch.setenv("META_REASONING_EFFORT", "none")
    svc.generate([{"role": "user", "content": "hi"}])
    assert captured["reasoning_effort"] == "low"


def test_r9_meta_reasoning_effort_bogus_falls_back(monkeypatch):
    """An unrecognised or empty value falls back to "low" and is
    never passed through unvalidated."""
    import requests
    captured = {}
    svc = MetaService(api_key="fake-key", model="m")
    monkeypatch.setattr(requests, "post", _fake_post_ok(captured))
    monkeypatch.setenv("META_REASONING_EFFORT", "turbo")
    svc.generate([{"role": "user", "content": "hi"}])
    assert captured["reasoning_effort"] == "low"
    monkeypatch.setenv("META_REASONING_EFFORT", "  HIGH  ")
    svc.generate([{"role": "user", "content": "hi"}])
    assert captured["reasoning_effort"] == "high"
    monkeypatch.setenv("META_REASONING_EFFORT", "")
    svc.generate([{"role": "user", "content": "hi"}])
    assert captured["reasoning_effort"] == "low"


def test_r9_meta_length_diagnostic_unchanged(monkeypatch):
    """Extraction/diagnostic behaviour is unchanged: a "length"
    reply with null content still records finish_reason "length",
    refusal_present False, and usage_keys with key names only —
    the token counts must never appear in the diagnostic."""
    import requests
    svc = MetaService(api_key="fake-key", model="muse-spark-1.3")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(_length_body()))
    monkeypatch.delenv("META_REASONING_EFFORT", raising=False)
    text, err = svc.generate([{"role": "user", "content": "Plan my day"}])
    assert text is None and err
    struct = svc.last_diagnostic["choice_structure"]
    assert struct["finish_reason"] == "length"
    assert struct["refusal_present"] is False
    assert svc.last_diagnostic["usage_keys"] == {
        "keys": ["completion_tokens", "completion_tokens_details",
                 "prompt_tokens", "total_tokens"],
        "nested": {"completion_tokens_details": [
            "accepted_prediction_tokens", "reasoning_tokens"]}}
    dumped = json.dumps(svc.last_diagnostic)
    for value in ("12345", "678", "13023"):
        assert value not in dumped


def test_r9_manager_meta_branch_sends_effort_and_floor(monkeypatch, tmp_path):
    """Through ModelManager with a tiny caller budget, the Meta
    request still carries both caps at 8192 and reasoning_effort
    "low"; the diagnostic still flows to status()."""
    import requests
    import services.model_manager as mm
    saved = dict(mm._LAST_PROVIDER_DIAGNOSTIC)
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setenv("META_API_KEY", "fake-key")
    monkeypatch.setenv("META_MODEL", "muse-spark-1.3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("META_REASONING_EFFORT", raising=False)
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
        assert captured["max_completion_tokens"] == 8192
        assert captured["reasoning_effort"] == "low"
        status = mm.ModelManager().status()
        error = status["provider_error"]
        assert error["choice_structure"]["finish_reason"] == "length"
        assert error["usage_keys"]["nested"] == {
            "completion_tokens_details": [
                "accepted_prediction_tokens", "reasoning_tokens"]}
    finally:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        mm._LAST_PROVIDER_DIAGNOSTIC.update(saved)
