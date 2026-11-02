"""Provider routing + fallback in ModelManager (2026-11-01, router
2026-11-03, Gemini replaces OpenAI 2026-11-03): the router decides
per request (short prompts go to Hugging Face first); when the chosen
provider fails, the other provider gets exactly one fallback call. If
both fail, one clean neutral error is returned — no provider
internals, no quota or billing wording. The detail stays in
structure-only diagnostics.
"""
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-fallback-tests.db")

from services.model_manager import ModelManager


class _FailingGemini:
    calls = 0

    def __init__(self, api_key=None, model=None):
        pass

    def generate(self, messages, max_tokens=256, temperature=0.7):
        type(self).calls += 1
        return None, ("Gemini returned HTTP 429: quota exhausted. "
                      "Check your Google AI Studio quota.")


def _manager(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    return ModelManager()


def test_gemini_quota_falls_back_to_hf(monkeypatch):
    import services.gemini_service as gemini_service
    monkeypatch.setattr(gemini_service, "GeminiService", _FailingGemini)
    manager = _manager(monkeypatch)
    assert manager.mode == "gemini"
    monkeypatch.setattr(ModelManager, "_generate_api",
                        lambda self, *a, **k: ("HF drafted answer.", None))
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text == "HF drafted answer." and err is None


def test_both_providers_failing_returns_clean_error(monkeypatch):
    import services.gemini_service as gemini_service
    monkeypatch.setattr(gemini_service, "GeminiService", _FailingGemini)
    manager = _manager(monkeypatch)
    monkeypatch.setattr(ModelManager, "_generate_api",
                        lambda self, *a, **k: (None, "HF router 503"))
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text is None and err == "The language model could not draft this."
    for forbidden in ("429", "quota", "aistudio.google.com", "HF router"):
        assert forbidden not in err


def test_no_hf_token_means_no_wasted_hf_attempt(monkeypatch):
    import services.gemini_service as gemini_service
    monkeypatch.setattr(gemini_service, "GeminiService", _FailingGemini)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    manager = ModelManager()
    assert manager.mode == "gemini" and not manager.token
    calls = {"n": 0}

    def _never(self, *a, **k):
        calls["n"] += 1
        return None, "should not be called"

    monkeypatch.setattr(ModelManager, "_generate_api", _never)
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text is None and calls["n"] == 0
