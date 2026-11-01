"""OpenAI -> Hugging Face fallback in ModelManager (2026-11-01):
when OpenAI fails with a quota/rate-limit error and an HF token is
also configured, the HF router gets a second chance instead of the
answer dying outright. If HF also fails, the original OpenAI error
is returned, never mislabelled.
"""
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-fallback-tests.db")

from services.model_manager import ModelManager


class _FailingOpenAI:
    calls = 0

    def __init__(self, api_key=None, model=None):
        pass

    def generate(self, messages, max_tokens=256, temperature=0.7):
        type(self).calls += 1
        return None, ("OpenAI returned HTTP 429: You have no credits remaining. "
                      "Add credits at https://platform.openai.com/settings/organization/billing/")


def _manager(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    return ModelManager()


def test_openai_quota_falls_back_to_hf(monkeypatch):
    import services.openai_service as openai_service
    monkeypatch.setattr(openai_service, "OpenAIService", _FailingOpenAI)
    manager = _manager(monkeypatch)
    assert manager.mode == "openai"
    monkeypatch.setattr(ModelManager, "_generate_api",
                        lambda self, *a, **k: ("HF drafted answer.", None))
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text == "HF drafted answer." and err is None


def test_both_providers_failing_returns_openai_error(monkeypatch):
    import services.openai_service as openai_service
    monkeypatch.setattr(openai_service, "OpenAIService", _FailingOpenAI)
    manager = _manager(monkeypatch)
    monkeypatch.setattr(ModelManager, "_generate_api",
                        lambda self, *a, **k: (None, "HF router 503"))
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text is None and "429" in err and "platform.openai.com" in err


def test_no_hf_token_means_no_wasted_hf_attempt(monkeypatch):
    import services.openai_service as openai_service
    monkeypatch.setattr(openai_service, "OpenAIService", _FailingOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    manager = ModelManager()
    assert manager.mode == "openai" and not manager.token
    calls = {"n": 0}

    def _never(self, *a, **k):
        calls["n"] += 1
        return None, "should not be called"

    monkeypatch.setattr(ModelManager, "_generate_api", _never)
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text is None and calls["n"] == 0
