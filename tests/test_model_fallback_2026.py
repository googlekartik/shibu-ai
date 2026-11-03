"""Gemini-only fallback (2026-11-03): Gemini is the only AI
provider. When Gemini fails, ModelManager makes exactly one call
and returns its error; there is no second provider to fall back
to. Callers use the deterministic fallback (rules / maths /
retrieved documents / search snippets, labelled) instead. No
provider internals leak into user-facing text beyond the
diagnostic detail the manager itself returns.
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


class _CountingGemini:
    calls = 0

    def __init__(self, api_key=None, model=None):
        pass

    def generate(self, messages, max_tokens=256, temperature=0.7):
        type(self).calls += 1
        return "Gemini drafted answer.", None


def _manager(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    monkeypatch.delenv("ROUTER_PREFERENCE", raising=False)
    return ModelManager()


def test_gemini_failure_returns_error_after_exactly_one_call(monkeypatch):
    import services.gemini_service as gemini_service
    _FailingGemini.calls = 0
    monkeypatch.setattr(gemini_service, "GeminiService", _FailingGemini)
    manager = _manager(monkeypatch)
    assert manager.mode == "gemini"
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text is None and err is not None
    assert _FailingGemini.calls == 1


def test_gemini_success_makes_exactly_one_call(monkeypatch):
    import services.gemini_service as gemini_service
    _CountingGemini.calls = 0
    monkeypatch.setattr(gemini_service, "GeminiService", _CountingGemini)
    manager = _manager(monkeypatch)
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text == "Gemini drafted answer." and err is None
    assert _CountingGemini.calls == 1


def test_off_mode_makes_no_call(monkeypatch):
    import services.gemini_service as gemini_service
    _CountingGemini.calls = 0
    monkeypatch.setattr(gemini_service, "GeminiService", _CountingGemini)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    manager = ModelManager()
    assert manager.mode == "off"
    text, err = manager.generate([{"role": "user", "content": "hi"}])
    assert text is None and err
    assert _CountingGemini.calls == 0
    assert "GEMINI_API_KEY" in manager.status()["how_to_enable"]


def test_failure_records_structure_only_diagnostic(monkeypatch):
    import services.gemini_service as gemini_service
    monkeypatch.setattr(gemini_service, "GeminiService", _FailingGemini)
    manager = _manager(monkeypatch)
    manager.generate([{"role": "user", "content": "hi"}])
    status = manager.status()
    assert status["provider_error"] is not None
    assert status["provider_error"]["provider"] == "gemini"
    assert "sk-test" not in str(status)
