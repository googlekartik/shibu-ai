"""Diagnostics recording (2026-11-03): provider diagnostics and
router decisions must be visible to /api/llm/status from ANY
ModelManager instance and must survive a process restart via the
durable sidecar — the live defect was last_decision always null
and a stale empty provider_error, because decisions lived only in
process memory and failures were never cleared by later success.
"""
import json
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-diag-tests.db")

import services.model_manager as mm
from services.diagnostic_store import diagnostic_path
from services.model_manager import ModelManager


@pytest.fixture(autouse=True)
def _isolated_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    mm._LAST_ROUTER_DECISION.clear()
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()
    yield
    mm._LAST_ROUTER_DECISION.clear()
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()


class _FakeGemini:
    """Stand-in for GeminiService: generate() returns the preset
    outcome and exposes last_diagnostic like the real service."""

    def __init__(self, outcome):
        self._outcome = outcome
        self.last_diagnostic = outcome.get("diagnostic")

    def generate(self, messages, max_tokens=256, temperature=0.7):
        return self._outcome["text"], self._outcome["error"]


def _fake_gemini(monkeypatch, outcome):
    def _factory(api_key=None, model=None, api_base=None):
        return _FakeGemini(outcome)

    monkeypatch.setattr("services.gemini_service.GeminiService", _factory)


def _keys(monkeypatch, gemini=True, hf=True):
    if gemini:
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CLOUD_ONLY", raising=False)


def test_two_instances_share_decision_and_diagnostic(monkeypatch):
    _keys(monkeypatch)
    _fake_gemini(monkeypatch, {"text": "drafted plan", "error": None,
                               "diagnostic": {"status": 200, "response_keys": ["candidates"]}})
    first = ModelManager()
    text, err = first.generate([{"role": "user", "content": "Plan my day"}],
                               task_kind="plan")
    assert text == "drafted plan" and err is None
    # A DIFFERENT instance (as /api/llm/status constructs) sees it.
    second = ModelManager()
    status = second.status()
    assert status["router"]["last_decision"]["provider"] == "gemini"
    assert status["router"]["last_decision"]["task_kind"] == "plan"
    assert status["router"]["last_decision"]["fallback_used"] is False
    assert status["provider_error"] is None


def test_decision_survives_fresh_process_via_sidecar(monkeypatch, tmp_path):
    _keys(monkeypatch)
    _fake_gemini(monkeypatch, {"text": "drafted plan", "error": None,
                               "diagnostic": {"status": 200, "response_keys": ["candidates"]}})
    ModelManager().generate([{"role": "user", "content": "Plan my day"}],
                            task_kind="plan")
    # Simulate a process restart: in-memory state is gone.
    mm._LAST_ROUTER_DECISION.clear()
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()
    status = ModelManager().status()
    assert status["router"]["last_decision"]["provider"] == "gemini"
    assert status["router"]["last_decision"]["task_kind"] == "plan"
    assert status["provider_error"] is None
    # The sidecar file itself carries a timestamp; status strips it.
    with open(diagnostic_path("router"), encoding="utf-8") as handle:
        stored = json.load(handle)
    assert isinstance(stored["recorded_at"], int)
    assert "recorded_at" not in status["router"]["last_decision"]


def test_gemini_failure_then_success_clears_error(monkeypatch):
    _keys(monkeypatch)
    _fake_gemini(monkeypatch, {"text": None, "error": "boom", "diagnostic": None})
    text, err = ModelManager().generate(
        [{"role": "user", "content": "Plan my day"}], task_kind="plan")
    assert text is None and err is not None
    status = ModelManager().status()
    assert status["router"]["last_decision"]["fallback_used"] is False
    assert status["provider_error"] is not None
    # A later Gemini success supersedes the failure: no stale error.
    _fake_gemini(monkeypatch, {"text": "gemini answer", "error": None,
                               "diagnostic": {"status": 200, "response_keys": ["candidates"]}})
    text, err = ModelManager().generate(
        [{"role": "user", "content": "Plan my day"}], task_kind="plan")
    assert text == "gemini answer" and err is None
    assert ModelManager().status()["provider_error"] is None


def test_no_http_response_failure_records_error_type(monkeypatch):
    _keys(monkeypatch, gemini=True)
    _fake_gemini(monkeypatch, {"text": None, "error": "boom", "diagnostic": None})
    text, err = ModelManager().generate(
        [{"role": "user", "content": "Plan my day"}], task_kind="plan")
    assert text is None and err is not None
    status = ModelManager().status()
    assert status["provider_error"]["provider"] == "gemini"
    assert status["provider_error"]["status"] is None
    assert status["provider_error"]["response_keys"] == []
    assert status["provider_error"]["error_type"] == "no_http_response"
    assert status["router"]["last_decision"]["provider"] == "gemini"


def test_stale_failure_cleared_by_later_success(monkeypatch):
    _keys(monkeypatch, gemini=True)
    failing = {"text": None, "error": "boom",
               "diagnostic": {"status": 403, "response_keys": ["error"]}}
    _fake_gemini(monkeypatch, failing)
    assert ModelManager().generate(
        [{"role": "user", "content": "Plan my day"}], task_kind="plan")[0] is None
    assert ModelManager().status()["provider_error"]["status"] == 403
    _fake_gemini(monkeypatch, {"text": "now it works", "error": None,
                               "diagnostic": {"status": 200, "response_keys": ["candidates"]}})
    text, _ = ModelManager().generate(
        [{"role": "user", "content": "Plan my day"}], task_kind="plan")
    assert text == "now it works"
    assert ModelManager().status()["provider_error"] is None


def test_diagnostic_never_contains_payload_values(monkeypatch):
    _keys(monkeypatch, gemini=True)
    _fake_gemini(monkeypatch, {"text": None, "error": "boom",
                               "diagnostic": {"status": 400, "response_keys": ["error"]}})
    ModelManager().generate([{"role": "user", "content": "Plan my secret day"}],
                            task_kind="plan")
    status = ModelManager().status()
    assert "secret" not in json.dumps(status["provider_error"])
    assert "secret" not in json.dumps(status["router"]["last_decision"])
