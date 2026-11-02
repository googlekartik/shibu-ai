"""Round-7 regression tests (2026-11-02 round-6 live re-test findings).

M7: the live /api/llm/status read during the round-6 re-test showed
the OLD choice_structure shape (no finish_reason / refusal_present /
logprobs_present) because the push landed mid-test and the first
check ran against round-5 code still serving (deploy race). These
tests lock in that the CURRENT read path returns the full new-shape
diagnostic from a fresh failure, from the durable sidecar after a
simulated restart, and that a stale old-shape sidecar never masks a
fresh diagnostic.

T7: "how to calibrate a flux capacitor" returned a Steps block +
duplicated web sentence + irrelevant Calibrate results, because
"calibrate" counted as a content noun, so verb-only results passed
the relevance gate, and because a stepless web answer was served
instead of the honest failure sentence. No network, no keys.
"""
import os
import sys
import tempfile

os.environ["RAG_BACKEND"] = "sqlite"
os.environ["SHIBU_SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "shibu-round7-test.db")
try:
    os.remove(os.environ["SHIBU_SQLITE_PATH"])
except OSError:
    pass
for _key in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "DEV_PANEL",
             "OPENAI_API_KEY", "OPENAI_MODEL", "HF_TOKEN",
             "HF_USE_LOCAL", "HF_FORCE_LOCAL"):
    os.environ.pop(_key, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests

import app as app_module  # noqa: F401  (kept for parity with round-6 tests)
from chatbot.context import ConversationContext
from chatbot.router import process_message
from services.content_processing_engine import process_presentation
from services.content_structuring_engine import structure
from services.web_answer import _content_nouns, filter_results


def _ctx(display_name=""):
    ctx = ConversationContext()
    ctx.display_name = display_name
    return ctx


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _openai_body(finish_reason, refusal):
    return {"choices": [{"finish_reason": finish_reason, "index": 0,
                         "logprobs": None,
                         "message": {"content": None, "refusal": refusal,
                                     "role": "assistant"}}],
            "created": 1, "id": "x", "model": "gpt-test",
            "object": "chat.completion", "usage": {}}


def _failing_llm(monkeypatch):
    import services.daily_task_service as daily

    class _Failing:
        mode = "api"
        model_id = "fake-model"

        def generate(self, *a, **k):
            return None, "API unavailable"

    monkeypatch.setattr(daily, "ModelManager", lambda: _Failing())


# ---------------------------------------------------------------- M7
def test_r7_fresh_openai_failure_status_has_new_diagnostic_keys(monkeypatch, tmp_path):
    """A fresh null-content OpenAI failure surfaces finish_reason,
    refusal_present and logprobs_present through ModelManager.status()
    — the exact read path behind /api/llm/status and /api/health."""
    import services.model_manager as mm
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(_openai_body("refusal", "I cannot help with that.")))
    saved = dict(mm._LAST_PROVIDER_DIAGNOSTIC)
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()
    try:
        mgr = mm.ModelManager()
        text, err = mgr.generate([{"role": "user", "content": "Plan my day"}],
                                 max_new_tokens=64)
        assert text is None and err
        cs = mgr.status()["provider_error"]["choice_structure"]
        assert cs["finish_reason"] == "refusal"
        assert cs["refusal_present"] is True
        assert cs["logprobs_present"] is False
        assert cs["content_type"] == "NoneType"
    finally:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        mm._LAST_PROVIDER_DIAGNOSTIC.update(saved or {})


def test_r7_stale_sidecar_never_masks_fresh_diagnostic(monkeypatch, tmp_path):
    """An old-shape sidecar left on disk (round-5 shape, no new
    keys) plus a fresh round-6 failure: the status read shows the
    fresh diagnostic, and the sidecar is overwritten with it."""
    import json
    import services.model_manager as mm
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(_openai_body("refusal", "Nope.")))
    old = {"provider": "openai", "status": 200, "response_keys": ["choices"],
           "choice_structure": {"choice_keys": ["finish_reason", "index", "logprobs", "message"],
                                "content_type": "NoneType",
                                "message_keys": ["content", "refusal", "role"],
                                "part_types": None}}
    (tmp_path / "provider_diagnostic.json").write_text(json.dumps(old))
    saved = dict(mm._LAST_PROVIDER_DIAGNOSTIC)
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()
    try:
        mgr = mm.ModelManager()
        mgr.generate([{"role": "user", "content": "Plan my day"}], max_new_tokens=64)
        cs = mgr.status()["provider_error"]["choice_structure"]
        assert cs["finish_reason"] == "refusal"
        assert cs["refusal_present"] is True
        stored = json.loads((tmp_path / "provider_diagnostic.json").read_text())
        assert stored["choice_structure"]["finish_reason"] == "refusal"
    finally:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        mm._LAST_PROVIDER_DIAGNOSTIC.update(saved)


def test_r7_restart_reads_new_shape_from_sidecar_only(monkeypatch, tmp_path):
    """Simulated restart: process memory cleared, only the durable
    sidecar remains — the status read still shows the new keys."""
    import services.model_manager as mm
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(_openai_body("stop", None)))
    saved = dict(mm._LAST_PROVIDER_DIAGNOSTIC)
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()
    try:
        mm.ModelManager().generate([{"role": "user", "content": "Plan my day"}],
                                   max_new_tokens=64)
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()  # the restart
        cs = mm.ModelManager().status()["provider_error"]["choice_structure"]
        assert cs["finish_reason"] == "stop"
        assert cs["refusal_present"] is False
        assert cs["logprobs_present"] is False
    finally:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        mm._LAST_PROVIDER_DIAGNOSTIC.update(saved)


# ---------------------------------------------------------------- T7
def test_r7_content_nouns_drop_leading_action_word():
    """The verb in the "how to X" slot is the action, not the
    topic: "calibrate" must not qualify results on its own."""
    assert _content_nouns("how to calibrate a flux capacitor") == ["flux", "capacitor"]
    assert _content_nouns("how to make tea") == ["tea"]
    assert _content_nouns("how to boil an egg") == ["egg"]
    assert _content_nouns("how to lose weight") == ["weight"]
    assert _content_nouns("how to meditate") == ["meditate"]
    # A repeated topic word survives (only the verb slot is dropped).
    assert _content_nouns("how to tie a tie") == []


def test_r7_filter_drops_verb_only_results():
    """Calibrate Weight Loss / dictionary results mention only the
    verb: for a flux-capacitor how-to they are dropped outright,
    even though a wiki definition of the topic may survive the
    noun gate (it still carries no steps — see the router test)."""
    results = [
        {"title": "Calibrate Weight Loss - Medical Weight Loss Programs",
         "url": "https://example.test/w",
         "snippet": "Calibrate is a medical weight loss program that combines medication with coaching."},
        {"title": "calibrate | Definition & Meaning - Dictionary",
         "url": "https://example.test/d",
         "snippet": "calibrate: to adjust a measuring instrument so that its readings are accurate."},
    ]
    assert filter_results("how to calibrate a flux capacitor", results) == []


_CALIBRATE_ONLY = [
    {"title": "Calibrate Weight Loss - Medical Weight Loss Programs",
     "url": "https://example.test/calibrate-weight",
     "snippet": "Calibrate is a medical weight loss program that combines medication with coaching to help you lose weight."},
    {"title": "calibrate | Definition & Meaning - Dictionary",
     "url": "https://example.test/dict",
     "snippet": "calibrate: to adjust a measuring instrument so that its readings are accurate."},
    {"title": "Flux Capacitor - Back to the Future Wiki",
     "url": "https://example.test/flux",
     "snippet": "The flux capacitor is a fictional device from the Back to the Future films that makes time travel possible."},
]


def test_r7_unknown_howto_verb_only_results_single_failure(monkeypatch):
    """The live defect, through the real router path: verb-only
    results plus a stepless topic definition yield ONE failure
    sentence — no Steps block, no Sources, no duplicated text,
    no Calibrate content."""
    from unittest.mock import patch
    from services.web_search_service import WebSearchService
    _failing_llm(monkeypatch)
    with patch.object(WebSearchService, "search", return_value=list(_CALIBRATE_ONLY)):
        out = process_message("how to calibrate a flux capacitor", _ctx())
    assert out["rule_id"] == "daily_howto_unavailable"
    assert out["response"].count("I could not look up steps") == 1
    assert "Calibrate" not in out["response"]
    assert "Sources" not in out["response"]
    assert "Steps" not in out["response"]
    assert "Here's what I found" not in out["response"]
    assert process_presentation("how to calibrate a flux capacitor",
                                out, context=_ctx()) is None


def test_r7_ontopic_howto_web_answer_intact(monkeypatch):
    """Control: results with real imperative steps still produce
    the normal web answer, sources included."""
    from unittest.mock import patch
    from services.web_search_service import WebSearchService
    _failing_llm(monkeypatch)
    good = [
        {"title": "How to Make Tea - WikiHow",
         "url": "https://example.test/tea",
         "snippet": "Boil fresh water in a kettle. Pour the hot water over the tea bag. Let it steep for 3 minutes."},
        {"title": "Tea Brewing Guide",
         "url": "https://example.test/guide",
         "snippet": "Place the tea leaves in the pot. Add boiling water and wait 4 minutes before serving."},
    ]
    with patch.object(WebSearchService, "search", return_value=list(good)):
        out = process_message("how to make tea", _ctx())
    assert out["rule_id"] == "web_fallback"
    assert "Boil fresh water" in out["response"]
    assert "Sources" in out["response"]


def test_r7_framing_sentence_never_a_step():
    """A web answer's framing sentence appears once, in the intro
    paragraph — never re-presented as a Steps item, even when it
    is the only sentence the steps fallback would pick up."""
    text = ("Here's what I found on the web about 'how to make tea':\n\n"
            "In short, according to the latest web results:\n"
            "Boil fresh water in a kettle. Pour the hot water over the tea bag.\n\n"
            "Key points:\n- Boil fresh water in a kettle.\n- Pour the hot water over the tea bag.")
    blocks = structure(text, {}, {"content_type": "process_steps", "topic": "how to make tea"},
                       sources=[])
    steps = [b for b in blocks if b.get("kind") == "steps"]
    assert steps, "real steps must still be presented"
    joined = " ".join(steps[0]["items"])
    assert "Here's what I found" not in joined
    assert "In short, according to" not in joined
    assert "Boil fresh water" in joined
    # Framing-only text yields no Steps block at all.
    bare = "Here's what I found on the web about 'how to calibrate a flux capacitor':"
    blocks = structure(bare, {}, {"content_type": "process_steps", "topic": "flux capacitor"},
                       sources=[])
    assert not any(b.get("kind") == "steps" for b in blocks)
