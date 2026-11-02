"""Round-6 regression tests (2026-11-02 round-5 live re-test findings).

Each test reproduces one defect from the 2026-11-02 14:32 IST
re-test of round 5 (commit 8cbab94b). No network, no keys: the
Gemini payloads below are mocked strings used only as test input.
Live Gemini behaviour is NOT tested here; the parent re-tests it
live after the push, reading the new diagnostic fields.
"""
import os
import sys
import tempfile

os.environ["RAG_BACKEND"] = "sqlite"
os.environ["SHIBU_SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "shibu-round6-test.db")
try:
    os.remove(os.environ["SHIBU_SQLITE_PATH"])
except OSError:
    pass
for _key in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "DEV_PANEL",
             "GEMINI_API_KEY", "GEMINI_MODEL", "HF_TOKEN",
             "HF_USE_LOCAL", "HF_FORCE_LOCAL"):
    os.environ.pop(_key, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import app as app_module
from chatbot.context import ConversationContext
from chatbot.router import process_message
from services.content_processing_engine import process_presentation
from services.content_structuring_engine import structure
from services.daily_task_service import _template_write, build_daily_task_payload
from services.gemini_service import GeminiService


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


# ---------------------------------------------------------------- M6
def _gemini_body(finish_reason, blocked):
    ratings = [{"category": "HARM_CATEGORY_HATE_SPEECH",
                "probability": "HIGH", "blocked": blocked}]
    return {"candidates": [{"content": {"parts": [], "role": "model"},
                            "finishReason": finish_reason,
                            "safetyRatings": ratings}],
            "usageMetadata": {}}


def test_r6_gemini_blocked_diagnostic(monkeypatch):
    """Empty parts + finishReason 'SAFETY' + a blocked safety
    rating: generate returns (None, error) and the diagnostic
    records blocked_present=True and the finishReason value.
    Rating details themselves are never recorded."""
    import requests
    svc = GeminiService(api_key="fake-key", model="gemini-test")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(_gemini_body("SAFETY", True)))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None and err
    struct = svc.last_diagnostic["choice_structure"]
    assert struct["content_type"] == "list"
    assert struct["finish_reason"] == "SAFETY"
    assert struct["blocked_present"] is True
    assert struct["text_present"] is False
    assert "HATE_SPEECH" not in str(svc.last_diagnostic)


def test_r6_gemini_empty_stop_diagnostic(monkeypatch):
    """Empty parts + no block + finishReason 'STOP': an empty
    generation, recorded as such — blocked_present False."""
    import requests
    svc = GeminiService(api_key="fake-key", model="gemini-test")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(_gemini_body("STOP", False)))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None and err
    struct = svc.last_diagnostic["choice_structure"]
    assert struct["finish_reason"] == "STOP"
    assert struct["blocked_present"] is False


def test_r6_gemini_text_extraction_unchanged(monkeypatch):
    """A normal text part still extracts as the answer."""
    import requests
    svc = GeminiService(api_key="fake-key", model="gemini-test")
    body = {"candidates": [{"content": {"parts": [{"text": "Hello there."}],
                                        "role": "model"},
                            "finishReason": "STOP"}],
            "usageMetadata": {}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(body))
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Hello there." and err is None


def test_r6_health_and_llm_status_agree(monkeypatch, tmp_path):
    """Both endpoints read the same diagnostic source: after a
    recorded failure, a fresh manager, /api/health and
    /api/llm/status all report the same provider_error."""
    import services.model_manager as mm
    saved = dict(mm._LAST_PROVIDER_DIAGNOSTIC)
    try:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
        mm._record_provider_diagnostic("gemini", {
            "status": 200, "response_keys": ["candidates"],
            "choice_structure": {"content_type": "list",
                                 "finish_reason": "SAFETY",
                                 "blocked_present": True,
                                 "text_present": False}})
        diag = mm.ModelManager().status().get("provider_error")
        assert diag is not None and diag["provider"] == "gemini"
        assert diag["choice_structure"]["finish_reason"] == "SAFETY"
        assert diag["choice_structure"]["blocked_present"] is True
        client = app_module.app.test_client()
        health = client.get("/api/health").get_json()
        status = client.get("/api/llm/status").get_json()
        assert health["llm"]["provider_error"] == diag
        assert status["provider_error"] == diag
    finally:
        mm._LAST_PROVIDER_DIAGNOSTIC.clear()
        mm._LAST_PROVIDER_DIAGNOSTIC.update(saved)


# ---------------------------------------------------------------- E6
def test_r6_email_template_is_finished_email():
    """No brackets, no 'fill in' lead-in, subject/greeting/body/
    closing on separate lines, and no invented day."""
    email = _template_write("Write a polite email to my teacher asking for leave tomorrow")
    assert "[" not in email and "]" not in email
    assert "Fill in" not in email
    lines = [ln for ln in email.split("\n")]
    assert lines[0].startswith("Subject: Request for leave tomorrow")
    assert "Dear Teacher," in lines
    assert any("I am writing to request leave tomorrow" in ln for ln in lines)
    assert any("Thank you for your understanding" in ln for ln in lines)
    assert email.rstrip().endswith("Yours sincerely")
    assert "on tomorrow" not in email


def test_r6_email_template_uses_display_name():
    """The signed-in name signs off the email; with no name known,
    the name line is omitted rather than placeholder text."""
    named = _template_write("Write an email to my teacher asking for leave tomorrow",
                            display_name="Kartik")
    assert "Kartik" in named and "[" not in named
    assert named.rstrip().endswith("Yours sincerely,\nKartik")
    unnamed = _template_write("Write an email to my teacher asking for leave tomorrow")
    assert "Kartik" not in unnamed
    assert unnamed.rstrip().endswith("Yours sincerely")


def test_r6_email_router_path_model_fails(monkeypatch):
    """Model failure -> template through the real router path, with
    the display name from context, no brackets, and no
    presentation wrapper flattening it."""
    import services.daily_task_service as daily

    class _Failing:
        mode = "api"
        model_id = "fake-model"

        def generate(self, *a, **k):
            return None, "API unavailable"

    monkeypatch.setattr(daily, "ModelManager", lambda: _Failing())
    ctx = _ctx(display_name="Kartik")
    out = process_message("Write an email to my teacher asking for leave tomorrow", ctx)
    assert out["rule_id"] == "daily_write"
    assert "[" not in out["response"] and "Fill in" not in out["response"]
    assert "Kartik" in out["response"]
    assert process_presentation("Write an email to my teacher asking for leave tomorrow",
                                out, context=ctx) is None


# ---------------------------------------------------------------- T6
_MAKE_ONLY = [
    {"title": "Make is an AI automation platform",
     "url": "https://example.test/make",
     "snippet": "Make is an AI automation platform that helps teams build workflows and automate tasks."},
    {"title": "AI Workflow Automation Software & Tools | Make",
     "url": "https://example.test/tools",
     "snippet": "Make is a visual platform for designing, building and automating workflows."},
]


def _failing_llm(monkeypatch):
    import services.daily_task_service as daily

    class _Failing:
        mode = "api"
        model_id = "fake-model"

        def generate(self, *a, **k):
            return None, "API unavailable"

    monkeypatch.setattr(daily, "ModelManager", lambda: _Failing())


def test_r6_tea_builtin_when_web_only_make_results(monkeypatch):
    """Both the original and the expanded query return only
    Make-platform results: the built-in tea steps answer, with no
    platform text and no failure sentence."""
    from unittest.mock import patch
    from services.web_search_service import WebSearchService
    _failing_llm(monkeypatch)
    with patch.object(WebSearchService, "search", return_value=list(_MAKE_ONLY)):
        out = process_message("how to make tea", _ctx())
    assert out["rule_id"] == "daily_howto_builtin"
    assert "Boil fresh water" in out["response"]
    assert "steep" in out["response"].lower()
    assert "Make is an AI automation platform" not in out["response"]
    assert "could not look up" not in out["response"]
    assert out["response"].count("Boil fresh water") == 1


def test_r6_tea_builtin_when_search_empty(monkeypatch):
    """Search failing entirely also yields the built-in tea steps."""
    from unittest.mock import patch
    from services.web_search_service import WebSearchService
    _failing_llm(monkeypatch)
    with patch.object(WebSearchService, "search", return_value=[]):
        out = process_message("how to make tea", _ctx())
    assert out["rule_id"] == "daily_howto_builtin"
    assert "Boil fresh water" in out["response"]
    assert "could not look up" not in out["response"]


def test_r6_unknown_howto_single_failure_sentence(monkeypatch):
    """A task outside the built-in table with no web results: one
    failure sentence, no Steps block, no duplicated text, and no
    presentation wrapper."""
    from unittest.mock import patch
    from services.web_search_service import WebSearchService
    _failing_llm(monkeypatch)
    with patch.object(WebSearchService, "search", return_value=[]):
        out = process_message("how to calibrate a flux capacitor", _ctx())
    assert out["rule_id"] == "daily_howto_unavailable"
    assert out["response"].count("I could not look up steps") == 1
    assert "\n1. " not in out["response"]
    assert process_presentation("how to calibrate a flux capacitor",
                                out, context=_ctx()) is None


def test_r6_steps_never_built_from_failure_sentences():
    """Structuring a process-steps answer whose only sentences are
    lookup-failure text produces no Steps block."""
    text = ("I could not look up steps for that right now, and no web results came back. "
            "Please try again in a moment, or rephrase with the exact task.")
    blocks = structure(text, {}, {"content_type": "process_steps", "topic": "how to make tea"},
                       sources=[])
    assert not any(b.get("kind") == "steps" for b in blocks)


def test_r6_builtin_howto_matching():
    """Content-noun matching: 'a cup of tea' and 'brew coffee'
    match; an unknown task does not."""
    from services.web_answer import builtin_howto_steps
    title, steps = builtin_howto_steps("how to make a cup of tea")
    assert title == "How to make tea" and len(steps) >= 5
    title, steps = builtin_howto_steps("how do I brew coffee")
    assert title == "How to make coffee" and len(steps) >= 4
    assert builtin_howto_steps("how to calibrate a flux capacitor") is None


# ------------------------------------------------- Pinecone no-regression
def test_r6_seeded_document_still_answers(monkeypatch):
    """A seeded SQLite document still answers through the Flask
    app — the round-5 RAG path is untouched by round 6."""
    from services.chunking import chunk_text
    from services.rag_service import document_id_for
    user = "google#r6-user"
    store = app_module._sqlite_store()
    store.get_or_create_user(user)
    text = ("Photosynthesis is the process by which green plants make their own food "
            "using sunlight, water and carbon dioxide. It takes place in chloroplasts.")
    doc_id = document_id_for(user, "notes.txt")
    store.save_document(user, doc_id, "notes.txt", ".txt", text,
                        chunk_text(text, doc_id), status="indexed")
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["google_user_id"] = user
        sess["google_display_name"] = "Round Six"
        sess["onboarding_completed"] = True
    resp = client.post("/api/chat", json={"message": "What does the document say about photosynthesis?"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "Photosynthesis" in body["response"]
