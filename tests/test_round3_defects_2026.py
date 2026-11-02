"""Round-3 regression tests (2026-11-02 live re-test findings).

Each test reproduces one defect from the 2026-11-02 02:19 IST
re-test of round 2 (commit 36199e32). No network, no keys: the
Pinecone error text below is a mocked string used only as test
input, never user-facing text. Live Pinecone is NOT tested here.
"""
import io
import json
import os
import sys
import tempfile

os.environ["RAG_BACKEND"] = "sqlite"
os.environ["SHIBU_SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "shibu-round3-test.db")
try:
    os.remove(os.environ["SHIBU_SQLITE_PATH"])
except OSError:
    pass
os.environ.pop("PINECONE_API_KEY", None)
os.environ.pop("PINECONE_INDEX_NAME", None)
os.environ.pop("DEV_PANEL", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from chatbot.context import ConversationContext
from chatbot.router import process_message
from services.content_processing_engine import process_presentation
from services.daily_task_service import build_daily_task_payload
from services.meta_service import MetaService
from services.model_manager import ModelManager
from services.web_answer import summarise_web_results

FORBIDDEN = ("could not draft", "not available", "Pinecone", "400",
             "/api/", "X-Pinecone", "language model", "unexpected format")

_PHOTO_TEXT = ("Photosynthesis is how green plants make food using sunlight, "
               "water and carbon dioxide, releasing oxygen as a by-product. "
               "Chloroplasts in the leaves capture sunlight for the process.")


def _ctx():
    return ConversationContext()


def _client(user="google#r3-user"):
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["google_user_id"] = user
        sess["google_display_name"] = "Round Three"
        sess["onboarding_completed"] = True
    return client


def _seed_document(user, name="shibu-test-upload.txt", text=_PHOTO_TEXT, status="failed"):
    store = app_module._sqlite_store()
    store.get_or_create_user(user)
    from services.chunking import chunk_text
    from services.rag_service import document_id_for
    doc_id = document_id_for(user, name)
    store.save_document(user, doc_id, name, ".txt", text,
                        chunk_text(text, doc_id), status=status)
    return doc_id


class _StubRAG:
    """RAG service stub: search finds nothing (optionally with an
    error), so the app's no-evidence / summary-fallback branches run."""

    def __init__(self, error=""):
        self._error = error

    def rag_answer(self, user_id, question, context=None):
        return {"answer": "", "sources": [], "backend": "sqlite",
                "evidence_found": False, "error": self._error}


# ---------------------------------------------------------------- F1
def test_r3_documents_api_upsert_failure_falls_back_gracefully(monkeypatch):
    """A failed Pinecone upsert must not lose the document: the
    text stays saved in SQLite (status failed) for the summary
    fallback, and ingest reports the failure by raising after
    saving, as the upload handler expects."""
    import pytest
    from services.rag_service import RAGService, document_id_for, get_rag_config
    from services.sqlite_store import SQLiteStore

    cfg = dict(get_rag_config())
    cfg["backend"] = "pinecone"
    cfg["index_type"] = "integrated"
    cfg["pinecone_configured"] = True
    cfg["index_name"] = "x"
    db_path = os.path.join(tempfile.gettempdir(), "shibu-r3-ingest.db")
    try:
        os.remove(db_path)
    except OSError:
        pass
    store = SQLiteStore(path=db_path)
    svc = RAGService(config=cfg, sqlite_store=store)

    def _boom(*a, **k):
        raise RuntimeError(
            "[400 INVALID_ARGUMENT] This index has a document schema, "
            "so writes must go through the documents API.")

    monkeypatch.setattr(svc, "_upsert_batches", _boom)
    monkeypatch.setattr(svc, "_pinecone_store", lambda: _FakeDescribe())
    with pytest.raises(RuntimeError):
        svc.ingest_text("google#r3-ingest", "notes.txt", _PHOTO_TEXT)
    saved = store.get_document_with_text(
        "google#r3-ingest", document_id_for("google#r3-ingest", "notes.txt"))
    assert saved is not None and _PHOTO_TEXT.split(".")[0] in saved["full_text"]
    assert saved["status"] == "failed"


class _FakeDescribe:
    def describe(self):
        return {"dimension": 1024, "metric": "cosine", "index_type": "integrated"}


def test_r3_documents_search_uses_sdk_documents_api(monkeypatch):
    """Documents-API search goes through the SDK index.documents
    namespace (pinecone>=10.1.0, API version 2026-07 handled by
    the SDK) with a TextQuery score_by clause, and parses
    result.matches hits with fields + nested metadata."""
    from pinecone.models.documents.score_by import TextQuery
    from services.vector_store import PineconeVectorStore

    seen = {}

    class FakeDocuments:
        def search(self, **kwargs):
            seen.update(kwargs)
            return {"result": {"hits": [
                {"_id": "c1", "_score": 0.95,
                 "fields": {"text": "sunlight sentence",
                            "metadata": {"document_id": "d1", "chunk_index": 2}}}]}}

    class FakeIndex:
        documents = FakeDocuments()

    store = PineconeVectorStore(api_key="x", index_name="x", index_type="integrated")
    store._index = FakeIndex()
    store._text_field = "text"
    store._schema_fields = {}
    hits = store.query("user-9", query_text="sunlight", top_k=5)
    assert seen["namespace"] == "user-9"
    assert seen["top_k"] == 5
    assert seen["include_fields"] == ["*"]
    assert len(seen["score_by"]) == 1
    assert isinstance(seen["score_by"][0], TextQuery)
    assert seen["score_by"][0].query == "sunlight"
    assert seen["score_by"][0].fields == ["text"]
    assert hits[0]["id"] == "c1" and hits[0]["text"] == "sunlight sentence"
    assert hits[0]["metadata"]["document_id"] == "d1"
    assert hits[0]["metadata"]["chunk_index"] == 2


# ---------------------------------------------------------------- F2
def test_r3_daily_task_fallbacks_never_disclose_provider_failure(monkeypatch):
    class _Failing:
        mode = "meta"
        model_id = "m"

        def generate(self, *a, **k):
            return None, "The Meta-compatible API returned a response in an unexpected format."

    monkeypatch.setattr("services.daily_task_service.ModelManager", _Failing)
    for message in ("Plan my day with gym and study",
                    "Write a leave email to my manager",
                    "Plan my week: work 9 to 5, gym 3 times, study for an exam, family dinner on Sunday"):
        payload, _needs_web = build_daily_task_payload(message, _ctx(), 0.0)
        assert payload is not None
        for forbidden in FORBIDDEN:
            assert forbidden not in payload["response"], (message, forbidden)


def test_r3_upload_failure_message_is_neutral(monkeypatch):
    """A raw provider error raised during indexing must stay in the
    log: the upload response carries only fixed neutral text."""
    raw_error = ("[400 INVALID_ARGUMENT] This index has a document schema, "
                 "so writes must go through the documents API. Use POST "
                 "/namespaces/{namespace}/documents/upsert with the "
                 "X-Pinecone-API-Version header set to 2026-07 or later.")

    class _FailingRAG:
        def ingest_text(self, *a, **k):
            raise RuntimeError(raw_error)

    monkeypatch.setattr(app_module, "_rag_service", lambda: _FailingRAG())
    client = _client("google#r3-upload")
    data = {"file": (io.BytesIO(_PHOTO_TEXT.encode()), "notes.txt")}
    resp = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Uploaded notes.txt" in body
    for forbidden in ("INVALID_ARGUMENT", "Pinecone", "400", "/namespaces/",
                      "X-Pinecone", "documents API"):
        assert forbidden not in body
    assert resp.get_json()["rag"]["error"] == \
        "Document search indexing is unavailable right now."


# ---------------------------------------------------------------- F3
def test_r3_meta_parses_every_remaining_envelope(monkeypatch):
    import requests

    def _run(body):
        class _Resp:
            status_code = 200

            def json(self):
                return body

        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
        svc = MetaService(api_key="k", model="m")
        text, err = svc.generate([{"role": "user", "content": "hi"}])
        assert err is None, body
        return text

    # content as a list of parts inside message.content
    assert _run({"choices": [{"message": {"content": [
        {"type": "text", "text": "hello parts"}]}}]}) == "hello parts"
    # choices[0].message.content as a plain list of strings
    assert _run({"choices": [{"message": {"content": ["hello list"]}}]}) == "hello list"
    # top-level data list
    assert _run({"data": [{"content": "hello data"}]}) == "hello data"
    # result / results envelopes
    assert _run({"result": {"text": "hello result"}}) == "hello result"
    assert _run({"results": [{"output_text": "hello results"}]}) == "hello results"
    # response object with text
    assert _run({"response": {"text": "hello response"}}) == "hello response"
    # streaming-style delta fragments joined
    assert _run({"choices": [{"delta": {"content": "hel"}},
                             {"delta": {"content": "lo"}}]}) == "hello"
    # bare JSON string body parsed once more
    assert _run('{"choices": [{"message": {"content": "hello string"}}]}') == "hello string"
    # nested content list inside a part
    assert _run({"output": [{"content": [{"content": [
        {"type": "text", "text": "hello nested"}]}]}]}) == "hello nested"


def test_r3_meta_diagnostic_is_status_and_keys_only(monkeypatch):
    import requests

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": {"unexpected": "shape"}}}],
                    "usage": {"tokens": 9}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    svc = MetaService(api_key="k", model="m")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "HTTP 200" in err and "response keys: choices, usage" in err
    # Keys only: the nested payload values never appear.
    assert "shape" not in err and "tokens" not in err
    assert svc.last_diagnostic["status"] == 200
    assert svc.last_diagnostic["response_keys"] == ["choices", "usage"]
    # Structure only: the dict-typed content is reported by its
    # Python type name, and the nested value ("shape") never
    # appears in the diagnostic.
    assert svc.last_diagnostic["choice_structure"] == {
        "choice_keys": ["message"], "message_keys": ["content"],
        "content_type": "dict", "part_types": None,
        "finish_reason": None, "refusal_present": False,
        "logprobs_present": False}
    assert "shape" not in str(svc.last_diagnostic)
    assert MetaService._response_keys(
        {"id": "x", "output": []}) == ["id", "output"]


def test_r3_llm_status_exposes_provider_error_without_values(monkeypatch):
    import requests

    class _Resp:
        status_code = 200

        def json(self):
            return {"output": [{"content": [{"type": "weird"}]}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    monkeypatch.setenv("META_API_KEY", "k")
    monkeypatch.setenv("META_MODEL", "m")
    manager = ModelManager()
    assert manager.mode == "meta"
    text, err = manager.generate("hello")
    assert text is None
    status = ModelManager().status()
    assert status["provider_error"]["status"] == 200
    assert status["provider_error"]["response_keys"] == ["output"]
    # Anonymous access to the status endpoint is unchanged.
    data = app_module.app.test_client().get("/api/llm/status").get_json()
    assert data["provider_error"]["status"] == 200
    assert data["provider_error"]["response_keys"] == ["output"]
    assert set(data["provider_error"]) <= {"status", "response_keys", "provider", "choice_structure"}
    # No "choices" list at all: there is no choices[0] structure
    # to report, so choice_structure is None rather than invented.
    assert data["provider_error"]["choice_structure"] is None
    assert "weird" not in json.dumps(data)
    assert "api_key" not in json.dumps(data).lower()
    assert "api_key" not in json.dumps(data).lower()


def test_r3_meta_unparseable_falls_through_to_openai(monkeypatch):
    import requests

    def _post(url, **kwargs):
        class _Resp:
            status_code = 200

            def json(self):
                if "meta" in url:
                    return {"choices": [{"message": {"content": {"weird": 1}}}]}
                return {"choices": [{"message": {"content": "openai wins"}}]}

        return _Resp()

    monkeypatch.setattr(requests, "post", _post)
    monkeypatch.setenv("META_API_KEY", "k")
    monkeypatch.setenv("META_MODEL", "m")
    monkeypatch.setenv("META_API_BASE", "https://meta.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    text, err = ModelManager().generate("hello")
    assert text == "openai wins"
    assert err is None


# ---------------------------------------------------------------- F4
def test_r3_document_question_with_docs_says_not_covered(monkeypatch):
    user = "google#r3-docs"
    _seed_document(user)
    monkeypatch.setattr(app_module, "_rag_service", lambda: _StubRAG())
    resp = _client(user).post("/api/chat", json={
        "message": "What does my document say about chlorophyll?"})
    assert resp.status_code == 200
    body = resp.get_json()["response"]
    assert "none of them cover" in body.lower()
    assert "upload a document first" not in body.lower()
    assert "not uploaded any documents" not in body.lower()


def test_r3_document_question_without_docs_mentions_upload(monkeypatch):
    user = "google#r3-nodocs"
    monkeypatch.setattr(app_module, "_rag_service", lambda: _StubRAG())
    resp = _client(user).post("/api/chat", json={
        "message": "What does my document say about chlorophyll?"})
    assert resp.status_code == 200
    body = resp.get_json()["response"]
    assert "upload" in body.lower()


def test_r3_search_error_never_claims_no_documents(monkeypatch):
    """When search itself fails, the answer must say search is
    unavailable — never 'no documents' and never 'not covered'."""
    user = "google#r3-errdocs"
    _seed_document(user)
    monkeypatch.setattr(app_module, "_rag_service", lambda: _StubRAG(error="boom"))
    resp = _client(user).post("/api/chat", json={
        "message": "What does my document say about chlorophyll?"})
    body = resp.get_json()["response"]
    assert "could not search your documents" in body.lower()
    assert "not uploaded" not in body.lower()
    assert "boom" not in body


# ---------------------------------------------------------------- F5
def test_r3_summary_uses_stored_text_when_indexing_failed(monkeypatch):
    user = "google#r3-summary"
    _seed_document(user)
    monkeypatch.setattr(app_module, "_rag_service", lambda: _StubRAG(error="index down"))
    resp = _client(user).post("/api/chat", json={
        "message": "Summarise the uploaded document"})
    assert resp.status_code == 200
    body = resp.get_json()["response"]
    assert "green plants make food" in body
    assert "shibu-test-upload.txt" in body
    assert "none of them cover" not in body.lower()


# ---------------------------------------------------------------- F6
def test_r3_week_plan_template_is_day_by_day(monkeypatch):
    class _Failing:
        mode = "meta"
        model_id = "m"

        def generate(self, *a, **k):
            return None, "boom"

    monkeypatch.setattr("services.daily_task_service.ModelManager", _Failing)
    payload, _needs_web = build_daily_task_payload(
        "Plan my week: work 9 to 5, gym 3 times, study for an exam, family dinner on Sunday",
        _ctx(), 0.0)
    text = payload["response"]
    for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"):
        assert day in text, day
    assert text.lower().count("gym") == 3
    assert "work" in text.lower() and "9am to 5pm" in text
    assert "family dinner" in text.lower()
    for forbidden in FORBIDDEN:
        assert forbidden not in text


# ---------------------------------------------------------------- F7
def test_r3_tea_steps_use_imperative_sentences_not_titles():
    results = [
        {"title": "How to Make Tea: Steeping, Brewing - Brew Buch",
         "url": "http://x/tea",
         "snippet": "How to Make Tea: Steeping, Brewing - Brew Buch. "
                    "Boil fresh water in a kettle. "
                    "Pour the hot water over the tea leaves. "
                    "The Ultimate Guide! - Brew Buch. "
                    "Steep the tea for 3 to 5 minutes. "
                    "Remove the tea bag and enjoy your tea."},
        {"title": "How to Make a Killing (film) - budget",
         "url": "http://x/film",
         "snippet": "How to Make a Killing is a film with a big budget."},
    ]
    summary = summarise_web_results("How to make tea", results)
    steps = summary["key_points"]
    assert steps, "no steps extracted"
    assert all(s[0].isupper() for s in steps)
    joined = " ".join(steps)
    for title_fragment in ("Brew Buch", "Ultimate Guide", "How to Make Tea",
                           "How to Make a Killing", "film", "budget"):
        assert title_fragment not in joined
    assert "Boil fresh water" in joined
    assert "Pour the hot water" in joined
    assert "Steep the tea" in joined


# ---------------------------------------------------------------- F8
def test_r3_simplify_shorten_chain_on_photosynthesis():
    ctx = _ctx()
    original = process_message("Explain photosynthesis", ctx)
    simpler = process_message("explain that more simply", ctx)
    shorter = process_message("make it shorter", ctx)
    assert simpler["response"] != original["response"]
    assert "It gives off oxygen." in simpler["response"]
    assert "with sunlight" in simpler["response"]
    assert "Equation" not in simpler["response"] and "->" not in simpler["response"]
    assert len(shorter["response"]) < len(simpler["response"]) < len(original["response"])
    assert shorter["response"] != original["response"].split(".")[0]


# ---------------------------------------------------------------- F9
def test_r3_reverse_string_answers_explain_each_line():
    ctx = _ctx()
    py = process_message("Write Python code to reverse a string", ctx)
    assert "```python" in py["response"]
    assert "s[::-1]" in py["response"]
    assert "- `s[::-1]`" in py["response"]
    assert py["response"].count("\n- ") >= 2

    # Same conversation: "the same" refers to the previous task.
    js = process_message("Write the same in JavaScript", ctx)
    assert "```javascript" in js["response"]
    assert ".reverse()" in js["response"]
    assert "split" in js["response"] and "join" in js["response"]
    assert js["response"].count("\n- ") >= 2


def test_r3_factorial_answer_explains_each_line():
    ctx = _ctx()
    payload = process_message("Write Python code for factorial", ctx)
    assert "```python" in payload["response"]
    assert "- `result = 1`" in payload["response"]
    assert "- `result *= i`" in payload["response"]


# ---------------------------------------------------------------- F10
def test_r3_dev_panel_chat_refusal_is_403_not_authorized(monkeypatch):
    monkeypatch.setenv("DEV_PANEL", "1")
    client = app_module.app.test_client()
    get_resp = client.get("/api/chat")
    assert get_resp.status_code == 403
    assert "not authorized" in get_resp.get_json()["error"]
    assert "Developer panel mode" in get_resp.get_json()["error"]
    post_resp = client.post("/api/chat", json={"message": "hello"})
    assert post_resp.status_code == 403
    assert "not authorized" in post_resp.get_json()["error"]


# ---------------------------------------------------------------- F11
def test_r3_comparison_quiz_has_exactly_one_quiz_block():
    ctx = _ctx()
    process_message("Compare Python and JavaScript", ctx)
    quiz = process_message("Give me a quiz on that comparison", ctx)
    assert quiz["rule_id"] == "comparison_quiz"
    # The quiz itself lives in the response text, exactly once.
    assert quiz["response"].count("Quiz on the comparison:") == 1
    # Round 5 (Q1): quiz payloads are exempt from presentation
    # re-structuring entirely — the raw text IS the rendering, so
    # no derived quiz/flashcards block can duplicate or mangle it.
    presentation = process_presentation("Give me a quiz on that comparison", quiz, ctx)
    assert presentation is None


# ---------------------------------------------------------------- F12
def test_r3_new_chat_keeps_profile_and_answers_normally():
    client = _client("google#r3-newchat")
    first = client.post("/api/chat", json={"message": "What is photosynthesis?"})
    assert first.status_code == 200
    assert "photosynthesis" in first.get_json()["response"].lower()
    cleared = client.post("/api/clear")
    assert cleared.status_code == 200
    second = client.post("/api/chat", json={"message": "What is 25% of 800?"})
    assert second.status_code == 200
    body = second.get_json()["response"]
    assert "200" in body
    assert "Choose your character" not in body
    assert "What should I call you" not in body


def test_r3_profile_syncs_back_after_sqlite_row_recreated():
    """A completed session profile survives a wiped/recreated
    SQLite row: _profile() writes the session's avatar/completed
    flags back, so the next session still skips onboarding."""
    user = "google#r3-syncback"
    store = app_module._sqlite_store()
    store.get_or_create_user(user)
    with app_module.app.test_request_context():
        from flask import session as flask_session
        flask_session["google_user_id"] = user
        flask_session["onboarding_completed"] = True
        flask_session["profile_avatar_id"] = "avatar-3"
        profile = app_module._profile()
    assert profile["onboarding_completed"] is True
    assert profile["avatar_id"] == "avatar-3"
    row = store.get_or_create_user(user)
    assert row["onboarding_completed"] == 1
    assert row["avatar_id"] == "avatar-3"
