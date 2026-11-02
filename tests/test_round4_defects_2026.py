"""Round-4 regression tests (2026-11-02 round-3 live re-test findings).

Each test reproduces one defect from the 2026-11-02 02:45 IST
re-test of round 3 (commit c58fc477). No network, no keys: the
Meta payloads below are mocked strings used only as test input.
Live Pinecone/Meta behaviour is NOT tested here; the parent
re-tests those live after the push.
"""
import io
import json
import os
import sys
import tempfile

os.environ["RAG_BACKEND"] = "sqlite"
os.environ["SHIBU_SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "shibu-round4-test.db")
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
from services.meta_service import MetaService
from services.model_manager import ModelManager

FORBIDDEN = ("could not draft", "not available", "Pinecone", "400",
             "/api/", "X-Pinecone", "language model", "unexpected format")

_PHOTO_TEXT = ("Photosynthesis is how green plants make food using sunlight, "
               "water and carbon dioxide, releasing oxygen as a by-product. "
               "Chloroplasts in the leaves capture sunlight for the process.")


def _ctx():
    return ConversationContext()


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _meta_post(payload):
    import requests
    return lambda *a, **k: _Resp(payload)


# ---------------------------------------------------------------- M1
def test_r4_meta_content_parts_output_text_skips_reasoning(monkeypatch):
    """Content as parts: output_text is read, reasoning/refusal
    parts are skipped, never leaked into the answer."""
    import requests
    payload = {"choices": [{"message": {"content": [
        {"type": "reasoning", "text": "SECRET REASONING"},
        {"type": "output_text", "text": "Visible answer"},
        {"type": "refusal", "text": "SECRET REFUSAL"}]}}]}
    monkeypatch.setattr(requests, "post", _meta_post(payload))
    svc = MetaService(api_key="k", model="m")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Visible answer"
    assert err is None
    assert "SECRET" not in (text or "")


def test_r4_meta_content_parts_input_text(monkeypatch):
    import requests
    payload = {"choices": [{"message": {"content": [
        {"type": "input_text", "text": "Part one "},
        {"type": "text", "text": "part two"}]}}]}
    monkeypatch.setattr(requests, "post", _meta_post(payload))
    svc = MetaService(api_key="k", model="m")
    text, _err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Part one part two"


def test_r4_meta_content_null_falls_back_to_message_text(monkeypatch):
    import requests
    payload = {"choices": [{"message": {"content": None, "text": "Fallback answer"}}]}
    monkeypatch.setattr(requests, "post", _meta_post(payload))
    svc = MetaService(api_key="k", model="m")
    text, _err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Fallback answer"


def test_r4_meta_choice_without_message_uses_text_field(monkeypatch):
    import requests
    payload = {"choices": [{"text": "Choice level text"}]}
    monkeypatch.setattr(requests, "post", _meta_post(payload))
    svc = MetaService(api_key="k", model="m")
    text, _err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Choice level text"


def test_r4_meta_delta_fragments_are_joined(monkeypatch):
    import requests
    payload = {"choices": [{"delta": {"content": [
        {"type": "text", "text": "Hel"},
        {"type": "text", "text": "lo"}]}, "finish_reason": "stop"}]}
    monkeypatch.setattr(requests, "post", _meta_post(payload))
    svc = MetaService(api_key="k", model="m")
    text, _err = svc.generate([{"role": "user", "content": "hi"}])
    assert text == "Hello"


def test_r4_meta_empty_content_is_not_success(monkeypatch):
    """An empty-string content must not count as a generated
    answer: fallthrough, not a blank reply."""
    import requests
    payload = {"choices": [{"message": {"content": ""}}]}
    monkeypatch.setattr(requests, "post", _meta_post(payload))
    svc = MetaService(api_key="k", model="m")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert err


def test_r4_meta_diagnostic_records_choice_structure_without_values(monkeypatch):
    """A response whose part text lives under an unhandled key
    still fails, but the diagnostic proves the structure: part
    type output_text, content a list — and no payload values."""
    import requests
    payload = {"choices": [{"message": {"content": [
        {"type": "output_text", "value": "SECRET VALUE TEXT"}]}}],
               "usage": {"tokens": 7}}
    monkeypatch.setattr(requests, "post", _meta_post(payload))
    monkeypatch.setenv("META_API_KEY", "k")
    monkeypatch.setenv("META_MODEL", "m")
    manager = ModelManager()
    text, err = manager.generate("hello")
    assert text is None
    status = ModelManager().status()
    diag = status["provider_error"]
    assert diag["status"] == 200
    assert diag["response_keys"] == ["choices", "usage"]
    assert diag["choice_structure"] == {
        "choice_keys": ["message"],
        "message_keys": ["content"],
        "content_type": "list",
        "part_types": ["output_text"],
    }
    dumped = json.dumps(diag)
    assert "SECRET VALUE TEXT" not in dumped
    assert "tokens" not in dumped


# ---------------------------------------------------------------- P1
def test_r4_pinecone_schema_discovery_uses_embed_field(monkeypatch):
    """describe() reads the index schema: the string field declared
    with `embed` carries the chunk text, and search uses an EmbedQuery
    on that field — the field name is the schema's, never assumed."""
    from pinecone.models.documents.score_by import EmbedQuery
    from services.vector_store import PineconeVectorStore

    seen = {}

    class FakeDocuments:
        def upsert(self, namespace, documents):
            seen["upsert"] = documents

        def search(self, namespace, score_by, top_k, include_fields, filter=None):
            seen["search"] = (namespace, score_by, top_k)
            return {"result": {"hits": []}}

    class FakeIndex:
        documents = FakeDocuments()

    class FakeIndexes:
        def describe(self, name):
            return {"name": name, "host": "h.pinecone.io", "metric": "cosine",
                    "schema": {"fields": {
                        "chunk_text": {"type": "string", "embed": True},
                        "document_id": {"type": "string"}}}}

    class FakeClient:
        indexes = FakeIndexes()

        def index(self, name):
            return FakeIndex()

    store = PineconeVectorStore(api_key="x", index_name="shibuai", index_type="integrated")
    store._client = FakeClient()
    info = store.describe()
    assert info["index_type"] == "integrated"
    assert info["text_field"] == "chunk_text"
    store.upsert("user-1", [{"id": "c1", "text": "sunlight sentence",
                             "metadata": {"document_id": "d1", "chunk_index": 0}}])
    assert "chunk_text" in seen["upsert"][0]
    assert "text" not in seen["upsert"][0]
    assert seen["upsert"][0]["chunk_text"] == "sunlight sentence"
    store.query("user-1", query_text="sunlight", top_k=3)
    _ns, score_by, _k = seen["search"]
    assert len(score_by) == 1
    assert isinstance(score_by[0], EmbedQuery)
    assert score_by[0].field == "chunk_text"
    assert score_by[0].query == "sunlight"


def test_r4_pinecone_failure_records_safe_diagnostic(monkeypatch):
    """A failed describe records endpoint + HTTP status only:
    no values, no key, no payload in /api/rag/status."""
    from services.vector_store import PineconeVectorStore, pinecone_last_diagnostic

    class _ApiErr(Exception):
        def __init__(self):
            self.status_code = 503
            super().__init__("service unavailable")

    class FakeIndexes:
        def describe(self, name):
            raise _ApiErr()

    class FakeClient:
        indexes = FakeIndexes()

        def describe_index(self, name):
            raise _ApiErr()

    store = PineconeVectorStore(api_key="x", index_name="shibuai", index_type="integrated")
    store._client = FakeClient()
    try:
        store.describe()
    except Exception:
        pass
    diag = pinecone_last_diagnostic()
    assert diag == {"endpoint": "describe", "status": 503}


def _pinecone_cfg(tmpdb):
    from services.rag_service import get_rag_config
    cfg = dict(get_rag_config())
    cfg["backend"] = "pinecone"
    cfg["index_type"] = "integrated"
    cfg["pinecone_configured"] = True
    cfg["index_name"] = "x"
    return cfg


def _seed_sqlite(store, user, text=_PHOTO_TEXT):
    from services.chunking import chunk_text
    from services.rag_service import document_id_for
    store.get_or_create_user(user)
    doc_id = document_id_for(user, "notes.txt")
    store.save_document(user, doc_id, "notes.txt", ".txt", text,
                        chunk_text(text, doc_id), status="failed")
    return doc_id


class _FailingPinecone:
    def describe(self):
        return {"dimension": 1024, "metric": "cosine", "index_type": "integrated"}

    def query(self, *a, **k):
        raise RuntimeError("Pinecone documents search unavailable")


def test_r4_pinecone_search_falls_back_to_sqlite(monkeypatch):
    """Pinecone search raising must not end the search: the local
    SQLite chunks answer the query, labelled sqlite_fallback."""
    from services.rag_service import RAGService
    from services.sqlite_store import SQLiteStore

    db_path = os.path.join(tempfile.gettempdir(), "shibu-r4-fallback.db")
    try:
        os.remove(db_path)
    except OSError:
        pass
    store = SQLiteStore(path=db_path)
    user = "google#r4-fallback"
    _seed_sqlite(store, user)
    svc = RAGService(config=_pinecone_cfg(db_path), sqlite_store=store)
    monkeypatch.setattr(svc, "_pinecone_store", lambda: _FailingPinecone())
    hits = svc.search(user, "What is photosynthesis?")
    assert hits, "SQLite fallback returned no hits"
    assert any("Photosynthesis" in h["text"] for h in hits)
    assert all(h["backend"] == "sqlite_fallback" for h in hits)


def test_r4_pinecone_and_sqlite_fail_means_not_covered(monkeypatch):
    """Pinecone fails AND SQLite has no matching chunk: the answer
    is evidence_found False with no error string and no
    upload-first advice — the document exists, it just does not
    cover the question. (The stored text mentions chloroplasts,
    not chlorophyll: 'not covered' is the correct answer.)"""
    from services.rag_service import RAGService
    from services.sqlite_store import SQLiteStore

    db_path = os.path.join(tempfile.gettempdir(), "shibu-r4-bothfail.db")
    try:
        os.remove(db_path)
    except OSError:
        pass
    store = SQLiteStore(path=db_path)
    user = "google#r4-bothfail"
    _seed_sqlite(store, user)
    svc = RAGService(config=_pinecone_cfg(db_path), sqlite_store=store)
    monkeypatch.setattr(svc, "_pinecone_store", lambda: _FailingPinecone())
    result = svc.rag_answer(user, "What is chlorophyll?")
    assert result["evidence_found"] is False
    assert result["error"] == ""
    assert "Pinecone" not in json.dumps(result)


def test_r4_rag_status_exposes_pinecone_error_field(monkeypatch, tmp_path):
    """pinecone_error is present on /api/rag/status in every
    backend mode. With no failure recorded (isolated durable
    sidecar + cleared process memory) it is None; when set it
    carries endpoint + status plus structure-only context (field
    names, query type, exception class) — never values or keys."""
    import services.vector_store as vs
    vs._LAST_PINECONE_DIAGNOSTIC.clear()
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))
    data = app_module.app.test_client().get("/api/rag/status").get_json()
    assert "pinecone_error" in data
    assert data["pinecone_error"] is None
    assert "PINECONE_API_KEY" not in str(data)


# ---------------------------------------------------------------- T1
_TEA_RESULTS = [
    {"title": "How to Make a Killing (2023 film)", "url": "https://example.invalid/film",
     "snippet": "How to Make a Killing is a thriller film directed by someone, "
                "starring actors. Box office $21.6 million against a production budget of $10 million."},
    {"title": "How to Make Tea: The Science Behind the Leaf (book)", "url": "https://example.invalid/book",
     "snippet": "How to Make Tea: The Science Behind the Leaf is a book by an author, "
                "a bestseller published by a publisher."},
    {"title": "How to Make Tea - Brew Guide", "url": "https://example.invalid/guide",
     "snippet": "Boil water in a kettle. Pour the hot water over the tea leaves. "
                "Steep the tea for 3 minutes."},
]


def test_r4_tea_through_real_router_has_no_film_book_budget(monkeypatch):
    """The fabricated set (film + book + budget + 2 real steps)
    through the REAL router path must yield only the steps —
    no film/book/budget text in the answer or its presentation."""
    import chatbot.router as router
    monkeypatch.setattr(router.search_service, "search", lambda query: list(_TEA_RESULTS))
    ctx = _ctx()
    payload = process_message("how to make tea", ctx)
    text = payload["response"]
    for forbidden in ("Make a Killing", "Science Behind the Leaf", "thriller",
                      "box office", "budget", "million", "bestseller", "published"):
        assert forbidden.lower() not in text.lower(), forbidden
    assert "Boil water" in text
    assert "Steep the tea" in text
    presentation = process_presentation("how to make tea", payload, ctx)
    assert presentation is not None
    dumped = json.dumps(presentation)
    for forbidden in ("Make a Killing", "Science Behind the Leaf", "thriller",
                      "box office", "budget", "million", "bestseller"):
        assert forbidden.lower() not in dumped.lower(), forbidden
    steps_blocks = [b for b in presentation["blocks"] if b.get("kind") == "steps"]
    assert steps_blocks, "no Steps block in the tea presentation"
    steps_text = " ".join(str(i) for b in steps_blocks for i in b.get("items", []))
    assert "Boil water" in steps_text and "Steep the tea" in steps_text


def test_r4_tea_second_query_does_not_resurrect_garbled_text(monkeypatch):
    """A second identical query must not resurrect stale cached
    garbled text: the filtered answer is served both times."""
    import chatbot.router as router
    monkeypatch.setattr(router.search_service, "search", lambda query: list(_TEA_RESULTS))
    ctx = _ctx()
    first = process_message("how to make tea", ctx)["response"]
    second = process_message("how to make tea", ctx)["response"]
    for text in (first, second):
        assert "Make a Killing" not in text
        assert "Science Behind the Leaf" not in text
        assert "Boil water" in text


# ---------------------------------------------------------------- B1
def test_r4_multipart_then_phrasing():
    payload = process_message(
        "Explain black holes to a 10 year old, then give 3 fun facts, "
        "then ask me one question about it", _ctx())
    text = payload["response"]
    assert text.count("Fun fact") == 3
    assert text.rstrip().endswith("?")
    assert text.index("Fun fact 1") < text.index("Fun fact 3") < text.rindex("?")


def test_r4_multipart_comma_and_phrasing():
    """The re-test phrasing: comma/and separators must yield the
    same explanation + 3 labelled facts + closing question."""
    payload = process_message(
        "Explain black holes, give me 3 interesting facts about black holes, "
        "and ask me a question about black holes", _ctx())
    text = payload["response"]
    assert "black hole" in text.lower()
    assert text.count("Fun fact") == 3
    assert "Fun fact 1" in text and "Fun fact 3" in text
    assert text.rstrip().endswith("?")
    assert text.index("Fun fact 1") < text.index("Fun fact 3") < text.rindex("?")
    assert "dictionary" not in text.lower()


# ---------------------------------------------------------------- E1
def test_r4_email_template_is_fluent_multiline(monkeypatch):
    """Subject, greeting, body and closing each on their own
    lines; 'leave tomorrow', never 'on tomorrow'; the signature
    is not a bullet."""
    class _Failing:
        mode = "meta"
        model_id = "m"

        def generate(self, *a, **k):
            return None, "The Meta-compatible API returned a response in an unexpected format."

    monkeypatch.setattr("services.daily_task_service.ModelManager", _Failing)
    from services.daily_task_service import build_daily_task_payload
    payload, _needs_web = build_daily_task_payload(
        "Write a leave email to my manager for tomorrow", _ctx(), 0.0)
    text = payload["response"]
    assert "on tomorrow" not in text.lower()
    assert "leave tomorrow" in text.lower()
    assert "3 November 2026" not in text
    lines = [line for line in text.split("\n")]
    assert any(line.startswith("Subject:") for line in lines)
    assert any(line.startswith("Dear Manager") for line in lines)
    assert any(line.startswith("Yours sincerely") for line in lines)
    assert any(line.strip() == "[Your name]" for line in lines)
    assert "- [Your name]" not in text and "• [Your name]" not in text
    for forbidden in FORBIDDEN:
        assert forbidden not in text


# ---------------------------------------------------------------- Q1
def test_r4_quiz_fragment_never_in_key_points():
    ctx = _ctx()
    process_message("Compare Python and JavaScript", ctx)
    quiz = process_message("Give me a quiz on that comparison", ctx)
    assert quiz["rule_id"] == "comparison_quiz"
    # Round 5 (Q1): quiz payloads are exempt from presentation, so
    # the quiz text can never be re-split into key-points blocks
    # or fragment flashcards — it renders exactly as built.
    presentation = process_presentation("Give me a quiz on that comparison", quiz, ctx)
    assert presentation is None
    assert quiz["response"].count("Quiz on the comparison:") == 1
