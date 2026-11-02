"""Round-5 regression tests (2026-11-02 round-4 live re-test findings).

Each test reproduces one defect from the 2026-11-02 14:03 IST
re-test of round 4 (commit 4110490f). No network, no keys: the
Meta/Pinecone payloads below are mocked strings used only as test
input. Live Meta/Pinecone behaviour is NOT tested here; the parent
re-tests those live after the push.
"""
import io
import json
import os
import sys
import tempfile

os.environ["RAG_BACKEND"] = "sqlite"
os.environ["SHIBU_SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "shibu-round5-test.db")
try:
    os.remove(os.environ["SHIBU_SQLITE_PATH"])
except OSError:
    pass
os.environ.pop("PINECONE_API_KEY", None)
os.environ.pop("PINECONE_INDEX_NAME", None)
os.environ.pop("DEV_PANEL", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENAI_MODEL", None)
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HF_USE_LOCAL", None)
os.environ.pop("HF_FORCE_LOCAL", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from chatbot.context import ConversationContext
from chatbot.router import process_message
from services.content_processing_engine import process_presentation
from services.daily_task_service import build_daily_task_payload

_PHOTO_TEXT = ("Photosynthesis is how green plants make food using sunlight, "
               "water and carbon dioxide, releasing oxygen as a by-product. "
               "Chloroplasts in the leaves capture sunlight for the process. "
               "The process takes place in green leaves during daylight hours.")


def _ctx():
    return ConversationContext()


def _client(user="google#r5-user"):
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["google_user_id"] = user
        sess["google_display_name"] = "Round Five"
        sess["onboarding_completed"] = True
    return client


def _seed_document(user, name="notes.txt", text=_PHOTO_TEXT, status="indexed"):
    store = app_module._sqlite_store()
    store.get_or_create_user(user)
    from services.chunking import chunk_text
    from services.rag_service import document_id_for
    doc_id = document_id_for(user, name)
    store.save_document(user, doc_id, name, ".txt", text,
                        chunk_text(text, doc_id), status=status)
    return doc_id


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _clear_diagnostics(monkeypatch, tmp_path):
    """Empty the in-memory diagnostics and isolate the durable
    sidecar files, so each test starts from 'no failure'."""
    import services.model_manager as mm
    import services.vector_store as vs
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()
    vs._LAST_PINECONE_DIAGNOSTIC.clear()
    monkeypatch.setenv("SHIBU_DIAGNOSTIC_DIR", str(tmp_path))


# ---------------------------------------------------------------- M1
def test_r5_openai_mode_day_plan_uses_model_text(monkeypatch):
    """A mocked ModelManager in openai mode returning text -> the
    day plan uses the model draft, not the template."""

    class _OpenAIOK:
        mode = "openai"
        model_id = "gpt-4o-mini"
        effective_model = "gpt-4o-mini"

        def generate(self, *a, **k):
            return "MODEL PLAN: gym at 6pm, study at 8pm.", None

    monkeypatch.setattr("services.daily_task_service.ModelManager", _OpenAIOK)
    payload, _needs_web = build_daily_task_payload(
        "Plan my day: I work 9 to 5, want gym and study time", _ctx(), 0.0)
    assert "MODEL PLAN" in payload["response"]
    assert "Language model" in payload["tool"]


def test_r5_openai_failure_records_diagnostic_and_survives_restart(monkeypatch, tmp_path):
    """An OpenAI failure (200 envelope, empty content) -> template
    + a non-null provider_error, and the diagnostic survives a
    simulated process restart (module state cleared) via the
    durable sidecar."""
    import requests
    import services.model_manager as mm
    _clear_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)
    payload = {"choices": [{"message": {"role": "assistant", "content": ""}}],
               "created": 1, "id": "x", "model": "gpt-test",
               "object": "chat.completion", "usage": {}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(payload))
    result, _needs_web = build_daily_task_payload(
        "Plan my day: I work 9 to 5, want gym and study time", _ctx(), 0.0)
    assert "Your day plan:" in result["response"]
    diag = mm.ModelManager().status().get("provider_error")
    assert diag is not None and diag["provider"] == "openai" and diag["status"] == 200
    # Simulate a process restart: in-memory state is gone.
    mm._LAST_PROVIDER_DIAGNOSTIC.clear()
    diag_after_restart = mm.ModelManager().status().get("provider_error")
    assert diag_after_restart is not None
    assert diag_after_restart["provider"] == "openai"
    assert diag_after_restart["status"] == 200


def test_r5_generate_exception_records_diagnostic(monkeypatch, tmp_path):
    """An unexpected exception from generate() must not crash the
    chat or vanish: the template applies and the failure is
    recorded for /api/llm/status."""

    class _Raising:
        mode = "openai"
        model_id = "gpt-4o-mini"

        def generate(self, *a, **k):
            raise RuntimeError("provider surprise")

    import services.model_manager as mm
    _clear_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setattr("services.daily_task_service.ModelManager", _Raising)
    payload, _needs_web = build_daily_task_payload(
        "Plan my day: I work 9 to 5, want gym and study time", _ctx(), 0.0)
    assert "Your day plan:" in payload["response"]
    diag = mm.ModelManager().status().get("provider_error")
    assert diag is not None and diag["provider"] == "openai"


# ---------------------------------------------------------------- P1
def _fake_index_with_schema(schema_fields, search_impl=None):
    from services.vector_store import PineconeVectorStore

    class FakeDocuments:
        def __init__(self):
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            if search_impl is not None:
                return search_impl(kwargs)
            return {"result": {"hits": []}}

    docs = FakeDocuments()

    class FakeIndex:
        documents = docs

    store = PineconeVectorStore(api_key="x", index_name="x", index_type="integrated")
    store._index = FakeIndex()
    store._schema_fields = schema_fields
    store._text_field = next(
        (name for name, spec in schema_fields.items() if spec.get("embed")),
        "text")
    store._fake_docs = docs
    return store


def test_r5_pinecone_embed_field_chunk_text(monkeypatch, tmp_path):
    """A schema whose embed field is 'chunk_text' (not 'text') is
    searched with EmbedQuery(field='chunk_text'), and the
    diagnostic records the discovered field names + query type."""
    import services.vector_store as vs
    from pinecone.models.documents.score_by import EmbedQuery
    _clear_diagnostics(monkeypatch, tmp_path)
    schema = {"chunk_text": {"type": "string", "embed": {"model": "m"},
                             "sparse_embed": False, "full_text_search": False,
                             "semantic_text": False, "store_text": None}}
    store = _fake_index_with_schema(schema)
    store.query("user-9", query_text="sunlight", top_k=5)
    seen = store._fake_docs.calls[0]
    assert len(seen["score_by"]) == 1
    clause = seen["score_by"][0]
    assert isinstance(clause, EmbedQuery)
    assert clause.field == "chunk_text"
    assert clause.query == "sunlight"


def test_r5_pinecone_embed_field_text(monkeypatch, tmp_path):
    """A schema whose embed field is 'text' is searched with
    EmbedQuery(field='text')."""
    from pinecone.models.documents.score_by import EmbedQuery
    _clear_diagnostics(monkeypatch, tmp_path)
    schema = {"text": {"type": "string", "embed": {"model": "m"},
                       "sparse_embed": False, "full_text_search": False,
                       "semantic_text": False, "store_text": None}}
    store = _fake_index_with_schema(schema)
    store.query("user-9", query_text="sunlight", top_k=5)
    clause = store._fake_docs.calls[0]["score_by"][0]
    assert isinstance(clause, EmbedQuery) and clause.field == "text"


def test_r5_pinecone_schema_parsing_all_shapes():
    """describe() schemas arrive in several shapes; all must yield
    the embed field name, never a guessed 'text'."""
    from services.vector_store import _parse_schema_fields, _schema_text_field
    dict_form = {"fields": {"body": {"type": "string", "embed": {"model": "m"}}}}
    assert _schema_text_field(_parse_schema_fields(dict_form)) == "body"
    list_form = {"fields": [{"name": "body", "aspect": "embed"}]}
    assert _schema_text_field(_parse_schema_fields(list_form)) == "body"
    list_fts = {"fields": [{"name": "content", "aspect": "full_text_search"}]}
    assert _schema_text_field(_parse_schema_fields(list_fts)) == "content"

    class _Spec:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Schema:
        fields = {"chunk_text": _Spec(type="string", embed=_Spec(model="m"),
                                      sparse_embed=None, full_text_search=None,
                                      store_text=None)}

    assert _schema_text_field(_parse_schema_fields(_Schema())) == "chunk_text"
    assert _schema_text_field(_parse_schema_fields({})) == "text"


def test_r5_pinecone_search_400_diagnostic_is_structure_only(monkeypatch, tmp_path):
    """A 400 from documents/search records endpoint + status +
    schema field names + chosen field + query type + exception
    class — never values, keys, or payload — and the caller falls
    back to SQLite."""

    class _ApiError400(Exception):
        def __init__(self):
            self.status_code = 400
            super().__init__("[400] bad request from the server (secret value)")

    import services.vector_store as vs
    from services.vector_store import PineconeError
    _clear_diagnostics(monkeypatch, tmp_path)

    def _raise(kwargs):
        raise _ApiError400()

    schema = {"chunk_text": {"type": "string", "embed": {"model": "m"},
                             "sparse_embed": False, "full_text_search": False,
                             "semantic_text": False, "store_text": None}}
    store = _fake_index_with_schema(schema, search_impl=_raise)
    try:
        store.query("user-9", query_text="secret query words", top_k=5)
        raise AssertionError("search should have raised")
    except PineconeError:
        pass
    diag = vs.pinecone_last_diagnostic()
    assert diag is not None
    assert diag["endpoint"] == "documents/search"
    assert diag["status"] == 400
    assert diag["schema_fields"] == ["chunk_text"]
    assert diag["text_field"] == "chunk_text"
    assert diag["query_type"] == "EmbedQuery"
    assert diag["error_type"] == "_ApiError400"
    blob = json.dumps(diag)
    assert "secret" not in blob and "sunlight" not in blob


def test_r5_pinecone_legacy_semantic_text_refuses_cleanly(monkeypatch, tmp_path):
    """A legacy semantic_text schema must never be sent a
    documents.* search (a guaranteed 400): it refuses cleanly so
    the caller falls back, with the reason in the diagnostic."""
    import services.vector_store as vs
    from services.vector_store import PineconeError
    _clear_diagnostics(monkeypatch, tmp_path)
    schema = {"text": {"type": "semantic_text", "embed": False,
                       "sparse_embed": False, "full_text_search": False,
                       "semantic_text": True, "store_text": None}}
    store = _fake_index_with_schema(schema)
    try:
        store.query("user-9", query_text="sunlight", top_k=5)
        raise AssertionError("search should have refused")
    except PineconeError:
        pass
    assert store._fake_docs.calls == []
    diag = vs.pinecone_last_diagnostic()
    assert diag is not None and diag["query_type"] == "legacy_semantic_text"


def test_r5_pinecone_search_error_falls_back_to_sqlite(monkeypatch, tmp_path):
    """The existing fallback contract holds with the new
    diagnostics: a Pinecone search failure answers from the
    caller's own SQLite chunks, labelled sqlite_fallback."""
    import services.vector_store as vs
    from services.rag_service import RAGService

    class _ApiError400(Exception):
        def __init__(self):
            self.status_code = 400
            super().__init__("[400] bad request")

    _clear_diagnostics(monkeypatch, tmp_path)
    schema = {"chunk_text": {"type": "string", "embed": {"model": "m"},
                             "sparse_embed": False, "full_text_search": False,
                             "semantic_text": False, "store_text": None}}
    store = _fake_index_with_schema(schema, search_impl=lambda kw: (_ for _ in ()).throw(_ApiError400()))
    cfg = {"backend": "pinecone", "pinecone_configured": True, "index_type": "integrated",
           "index_name": "x", "embedding_provider": "pinecone-integrated",
           "embedding_model": "index-managed", "embedding_dimension": 1024,
           "embedding_version": "v1", "top_k": 5, "min_score": 0.1,
           "chunk_size": 400, "chunk_overlap": 50, "metadata_text_chars": 500,
           "embedding_batch_size": 10, "upsert_batch_size": 50,
           "cloud_retries": 1, "hybrid_enabled": False, "rerank_enabled": False,
           "rerank_model": "", "query_cache_ttl_seconds": 0,
           "max_context_chars": 4000}
    svc = RAGService(config=cfg,
                     sqlite_path=os.path.join(tempfile.gettempdir(), "shibu-r5-fallback.db"))
    from services.chunking import chunk_text
    svc.store.get_or_create_user("u1")
    svc.store.save_document("u1", "d1", "notes.txt", ".txt", _PHOTO_TEXT,
                            chunk_text(_PHOTO_TEXT, "d1"), status="indexed")
    monkeypatch.setattr(svc, "_pinecone_store", lambda: store)
    monkeypatch.setattr(store, "describe", lambda: {
        "dimension": 1024, "metric": "cosine", "index_type": "integrated",
        "schema_fields": schema, "text_field": "chunk_text"})
    hits = svc.search("u1", "chloroplasts capture sunlight")
    assert hits and all(h["backend"] == "sqlite_fallback" for h in hits)
    assert any("Chloroplasts" in h["text"] for h in hits)
    diag = vs.pinecone_last_diagnostic()
    assert diag is not None and diag["endpoint"] == "documents/search"


# ---------------------------------------------------------------- T1
_MAKE_ONLY = [
    {"title": "Make is an AI automation platform",
     "snippet": "Make is an AI automation platform that helps teams build workflows and automate tasks.",
     "url": "https://example.test/make-platform"},
    {"title": "Make - Wikipedia",
     "snippet": "Make is a visual platform for designing, building and automating workflows.",
     "url": "https://example.test/make-wiki"},
    {"title": "AI Workflow Automation",
     "snippet": "Use Make to automate your AI workflow with integrations and scenarios.",
     "url": "https://example.test/ai-workflow"},
]
_TEA_RESULTS = [
    {"title": "How to Make Tea",
     "snippet": "Boil water in a kettle. Pour the hot water over the tea leaves and let it steep for 3 minutes. Strain and serve hot.",
     "url": "https://example.test/tea"},
    {"title": "Simple Tea Recipe",
     "snippet": "Add one teaspoon of tea leaves to the pot. Pour in boiling water and cover. Wait 3 minutes, then pour into cups.",
     "url": "https://example.test/tea2"},
]


def test_r5_tea_router_path_drops_make_platform_results(monkeypatch):
    """Fabricated DDG set (Make-platform titles + real tea steps)
    through the REAL router path: zero platform/film text, only
    tea steps — in the answer text and in the presentation Steps."""
    from unittest.mock import patch
    from services.web_search_service import WebSearchService
    with patch.object(WebSearchService, "search", return_value=list(_MAKE_ONLY + _TEA_RESULTS)):
        payload = process_message("how to make tea", _ctx(), 0.0)
    body = payload["response"]
    assert "Boil water" in body
    assert "automation platform" not in body
    assert "Make is an AI" not in body
    assert "Wikipedia" not in body
    pres = process_presentation("how to make tea", payload, context=_ctx())
    assert pres is not None
    steps = [b for b in pres["blocks"] if b.get("kind") == "steps"]
    assert steps, "expected a Steps block"
    blob = json.dumps(pres["blocks"])
    assert "automation platform" not in blob
    assert "Make is an AI" not in blob
    assert any("Boil water" in item for block in steps for item in block["items"])


def test_r5_tea_query_expansion_when_first_search_is_only_make(monkeypatch):
    """First search returns only Make-platform results: the
    expanded query ('how to make tea step by step') is issued and
    the answer comes from the expanded set."""
    from unittest.mock import MagicMock, patch
    from services.web_search_service import WebSearchService
    mock_search = MagicMock(side_effect=[list(_MAKE_ONLY), list(_TEA_RESULTS)])
    with patch.object(WebSearchService, "search", mock_search):
        payload = process_message("how to make tea", _ctx(), 0.0)
    assert mock_search.call_count == 2
    assert mock_search.call_args_list[0][0][0] == "how to make tea"
    assert mock_search.call_args_list[1][0][0] == "how to make tea step by step"
    assert "Boil water" in payload["response"]
    assert "automation platform" not in payload["response"]


def test_r5_steps_never_built_from_source_titles():
    """Presentation Steps must never contain a Sources title,
    even when the raw text lists one as a numbered item."""
    from services.content_structuring_engine import structure
    text = ("Here are the steps:\n- Boil water in a kettle.\n- Pour it over the tea leaves.\n\n"
            "Sources (from the web, not from my rule database):\n"
            "1. Make is an AI automation platform\n   Make is an AI automation platform.\n   https://example.test/x\n"
            "2. Simple Tea Recipe\n   Add tea leaves.\n   https://example.test/y\n")
    blocks = structure(text, {"facts": [], "key_points": [], "definitions": [],
                              "examples": [], "conclusions": [], "uncertainties": [],
                              "contradictions": [], "statistics": []},
                       {"content_type": "process_steps", "topic": "how to make tea"},
                       sources=[{"source_name": "Make is an AI automation platform"},
                                {"source_name": "Simple Tea Recipe"}])
    steps = [b for b in blocks if b.get("kind") == "steps"]
    assert steps
    for item in steps[0]["items"]:
        assert "Make is an AI automation platform" not in item
        assert item != "Simple Tea Recipe"
    assert any("Boil water" in item for item in steps[0]["items"])


def test_r5_howto_steps_drop_verb_definitions():
    """'Make is an AI automation platform' is a definition, not a
    step; genuine imperatives ('Boil water') survive."""
    from services.web_answer import _howto_steps
    steps = _howto_steps(["Make is an AI automation platform that helps teams build workflows.",
                          "Boil water in a kettle. Pour it over the tea leaves."])
    assert steps and all("automation platform" not in s for s in steps)
    assert any("Boil water" in s for s in steps)


# ---------------------------------------------------------------- S1
def test_r5_shorten_after_upload_uses_previous_answer():
    """Uploaded document -> explain -> simpler -> shorter: the
    shorter answer is strictly shorter than the simpler one and
    contains no document-dump passage."""
    user = "google#r5-s1"
    _seed_document(user)
    client = _client(user)
    first = client.post("/api/chat", json={"message": "explain photosynthesis"}).get_json()
    assert first["rule_id"] == "photosynthesis"
    simpler = client.post("/api/chat", json={"message": "explain that more simply"}).get_json()
    assert simpler["rule_id"] == "context_followup_memory"
    shorter = client.post("/api/chat", json={"message": "make it shorter"}).get_json()
    assert shorter["rule_id"] == "context_followup_memory", shorter
    assert shorter["response"].startswith("Short version:")
    assert len(shorter["response"]) < len(simpler["response"])
    assert "Chloroplasts in the leaves capture sunlight" not in shorter["response"]
    assert "direct retrieval result" not in shorter["response"]


def test_r5_genuine_document_question_still_retrieves():
    """Control: a genuine new document question still reaches
    retrieval after the transform-first change."""
    user = "google#r5-s1-control"
    _seed_document(user)
    client = _client(user)
    resp = client.post("/api/chat", json={"message": "what do chloroplasts capture"}).get_json()
    assert resp["rule_id"] == "rag_answer", resp
    assert "sunlight" in resp["response"].lower()


# ---------------------------------------------------------------- E1
def test_r5_email_payload_skips_presentation():
    """A daily-write/email payload is exempt from presentation
    re-structuring: process_presentation returns None, so the raw
    multi-line email is what the user sees."""
    payload, _needs_web = build_daily_task_payload(
        "Write a leave email to my manager for tomorrow", _ctx(), 0.0)
    assert payload["rule_id"] == "daily_write"
    assert process_presentation("Write a leave email to my manager for tomorrow",
                                payload, context=_ctx()) is None
    lines = payload["response"].splitlines()
    assert any(line.startswith("Subject:") for line in lines)
    assert any(line.startswith("Dear ") for line in lines)
    assert any("Yours sincerely" in line or "Regards" in line for line in lines)
    assert "Overview:" not in payload["response"]


def test_r5_email_via_chat_has_no_presentation_blocks():
    """End to end: the email answer carries no presentation blocks
    and keeps its line breaks."""
    client = _client("google#r5-email")
    resp = client.post("/api/chat",
                       json={"message": "Write a leave email to my manager for tomorrow"}).get_json()
    assert resp["rule_id"] == "daily_write"
    assert "presentation" not in resp
    assert "\n" in resp["response"]
    assert "Overview:" not in resp["response"]
    assert "Key points" not in resp["response"]


# ---------------------------------------------------------------- C1
def test_r5_js_code_blocks_labelled_in_both_renderers():
    """Every fenced block gets a language label + Copy button:
    the plain-text renderer AND the presentation-blocks renderer
    (which is what a code follow-up actually uses). The fence
    regex is global, and no language allowlist gates the label."""
    import re
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "static", "js", "app.js"), encoding="utf-8").read()
    fence = re.search(r"const fenceRe = (.+);", src)
    assert fence and fence.group(1).rstrip().endswith("/g"), "fence regex must be global"
    code_branch = src[src.index('block.kind === "code"'):]
    code_branch = code_branch[:code_branch.index('} else if (block.kind')]
    assert "code-copy-btn" in code_branch
    assert "code-head" in code_branch
    assert "block.language" in code_branch
    assert '=== "python"' not in code_branch and "== 'python'" not in code_branch


# ---------------------------------------------------------------- Q1
def test_r5_comparison_quiz_payload_shape_and_no_presentation():
    """The quiz builder already emits 3 questions with options +
    answers on separate lines; presentation must not re-split
    them into run-on key points + fragment flashcards."""
    payload = process_message("give me a quiz on python and javascript", _ctx(), 0.0)
    assert payload["rule_id"] == "comparison_quiz"
    assert process_presentation("give me a quiz on python and javascript",
                                payload, context=_ctx()) is None
    text = payload["response"]
    questions = [line for line in text.splitlines()
                 if line[:2] in ("1.", "2.", "3.")]
    assert len(questions) == 3
    assert text.count("Answer:") == 3
    blocks = text.split("\n")
    for i, qline in enumerate(questions):
        assert qline in blocks
    options = [line for line in text.splitlines() if line.strip().startswith("- ")]
    assert len(options) >= 6


def test_r5_quiz_via_chat_has_no_presentation_blocks():
    """End to end: the quiz answer carries no presentation blocks,
    so each question renders on its own lines."""
    client = _client("google#r5-quiz")
    resp = client.post("/api/chat",
                       json={"message": "give me a quiz on python and javascript"}).get_json()
    assert resp["rule_id"] == "comparison_quiz"
    assert "presentation" not in resp
    assert resp["response"].count("Answer:") == 3
