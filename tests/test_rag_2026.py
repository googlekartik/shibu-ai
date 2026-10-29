"""Pinecone-backed RAG tests (2026) — SQLite fallback path only.

Live Pinecone is NOT tested here: PINECONE_API_KEY / INDEX are
absent in this workspace, so /api/rag/status must report live
Pinecone as blocked, not successful. That blocked result IS the
live test result for this environment. Pinecone API selection
(dense vs integrated) is unit-tested with a fake index object,
without any network call.

All other tests use a temporary SQLite file, RAG_BACKEND=sqlite,
and a logged-in Flask session (google#sub). No OpenAI/HF key is
needed: the SQLite fallback uses token-overlap retrieval, labelled
sqlite_fallback, and the LLM-off RAG path returns retrieved
passages directly.
"""
import os
import sys
import tempfile

os.environ["RAG_BACKEND"] = "sqlite"
os.environ["SHIBU_SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "shibu-rag-test.db")
# Fresh DB for this test module.
try:
    os.remove(os.environ["SHIBU_SQLITE_PATH"])
except OSError:
    pass
os.environ.pop("PINECONE_API_KEY", None)
os.environ.pop("PINECONE_INDEX_NAME", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from services.chunking import chunk_text
from services.rag_config import namespace_for_user, validate_index_info
from services.vector_store import PineconeVectorStore


def _client(user="google#rag-test-1"):
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["google_user_id"] = user
        sess["google_display_name"] = "RAG Tester"
        sess["onboarding_completed"] = True
    return client


def test_chunking_stable_ids_and_overlap():
    text = "Alpha beta gamma. " * 100
    chunks = chunk_text(text, "doc1", chunk_size=800, chunk_overlap=100)
    assert len(chunks) > 1
    assert chunks[0]["id"] == chunk_text(text, "doc1", chunk_size=800, chunk_overlap=100)[0]["id"]
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert chunk_text("", "doc1") == []


def test_namespace_is_sub_not_name():
    assert namespace_for_user("google#12345") == "user-12345"
    assert namespace_for_user("google#12345") != namespace_for_user("google#99999")


def test_dimension_mismatch_is_migration_not_silent_fix():
    cfg = {"pinecone_configured": True, "index_type": "dense",
           "embedding_model": "text-embedding-3-small", "embedding_dimension": 1536}
    ok, errors = validate_index_info({"dimension": 1024, "metric": "cosine", "index_type": "dense"}, cfg)
    assert ok is False
    assert any("Migration required" in e for e in errors)
    ok2, _ = validate_index_info({"dimension": 1536, "metric": "cosine", "index_type": "dense"}, cfg)
    assert ok2 is True


def test_pinecone_api_selection_by_index_type():
    """Fake index objects prove dense uses upsert/query and
    integrated uses upsert_records/search, without network."""
    calls = []

    class FakeIndex:
        def upsert(self, vectors, namespace):
            calls.append(("upsert", namespace, len(vectors)))
        def upsert_records(self, records, namespace):
            calls.append(("upsert_records", namespace, len(records)))
        def query(self, vector, top_k, namespace, include_metadata, filter=None):
            calls.append(("query", namespace))
            return {"matches": []}
        def search(self, namespace, inputs, top_k, filter=None):
            calls.append(("search", namespace))
            return {"result": {"hits": []}}

    dense = PineconeVectorStore(api_key="x", index_name="x", index_type="dense")
    dense._index = FakeIndex()
    dense.upsert("user-1", [{"id": "a", "values": [0.1, 0.2], "metadata": {}}])
    dense.query("user-1", vector=[0.1, 0.2], top_k=3)
    assert ("upsert", "user-1", 1) in calls and ("query", "user-1") in calls

    integ = PineconeVectorStore(api_key="x", index_name="x", index_type="integrated")
    integ._index = FakeIndex()
    integ.upsert("user-1", [{"id": "b", "text": "hello world"}])
    integ.query("user-1", query_text="hello", top_k=3)
    assert ("upsert_records", "user-1", 1) in calls and ("search", "user-1") in calls


def test_rag_status_reports_live_pinecone_blocked_not_success():
    client = _client()
    data = client.get("/api/rag/status").get_json()
    assert data["backend"] == "sqlite"
    assert data["pinecone_configured"] is False
    assert data["live_pinecone"] in ("blocked", "not_used")
    assert "PINECONE_API_KEY" not in str(data)


def test_ingest_search_delete_reindex_and_isolation():
    alice = _client("google#alice-rag")
    bob = _client("google#bob-rag")
    # Ingest as Alice.
    res = alice.post("/api/rag/ingest", json={
        "text": "The Shibu River flows through Patna and is famous for its blue water and river festivals. "
                "The river festival happens every spring in Patna.",
        "source_name": "patna-river.txt"})
    assert res.status_code == 200, res.get_json()
    doc_id = res.get_json()["document_id"]
    assert res.get_json()["chunk_count"] >= 1
    # Duplicate ingest: same text -> same document_id, no duplicate chunks.
    res2 = alice.post("/api/rag/ingest", json={
        "text": "The Shibu River flows through Patna and is famous for its blue water and river festivals. "
                "The river festival happens every spring in Patna.",
        "source_name": "patna-river.txt"})
    assert res2.get_json()["document_id"] == doc_id
    docs = alice.get("/api/rag/documents").get_json()["documents"]
    assert len([d for d in docs if d["document_id"] == doc_id]) == 1
    # Alice can search it; Bob cannot (cross-user isolation).
    hits = alice.post("/api/rag/search", json={"q": "Shibu River Patna festival"}).get_json()["results"]
    assert hits and hits[0]["source_name"] == "patna-river.txt"
    assert hits[0]["backend"] == "sqlite_fallback"
    assert bob.post("/api/rag/search", json={"q": "Shibu River Patna festival"}).get_json()["results"] == []
    assert bob.delete(f"/api/rag/documents/{doc_id}").status_code == 404
    # Re-index, then delete as owner.
    assert alice.post(f"/api/rag/documents/{doc_id}/reindex").status_code == 200
    assert alice.delete(f"/api/rag/documents/{doc_id}").status_code == 200
    assert alice.post("/api/rag/search", json={"q": "Shibu River Patna festival"}).get_json()["results"] == []


def test_rag_chat_answer_with_sources_and_empty_result():
    client = _client("google#chat-rag")
    client.post("/api/rag/ingest", json={
        "text": "Zorblax is a fictional alloy invented in 2024 that conducts heat but not electricity. "
                "Zorblax is used in spacecraft heat shields.",
        "source_name": "zorblax.txt"})
    res = client.post("/api/chat", json={"message": "Explain Zorblax heat conduction"})
    data = res.get_json()
    assert data["intent"] == "RAG_ANSWER", data
    assert data["sources"] and data["sources"][0]["source_name"] == "zorblax.txt"
    assert "Zorblax" in data["response"] or "zorblax" in data["response"].lower()
    # Empty result for a document question with no evidence.
    res2 = client.post("/api/chat", json={"message": "What does my document say about quantum bananas?"})
    assert res2.get_json()["intent"] == "RAG_NO_EVIDENCE"


def test_rules_still_win_over_rag_and_anonymous_rag_is_blocked():
    client = _client("google#rules-rag")
    client.post("/api/rag/ingest", json={"text": "Python is a banana.", "source_name": "wrong.txt"})
    res = client.post("/api/chat", json={"message": "What is Python?"})
    # Rule answer, not the (wrong) document answer.
    assert "banana" not in res.get_json()["response"]
    anon = app_module.app.test_client()
    assert anon.post("/api/rag/ingest", json={"text": "x", "source_name": "x.txt"}).status_code == 401
    assert anon.post("/api/rag/search", json={"q": "x"}).status_code == 401


def test_explicit_migration_only_and_persistence():
    client = _client("google#migrate-rag")
    # Without confirm, nothing migrates.
    res = client.post("/api/history/migrate", json={"exchanges": [{"role": "user", "text": "Hello"}]})
    assert res.get_json()["migrated"] == 0
    res = client.post("/api/history/migrate", json={"confirm": True, "exchanges": [
        {"role": "user", "text": "Hello"}, {"role": "bot", "text": "Hi there"}]})
    assert res.get_json()["migrated"] == 2
    convs = client.get("/api/conversations").get_json()["conversations"]
    assert convs
    msgs = client.get(f"/api/conversations/{convs[0]['conversation_id']}/messages").get_json()["messages"]
    assert len(msgs) >= 2
    # Chat is also persisted for logged-in users.
    client.post("/api/chat", json={"message": "What is Python?"})
    convs2 = client.get("/api/conversations").get_json()["conversations"]
    assert len(convs2) >= 1
