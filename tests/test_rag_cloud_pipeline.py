"""Offline tests for the cloud-first RAG control plane.

No live provider calls or credentials are used here.  Provider SDK calls are
exercised through fakes; the live index remains validated only after its owner
sets secrets in Render.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.document_service import extract_text
from services.rag_config import get_rag_config
from services.rag_service import RAGService, document_id_for
from services.request_router import plan_request
from services.sqlite_store import SQLiteStore


def _sqlite_service(monkeypatch):
    handle, path = tempfile.mkstemp(prefix="shibu-rag-upgrade-", suffix=".db")
    os.close(handle)
    monkeypatch.setenv("RAG_BACKEND", "sqlite")
    monkeypatch.setenv("SHIBU_SQLITE_PATH", path)
    return RAGService(config=get_rag_config(), sqlite_store=SQLiteStore(path)), path


def test_revised_source_replaces_instead_of_creating_a_duplicate(monkeypatch):
    service, path = _sqlite_service(monkeypatch)
    try:
        first = service.ingest_text("google#revision", "notes.txt", "Aster is the old internal project codename.")
        second = service.ingest_text("google#revision", "notes.txt", "Borealis is the current internal project codename.")
        assert first["document_id"] == second["document_id"]
        assert second["updated"] is True
        docs = service.store.list_documents("google#revision")
        assert len(docs) == 1
        assert all("aster" not in hit["text"].lower()
                   for hit in service.search("google#revision", "Aster codename"))
        assert service.search("google#revision", "Borealis codename")
    finally:
        os.remove(path)


def test_duplicate_ingestion_short_circuits_cloud_work(monkeypatch):
    service, path = _sqlite_service(monkeypatch)
    try:
        service.ingest_text("google#duplicate", "same.txt", "The moon has no breathable atmosphere.")
        duplicate = service.ingest_text("google#duplicate", "same.txt", "The moon has no breathable atmosphere.")
        assert duplicate["status"] == "unchanged"
        assert duplicate["deduplicated"] is True
    finally:
        os.remove(path)


def test_metadata_filter_and_citations_are_scoped(monkeypatch):
    service, path = _sqlite_service(monkeypatch)
    try:
        one = service.ingest_text("google#filter", "biology.txt", "Mitochondria make ATP for cells.")
        service.ingest_text("google#filter", "history.txt", "The treaty was signed in 1919.")
        hits = service.search("google#filter", "what makes ATP", filters={"document_id": one["document_id"]})
        assert hits and all(hit["source_name"] == "biology.txt" for hit in hits)
        answer = service.rag_answer("google#filter", "What makes ATP?", filters={"source_name": "biology.txt"})
        assert answer["evidence_found"] is True
        assert answer["sources"][0]["citation"] == "S1"
        assert "mitochondria" in answer["answer"].lower()
    finally:
        os.remove(path)


def test_router_uses_both_sources_only_when_explicit():
    assert plan_request("What does my document say about mitochondria?")["route"] == "rag"
    assert plan_request("Search the web for current AI news")["route"] == "web"
    assert plan_request("Compare my document with the latest web information")["route"] == "rag_and_web"
    assert plan_request("What is Python?")["route"] == "standard"


def test_txt_and_html_extraction_share_the_document_pipeline(tmp_path):
    text_file = tmp_path / "notes.txt"
    html_file = tmp_path / "notes.html"
    text_file.write_text("Cloud embeddings are hosted APIs.", encoding="utf-8")
    html_file.write_text("<html><script>ignore()</script><body><h1>Semantic search</h1></body></html>", encoding="utf-8")
    text, text_info = extract_text(str(text_file), ".txt")
    html, html_info = extract_text(str(html_file), ".html")
    assert text_info["type"] == "txt" and "hosted APIs" in text
    assert html_info["type"] == "html" and "Semantic search" in html and "ignore" not in html


def test_document_identity_is_not_content_identity():
    assert document_id_for("google#u", "notes.txt", "first") == document_id_for("google#u", "notes.txt", "revised")
    assert document_id_for("google#u", "notes.txt") != document_id_for("google#u", "other.txt")
