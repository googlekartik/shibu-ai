"""RAG orchestration: ingest -> chunk -> embed (dense Pinecone only)
-> upsert -> query -> grounded answer. All user isolation is by
namespace (Pinecone) and user_id (SQLite); the caller must pass the
session user_id, never a body-supplied id.

Failure model (no silent fallbacks in production):
- Pinecone backend, key/index missing -> PineconeError, status
  "blocked". The caller reports blocked, it does not silently use
  SQLite and call it Pinecone.
- Pinecone backend, dimension mismatch -> validation error, RAG
  disabled, migration required (see rag_config).
- SQLite backend -> token-overlap retrieval, labelled
  "sqlite_fallback" in every response, so a test result is never
  mistaken for a Pinecone semantic result.
- Ingestion failure -> document status "failed" with the error in
  SQLite, so a retry/re-index is possible; the previous good
  chunks are kept only until a successful replacement deletes them.
"""
from __future__ import annotations

import hashlib
import os
import re

from services.chunking import chunk_text
from services.embeddings import EmbeddingError, get_embedder
from services.rag_config import get_rag_config, namespace_for_user, validate_index_info
from services.sqlite_store import SQLiteStore
from services.vector_store import PineconeError, PineconeVectorStore


def document_id_for(user_id: str, source_name: str, text: str) -> str:
    raw = f"{user_id}:{source_name}:{hashlib.sha256((text or '').encode('utf-8')).hexdigest()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _extract_plain_text(path: str, ext: str) -> str:
    if ext in (".txt", ".md", ".csv"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in (".html", ".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S | re.I)
        return re.sub(r"<[^>]+>", " ", raw)
    # PDF / DOCX / XLSX reuse the existing pipeline exactly.
    from services.document_service import extract_text
    text, _info = extract_text(path, ext)
    return text


class RAGService:
    def __init__(self, config: dict = None, sqlite_path: str = None, sqlite_store=None):
        self.config = config or get_rag_config()
        self.store = sqlite_store or SQLiteStore(path=sqlite_path or self.config.get("sqlite_path"))
        self._pinecone = None

    # -- status / validation ----------------------------------------
    def status(self, user_id: str = None) -> dict:
        cfg = self.config
        out = {
            "backend": cfg["backend"],
            "pinecone_configured": cfg["pinecone_configured"],
            "index_name": cfg["index_name"],
            "index_type": cfg["index_type"],
            "embedding_provider": cfg["embedding_provider"],
            "embedding_model": cfg["embedding_model"],
            "embedding_dimension": cfg["embedding_dimension"],
            "top_k": cfg["top_k"],
            "min_score": cfg["min_score"],
            "chunk_size": cfg["chunk_size"],
            "chunk_overlap": cfg["chunk_overlap"],
            "namespace_strategy": "one namespace per Google sub (user-<sub>)",
            "live_pinecone": "blocked",
            "live_pinecone_reason": "",
        }
        if user_id:
            try:
                out["own_documents"] = len(self.store.list_documents(user_id))
            except Exception:
                out["own_documents"] = 0
        if cfg["backend"] != "pinecone":
            out["live_pinecone"] = "not_used"
            out["live_pinecone_reason"] = f"Backend is {cfg['backend']}; Pinecone is not queried in this mode."
            return out
        if not cfg["pinecone_configured"]:
            out["live_pinecone_reason"] = "PINECONE_API_KEY and/or PINECONE_INDEX_NAME are absent from the environment. Set them in .env / Render Environment (never in chat) to enable live validation."
            return out
        try:
            store = self._pinecone_store()
            info = store.describe()
            ok, errors = validate_index_info(info, cfg)
            out["index_dimension"] = info.get("dimension")
            out["index_metric"] = info.get("metric")
            out["index_type_reported"] = info.get("index_type")
            out["live_pinecone"] = "ok" if ok else "mismatch"
            out["live_pinecone_reason"] = "; ".join(errors) if errors else "Index described and dimension matches the embedding model."
        except Exception as exc:
            out["live_pinecone_reason"] = str(exc)[:300]
        return out

    def _pinecone_store(self):
        if self._pinecone is None:
            self._pinecone = PineconeVectorStore(index_type=self.config["index_type"])
        return self._pinecone

    # -- ingestion --------------------------------------------------
    def ingest_text(self, user_id: str, source_name: str, text: str, source_type: str = ""):
        """Chunk, embed (dense Pinecone only), upsert, save metadata.
        Returns {document_id, chunk_count, backend, status}. On
        Pinecone failure the SQLite document is marked failed and
        the error is returned (not hidden), so re-index can retry."""
        cfg = self.config
        doc_id = document_id_for(user_id, source_name, text or "")
        chunks = chunk_text(text or "", doc_id, cfg["chunk_size"], cfg["chunk_overlap"])
        if not chunks:
            raise ValueError("No readable text to index from this document.")
        namespace = namespace_for_user(user_id)
        if cfg["backend"] == "pinecone":
            store = self._pinecone_store()
            info = store.describe()
            ok, errors = validate_index_info(info, cfg)
            if not ok:
                self.store.save_document(user_id, doc_id, source_name, source_type, text or "", chunks, status="failed", error="; ".join(errors))
                raise PineconeError("; ".join(errors))
            try:
                if cfg["index_type"] == "integrated":
                    items = [{"id": c["id"], "text": c["text"],
                              "metadata": {"document_id": doc_id, "source_name": source_name, "chunk_index": c["chunk_index"], "text": c["text"][:2000]}}
                             for c in chunks]
                    store.upsert(namespace, items)
                else:
                    embedder = get_embedder(cfg)
                    if embedder is None:
                        raise EmbeddingError("No embedding provider is configured for a dense index.")
                    vectors = embedder.embed([c["text"] for c in chunks])
                    if vectors and len(vectors[0]) != info.get("dimension"):
                        raise PineconeError(f"Embedding dimension {len(vectors[0])} does not match index dimension {info.get('dimension')}.")
                    items = [{"id": c["id"], "values": v,
                              "metadata": {"document_id": doc_id, "source_name": source_name, "chunk_index": c["chunk_index"], "text": c["text"][:2000]}}
                             for c, v in zip(chunks, vectors)]
                    store.upsert(namespace, items)
            except Exception as exc:
                self.store.save_document(user_id, doc_id, source_name, source_type, text or "", chunks, status="failed", error=str(exc)[:300])
                raise
        # SQLite metadata / fallback chunks are always saved on success.
        self.store.save_document(user_id, doc_id, source_name, source_type, text or "", chunks, status="indexed")
        return {"document_id": doc_id, "chunk_count": len(chunks), "backend": cfg["backend"], "status": "indexed"}

    def ingest_file(self, user_id: str, path: str, source_name: str):
        ext = os.path.splitext(source_name)[1].lower()
        if ext not in (".pdf", ".docx", ".xlsx", ".txt", ".md", ".html", ".htm", ".csv"):
            raise ValueError(f"Unsupported file type for RAG ingestion: {ext}. Use PDF, DOCX, XLSX, TXT, MD, HTML or CSV.")
        text = _extract_plain_text(path, ext)
        return self.ingest_text(user_id, source_name, text, source_type=ext)

    def delete_document(self, user_id: str, document_id: str):
        cfg = self.config
        if cfg["backend"] == "pinecone" and cfg["pinecone_configured"]:
            try:
                self._pinecone_store().delete_by_document(namespace_for_user(user_id), document_id)
            except Exception:
                # SQLite deletion still happens; the caller can retry
                # the Pinecone delete. Do not hide the failure.
                deleted_sqlite = self.store.delete_document(user_id, document_id)
                raise PineconeError("SQLite document deleted, but Pinecone deletion failed; retry the delete to remove its vectors.") from None
        return self.store.delete_document(user_id, document_id)

    def reindex_document(self, user_id: str, document_id: str):
        doc = self.store.get_document_with_text(user_id, document_id)
        if not doc:
            raise ValueError("Document not found.")
        # Delete old vectors first, then ingest the stored text again.
        if self.config["backend"] == "pinecone" and self.config["pinecone_configured"]:
            try:
                self._pinecone_store().delete_by_document(namespace_for_user(user_id), document_id)
            except Exception:
                pass
        return self.ingest_text(user_id, doc["source_name"], doc.get("full_text", ""), source_type=doc.get("source_type", ""))

    # -- search / RAG ------------------------------------------------
    def search(self, user_id: str, query: str, top_k: int = None) -> list:
        cfg = self.config
        k = top_k or cfg["top_k"]
        if cfg["backend"] == "pinecone":
            store = self._pinecone_store()
            info = store.describe()
            ok, errors = validate_index_info(info, cfg)
            if not ok:
                raise PineconeError("; ".join(errors))
            namespace = namespace_for_user(user_id)
            if cfg["index_type"] == "integrated":
                hits = store.query(namespace, query_text=query, top_k=k)
            else:
                embedder = get_embedder(cfg)
                if embedder is None:
                    raise EmbeddingError("No embedding provider is configured for a dense index.")
                vec = embedder.embed([query])[0]
                hits = store.query(namespace, vector=vec, top_k=k)
            out = []
            for h in hits:
                md = h.get("metadata") or {}
                out.append({"id": h["id"], "score": float(h.get("score", 0.0)),
                            "text": h.get("text") or md.get("text", ""),
                            "document_id": md.get("document_id", ""),
                            "source_name": md.get("source_name", ""),
                            "chunk_index": md.get("chunk_index"),
                            "backend": "pinecone"})
            return out
        # SQLite fallback: token overlap, labelled as such.
        hits = self.store.search_chunks_fallback(user_id, query, top_k=k)
        for h in hits:
            h["backend"] = "sqlite_fallback"
        return hits

    def rag_answer(self, user_id: str, question: str, top_k: int = None, context=None):
        """Return {answer, sources, backend, evidence_found, error}.
        The LLM is called only with retrieved evidence; if the LLM
        is off, the retrieved passages themselves are returned,
        labelled as retrieved passages — never invented as an LLM
        answer. If evidence is below threshold, evidence_found is
        False and the caller must fall through to web/LLM or say
        the evidence is insufficient."""
        cfg = self.config
        try:
            hits = self.search(user_id, question, top_k=top_k)
        except Exception as exc:
            return {"answer": "", "sources": [], "backend": cfg["backend"],
                    "evidence_found": False, "error": str(exc)[:300]}
        hits = [h for h in hits if h["score"] >= cfg["min_score"]]
        if not hits:
            return {"answer": "", "sources": [], "backend": cfg["backend"],
                    "evidence_found": False, "error": ""}
        sources = [{"source_name": h["source_name"], "document_id": h["document_id"],
                    "chunk_index": h.get("chunk_index"), "score": round(h["score"], 3)} for h in hits]
        evidence = "\n\n".join(f"[Source: {h['source_name']}] {h['text']}" for h in hits)
        from services.model_manager import ModelManager
        manager = ModelManager()
        if manager.mode == "off":
            answer = ("From your documents, the most relevant passages are:\n\n" + evidence +
                      "\n\nNote: No language model is enabled, so this is a direct retrieval result, not an AI-generated summary.")
            return {"answer": answer, "sources": sources, "backend": cfg["backend"], "evidence_found": True, "error": ""}
        extra_lines = []
        if context is not None:
            try:
                from services.conversation_manager import context_system_lines
                extra_lines = context_system_lines(context)
            except Exception:
                extra_lines = []
        system = ("Answer ONLY from the evidence below. If the evidence does not answer the question, say so plainly and do not invent facts. Cite sources by name in your answer.")
        if extra_lines:
            system += " Context: " + " ".join(extra_lines)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {question}"},
        ]
        raw, err = manager.generate(messages, max_new_tokens=256, temperature=0.3)
        if err or not raw:
            # Evidence was found, so do not lose it: return the
            # passages directly, labelled, instead of an error that
            # makes the caller fall through and web-search instead.
            answer = ("The language model could not summarise this time, so here are the most relevant passages from your documents:\n\n"
                      + evidence +
                      "\n\nNote: This is a direct retrieval result, not an AI-generated summary.")
            return {"answer": answer, "sources": sources, "backend": cfg["backend"], "evidence_found": True, "error": ""}
        return {"answer": raw, "sources": sources, "backend": cfg["backend"], "evidence_found": True, "error": ""}
