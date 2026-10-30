"""Smaller memory upgrade (2026-10-31): LLM context, explicit
memories, restore-from-SQLite, and RAG passages fallback.

Scope is deliberately small: no resume-UI, no new-chat button, no
LLM polishing of rule answers, no auto-learning. Explicit memories
only ("remember that ..."), per-user, never secrets.
"""
import os
import sys
import tempfile

os.environ["RAG_BACKEND"] = "sqlite"
os.environ.pop("PINECONE_API_KEY", None)
os.environ.pop("PINECONE_INDEX_NAME", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from chatbot.context import ConversationContext
from chatbot.router import process_message
from services.conversation_manager import build_messages, context_system_lines
from services.rag_config import get_rag_config
from services.rag_service import RAGService
from services.sqlite_store import SQLiteStore


def _tmp_store(tmp_path, name):
    return SQLiteStore(path=str(tmp_path / name))


# --- LLM context -------------------------------------------------------

def test_build_messages_includes_summary_name_preferences_and_memories():
    ctx = ConversationContext()
    ctx.display_name = "Kartik"
    ctx.preferences["tone"] = "simple"
    ctx.preferences["level"] = "beginner"
    ctx.remember("I prefer simple explanations")
    ctx.add_exchange("What is Python?", "Python is a programming language.", "PYTHON", "python", 0.95, topic="python")
    messages, _lang = build_messages(ctx, "Tell me more")
    system = messages[0]["content"]
    assert "Kartik" in system
    assert "simple" in system
    assert "beginner" in system
    assert "I prefer simple explanations" in system
    assert "python" in system.lower()
    lines = context_system_lines(ctx)
    assert any("Kartik" in line for line in lines)


def test_build_messages_without_context_is_unchanged():
    ctx = ConversationContext()
    messages, _lang = build_messages(ctx, "Hello")
    assert messages[0]["role"] == "system"
    assert "Shibu AI" in messages[0]["content"]
    assert messages[-1]["content"] == "Hello"


# --- Explicit memories in the session context ---------------------------

def test_remember_forget_and_clear_preserves_memories():
    ctx = ConversationContext()
    assert ctx.remember("I like short answers") is True
    assert ctx.remember("I like short answers") is False  # duplicate
    assert ctx.remember("I like short answers") is False
    assert ctx.forget_matching("short") == 1
    assert ctx.memories == []
    ctx.remember("Keep answers simple")
    ctx.add_exchange("hi", "hello", "X", "x", 0.9)
    ctx.clear()
    # Clear chat keeps explicit memories; only "forget" removes them.
    assert ctx.memories == ["Keep answers simple"]
    assert ctx.forget_all_memories() == 1
    assert ctx.memories == []


def test_router_remember_list_forget_and_secret_refusal(monkeypatch):
    import chatbot.router as router
    monkeypatch.setattr(router.search_service, "search", lambda q: [])
    ctx = ConversationContext()
    out = process_message("Remember that I prefer simple explanations", ctx)
    assert out["intent"] == "MEMORY" and out["rule_id"] == "memory_save"
    out = process_message("What do you remember about me?", ctx)
    assert "simple explanations" in out["response"]
    out = process_message("Remember that my password is hunter2-fake", ctx)
    assert out["rule_id"] == "memory_refused"
    assert "hunter2" not in str(ctx.memories)
    out = process_message("Forget simple explanations", ctx)
    assert out["rule_id"] == "memory_forget"
    assert ctx.memories == []


def test_restore_from_messages_caps_and_sets_running_total():
    ctx = ConversationContext()
    msgs = [{"role": "user", "text": f"question {i}"} for i in range(20)]
    msgs += [{"role": "bot", "text": "Answer: 49933"}]
    assert ctx.restore_from_messages(msgs) is True
    assert len(ctx.conversation_history) <= 12
    assert ctx.last_numeric_value == 49933
    assert ctx.restore_from_messages([]) is False


# --- SQLite memories: per-user isolation --------------------------------

def test_sqlite_memories_isolated_and_replaced(tmp_path):
    store = _tmp_store(tmp_path, "mem.db")
    store.replace_memories("google#alice", ["I like simple answers", "I like simple answers"])
    store.replace_memories("google#bob", ["I am Bob"])
    assert store.list_memories("google#alice") == ["I like simple answers"]
    assert store.list_memories("google#bob") == ["I am Bob"]
    assert store.list_memories("google#nobody") == []
    store.replace_memories("google#alice", [])
    assert store.list_memories("google#alice") == []
    assert store.list_memories("google#bob") == ["I am Bob"]


def test_sqlite_latest_conversation_messages(tmp_path):
    store = _tmp_store(tmp_path, "conv.db")
    cid = store.create_conversation("google#alice", title="first")
    store.save_message("google#alice", cid, "user", "564+2674")
    store.save_message("google#alice", cid, "bot", "Answer: 3238")
    msgs = store.latest_conversation_messages("google#alice", limit=12)
    assert [m["text"] for m in msgs] == ["564+2674", "Answer: 3238"]
    assert store.latest_conversation_messages("google#bob") == []


# --- RAG: context + passages fallback when the LLM fails -----------------

def test_rag_answer_passes_context_and_falls_back_to_passages(tmp_path, monkeypatch):
    import services.model_manager as mm

    class FailingManager:
        mode = "api"

        def generate(self, messages, max_new_tokens=256, temperature=0.3):
            # The grounded prompt must carry the user's context.
            assert any("Kartik" in m["content"] for m in messages)
            return None, "Model failed for test."

    monkeypatch.setattr(mm, "ModelManager", lambda: FailingManager())
    store = _tmp_store(tmp_path, "rag.db")
    cfg = get_rag_config()
    cfg["backend"] = "sqlite"
    svc = RAGService(config=cfg, sqlite_store=store)
    svc.ingest_text("google#rag-memory", "zorblax.txt",
                    "Zorblax is a fictional alloy that conducts heat but not electricity.")
    ctx = ConversationContext()
    ctx.display_name = "Kartik"
    result = svc.rag_answer("google#rag-memory", "What is Zorblax?", context=ctx)
    assert result["evidence_found"] is True
    assert result["error"] == ""
    assert "Zorblax" in result["answer"]
    assert "direct retrieval result" in result["answer"]


# --- Flask: restore + memory persistence across sessions -----------------

def _client(user):
    import app as app_module
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["google_user_id"] = user
        sess["google_display_name"] = "Memory Tester"
        sess["onboarding_completed"] = True
    return client


def test_flask_memory_persists_and_history_restores_across_sessions(monkeypatch):
    import chatbot.router as router
    monkeypatch.setattr(router.search_service, "search", lambda q: [])
    user = "google#memory-flask-1"
    first = _client(user)
    res = first.post("/api/chat", json={"message": "Remember that I prefer simple explanations"})
    assert res.get_json()["intent"] == "MEMORY", res.get_json()
    # A fresh browser session for the same user: memory is restored.
    second = _client(user)
    res = second.post("/api/chat", json={"message": "What do you remember about me?"})
    assert "simple explanations" in res.get_json()["response"], res.get_json()
    # Fresh session again: forget works and persists.
    third = _client(user)
    res = third.post("/api/chat", json={"message": "Forget simple explanations"})
    assert res.get_json()["rule_id"] == "memory_forget"
    fourth = _client(user)
    res = fourth.post("/api/chat", json={"message": "What do you remember about me?"})
    assert res.get_json()["response"].startswith("I haven't been asked to remember")


def test_flask_restore_continues_running_total_across_sessions():
    user = "google#memory-restore-1"
    first = _client(user)
    res = first.post("/api/chat", json={"message": "564+2674+679+45467+549"})
    assert "49933" in res.get_json()["response"], res.get_json()
    # Fresh session, same user: the previous numeric answer is restored.
    second = _client(user)
    res = second.post("/api/chat", json={"message": "add 967464 in previous answer"})
    assert "1017397" in res.get_json()["response"], res.get_json()
