"""Rule-Based Intelligent Chatbot — Flask entry point.

Run:  python app.py
Then open: http://127.0.0.1:5000
"""
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session
from flask_cors import CORS

load_dotenv()

from chatbot.context import ContextStore
from chatbot.router import process_message
from services.huggingface_service import HuggingFaceService
from services.model_manager import ModelManager
from services.conversation_manager import build_messages, detect_language
from services.response_generator import validate_llm_answer, format_llm_answer
from services.math_service import MathService
from services.pdf_service import PDFService
from services.web_search_service import WebSearchService
from services.document_service import (extract_text, summarize as summarize_doc,
                                       answer_from_document, SUPPORTED_DOCUMENT_EXTENSIONS)
from services.file_generation_service import generate_pptx, generate_docx, generate_xlsx
from services.cache_service import (cached_answer_lookup, cached_answer_store,
                                    looks_personal, normalize_query)
from services.rag_config import get_rag_config
from services.rag_service import RAGService
from services.request_router import plan_request
from services.request_limiter import request_limiter
from services.retrieval_metrics import retrieval_metrics
from services.sqlite_store import SQLiteStore
from services.math_service import looks_like_math
from chatbot.rule_engine import match_rule
from services.google_auth import (build_authorization_url, exchange_code,
                                  google_config, google_user_id, new_state,
                                  verify_id_token)
from utils.security import sanitize_filename, safe_join
from utils.logger import Timer, get_logger, log_chat
from utils.security import MAX_PDF_TEXT_LENGTH, validate_message

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB request limit
CORS(app)

logger = get_logger()
contexts = ContextStore()
hf = HuggingFaceService()
math_service = MathService()
pdf_service = PDFService()
search_service = WebSearchService()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _session_id():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def _context():
    return contexts.get(_session_id())


_RAG_CACHE = {}


def _allow_costly_request(category: str) -> bool:
    """Bound repeat requests to cloud-backed endpoints per user/IP process."""
    try:
        limit = int(os.environ.get("CLOUD_REQUEST_LIMIT", "60"))
        window = int(os.environ.get("CLOUD_REQUEST_WINDOW_SECONDS", "60"))
    except ValueError:
        limit, window = 60, 60
    identity = _session_user_id() or request.remote_addr or "anonymous"
    return request_limiter.allow(f"{category}:{identity}", limit, window)


def _sqlite_store():
    path = get_rag_config().get("sqlite_path")
    if path not in _RAG_CACHE:
        _RAG_CACHE[path] = SQLiteStore(path=path)
    return _RAG_CACHE[path]


def _rag_service():
    # Share the app's single SQLite connection with RAGService:
    # two connections to the same file caused "database is locked"
    # in tests. One connection per process, sequential Flask use.
    key = ("rag", get_rag_config().get("sqlite_path"))
    if key not in _RAG_CACHE:
        _RAG_CACHE[key] = RAGService(sqlite_store=_sqlite_store())
    return _RAG_CACHE[key]


def _save_chat_to_sqlite(user_message, payload):
    """Persistent chat history for logged-in users only, one
    conversation per Flask session. Anonymous chat is never saved
    to SQLite (no user_id to scope it to)."""
    user_id = _session_user_id()
    if not user_id:
        return
    try:
        store = _sqlite_store()
        store.get_or_create_user(user_id)
        conv_id = session.get("sqlite_conversation_id")
        if not conv_id:
            conv_id = store.create_conversation(user_id, title=user_message[:80])
            session["sqlite_conversation_id"] = conv_id
        store.save_message(user_id, conv_id, "user", user_message)
        store.save_message(user_id, conv_id, "bot", payload.get("response", ""),
                           intent=payload.get("intent", ""), rule_id=payload.get("rule_id", ""),
                           sources=payload.get("sources", []))
    except Exception:
        logger.exception("SQLite chat save failed (chat still returned)")
    # Persist explicit memories only after the context has been
    # restored for this session (see _restore_context), so an
    # unrestored empty list can never wipe stored memories.
    if user_id and session.get("context_restored_v1"):
        try:
            _sqlite_store().replace_memories(user_id, list(getattr(_context(), "memories", []) or []))
        except Exception:
            logger.exception("SQLite memory save failed (chat still returned)")


def _restore_context(ctx, user_id, display_name=""):
    """First logged-in chat in a session: load explicit memories and
    the most recent saved conversation into the in-memory context,
    so follow-ups and the LLM see more than this browser session.
    Runs once per Flask session (session flag); /api/clear sets the
    flag too, so clearing is not undone by a restore."""
    if session.get("context_restored_v1"):
        return
    session["context_restored_v1"] = True
    if display_name:
        ctx.display_name = display_name
    try:
        store = _sqlite_store()
        ctx.memories = store.list_memories(user_id)
        msgs = store.latest_conversation_messages(user_id, limit=12)
        ctx.restore_from_messages([{"role": m.get("role"), "text": m.get("text", "")} for m in msgs])
    except Exception:
        logger.exception("SQLite context restore failed (chat still returned)")


def _session_user_id():
    """Logged-in Google user ID from the session, or None. Never
    trust a user_id sent in a request body — the session is the source."""
    return session.get("google_user_id")


# Onboarding avatars: illustrated assets in static/avatars/. The first
# avatar is also the welcome image on /login. Files are swapped by
# replacing these SVGs — no code change is needed for new artwork.
AVATARS = [
    {"id": "maya", "name": "Maya", "file": "avatars/avatar-1.svg",
     "style": "Short black bob, blue outfit", "color": "#7c5cff"},
    {"id": "aria", "name": "Aria", "file": "avatars/avatar-2.svg",
     "style": "Long brown wavy hair, green outfit", "color": "#10b981"},
    {"id": "zara", "name": "Zara", "file": "avatars/avatar-3.svg",
     "style": "Curly red hair, purple outfit", "color": "#a855f7"},
    {"id": "luna", "name": "Luna", "file": "avatars/avatar-4.svg",
     "style": "Blonde pixie cut, pink outfit", "color": "#f472b6"},
    {"id": "nia", "name": "Nia", "file": "avatars/avatar-5.svg",
     "style": "Braided hair, teal outfit", "color": "#0d9488"},
]
AVATAR_IDS = {a["id"] for a in AVATARS}
QUICK_ACTIONS = ["Search the web", "Ask a question", "Learn something new",
                 "Write or create something", "Help me with something else"]


def _avatar_url(avatar_id):
    for a in AVATARS:
        if a["id"] == avatar_id:
            return "/static/" + a["file"]
    return "/static/shibu.png"


def _profile():
    """Logged-in profile, from the session merged with the SQLite
    user row (same host only; SQLite on Render free is ephemeral).
    Server-side authorisation: the user_id always comes from the
    session, never from a request body, so a user can only read or
    change their own profile."""
    user_id = _session_user_id()
    if not user_id:
        return {"logged_in": False, "user_id": None, "display_name": "",
                "avatar_id": None, "avatar_url": "/static/shibu.png",
                "onboarding_completed": False, "onboarding_stage": None}
    profile = {"logged_in": True, "user_id": user_id,
               "display_name": session.get("google_display_name", "") or "",
               "avatar_id": session.get("profile_avatar_id"),
               "onboarding_completed": bool(session.get("onboarding_completed")),
               "onboarding_stage": session.get("onboarding_stage")}
    try:
        user = _sqlite_store().get_or_create_user(user_id)
        if user.get("avatar_id") in AVATAR_IDS:
            profile["avatar_id"] = user["avatar_id"]
        if user.get("display_name"):
            profile["display_name"] = user["display_name"]
        if user.get("onboarding_completed"):
            profile["onboarding_completed"] = True
            profile["onboarding_stage"] = None
    except Exception:
        pass
    if profile["avatar_id"] not in AVATAR_IDS:
        profile["avatar_id"] = None
    if not profile["onboarding_completed"] and not profile["onboarding_stage"]:
        profile["onboarding_stage"] = "ask_name" if profile["avatar_id"] else "need_avatar"
    profile["avatar_url"] = _avatar_url(profile["avatar_id"])
    return profile


def _save_profile(display_name=None, avatar_id=None, onboarding_completed=None,
                  onboarding_stage=None):
    """Save to the session and to SQLite (logged-in users only).
    SQLite is the persistent profile store on this host; the
    session alone would lose it when the browser session ends."""
    user_id = _session_user_id()
    if display_name is not None:
        session["google_display_name"] = display_name
    if avatar_id is not None:
        session["profile_avatar_id"] = avatar_id
    if onboarding_completed is not None:
        session["onboarding_completed"] = bool(onboarding_completed)
    if onboarding_stage is not None:
        session["onboarding_stage"] = onboarding_stage
    if user_id:
        try:
            _sqlite_store().update_user(user_id, display_name=display_name,
                                        avatar_id=avatar_id,
                                        onboarding_completed=onboarding_completed)
        except Exception:
            pass
    return _profile()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    rag_cfg = get_rag_config()
    return jsonify({
        "status": "ok",
        "rule_engine": True,
        "math_service": True,
        "pdf_service": True,
        "search_service": True,
        "huggingface": hf.enabled,
        "llm": ModelManager().status(),
        "rag": {"backend": rag_cfg["backend"], "pinecone_configured": rag_cfg["pinecone_configured"],
                "embedding_provider": rag_cfg["embedding_provider"]},
    })


@app.route("/api/llm/status")
def llm_status():
    """Safe LLM status: model, mode, RAM/GPU, cache — never the token."""
    return jsonify(ModelManager().status())


@app.route("/api/llm/chat", methods=["POST"])
def llm_chat():
    """Direct conversational LLM endpoint (bypasses the rule engine).

    Use /api/chat for the normal rule-first flow; use this endpoint to
    test the language model itself. Returns 503 with how-to-enable
    instructions when no model is configured, instead of a fake answer.
    """
    data = request.get_json(silent=True) or {}
    ok, message_or_error = validate_message(data.get("message"))
    if not ok:
        return jsonify({"error": message_or_error}), 400
    manager = ModelManager()
    if manager.mode == "off":
        return jsonify({"error": "No language model is enabled.",
                        "how_to_enable": manager.status()["how_to_enable"]}), 503
    ctx = _context()
    messages, language = build_messages(ctx, message_or_error)
    raw, err = manager.generate(messages, max_new_tokens=256, temperature=0.7)
    if err or raw is None:
        return jsonify({"error": err or "Language model returned nothing."}), 502
    ok2, cleaned, note = validate_llm_answer(raw, previous_bot_answer=ctx.last_bot_response)
    if not ok2:
        return jsonify({"error": note}), 502
    answer = format_llm_answer(cleaned, model_id=manager.model_id)
    ctx.add_exchange(message_or_error, answer, "LLM_ANSWER", "llm_answer", 0.65)
    return jsonify({"response": answer, "intent": "LLM_ANSWER",
                    "model": getattr(manager, "effective_model", None) or manager.model_id, "language": language})


@app.route("/api/llm/test", methods=["POST"])
def llm_test():
    """Run the model manager's one-sentence smoke test.

    Only call this when you intend to load a model: in local mode it
    will download the selected model on first use (once, cached after).
    """
    manager = ModelManager()
    if manager.mode == "off":
        return jsonify({"error": "No language model is enabled.",
                        "how_to_enable": manager.status()["how_to_enable"]}), 503
    return jsonify(manager.test_prompt())


@app.route("/api/chat", methods=["POST"])
def chat():
    timer = Timer()
    data = request.get_json(silent=True) or {}
    ok, message_or_error = validate_message(data.get("message"))
    if not ok:
        return jsonify({"error": message_or_error}), 400
    ctx = _context()
    # First-time welcome conversation (logged-in users only, and only
    # until onboarding completes). Anonymous chat is untouched.
    # Stage ask_name: the reply is the preferred name, saved only here
    # after the user has typed it in this step. Stage ask_goal: the
    # next reply is their first real request, so onboarding completes
    # and the message falls through to the normal router below.
    profile = _profile()
    if profile["logged_in"] and not profile["onboarding_completed"]:
        if profile["onboarding_stage"] == "need_avatar":
            return jsonify({"response": "Please choose an avatar first.",
                            "intent": "ONBOARDING", "rule_id": "onboarding_avatar",
                            "confidence": 1.0, "confidence_label": "HIGH",
                            "tool": "Onboarding", "action": "Avatar selection required",
                            "needs_avatar": True}), 200
        if profile["onboarding_stage"] == "ask_name":
            name = message_or_error.strip()
            if len(name) > 60 or "?" in name:
                return jsonify({"response": "Hi! Welcome! What should I call you? "
                                "Just reply with your name, for example: Alex",
                                "intent": "ONBOARDING", "rule_id": "onboarding_name",
                                "confidence": 1.0, "confidence_label": "HIGH",
                                "tool": "Onboarding", "action": "Ask preferred name"}), 200
            _save_profile(display_name=name, onboarding_stage="ask_goal")
            payload = {"response": f"Nice to meet you, {name}! What would you like to do today?",
                       "intent": "ONBOARDING", "rule_id": "onboarding_goal",
                       "confidence": 1.0, "confidence_label": "HIGH",
                       "tool": "Onboarding", "action": "Ask first goal",
                       "quick_actions": QUICK_ACTIONS}
            return jsonify(payload)
        if profile["onboarding_stage"] == "ask_goal":
            _save_profile(onboarding_completed=True, onboarding_stage=None)
    # Smaller memory upgrade: restore memories + recent history once
    # per session for logged-in users, before RAG / router run.
    if profile["logged_in"]:
        _restore_context(ctx, _session_user_id(), profile["display_name"])
        if profile["display_name"]:
            ctx.display_name = profile["display_name"]
    # RAG attempt, AFTER onboarding and BEFORE the cache/router:
    # rules and maths keep priority (see below: RAG only runs when
    # the rule match is UNKNOWN / low-confidence, or the user
    # explicitly asks about their documents). If RAG finds evidence
    # above threshold, it answers with sources. If it finds nothing
    # for a document question, say so — do not web-search a private
    # document question. Otherwise fall through to the normal pipeline.
    if profile["logged_in"]:
        _route = plan_request(message_or_error)
        _forced_doc = _route["use_rag"]
        try:
            _rule_rag = match_rule(message_or_error)
        except Exception:
            _rule_rag = {"intent": "UNKNOWN", "confidence": 0.0}
        if (_forced_doc or _rule_rag.get("intent") == "UNKNOWN" or _rule_rag.get("rule_id") == "definition"
                or float(_rule_rag.get("confidence") or 0) < 0.45):
            try:
                _rag = _rag_service().rag_answer(_session_user_id(), message_or_error, context=ctx)
            except Exception:
                _rag = {"evidence_found": False, "error": "RAG unavailable"}
            if _rag.get("evidence_found") and not _rag.get("error"):
                _response = _rag["answer"]
                _sources = list(_rag["sources"])
                _tool = "Pinecone RAG" if _rag["backend"] == "pinecone" else "SQLite fallback RAG"
                _action = "Answer from your documents"
                # Combined routing is intentionally explicit: it does not add a
                # web request unless the user asks to compare/search online.
                if _route["route"] == "rag_and_web":
                    _web_results = search_service.search(message_or_error)
                    if _web_results:
                        from services.web_answer import format_web_answer
                        _response += "\n\n--- Current web results (separate from your document evidence) ---\n" + format_web_answer(message_or_error, _web_results)
                        _sources.extend({"citation": f"W{i}", "source_name": item.get("title", "Web result"),
                                         "url": item.get("url", ""), "type": "web"}
                                        for i, item in enumerate(_web_results[:5], 1))
                        _tool = _tool + " + DuckDuckGo"
                        _action = "Combine document retrieval with current web results"
                _payload = {"response": _rag["answer"], "intent": "RAG_ANSWER",
                            "rule_id": "rag_answer", "confidence": 0.85,
                            "confidence_label": "HIGH", "tool": "Pinecone RAG" if _rag["backend"] == "pinecone" else "SQLite fallback RAG",
                            "action": _action, "sources": _sources,
                            "backend": _rag["backend"], "processing_ms": round(timer.elapsed_ms(), 1)}
                _payload["response"] = _response
                _payload["tool"] = _tool
                ctx.add_exchange(message_or_error, _payload["response"], "RAG_ANSWER", "rag_answer", 0.85)
                _save_chat_to_sqlite(message_or_error, _payload)
                return jsonify(_payload)
            if _forced_doc and not _rag.get("evidence_found"):
                _payload = {"response": "I searched your documents but did not find passages that answer that question. Try rephrasing with words from the document, or upload the document first with /api/rag/ingest.",
                            "intent": "RAG_NO_EVIDENCE", "rule_id": "rag_no_evidence", "confidence": 0.90,
                            "confidence_label": "HIGH", "tool": "Pinecone RAG" if _rag.get("backend") == "pinecone" else "SQLite fallback RAG",
                            "action": "No relevant document evidence", "sources": [],
                            "backend": _rag.get("backend"), "processing_ms": round(timer.elapsed_ms(), 1)}
                ctx.add_exchange(message_or_error, _payload["response"], "RAG_NO_EVIDENCE", "rag_no_evidence", 0.90)
                _save_chat_to_sqlite(message_or_error, _payload)
                return jsonify(_payload)
    # In-memory cache, before the (slow) web search: only for
    # questions with no rule and no maths, so a rule/math answer is
    # never replaced by a cached web answer. Personal queries are
    # never cached. This cache lives only in this process (5 min).
    normalized = normalize_query(message_or_error)
    if (normalized and not looks_personal(message_or_error)
            and not looks_like_math(message_or_error)):
        try:
            rule_for_cache = match_rule(message_or_error)
        except Exception:
            rule_for_cache = {"intent": "UNKNOWN"}
        if rule_for_cache.get("intent") == "UNKNOWN":
            hit = cached_answer_lookup(normalized)
            if hit:
                ctx.add_exchange(message_or_error, hit["summary"],
                                 "WEB_SEARCH", "cached_web_answer", 0.85)
                payload = {
                    "response": hit["summary"], "intent": "WEB_SEARCH",
                    "rule_id": "cached_web_answer", "confidence": 0.85,
                    "confidence_label": "HIGH", "tool": "Cached web answer",
                    "action": "Answer from cache", "sources": hit.get("sources", []),
                    "cached": True, "processing_ms": round(timer.elapsed_ms(), 1),
                }
                return jsonify(payload)
    try:
        payload = process_message(message_or_error, ctx, processing_ms=0.0)
        payload["processing_ms"] = round(timer.elapsed_ms(), 1)
        # No extra HF call here: the router already tries the LLM (via
        # ModelManager) for UNKNOWN / low-confidence answers, labelled
        # as an AI-generated answer. Calling it again here would double
        # the API cost and could double the answer.
        log_chat(logger, payload["intent"], payload["rule_id"],
                 payload["confidence"], payload["tool"], payload["processing_ms"])
        # In-memory cache, after a successful web answer: store it
        # for the next identical question (summary only, max 5 sources).
        if (payload.get("intent") == "WEB_SEARCH"
                and float(payload.get("confidence") or 0) >= 0.85
                and not looks_personal(message_or_error)):
            try:
                cached_answer_store(
                    normalized, payload.get("response", ""),
                    payload.get("search_results", []),
                    tool=payload.get("tool", "web"))
            except Exception:
                pass
        _save_chat_to_sqlite(message_or_error, payload)
        return jsonify(payload)
    except Exception as exc:
        logger.exception("Chat failed")
        return jsonify({"error": "Sorry, something went wrong on the server. Please try again."}), 500


@app.route("/api/calculate", methods=["POST"])
def calculate():
    data = request.get_json(silent=True) or {}
    ok, message_or_error = validate_message(data.get("expression") or data.get("message"))
    if not ok:
        return jsonify({"error": message_or_error}), 400
    result = math_service.calculate(message_or_error)
    if not result.get("success"):
        return jsonify({"error": result.get("error", "Could not parse that calculation.")}), 400
    return jsonify({"intent": "MATHEMATICS", "confidence": 0.99, **result})


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(silent=True) or {}
    ok, query_or_error = validate_message(data.get("query") or data.get("message"))
    if not ok:
        return jsonify({"error": query_or_error}), 400
    results = search_service.search(query_or_error)
    return jsonify({"query": query_or_error, "results": results, "source": "DuckDuckGo"})


@app.route("/api/generate-pdf", methods=["POST"])
def generate_pdf_route():
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    title = (data.get("title") or "Chatbot Export").strip()
    if not content:
        # Fall back to the whole conversation
        ctx = _context()
        content = "\n\n".join(f"{i['role'].title()}: {i['text']}" for i in ctx.conversation_history)
    if not content:
        return jsonify({"error": "Nothing to export yet. Send a message first."}), 400
    if len(content) > MAX_PDF_TEXT_LENGTH:
        return jsonify({"error": "Content is too long for one PDF."}), 400
    try:
        _path, filename = pdf_service.generate_pdf(content, title)
        return jsonify({"pdf_file": filename, "pdf_url": f"/api/download/{filename}"})
    except Exception:
        logger.exception("PDF generation failed")
        return jsonify({"error": "PDF generation failed. Please try again."}), 500


@app.route("/api/download/<filename>")
def download(filename):
    # send_from_directory + a generated/ dir prevents path traversal;
    # only files actually in generated/ can be served.
    safe = os.path.basename(filename)
    return send_from_directory(os.path.join(BASE_DIR, "generated"), safe, as_attachment=True)



@app.route("/api/upload", methods=["POST"])
def upload():
    """Upload a supported document for preview and (when signed in) RAG."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use the paperclip button and choose a PDF, DOCX or XLSX file."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        return jsonify({"error": "Supported files: PDF, DOCX, XLSX, TXT, Markdown, HTML and CSV."}), 400
    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = sanitize_filename(f.filename)
    path = safe_join(upload_dir, safe_name)
    f.save(path)
    try:
        text, info = extract_text(path, ext)
    except Exception as exc:
        logger.exception("Document parse failed")
        return jsonify({"error": f"Could not read that file: {exc}"}), 400
    ctx = _context()
    ctx.last_document = safe_name
    ctx.last_document_text = text
    ctx.last_document_info = info
    preview = summarize_doc(text, max_sentences=3) if text.strip() else "No readable text found."
    rag_result = None
    # Preserve anonymous preview-only uploads. Signed-in users receive the
    # requested end-to-end ingest automatically, using their session identity.
    if _session_user_id() and text.strip():
        if not _allow_costly_request("rag-ingest"):
            return jsonify({"error": "Too many document requests. Please wait a minute and try again."}), 429
        try:
            rag_result = _rag_service().ingest_text(_session_user_id(), safe_name, text, source_type=ext)
        except Exception as exc:
            logger.warning("Upload parsed but RAG indexing failed: %s", str(exc)[:300])
            rag_result = {"status": "failed", "error": str(exc)[:300]}
    message = (f"Uploaded {safe_name}. You can now say 'Summarize my document', "
               "'Make notes from my document', or ask a question about it.")
    if rag_result and rag_result.get("status") in ("indexed", "unchanged"):
        message += " It is also available for document-grounded answers."
    return jsonify({"filename": safe_name, "info": info, "chars": len(text),
                    "message": message, "summary_preview": preview, "rag": rag_result})


@app.route("/api/documents")
def documents():
    ctx = _context()
    return jsonify({"last_document": ctx.last_document, "info": ctx.last_document_info})


@app.route("/api/generate-file", methods=["POST"])
def generate_file():
    """Generate a file in the format the user specifies: pptx, docx, xlsx or pdf."""
    data = request.get_json(silent=True) or {}
    file_type = (data.get("type") or "").lower()
    title = (data.get("title") or data.get("topic") or "Shibu AI Document").strip()
    content = (data.get("content") or "").strip()
    if file_type not in ("pptx", "docx", "xlsx", "pdf"):
        return jsonify({"error": "type must be one of: pptx, docx, xlsx, pdf."}), 400
    if not content and file_type != "xlsx":
        # Fall back to a rule answer for the topic, if there is one.
        from chatbot.rule_engine import match_rule
        rule = match_rule(title)
        content = rule["response"] if rule["intent"] != "UNKNOWN" else f"Notes about {title}."
    try:
        out_dir = os.path.join(BASE_DIR, "generated")
        if file_type == "pptx":
            points = [p.strip() for p in content.split("\n") if p.strip()] or [content]
            _p, filename = generate_pptx(title, points[:10], out_dir)
        elif file_type == "docx":
            _p, filename = generate_docx(title, content, out_dir)
        elif file_type == "xlsx":
            _p, filename = generate_xlsx(title, ["Item", "Value"], [["Example A", 10], ["Example B", 25], ["Example C", 15]], out_dir)
        else:
            _p, filename = pdf_service.generate_pdf(content, title)
        return jsonify({"file": filename, "file_url": f"/api/download/{filename}", "type": file_type})
    except Exception:
        logger.exception("File generation failed")
        return jsonify({"error": "File generation failed. Please try again."}), 500


@app.route("/api/preferences", methods=["GET", "POST"])
def preferences():
    ctx = _context()
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if data.get("tone") in ("normal", "simple", "concise", "detailed"):
            ctx.preferences["tone"] = data["tone"]
        if data.get("level") in ("normal", "beginner", "expert"):
            ctx.preferences["level"] = data["level"]
    return jsonify(ctx.preferences)


@app.route("/api/history")
def history():
    return jsonify(_context().to_dict())


@app.route("/api/clear", methods=["POST"])
def clear():
    contexts.clear(_session_id())
    # Do not restore the cleared history on the next chat in this
    # session; explicit memories are kept (see ConversationContext.clear).
    session["context_restored_v1"] = True
    return jsonify({"status": "cleared"})


@app.route("/login")
def login_page():
    """Dedicated login page (clean, black, same theme as chat).
    The actual sign-in is /auth/google; this page only explains
    what signing in does and does not do, and links to it."""
    return render_template("login.html")


@app.route("/auth/google")
def google_login():
    cfg = google_config()
    if not cfg["client_id"]:
        return jsonify({"error": "Google login is not configured. Set GOOGLE_CLIENT_ID in .env."}), 503
    state = new_state()
    session["google_oauth_state"] = state
    return redirect(build_authorization_url(cfg["client_id"], cfg["redirect_uri"], state))


@app.route("/auth/callback")
def google_callback():
    if request.args.get("state") != session.get("google_oauth_state"):
        return jsonify({"error": "Invalid login state. Please try signing in again."}), 400
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Google login was not completed."}), 400
    cfg = google_config()
    try:
        tokens = exchange_code(cfg["client_id"], cfg["client_secret"], code, cfg["redirect_uri"])
        claims = verify_id_token(tokens["id_token"], cfg["client_id"])
    except Exception as exc:
        logger.exception("Google login failed")
        return jsonify({"error": str(exc)}), 400
    user_id = google_user_id(claims["sub"])
    session["google_user_id"] = user_id
    session["google_display_name"] = claims.get("name", "")
    session["google_email"] = claims.get("email", "")
    session.pop("google_oauth_state", None)
    # Persistent SQLite user (profile + history), scoped by google#sub.
    try:
        _sqlite_store().get_or_create_user(user_id, display_name=claims.get("name", ""))
    except Exception:
        pass
    # First-time flow: no avatar yet -> avatar selection. Returning
    # users with a saved avatar (and completed onboarding) go straight
    # to chat, and are never asked to choose again.
    profile = _profile()
    if not profile["avatar_id"]:
        return redirect("/avatar")
    return redirect("/")


@app.route("/auth/logout", methods=["POST"])
def google_logout():
    session.pop("google_user_id", None)
    session.pop("google_display_name", None)
    session.pop("google_email", None)
    session.pop("sqlite_conversation_id", None)
    session.pop("context_restored_v1", None)
    session.pop("profile_avatar_id", None)
    session.pop("onboarding_completed", None)
    session.pop("onboarding_stage", None)
    return jsonify({"status": "logged out"})


@app.route("/api/me")
def me():
    return jsonify(_profile())


@app.route("/avatar")
def avatar_page():
    """Avatar selection, also used later from the chat header to
    change an avatar. Login is required: the avatar belongs to the
    authenticated user_id in the session, never to a request body."""
    if not _session_user_id():
        return redirect("/login")
    return render_template("avatar_select.html", avatars=AVATARS)


@app.route("/api/avatars")
def avatars_list():
    return jsonify({"avatars": AVATARS})


@app.route("/api/profile")
def profile_api():
    return jsonify(_profile())


@app.route("/api/profile/avatar", methods=["POST"])
def set_avatar():
    """Save the selected avatar to the logged-in user's profile.
    Anonymous requests are rejected: there is no profile to save to."""
    profile = _profile()
    if not profile["logged_in"]:
        return jsonify({"error": "Please sign in first."}), 401
    data = request.get_json(silent=True) or {}
    avatar_id = data.get("avatar_id")
    if avatar_id not in AVATAR_IDS:
        return jsonify({"error": "Unknown avatar. Choose one of: " + ", ".join(sorted(AVATAR_IDS))}), 400
    # Changing an avatar later must not restart the welcome flow for
    # a user who already completed onboarding.
    stage = None if profile["onboarding_completed"] else "ask_name"
    saved = _save_profile(avatar_id=avatar_id, onboarding_stage=stage)
    return jsonify(saved)


@app.route("/api/onboarding/status")
def onboarding_status():
    """What the chat UI should show on load. Returning users with a
    completed onboarding get needs_welcome False and skip straight
    to their previous chat; first-time users get the welcome question."""
    profile = _profile()
    if not profile["logged_in"]:
        return jsonify({"logged_in": False, "needs_welcome": False})
    if profile["onboarding_completed"]:
        return jsonify({"logged_in": True, "needs_welcome": False,
                        "profile": profile})
    if profile["onboarding_stage"] == "need_avatar":
        return jsonify({"logged_in": True, "needs_welcome": False,
                        "needs_avatar": True, "profile": profile})
    return jsonify({"logged_in": True, "needs_welcome": True,
                    "welcome_message": "Hi! Welcome! What should I call you?",
                    "quick_actions": [], "profile": profile})



@app.route("/api/rag/status")
def rag_status():
    return jsonify(_rag_service().status(user_id=_session_user_id()))


@app.route("/api/rag/ingest", methods=["POST"])
def rag_ingest():
    """Ingest a document for RAG. Login required; the document
    belongs to the session user_id only. Accepts a file upload
    (PDF/DOCX/XLSX/TXT/MD/HTML/CSV, reusing document extraction)
    or JSON {"text": ..., "source_name": ...}. On failure the
    document is marked failed in SQLite so it can be re-indexed."""
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    if not _allow_costly_request("rag-ingest"):
        return jsonify({"error": "Too many document requests. Please wait a minute and try again."}), 429
    try:
        if "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
                return jsonify({"error": "Supported files: PDF, DOCX, XLSX, TXT, Markdown, HTML and CSV."}), 400
            upload_dir = os.path.join(BASE_DIR, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            safe_name = sanitize_filename(f.filename)
            path = safe_join(upload_dir, safe_name)
            f.save(path)
            result = _rag_service().ingest_file(user_id, path, safe_name)
            return jsonify(result)
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        source_name = sanitize_filename((data.get("source_name") or "pasted-text.txt").strip())
        if not text:
            return jsonify({"error": "Provide a file or JSON text + source_name."}), 400
        return jsonify(_rag_service().ingest_text(user_id, source_name, text))
    except Exception as exc:
        logger.exception("RAG ingest failed")
        return jsonify({"error": str(exc)[:300]}), 400


@app.route("/api/rag/search", methods=["POST"])
def rag_search():
    """Semantic search over the caller's own documents only.
    Login required; the namespace/user filter is the session id,
    never a body-supplied user_id."""
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    data = request.get_json(silent=True) or {}
    ok, query = validate_message(data.get("q") or data.get("query"))
    if not ok:
        return jsonify({"error": query}), 400
    if not _allow_costly_request("rag-search"):
        return jsonify({"error": "Too many search requests. Please wait a minute and try again."}), 429
    try:
        top_k = int(data.get("top_k", 5))
    except Exception:
        top_k = 5
    try:
        raw_filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}
        filters = {key: raw_filters[key] for key in ("document_id", "source_name") if raw_filters.get(key)}
        hits = _rag_service().search(user_id, query, top_k=top_k, filters=filters)
    except Exception as exc:
        return jsonify({"error": str(exc)[:300], "backend": get_rag_config()["backend"]}), 502
    return jsonify({"query": query, "results": hits, "filters": filters,
                    "backend": get_rag_config()["backend"]})


@app.route("/api/rag/metrics")
def rag_metrics():
    """Aggregate, process-local retrieval observability (no query/document data)."""
    if not _session_user_id():
        return jsonify({"error": "Please sign in first."}), 401
    return jsonify(retrieval_metrics.snapshot())


@app.route("/api/rag/documents")
def rag_documents():
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    return jsonify({"documents": _sqlite_store().list_documents(user_id)})


@app.route("/api/rag/documents/<document_id>", methods=["DELETE"])
def rag_delete_document(document_id):
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    try:
        deleted = _rag_service().delete_document(user_id, document_id)
    except Exception as exc:
        return jsonify({"error": str(exc)[:300]}), 502
    if not deleted:
        return jsonify({"error": "Document not found."}), 404
    return jsonify({"status": "deleted", "document_id": document_id})


@app.route("/api/rag/documents/<document_id>/reindex", methods=["POST"])
def rag_reindex_document(document_id):
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    try:
        return jsonify(_rag_service().reindex_document(user_id, document_id))
    except Exception as exc:
        return jsonify({"error": str(exc)[:300]}), 400


@app.route("/api/conversations")
def list_conversations():
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    return jsonify({"conversations": _sqlite_store().list_conversations(user_id)})


@app.route("/api/conversations/<conversation_id>/messages")
def list_conversation_messages(conversation_id):
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    return jsonify({"messages": _sqlite_store().list_messages(user_id, conversation_id)})


@app.route("/api/history/migrate", methods=["POST"])
def migrate_history():
    """EXPLICIT migration only: localStorage/session history is
    never migrated automatically. Send {"confirm": true,
    "exchanges": [{"role": "user"|"bot", "text": ...}]}. Without
    confirm=true, nothing is migrated."""
    user_id = _session_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in first."}), 401
    data = request.get_json(silent=True) or {}
    if data.get("confirm") is not True:
        return jsonify({"migrated": 0, "conversation_id": None,
                        "note": "Send confirm=true to migrate explicitly."})
    result = _sqlite_store().migrate_session_history(user_id, data.get("exchanges") or [], confirm=True)
    return jsonify(result)


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Request is too large."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")  # 0.0.0.0 works locally AND on a cloud host; 127.0.0.1 would not accept cloud traffic
    debug = os.environ.get("FLASK_DEBUG", "").strip() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug)
