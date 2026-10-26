"""Rule-Based Intelligent Chatbot — Flask entry point.

Run:  python app.py
Then open: http://127.0.0.1:5000
"""
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory, session
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
from services.document_service import extract_text, summarize as summarize_doc, answer_from_document
from services.file_generation_service import generate_pptx, generate_docx, generate_xlsx
from utils.security import sanitize_filename
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "rule_engine": True,
        "math_service": True,
        "pdf_service": True,
        "search_service": True,
        "huggingface": hf.enabled,
        "llm": ModelManager().status(),
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
    try:
        payload = process_message(message_or_error, ctx, processing_ms=0.0)
        payload["processing_ms"] = round(timer.elapsed_ms(), 1)
        # No extra HF call here: the router already tries the LLM (via
        # ModelManager) for UNKNOWN / low-confidence answers, labelled
        # as an AI-generated answer. Calling it again here would double
        # the API cost and could double the answer.
        log_chat(logger, payload["intent"], payload["rule_id"],
                 payload["confidence"], payload["tool"], payload["processing_ms"])
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
    """Upload a PDF, DOCX or XLSX for document intelligence."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use the paperclip button and choose a PDF, DOCX or XLSX file."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".pdf", ".docx", ".xlsx"):
        return jsonify({"error": "Only PDF, DOCX and XLSX files are supported."}), 400
    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = sanitize_filename(f.filename)
    path = os.path.join(upload_dir, safe_name)
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
    return jsonify({"filename": safe_name, "info": info, "chars": len(text),
                    "message": f"Uploaded {safe_name}. You can now say 'Summarize my document', 'Make notes from my document', or ask a question about it.",
                    "summary_preview": preview})


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
    return jsonify({"status": "cleared"})


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
